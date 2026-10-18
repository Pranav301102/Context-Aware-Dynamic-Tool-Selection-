"""OpenTelemetry export of CATS trace spans (workbook Figure 3 observability layer, NFR8).

CATS records its own light spans in-process (``ledger.Tracer`` -> ``Selection.spans``), so the
pipeline itself has no tracing dependency and its latency numbers stay comparable with the
benchmarks. This module turns those records into OpenTelemetry spans after the fact:

    cats.select                      one root span per request
      ├─ cats.stage.analyze          one child span per recorded stage
      ├─ cats.stage.route+retrieve
      └─ cats.stage.assemble

Stage spans are laid end to end from the request's start time, using each stage's measured
latency. Attributes (``cats.*``): stage, latency_ms, tokens_in/out, route (selected branches,
conformal set, threshold, fallback, top branch and its probability), k, tool ids, tool and
router tokens, and anything else the stage put in its ``meta``.

Dispatcher runs (``dispatcher.RunResult``) become a ``cats.dispatch`` span with the run's
turns, escalations, calls and cost ledger, its initial selection as a child tree, and one
``tool_call`` event per call.

Exporters (``make_provider``):
  console     ConsoleSpanExporter, one JSON document per span on stdout
  otlp        OTLP/HTTP (default endpoint from OTEL_EXPORTER_OTLP_* or localhost:4318)
  otlp-grpc   OTLP/gRPC (localhost:4317)
  langfuse    Langfuse's OTLP endpoint; uses the Langfuse SDK's span processor when the SDK is
              installed (``pip install langfuse``), else a plain OTLP/HTTP exporter with the same
              Basic-auth header. Keys from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY,
              host from LANGFUSE_HOST (or LANGFUSE_BASE_URL).
  none        spans are created and dropped

    provider = make_provider("console")
    tx = TraceExporter(provider)
    tx.export_selection(cats.select(query), query=query)
    tx.shutdown()

Needs ``pip install opentelemetry-sdk opentelemetry-exporter-otlp`` (extra ``tracing``).
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Status, StatusCode

from .types import Selection, TraceSpan

if TYPE_CHECKING:
    from .dispatcher import RunResult

SCOPE = "cats"  # instrumentation scope of every span this module creates
EXPORTERS = ("none", "console", "otlp", "otlp-grpc", "langfuse")
_META_KEYS = {"selected": "cats.route.selected", "fallback": "cats.route.fallback", "k": "cats.k",
              "parts": "cats.parts"}
_MAX_STR = 2000


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cats")
    except PackageNotFoundError:
        return "0"


# ------------------------------------------------------------------ attribute coercion
def attr_value(v: Any) -> Any:
    """Coerce a value into an OpenTelemetry attribute value (None means: leave it out).

    OTel accepts str, bool, int, float and homogeneous sequences of those; everything else
    is stored as compact JSON.
    """
    if v is None:
        return None
    if isinstance(v, bool | int | float):
        return v
    if isinstance(v, str):
        return v[:_MAX_STR]
    if isinstance(v, Sequence) and not isinstance(v, bytes | bytearray):
        items = list(v)
        for kind in (bool, int, float, str):
            if items and all(isinstance(x, kind) and (kind is bool or not isinstance(x, bool)) for x in items):
                return [x[:_MAX_STR] if isinstance(x, str) else x for x in items]
        if not items:
            return []
    try:
        return json.dumps(v, default=str, separators=(",", ":"))[:_MAX_STR]
    except (TypeError, ValueError):
        return str(v)[:_MAX_STR]


def _set_attrs(span: trace.Span, attrs: Mapping[str, Any]) -> None:
    for k, v in attrs.items():
        val = attr_value(v)
        if val is not None:
            span.set_attribute(k, val)


def stage_attributes(sp: TraceSpan) -> dict[str, Any]:
    """OTel attributes for one recorded CATS stage."""
    attrs: dict[str, Any] = {"cats.stage": sp.stage, "cats.latency_ms": float(sp.latency_ms),
                             "cats.tokens_in": int(sp.tokens_in), "cats.tokens_out": int(sp.tokens_out)}
    for k, v in sp.meta.items():
        attrs[_META_KEYS.get(k, f"cats.meta.{k}")] = v
    return attrs


def selection_attributes(sel: Selection, *, query: str | None = None,
                         static_tokens: int | None = None) -> dict[str, Any]:
    """OTel attributes for a whole selection: route, k, tools, tokens and latency."""
    attrs: dict[str, Any] = {
        "cats.strategy": sel.strategy, "cats.k": len(sel.tool_ids), "cats.tool_ids": list(sel.tool_ids),
        "cats.tool_tokens": int(sel.tool_tokens), "cats.overhead_tokens": int(sel.overhead_tokens),
        "cats.router_calls": int(sel.router_calls), "cats.router_kind": sel.router_kind or "none",
        "cats.latency_ms": float(sum(sp.latency_ms for sp in sel.spans)),
    }
    r = sel.route
    if r is not None:
        attrs.update({"cats.route.selected": list(r.selected), "cats.route.conformal_set": list(r.conformal_set),
                      "cats.route.fallback": bool(r.fallback), "cats.route.escalations": int(r.escalations),
                      "cats.route.n_branches": len(r.conformal_set or r.selected)})
        if r.threshold == r.threshold and abs(r.threshold) != float("inf"):
            attrs["cats.route.threshold"] = float(r.threshold)
        if r.branch_scores:
            top = max(r.branch_scores, key=lambda b: r.branch_scores[b])
            attrs["cats.route.top_branch"] = top
            attrs["cats.route.top_p"] = float(r.branch_scores[top])
    if query is not None:
        attrs["cats.query"] = query
    if static_tokens is not None:
        attrs["cats.static_tokens"] = int(static_tokens)
        attrs["cats.tokens_saved"] = int(static_tokens - sel.tool_tokens)
    return attrs


# --------------------------------------------------------------------------- providers
def _is_cats_span(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope
    return scope is not None and (scope.name == SCOPE or scope.name.startswith(SCOPE + "."))


def langfuse_processor(public_key: str | None = None, secret_key: str | None = None, host: str | None = None,
                       *, span_exporter: SpanExporter | None = None) -> SpanProcessor:
    """A span processor that ships CATS spans to Langfuse (self-hosted or cloud).

    With the Langfuse SDK (v3 or later) installed, its ``LangfuseSpanProcessor`` is used, told to
    keep CATS spans (by default it only forwards LLM-framework spans). Without the SDK, spans go
    to the same documented endpoint, ``<host>/api/public/otel/v1/traces``, over OTLP/HTTP with
    Basic auth. ``span_exporter`` replaces the network exporter (tests).
    """
    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = (host or os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
            or "http://localhost:3000").rstrip("/")
    if not (public_key and secret_key) and span_exporter is None:
        raise ValueError("Langfuse export needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    try:
        from langfuse._client.span_processor import LangfuseSpanProcessor
    except ImportError:  # SDK not installed (or a layout this code does not know)
        LangfuseSpanProcessor = None
    if LangfuseSpanProcessor is not None:
        params = inspect.signature(LangfuseSpanProcessor.__init__).parameters
        kw: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
        kw["base_url" if "base_url" in params else "host"] = host
        if "should_export_span" in params:
            kw["should_export_span"] = _is_cats_span
        if span_exporter is not None and "span_exporter" in params:
            kw["span_exporter"] = span_exporter
        return LangfuseSpanProcessor(**kw)
    if span_exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
        span_exporter = OTLPSpanExporter(endpoint=f"{host}/api/public/otel/v1/traces",
                                         headers={"Authorization": f"Basic {auth}"})
    return BatchSpanProcessor(span_exporter)


def make_provider(exporter: str = "console", *, endpoint: str | None = None,
                  headers: Mapping[str, str] | None = None, service_name: str = "cats",
                  resource: Mapping[str, Any] | None = None, span_exporter: SpanExporter | None = None,
                  simple: bool | None = None) -> TracerProvider:
    """A TracerProvider with one exporter (see the module docstring for the kinds).

    ``span_exporter`` adds an exporter object directly (e.g. an ``InMemorySpanExporter``) and
    ``simple`` forces synchronous export (default: synchronous for console and in-memory
    exporters, batched for network exporters). The provider is not installed globally.
    """
    if exporter not in EXPORTERS:
        raise ValueError(f"unknown exporter {exporter!r}; choose one of {', '.join(EXPORTERS)}")
    res = Resource.create({"service.name": service_name, "service.version": _version(), **(resource or {})})
    provider = TracerProvider(resource=res)
    exp: SpanExporter | None = span_exporter
    batched = False
    if exp is None:
        if exporter == "console":
            exp = ConsoleSpanExporter(service_name=service_name, out=sys.stdout)  # current stdout, not import-time
        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exp, batched = OTLPSpanExporter(endpoint=endpoint, headers=dict(headers or {}) or None), True
        elif exporter == "otlp-grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter

            exp, batched = GrpcExporter(endpoint=endpoint, headers=dict(headers or {}) or None), True
        elif exporter == "langfuse":
            provider.add_span_processor(langfuse_processor())
            return provider
    if exp is not None:
        sync = simple if simple is not None else not batched
        provider.add_span_processor(SimpleSpanProcessor(exp) if sync else BatchSpanProcessor(exp))
    return provider


# ---------------------------------------------------------------------------- export
def _hex_trace_id(span: trace.Span) -> str:
    return trace.format_trace_id(span.get_span_context().trace_id)


def export_selection(tracer: trace.Tracer, sel: Selection, *, query: str | None = None,
                     start_ns: int | None = None, name: str = "cats.select",
                     attributes: Mapping[str, Any] | None = None, static_tokens: int | None = None,
                     context: Any = None) -> str:
    """Emit one root span for ``sel`` and one child span per recorded stage; returns the trace id.

    ``start_ns`` is the request's start (``time.time_ns()``); without it the spans are placed so
    that the last stage ends now. ``context`` parents the root span (e.g. a dispatcher span).
    """
    total_ns = int(sum(sp.latency_ms for sp in sel.spans) * 1e6)
    t0 = start_ns if start_ns is not None else time.time_ns() - total_ns
    root = tracer.start_span(name, context=context, start_time=t0, kind=trace.SpanKind.INTERNAL)
    _set_attrs(root, selection_attributes(sel, query=query, static_tokens=static_tokens))
    _set_attrs(root, attributes or {})
    ctx = trace.set_span_in_context(root)
    t = t0
    for sp in sel.spans:
        dur = max(0, int(sp.latency_ms * 1e6))
        child = tracer.start_span(f"cats.stage.{sp.stage}", context=ctx, start_time=t, kind=trace.SpanKind.INTERNAL)
        _set_attrs(child, stage_attributes(sp))
        child.set_status(Status(StatusCode.OK))
        child.end(end_time=t + dur)
        t += dur
    root.set_status(Status(StatusCode.OK))
    root.end(end_time=max(t, t0 + total_ns))
    return _hex_trace_id(root)


def export_selections(tracer: trace.Tracer, selections: Iterable[Selection], **kw: Any) -> list[str]:
    """``export_selection`` for many selections (e.g. a benchmark's), one trace each."""
    return [export_selection(tracer, s, **kw) for s in selections]


def run_attributes(run: RunResult) -> dict[str, Any]:
    """OTel attributes for one dispatcher run: outcome, turns, escalations and the cost ledger."""
    c = run.cost
    return {
        "cats.task_id": run.task_id, "cats.success": bool(run.success), "cats.turns": int(run.turns),
        "cats.escalations": int(run.escalations), "cats.calls": list(run.calls),
        "cats.invalid_calls": list(run.invalid_calls), "cats.loaded_k": len(run.loaded), "cats.usd": float(run.usd),
        "cats.cost.input_tokens": int(c.input_tokens), "cats.cost.cache_write": int(c.cache_write),
        "cats.cost.cache_read": int(c.cache_read), "cats.cost.output_tokens": int(c.output_tokens),
        "cats.cost.router_tokens": int(c.router_tokens),
    }


def export_run(tracer: trace.Tracer, run: RunResult, *, selection: Selection | None = None,
               escalations: Sequence[Selection] = (), start_ns: int | None = None, duration_ms: float | None = None,
               attributes: Mapping[str, Any] | None = None) -> str:
    """Emit a ``cats.dispatch`` span for a sub-agent run; returns the trace id.

    The run's initial ``selection`` and any ``escalations`` (``CATS.escalate`` results) become
    child trees; every tool call becomes a ``tool_call`` event. RunResult carries no per-turn
    timing, so the run span lasts ``duration_ms`` (default: the selections' own latency).
    """
    sels = ([selection] if selection is not None else []) + list(escalations)
    inner_ns = int(sum(sp.latency_ms for s in sels for sp in s.spans) * 1e6)
    dur_ns = int(duration_ms * 1e6) if duration_ms is not None else inner_ns
    t0 = start_ns if start_ns is not None else time.time_ns() - dur_ns
    root = tracer.start_span("cats.dispatch", start_time=t0, kind=trace.SpanKind.INTERNAL)
    _set_attrs(root, run_attributes(run))
    _set_attrs(root, attributes or {})
    ctx = trace.set_span_in_context(root)
    t = t0
    for i, s in enumerate(sels):
        name = "cats.select" if (selection is not None and i == 0) else "cats.escalate"
        export_selection(tracer, s, start_ns=t, name=name, context=ctx)
        t += int(sum(sp.latency_ms for sp in s.spans) * 1e6)
    for tool in run.calls:
        root.add_event("tool_call", {"cats.tool": tool, "cats.valid": True})
    for tool in run.invalid_calls:
        root.add_event("tool_call", {"cats.tool": tool, "cats.valid": False})
    root.set_status(Status(StatusCode.OK if run.success else StatusCode.ERROR))
    root.end(end_time=max(t, t0 + dur_ns))
    return _hex_trace_id(root)


class TraceExporter:
    """A tracer bound to a provider, with the export helpers as methods.

    Used by the REST gateway (``cats.api``): every served selection and escalation becomes a
    trace. ``exporter`` / ``endpoint`` are passed to ``make_provider`` when no provider is given.
    """

    def __init__(self, provider: TracerProvider | None = None, *, exporter: str = "console",
                 endpoint: str | None = None, service_name: str = "cats"):
        self.provider = provider or make_provider(exporter, endpoint=endpoint, service_name=service_name)
        self.tracer = self.provider.get_tracer(SCOPE, _version())

    def export_selection(self, sel: Selection, **kw: Any) -> str:
        return export_selection(self.tracer, sel, **kw)

    def export_run(self, run: RunResult, **kw: Any) -> str:
        return export_run(self.tracer, run, **kw)

    def force_flush(self, timeout_ms: int = 5000) -> bool:
        return self.provider.force_flush(timeout_ms)

    def shutdown(self) -> None:
        self.provider.shutdown()


def main(argv: list[str] | None = None) -> None:
    """``python -m cats.tracing "query" --exporter console``: build CATS, select, export one trace."""
    import argparse
    from pathlib import Path

    from .build import build_system

    p = argparse.ArgumentParser(prog="cats-tracing", description=main.__doc__)
    p.add_argument("query")
    p.add_argument("--exporter", choices=EXPORTERS, default="console")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--enrich", default=None)
    a = p.parse_args(argv)
    s = build_system(enrichment=Path(a.enrich) if a.enrich else None)
    tx = TraceExporter(exporter=a.exporter, endpoint=a.endpoint)
    t0 = time.time_ns()
    sel = s.cats.select(a.query)
    print("trace", tx.export_selection(sel, query=a.query, start_ns=t0, static_tokens=s.registry.total_tokens()))
    tx.shutdown()


if __name__ == "__main__":
    main()
