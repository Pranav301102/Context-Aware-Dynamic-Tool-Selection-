"""S7 - One traced agent run, exported as OpenTelemetry spans to the console.

Shows
  The trace of one task through CATS and the sub-agent loop (workbook NFR8, observability):

    cats.task                    root: task id, success, escalations, turns, simulated $ cost
      cats.select                k, tool tokens, routed branches
        cats.analyze / cats.route+retrieve / cats.assemble   CATS's own stage timings (TraceSpan)
      gen_ai.chat (one per model turn)  OTel GenAI attributes: model, tools offered, token usage, tool calls
      cats.escalate              when the model calls request_tools

  Spans go through the OpenTelemetry SDK's ConsoleSpanExporter (one compact JSON line per span here);
  swap in an OTLP exporter to send the same spans to Jaeger, Langfuse or any collector. CATS's stage
  spans are re-emitted with their measured start and end times, so the waterfall is exact.

Needs
  ``pip install opentelemetry-sdk`` (the ``tracing`` extra). The model is cats.llm.HeuristicLLM, so no
  key; ``--llm anthropic:<model>`` etc. traces a real model the same way. Uses the 9-tool sample catalog.

  python samples/s07_traced_run.py ["request"] [--llm heuristic]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from _mini import mini_cats
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from cats.dispatcher import Dispatcher
from cats.llm import HeuristicLLM, make_llm
from cats.types import Task


def compact(span) -> str:
    d = json.loads(span.to_json())
    ms = (span.end_time - span.start_time) / 1e6
    parent = d.get("parent_id") or "-"
    return json.dumps({"name": d["name"], "span": d["context"]["span_id"][-6:], "parent": parent[-6:],
                       "ms": round(ms, 2), "attrs": d.get("attributes", {})}) + os.linesep


class TracedLLM:
    def __init__(self, llm, tracer):
        self.llm, self.tracer, self.name = llm, tracer, getattr(llm, "name", "llm")

    def complete(self, system, messages, tools):
        with self.tracer.start_as_current_span("gen_ai.chat") as sp:
            sp.set_attribute("gen_ai.operation.name", "chat")
            sp.set_attribute("gen_ai.request.model", self.name)
            sp.set_attribute("cats.tools_offered", [t["name"] for t in tools])
            r = self.llm.complete(system, messages, tools)
            sp.set_attribute("gen_ai.usage.input_tokens", r.input_tokens)
            sp.set_attribute("gen_ai.usage.output_tokens", r.output_tokens)
            sp.set_attribute("cats.tool_calls", [c.name for c in r.tool_calls])
            return r


class TracedCATS:
    """Proxy around CATS: select / escalate become spans, with CATS's stage timings as children."""

    def __init__(self, cats, tracer):
        self._cats, self._tracer = cats, tracer

    def __getattr__(self, name):
        return getattr(self._cats, name)

    def _emit_stages(self, sel, t0_ns: int) -> None:
        t = t0_ns
        for st in sel.spans:
            end = t + int(st.latency_ms * 1e6)
            sp = self._tracer.start_span(f"cats.{st.stage}", start_time=t)
            for k, v in st.meta.items():
                sp.set_attribute(f"cats.{k}", v if isinstance(v, (str, int, float, bool)) else json.dumps(v))
            if st.tokens_in:
                sp.set_attribute("cats.tokens", st.tokens_in)
            sp.end(end_time=end)
            t = end

    def select(self, query, history=None, **kw):
        with self._tracer.start_as_current_span("cats.select") as sp:
            t0 = time.time_ns()
            sel = self._cats.select(query, history, **kw)
            self._emit_stages(sel, t0)
            sp.set_attribute("cats.k", len(sel.tool_ids))
            sp.set_attribute("cats.tool_tokens", sel.tool_tokens)
            sp.set_attribute("cats.branches", list(sel.route.selected) if sel.route else [])
            return sel

    def escalate(self, capability, exclude, k=None):
        with self._tracer.start_as_current_span("cats.escalate") as sp:
            sel = self._cats.escalate(capability, exclude, k)
            sp.set_attribute("cats.capability", capability)
            sp.set_attribute("cats.added", list(sel.tool_ids))
            return sel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("request", nargs="?", default="are there flood alerts near the coast")
    ap.add_argument("--llm", default="heuristic")
    a = ap.parse_args()
    provider = TracerProvider(resource=Resource.create({"service.name": "cats-sample"}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(formatter=compact)))
    tracer = provider.get_tracer("cats.samples")
    _, cats, tasks = mini_cats()
    task = next((t for t in tasks if t.query == a.request), Task(id="adhoc", query=a.request, gold=[]))
    llm = TracedLLM(HeuristicLLM() if a.llm == "heuristic" else make_llm(a.llm), tracer)
    with tracer.start_as_current_span("cats.task") as root:
        res = Dispatcher(TracedCATS(cats, tracer), llm).run(task)
        root.set_attribute("cats.task_id", task.id)
        root.set_attribute("cats.success", res.success)
        root.set_attribute("cats.turns", res.turns)
        root.set_attribute("cats.escalations", res.escalations)
        root.set_attribute("cats.calls", res.calls)
        root.set_attribute("cats.usd", round(res.usd, 6))
    provider.shutdown()  # flushes; the provider stays local (no global set_tracer_provider needed)


if __name__ == "__main__":
    main()
