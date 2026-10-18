"""Parquet metrics store (workbook Table 9: evaluation results in columnar files; NFR8).

Two kinds of tables live under one root (default ``results/parquet/``):

* **Append-only tables**, one directory each (``<root>/<table>/*.parquet``). Every append writes
  a new part file, so concurrent writers never touch each other's files.
    records      ``cats.bench.harness.Record`` rows, one per (strategy, task)
    agent_runs   sub-agent runs: ``agent_cli.score_run`` rows or ``dispatcher.RunResult``
  Every row carries its run metadata: run_id, experiment, created_at, seed, git_rev, config
  (JSON), plus any extra keyword metadata given to the append call.

* **Snapshot tables**, one file each (``<root>/<table>.parquet``), rebuilt from
  ``results/*.json`` by ``convert_results``:
    files       one row per JSON file: kind detected, sha256, rows extracted, the raw JSON text
    summaries   one row per strategy summary (bench and ablation files); CIs as _lo/_hi columns
    by_kind     summaries split by request kind
    paired      paired bootstrap differences against CATS
    validity    per-seed coverage rows of the validity runs
    scale       one row per (pool size, strategy) of the scaling runs
    cache       one row per caching policy
    agent       per-run summaries of the subagent end-to-end runs

    store = MetricsStore()                                   # results/parquet
    store.append_records(evaluate(strategy, tasks, reg), run_id="bench-s0", experiment="bench", seed=0)
    store.summarize_records(by=("run_id", "strategy"))
    convert_results()                                       # results/*.json -> results/parquet/*.parquet

Command line: ``python -m cats.metrics_store convert|tables|show|summary``.
Needs ``pip install pyarrow`` (extra ``metrics``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .build import ROOT

RESULTS = ROOT / "results"
DEFAULT_ROOT = RESULTS / "parquet"

# Columns of cats.bench.harness.Record, typed explicitly so all-null columns keep their type.
RECORD_FIELDS: list[tuple[str, pa.DataType]] = [
    ("strategy", pa.string()), ("task_id", pa.string()), ("kind", pa.string()), ("k", pa.int64()),
    ("tool_tokens", pa.int64()), ("router_tokens", pa.int64()), ("router_calls", pa.int64()),
    ("slot_recall", pa.float64()), ("full_cover", pa.bool_()), ("precision", pa.float64()),
    ("branch_cover", pa.float64()), ("n_branches", pa.int64()), ("fallback", pa.bool_()),
    ("latency_ms", pa.float64()),
]
META_FIELDS: list[tuple[str, pa.DataType]] = [
    ("run_id", pa.string()), ("experiment", pa.string()), ("created_at", pa.timestamp("us", tz="UTC")),
    ("seed", pa.int64()), ("git_rev", pa.string()), ("config", pa.string()),
]


# ------------------------------------------------------------------------ helpers
def _clean(v: Any) -> Any:
    """A value Arrow can store: NaN/inf -> None, numpy scalars -> Python, nested -> JSON text."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, Mapping) or (isinstance(v, list | tuple) and any(isinstance(x, Mapping | list | tuple)
                                                                      for x in v)):
        return json.dumps(v, default=str, sort_keys=True)
    if isinstance(v, tuple):
        return list(v)
    return v


def _column_type(values: list[Any]) -> pa.DataType | None:
    """One Arrow type for a column of mixed Python values (None when Arrow can infer it)."""
    kinds = {type(v) for v in values if v is not None}
    if not kinds:
        return pa.null()
    if kinds <= {datetime}:
        return pa.timestamp("us", tz="UTC")
    if kinds <= {bool}:
        return pa.bool_()
    if kinds <= {int}:
        return pa.int64()
    if kinds <= {int, float}:
        return pa.float64()
    if list in kinds and kinds != {list}:
        return pa.string()
    if kinds == {list}:
        return None
    return pa.string()


def table_from_rows(rows: Sequence[Mapping[str, Any]],
                    fixed: Sequence[tuple[str, pa.DataType]] = ()) -> pa.Table:
    """An Arrow table from heterogeneous dict rows: union of keys (missing -> null), one type per
    column (int+float -> float, anything mixed with strings -> string), nested values as JSON."""
    fixed_types = dict(fixed)
    cols: dict[str, None] = dict.fromkeys(fixed_types)
    clean = [{k: _clean(v) for k, v in r.items()} for r in rows]
    for r in clean:
        cols.update(dict.fromkeys(r))
    arrays, fields = [], []
    for c in cols:
        vals = [r.get(c) for r in clean]
        typ = fixed_types.get(c) or _column_type(vals)
        if typ == pa.string():
            vals = [None if v is None else (v if isinstance(v, str) else json.dumps(v, default=str)) for v in vals]
        elif typ == pa.float64():
            vals = [None if v is None else float(v) for v in vals]
        try:
            arr = pa.array(vals, type=typ) if typ is not None else pa.array(vals)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            arr = pa.array([None if v is None else json.dumps(v, default=str) for v in vals], type=pa.string())
        arrays.append(arr)
        fields.append(pa.field(c, arr.type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _record_dict(r: Any) -> dict[str, Any]:
    """A Record dataclass, pydantic model or mapping as a plain dict."""
    if isinstance(r, Mapping):
        return dict(r)
    if hasattr(r, "model_dump"):
        return r.model_dump()
    return dataclasses.asdict(r)


def run_result_row(run: Any) -> dict[str, Any]:
    """A ``dispatcher.RunResult`` as a flat row (cost ledger split into columns)."""
    c = run.cost
    return {"task": run.task_id, "success": bool(run.success), "escalations": run.escalations, "turns": run.turns,
            "invalid": len(run.invalid_calls), "calls": list(run.calls), "loaded_k": len(run.loaded),
            "usd": float(run.usd), "input_tokens": c.input_tokens + c.cache_write + c.cache_read,
            "uncached_input_tokens": c.input_tokens, "cache_write": c.cache_write, "cache_read": c.cache_read,
            "output_tokens": c.output_tokens, "router_tokens": c.router_tokens}


def _git_rev() -> str | None:
    try:
        from .manifest import git_info

        return git_info().get("rev")
    except (ImportError, OSError):
        return None


# --------------------------------------------------------------------------- store
class MetricsStore:
    """Append-only Parquet tables plus converted snapshots under one root directory."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else DEFAULT_ROOT

    # ------------------------------------------------------------------- writing
    def append(self, table: str, rows: Iterable[Mapping[str, Any]], *, run_id: str | None = None,
               experiment: str = "", seed: int | None = None, config: Any = None, git_rev: str | None = None,
               fixed: Sequence[tuple[str, pa.DataType]] = (), **meta: Any) -> Path | None:
        """Append rows (with run metadata on every row) as a new part file; returns its path."""
        rows = list(rows)
        if not rows:
            return None
        if config is not None and not isinstance(config, str):
            from .manifest import to_plain

            config = json.dumps(to_plain(config), sort_keys=True)
        run_id = run_id or f"{experiment or table}-{uuid.uuid4().hex[:8]}"
        stamp = datetime.now(UTC)
        head = {"run_id": run_id, "experiment": experiment, "created_at": stamp, "seed": seed,
                "git_rev": git_rev if git_rev is not None else _git_rev(), "config": config}
        full = [{**head, **{k: _clean(v) for k, v in meta.items()}, **r} for r in rows]
        tbl = table_from_rows(full, [*META_FIELDS, *fixed])
        d = self.root / table
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_id)[:80]
        path = d / f"{stamp.strftime('%Y%m%dT%H%M%S')}-{safe}-{uuid.uuid4().hex[:6]}.parquet"
        pq.write_table(tbl, path)
        return path

    def append_records(self, records: Iterable[Any], **kw: Any) -> Path | None:
        """Append ``cats.bench.harness.Record`` objects (or dicts with the same fields)."""
        return self.append("records", (_record_dict(r) for r in records), fixed=RECORD_FIELDS, **kw)

    def append_agent_runs(self, rows: Iterable[Any], *, mode: str = "", model: str = "", **kw: Any) -> Path | None:
        """Append agent runs: ``agent_cli.score_run(...)["rows"]`` dicts or ``RunResult`` objects."""
        flat = [run_result_row(r) if hasattr(r, "cost") and hasattr(r, "task_id") else dict(r) for r in rows]
        return self.append("agent_runs", flat, mode=mode, model=model, **kw)

    def write_snapshot(self, table: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Replace ``<root>/<table>.parquet`` with these rows."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{table}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table_from_rows(rows), tmp)
        tmp.replace(path)
        return path

    # ------------------------------------------------------------------- reading
    def tables(self) -> list[str]:
        if not self.root.exists():
            return []
        names = {p.stem for p in self.root.glob("*.parquet")}
        names |= {p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("*.parquet"))}
        return sorted(names)

    def read(self, table: str, *, where: Mapping[str, Any] | None = None,
             columns: Sequence[str] | None = None) -> pa.Table:
        """The whole table (all parts, schemas unified), filtered by equality on ``where``
        (a list value means "any of")."""
        parts: list[pa.Table] = []
        snap = self.root / f"{table}.parquet"
        if snap.exists():
            parts.append(pq.read_table(snap))
        d = self.root / table
        if d.is_dir():
            parts += [pq.read_table(p) for p in sorted(d.glob("*.parquet"))]
        if not parts:
            raise KeyError(f"no table {table!r} under {self.root}")
        tbl = pa.concat_tables(parts, promote_options="permissive") if len(parts) > 1 else parts[0]
        for col, val in (where or {}).items():
            if col not in tbl.column_names:
                return tbl.slice(0, 0)
            vals = val if isinstance(val, list | tuple | set) else [val]
            tbl = tbl.filter(pc.field(col).isin(pa.array(list(vals), type=tbl[col].type)))
        if columns:
            tbl = tbl.select([c for c in columns if c in tbl.column_names])
        return tbl

    def query(self, table: str, where: Mapping[str, Any] | None = None,
              columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        return self.read(table, where=where, columns=columns).to_pylist()

    def runs(self, table: str = "records") -> list[dict[str, Any]]:
        """One row per run_id: experiment, first created_at, seed, git_rev and row count."""
        tbl = self.read(table, columns=["run_id", "experiment", "created_at", "seed", "git_rev"])
        agg = tbl.group_by(["run_id", "experiment"]).aggregate(
            [("created_at", "min"), ("seed", "max"), ("git_rev", "max"), ("run_id", "count")])
        return sorted(agg.to_pylist(), key=lambda r: str(r.get("created_at_min")))

    def summarize_records(self, by: Sequence[str] = ("run_id", "strategy"),
                          where: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Per-group means of the evaluation records (the harness's headline metrics)."""
        tbl = self.read("records", where=where)
        if tbl.num_rows == 0:
            return []
        tbl = tbl.set_column(tbl.column_names.index("full_cover"), "full_cover",
                             pc.cast(tbl["full_cover"], pa.float64()))
        aggs = [("task_id", "count"), ("full_cover", "mean"), ("slot_recall", "mean"), ("precision", "mean"),
                ("k", "mean"), ("tool_tokens", "mean"), ("router_tokens", "mean"), ("branch_cover", "mean"),
                ("latency_ms", "tdigest", pc.TDigestOptions(q=[0.5, 0.95]))]
        rows = tbl.group_by(list(by)).aggregate(aggs).to_pylist()
        out = []
        for r in rows:
            q = r.pop("latency_ms_tdigest") or [None, None]
            r["n"] = r.pop("task_id_count")
            r["latency_p50_ms"], r["latency_p95_ms"] = q[0], q[1]
            out.append(r)
        return sorted(out, key=lambda r: tuple(str(r.get(k)) for k in by))

    def to_pandas(self, table: str, **kw: Any):
        return self.read(table, **kw).to_pandas()


# ------------------------------------------------------------------- JSON conversion
def _mapping(x: Any) -> Mapping[str, Any]:
    """``x`` if it is a mapping; for a list of mappings (e.g. one calibration report per fold), the
    first one; else empty."""
    if isinstance(x, Mapping):
        return x
    if isinstance(x, list) and x and isinstance(x[0], Mapping):
        return x[0]
    return {}


def _summary_row(s: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for k, v in s.items():
        if k == "by_kind":
            continue
        if k == "calibration" and isinstance(v, Mapping):
            row.update({f"cal_{ck}": cv for ck, cv in v.items() if not isinstance(cv, Mapping | list)})
        elif k.endswith("_ci") and isinstance(v, list | tuple) and len(v) == 2:
            row[f"{k[:-3]}_ci_lo"], row[f"{k[:-3]}_ci_hi"] = v
        elif not isinstance(v, Mapping):
            row[k] = v
    return row


def _by_kind_rows(s: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"strategy": s.get("strategy"), "kind": kind, **vals}
            for kind, vals in (s.get("by_kind") or {}).items()]


def detect_kind(obj: Any) -> str:
    """Which experiment wrote this results JSON (bench, ablate, validity, scale, cache, agent)."""
    if isinstance(obj, Mapping):
        if "summaries" in obj:
            return "bench"
        if "rows" in obj and "slot_miss_mean" in obj:
            return "validity"
        if "policies" in obj:
            return "cache"
        if "summary" in obj and "rows" in obj:
            return "agent_run"
        if "result" in obj and isinstance(obj["result"], Mapping | list):
            return "wrapped"
        return "unknown"
    if isinstance(obj, list) and obj and all(isinstance(x, Mapping) for x in obj):
        first = obj[0]
        if "strategy" in first and "full_cover" in first:
            return "ablate"
        if "size" in first and "static_tokens" in first:
            return "scale"
        if "mode" in first and "success" in first:
            return "agent"
    return "unknown"


def extract_rows(obj: Any, kind: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Table rows (by snapshot table name) from one results JSON document."""
    kind = kind or detect_kind(obj)
    out: dict[str, list[dict[str, Any]]] = {}
    if kind == "wrapped":
        return extract_rows(obj["result"])
    if kind == "bench":
        base = {"embedder": obj.get("embedder"), **{f"registry_{k}": v for k, v in _mapping(obj.get("registry")).items()}}
        cal = {f"cal_{k}": v for k, v in _mapping(obj.get("calibration")).items() if not isinstance(v, Mapping | list)}
        out["summaries"] = [{**base, **cal, **_summary_row(s)} for s in obj["summaries"]]
        out["by_kind"] = [r for s in obj["summaries"] for r in _by_kind_rows(s)]
        out["paired"] = [{"baseline": b, "mean": d.get("mean"), "ci_lo": (d.get("ci") or [None, None])[0],
                          "ci_hi": (d.get("ci") or [None, None])[1], "n": d.get("n")}
                         for b, d in _mapping(obj.get("paired_full_cover_vs_cats")).items() if isinstance(d, Mapping)]
    elif kind == "ablate":
        out["summaries"] = [_summary_row(s) for s in obj]
        out["by_kind"] = [r for s in obj for r in _by_kind_rows(s)]
    elif kind == "validity":
        extra = {k: obj.get(k) for k in ("branch_cover_mean", "branch_target", "slot_miss_mean", "slot_miss_target")}
        cfg = json.dumps(obj.get("config"), sort_keys=True, default=str)
        out["validity"] = [{**r, **extra, "config": cfg} for r in obj["rows"]]
    elif kind == "scale":
        rows = []
        for r in obj:
            head = {k: v for k, v in r.items() if not isinstance(v, Mapping)}
            cal = {f"cal_{k}": v for k, v in (r.get("cats_calibration") or {}).items()}
            for strat, m in r.items():
                if isinstance(m, Mapping) and "full_cover" in m:
                    rows.append({**head, "strategy": strat, **m, **cal})
        out["scale"] = rows
    elif kind == "cache":
        out["cache"] = [{"session_len": obj.get("session_len"), "n_sessions": obj.get("n_sessions"),
                         "policy": p, **m} for p, m in obj["policies"].items()]
    elif kind == "agent":
        out["agent"] = [dict(r) for r in obj]
    elif kind == "agent_run":
        out["agent"] = [dict(obj["summary"])]
        out["agent_rows"] = [{"run": obj["summary"].get("run"), **r} for r in obj["rows"]]
    return out


def convert_results(results_dir: Path | str | None = None, out_dir: Path | str | None = None) -> dict[str, int]:
    """Load every ``results/*.json`` into snapshot Parquet tables (``results/parquet/<table>.parquet``).

    Unknown shapes still get a ``files`` row (with the raw JSON), so nothing is lost. Returns the
    row count per table written.
    """
    src = Path(results_dir) if results_dir is not None else RESULTS
    store = MetricsStore(out_dir if out_dir is not None else src / "parquet")
    tables: dict[str, list[dict[str, Any]]] = {}
    files = []
    for path in sorted(src.glob("*.json")):
        raw = path.read_bytes()
        info: dict[str, Any] = {"file": path.name, "name": path.stem, "sha256": hashlib.sha256(raw).hexdigest(),
                                "bytes": len(raw), "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC)}
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            files.append({**info, "kind": "invalid", "rows": 0, "error": str(e), "json": raw.decode(errors="replace")})
            continue
        kind = detect_kind(obj)
        try:
            extracted, error = extract_rows(obj, kind), None
        except (KeyError, TypeError, AttributeError, IndexError, ValueError) as e:  # a shape this code misreads
            extracted, error = {}, f"{type(e).__name__}: {e}"
        for table, rows in extracted.items():
            tables.setdefault(table, []).extend({"file": path.name, "name": path.stem, **r} for r in rows)
        files.append({**info, "kind": kind, "rows": sum(len(r) for r in extracted.values()), "error": error,
                      "json": raw.decode(errors="replace")})
    counts = {"files": len(files)}
    store.write_snapshot("files", files)
    for table, rows in sorted(tables.items()):
        if rows:
            store.write_snapshot(table, rows)
            counts[table] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="cats-metrics", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help=f"store root (default {DEFAULT_ROOT})")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="results/*.json -> Parquet snapshot tables")
    c.add_argument("--results", default=None)
    sub.add_parser("tables")
    s = sub.add_parser("show")
    s.add_argument("table")
    s.add_argument("--where", nargs="*", default=[], help="col=value filters")
    s.add_argument("--columns", default="")
    s.add_argument("--limit", type=int, default=20)
    sm = sub.add_parser("summary", help="per-run, per-strategy means of the records table")
    sm.add_argument("--by", default="run_id,strategy")
    a = p.parse_args(argv)
    if a.cmd == "convert":
        counts = convert_results(a.results, a.root)
        print(json.dumps(counts))
        return
    store = MetricsStore(a.root)
    if a.cmd == "tables":
        for t in store.tables():
            print(t, store.read(t).num_rows)
    elif a.cmd == "show":
        where: dict[str, Any] = {}
        for kv in a.where:
            k, _, v = kv.partition("=")
            where[k] = int(v) if v.lstrip("-").isdigit() else v
        cols = [c for c in a.columns.split(",") if c] or None
        for r in store.query(a.table, where, cols)[: a.limit]:
            print(json.dumps({k: v for k, v in r.items() if k != "json"}, default=str))
    else:
        for r in store.summarize_records(by=tuple(a.by.split(","))):
            print(json.dumps(r, default=str))


if __name__ == "__main__":
    main()
