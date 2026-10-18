"""Selection-level evaluation harness (FR9): same tasks, same token counter, every strategy."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from ..calibration import slot_miss
from ..registry import Registry
from ..types import Task


@dataclass
class Record:
    strategy: str
    task_id: str
    kind: str
    k: int
    tool_tokens: int
    router_tokens: int  # LLM routing overhead (0 for retrieval-only routers)
    router_calls: int  # 1 if an external router (LLM or Jev) was called
    slot_recall: float  # fraction of gold slots covered
    full_cover: bool  # every slot covered
    precision: float  # selected tools that are gold
    branch_cover: float | None  # gold branches inside the routed set (routed strategies only)
    n_branches: int | None
    fallback: bool
    latency_ms: float


def evaluate(strategy, tasks: list[Task], registry: Registry) -> list[Record]:
    out = []
    for t in tasks:
        t0 = time.perf_counter()
        sel = strategy.select(t.query, t.history)
        ms = (time.perf_counter() - t0) * 1000
        chosen = set(sel.tool_ids)
        miss = slot_miss(chosen, t.gold)
        gold = t.gold_tools()
        bc = nb = None
        fb = False
        if sel.route is not None:
            gb = set(t.branches)
            routed = sel.route.conformal_set or sel.route.selected  # the set the guarantee is about
            bc = len(gb & set(routed)) / max(1, len(gb))
            nb = len(routed)
            fb = sel.route.fallback
        out.append(Record(
            strategy=sel.strategy, task_id=t.id, kind=t.kind, k=len(chosen), tool_tokens=sel.tool_tokens,
            router_tokens=sel.overhead_tokens, router_calls=getattr(sel, "router_calls", 0),
            slot_recall=1 - miss, full_cover=miss == 0, precision=len(chosen & gold) / max(1, len(chosen)),
            branch_cover=bc, n_branches=nb, fallback=fb, latency_ms=ms,
        ))
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(records: list[Record], static_tokens: int) -> dict:
    def x(f):
        return np.array([f(r) for r in records], dtype=float)

    recall, cover, toks = x(lambda r: r.slot_recall), x(lambda r: r.full_cover), x(lambda r: r.tool_tokens)
    lat = x(lambda r: r.latency_ms)
    routed = [r for r in records if r.branch_cover is not None]
    s = {
        "strategy": records[0].strategy if records else "",
        "n": len(records),
        "slot_recall": recall.mean(),
        "slot_recall_ci": bootstrap_ci(recall),
        "full_cover": cover.mean(),
        "full_cover_ci": bootstrap_ci(cover),
        "precision": x(lambda r: r.precision).mean(),
        "k_mean": x(lambda r: r.k).mean(),
        "tool_tokens": toks.mean(),
        "token_pct_static": 100 * toks.mean() / max(1, static_tokens),
        "router_tokens": x(lambda r: r.router_tokens).mean(),
        "router_call_rate": x(lambda r: r.router_calls).mean(),
        "latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else 0.0,
        "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else 0.0,
    }
    if routed:
        s["branch_cover"] = float(np.mean([r.branch_cover == 1.0 for r in routed]))
        s["branches_mean"] = float(np.mean([r.n_branches for r in routed]))
        s["fallback_rate"] = float(np.mean([r.fallback for r in routed]))
    by_kind = defaultdict(list)
    for r in records:
        by_kind[r.kind].append(r)
    s["by_kind"] = {
        k: {"n": len(v), "slot_recall": float(np.mean([r.slot_recall for r in v])),
            "full_cover": float(np.mean([r.full_cover for r in v])), "k_mean": float(np.mean([r.k for r in v]))}
        for k, v in sorted(by_kind.items())
    }
    return s


def paired_diff(a: list[Record], b: list[Record], field: str = "full_cover", n_boot: int = 2000) -> dict:
    """Paired bootstrap of mean(a - b) over the same tasks."""
    bm = {r.task_id: getattr(r, field) for r in b}
    d = np.array([float(getattr(r, field)) - float(bm[r.task_id]) for r in a if r.task_id in bm])
    lo, hi = bootstrap_ci(d, n_boot)
    return {"mean": float(d.mean()) if len(d) else float("nan"), "ci": (lo, hi), "n": len(d)}


def format_table(summaries: list[dict]) -> str:
    cols = [("strategy", "{:<26}"), ("full_cover", "{:>9.3f}"), ("slot_recall", "{:>9.3f}"),
            ("precision", "{:>8.3f}"), ("k_mean", "{:>6.1f}"), ("tool_tokens", "{:>9.0f}"),
            ("token_pct_static", "{:>7.2f}%"), ("latency_p50_ms", "{:>7.1f}"), ("latency_p95_ms", "{:>7.1f}")]
    head = ["strategy", "cover", "recall", "prec", "k", "tokens", "% static", "p50 ms", "p95 ms"]
    widths = [26, 9, 9, 8, 6, 9, 8, 7, 7]
    lines = [" ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(head, widths)))]
    for s in summaries:
        lines.append(" ".join(fmt.format(s.get(c, float("nan"))) for c, fmt in cols))
    return "\n".join(lines)


def records_to_dicts(records: list[Record]) -> list[dict]:
    return [asdict(r) for r in records]
