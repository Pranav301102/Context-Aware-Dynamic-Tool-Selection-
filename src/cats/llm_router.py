"""LLM branch router as an extra router signal (workbook: "small LLM classifier").

The prompt lists each branch with its summary and server names; the model returns a
probability per branch. The probabilities enter the conditional-logit router as one
more feature, so the calibrated set still comes from conformal prediction.

Offline, a subagent produces the scores: ``export_batches`` writes the prompt and the
queries, the subagent writes ``{"key": ..., "p": {branch: prob}}`` lines, and
``load_scores`` reads them back. Online, ``LLMRouterClient`` asks any cats.llm backend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .registry import Registry
from .tokens import default_counter
from .types import Task, Turn


def request_key(query: str, history: list[Turn] | None = None) -> str:
    text = " || ".join([h.content for h in (history or [])] + [query]).strip().lower()
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def router_prompt(reg: Registry) -> str:
    lines = [
        "You route a user's request to the tool domains (branches) that can serve it.",
        "Return a probability for each branch that it holds a tool the request needs. A request can",
        "need several branches; probabilities need not sum to 1. Use 0.0 for clearly irrelevant branches.",
        "",
        "Branches:",
    ]
    for b in sorted(reg.branches.values(), key=lambda x: x.id):
        servers = ", ".join(reg.toolkits[t].name for t in reg.branch_toolkits(b.id))
        lines.append(f"- {b.id}: {b.summary} Servers: {servers}.")
    return "\n".join(lines)


def router_overhead_tokens(reg: Registry, query: str) -> int:
    """Input tokens of one routing call (prompt + request), billed as router overhead."""
    c = default_counter()
    return c.count(router_prompt(reg)) + c.count(query) + 30


def export_batches(reg: Registry, tasks: list[Task], out_dir: Path, n_batches: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.md").write_text(router_prompt(reg))
    paths = []
    for i in range(n_batches):
        p = out_dir / f"queries_{i}.jsonl"
        with p.open("w") as f:
            for t in tasks[i::n_batches]:
                rec = {"key": request_key(t.query, t.history), "query": t.query,
                       "history": [h.model_dump() for h in t.history]}
                f.write(json.dumps(rec) + "\n")
        paths.append(p)
    return paths


def load_scores(path: Path, reg: Registry) -> dict[str, dict[str, float]]:
    files = sorted(Path(path).glob("scores_*.jsonl")) if Path(path).is_dir() else [Path(path)]
    out = {}
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p = {b: float(rec.get("p", {}).get(b, 0.0)) for b in reg.branches}
            out[rec["key"]] = p
    return out
