"""S10 - A LiveMCPBench-style pilot over the project's tasks: agent runs over a large tool pool, judged.

Shows
  LiveMCPBench (Mo et al., 2025) evaluates an agent that must find and call tools from a pool of ~70 MCP
  servers on real-world tasks, judged by an LLM (LiveMCPEval) for task completion. This pilot runs the
  same protocol on the project's 477-tool pool and held-out tasks, with three ways of giving the agent
  its tools:

    cats     CATS-selected tools + request_tools (at most 2 escalations)       (this project)
    search   no tools at first, only request_tools (at most 3)                (B6; LiveMCPBench's own
                                                                                MCP Copilot Agent style)
    static   every tool definition in the prompt                               (B0)

  Each task runs through cats.dispatcher (every model turn billed by the prefix-cache simulator).
  The judge checks the trajectory against the task's key points: a call to an acceptable tool for
  every gold slot and no call to a tool that was not offered (LiveMCPEval uses an LLM judge for the
  same decision; pass ``--judge <llm spec>`` to use one). Reported per mode: success, turns,
  escalations, tool tokens per prompt, input tokens and $ per task.

Needs
  Nothing beyond the repository for the default model, cats.llm.HeuristicLLM (lexical, so absolute
  success rates only exercise the harness). With a key, ``--llm anthropic:claude-sonnet-5-5``,
  ``--llm vllm:Qwen/Qwen3-8B`` (OPENAI_BASE_URL) or ``--llm litellm:<provider>/<model>`` run a real model.
  The real LiveMCPBench tasks and tool list are in data/external/livemcpbench (selection-level
  evaluation: ``python -m cats.bench.external.run livemcpbench``); running them end to end needs the
  benchmark's live MCP servers and their API keys. ~2 minutes (building and fitting CATS).

  python samples/s10_livemcpbench_pilot.py [--n 30] [--llm heuristic] [--modes cats,search,static]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

from cats.assembler import ContextAssembler
from cats.build import DATA, build_system
from cats.dispatcher import Dispatcher
from cats.llm import make_llm
from cats.types import Selection

JUDGE_SYSTEM = ("You judge whether an agent completed a task. Given the task and the tool calls it made, "
                'answer with JSON only: {"success": true|false, "reason": "..."}')


class StaticSelector:
    """B0 as a dispatcher selector: every tool, no budget, no request_tools."""

    def __init__(self, cats):
        self.cats, self.registry = cats, cats.registry
        self.assembler = ContextAssembler(cats.registry, budget=10**9, meta_tool=False)

    def select(self, query, history=None):
        ids = sorted(self.registry.tools)
        return Selection(strategy="B0_static", tool_ids=ids, tool_tokens=self.registry.total_tokens(ids),
                         scores=[1.0] * len(ids))

    def escalate(self, capability, exclude, k=None):
        return Selection(strategy="B0_static", tool_ids=[])


def llm_judge(judge, task, res) -> bool:
    calls = [{"tool": t} for t in res.calls]
    r = judge.complete(JUDGE_SYSTEM, [{"role": "user", "content": json.dumps({"task": task.query, "calls": calls})}], [])
    try:
        return bool(json.loads(r.text[r.text.index("{"): r.text.rindex("}") + 1]).get("success"))
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--llm", default="heuristic")
    ap.add_argument("--judge", default="", help="LLM spec for an LLM judge (default: rule-based key points)")
    ap.add_argument("--modes", default="cats,search,static")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    s = build_system(seed=a.seed, enrichment=DATA / "enrichment")
    print(f"built and fitted CATS on {len(s.registry.tools)} tools in {time.perf_counter() - t0:.0f}s")
    rng = random.Random(a.seed)
    by_kind = defaultdict(list)
    for t in s.test:
        by_kind[t.kind].append(t)
    tasks = []
    while len(tasks) < min(a.n, len(s.test)):  # round-robin over request kinds, like a stratified sample
        for kind in sorted(by_kind):
            if by_kind[kind] and len(tasks) < a.n:
                tasks.append(by_kind[kind].pop(rng.randrange(len(by_kind[kind]))))
    judge = make_llm(a.judge) if a.judge else None
    summary = {}
    for mode in a.modes.split(","):
        llm = make_llm(a.llm)
        if mode == "static":
            disp = Dispatcher(StaticSelector(s.cats), llm, max_escalations=0)
        else:
            disp = Dispatcher(s.cats, llm, max_escalations=2 if mode == "cats" else 3, select=mode == "cats")
        rows, kinds = [], defaultdict(list)
        for t in tasks:
            res = disp.run(t)
            ok = llm_judge(judge, t, res) if judge else res.success
            rows.append((res, ok))
            kinds[t.kind].append(ok)
        n = len(rows)
        summary[mode] = {
            "n": n, "success": sum(ok for _, ok in rows) / n,
            "turns": sum(r.turns for r, _ in rows) / n,
            "escalation_rate": sum(r.escalations > 0 for r, _ in rows) / n,
            "invalid_calls": sum(len(r.invalid_calls) for r, _ in rows),
            "tool_tokens_per_prompt": sum(s.registry.total_tokens(r.loaded) for r, _ in rows) / n,
            "input_tokens_per_task": sum(r.cost.input_tokens + r.cost.cache_read + r.cost.cache_write
                                         for r, _ in rows) / n,
            "usd_per_task": sum(r.usd for r, _ in rows) / n,
            "success_by_kind": {k: sum(v) / len(v) for k, v in sorted(kinds.items())},
        }
        print(f"{mode:7s} " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in summary[mode].items() if k != "success_by_kind"}))
    out = {"model": a.llm, "judge": a.judge or "rule:key-points", "seed": a.seed,
           "tasks": [t.id for t in tasks], "modes": summary}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print("success by kind:", json.dumps({m: v["success_by_kind"] for m, v in summary.items()}))


if __name__ == "__main__":
    main()
