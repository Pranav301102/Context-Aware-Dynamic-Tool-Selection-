"""Baseline B5: an AnyTool-style hierarchy of LLM agents over the branch -> toolkit -> tool tree.

AnyTool (Du, Wei and Zhang, "AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API
Calls", ICML 2024) searches 16K RapidAPI tools with a meta-agent that picks categories, category
agents that pick tools, and tool agents that pick APIs into a candidate pool, re-activating the
hierarchy when the solver reports that the pool is insufficient. The same hierarchy on CATS's tree:

    meta agent      the request + every branch (id, summary)            -> up to ``max_branches``
    category agent  one per chosen branch: its toolkits (id, summary)   -> up to ``max_toolkits``
    tool agent      one per chosen toolkit: its tools (name, 1st sentence) -> up to ``max_tools``
    pool            union in agent order, capped at ``k_max``
    reflection      if the pool is empty, or ``check(pool)`` says it is insufficient, the meta agent is
                    asked once more with the branches already tried marked as excluded

Each agent replies with JSON {"choose": [ids]}; anything else counts as an empty choice. Every call is
billed: the selection's ``overhead_tokens`` hold the agents' prompt and reply tokens and
``router_calls`` the number of agent calls, so B5 can be compared with CATS on cost as well as cover.

Backends: any cats.llm backend (AnthropicLLM, OpenAICompatLLM, cats.llm.extra.LiteLLMBackend),
``ExchangeLLM`` (offline: prompts are written to files and an external agent such as a Claude Code
subagent writes the answers, the same protocol as cats.llm_router and cats.jev), or
``OverlapChooser`` (a deterministic lexical stand-in for tests and demos; not a model).

Exchange mode runs level by level, because each level's prompts depend on the previous answers:

    python -m cats.baselines_llm round --dir runs/anytool --n 40 --seed 2   # writes requests_<r>.jsonl
    # an agent answers each line: {"key": ..., "text": "{\\"choose\\": [...]}"} -> answers_<r>.jsonl
    python -m cats.baselines_llm round --dir runs/anytool --n 40 --seed 2   # next level ... until done
    python -m cats.baselines_llm score --dir runs/anytool
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import LLMResponse
from .registry import Registry
from .text import tokenize
from .tokens import default_counter
from .types import Selection, Task, TraceSpan, Turn

META_SYSTEM = (
    "You are the meta agent of a hierarchical tool finder. Choose the tool categories that hold the tools "
    "the user's request needs. Choose several only when the request needs several kinds of tool. "
    'Reply with JSON only: {"choose": ["<category id>", ...]} with at most MAX ids from the options.'
)
CATEGORY_SYSTEM = (
    "You are a category agent of a hierarchical tool finder, in charge of one category. Choose the servers "
    "in it whose tools can serve the user's request. "
    'Reply with JSON only: {"choose": ["<server id>", ...]} with at most MAX ids from the options; [] if none fits.'
)
TOOL_SYSTEM = (
    "You are a tool agent of a hierarchical tool finder, in charge of one server. Choose the tools the "
    "user's request needs from this server. "
    'Reply with JSON only: {"choose": ["<tool name>", ...]} with at most MAX names from the options; [] if none fits.'
)


# =============================================================================== backends
class ExchangePending(Exception):
    """Raised when the answer to a prompt has not been written yet (exchange mode)."""

    def __init__(self, keys: Sequence[str]):
        super().__init__(f"{len(keys)} prompt(s) awaiting answers")
        self.keys = list(keys)


def prompt_key(system: str, messages: list[dict]) -> str:
    return hashlib.sha1((system + "\n" + json.dumps(messages, sort_keys=True)).encode()).hexdigest()[:16]


class ExchangeLLM:
    """Offline backend: answers come from ``<dir>/answers_*.jsonl`` ({"key", "text"} or {"key", "choose"});
    unanswered prompts are collected in ``pending`` and written by ``flush``."""

    name = "exchange"

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.answers: dict[str, str] = {}
        self.pending: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        for f in sorted(self.dir.glob("answers_*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.answers[rec["key"]] = rec["text"] if "text" in rec else json.dumps({"choose": rec.get("choose", [])})

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        key = prompt_key(system, messages)
        if key in self.answers:
            return LLMResponse(text=self.answers[key])
        self.pending[key] = {"key": key, "system": system, "prompt": messages[-1]["content"]}
        raise ExchangePending([key])

    def flush(self, round_id: int | str) -> Path | None:
        """Write pending prompts to requests_<round>.jsonl (and the answering instructions once)."""
        if not self.pending:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "README.md").write_text(
            "Answer every line of requests_*.jsonl: read `system` and `prompt` (JSON with the request and the "
            "options), decide as that agent would, and append one line per request to answers_<round>.jsonl:\n"
            '{"key": "<key>", "choose": ["<id>", ...]}\nChoose only ids listed in the options. Use judgment, '
            "not keyword matching code.\n")
        p = self.dir / f"requests_{round_id}.jsonl"
        with p.open("w") as f:
            for rec in self.pending.values():
                f.write(json.dumps(rec) + "\n")
        self.pending = {}
        return p


class OverlapChooser:
    """Deterministic stand-in for an LLM agent: picks the options whose words overlap the request most.
    For tests and offline demos only; it measures the hierarchy's plumbing, not B5's accuracy."""

    name = "overlap-chooser"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        p = json.loads(messages[-1]["content"])
        q = set(tokenize(p["request"] + " " + " ".join(h["content"] for h in p.get("history", []))))
        scored = []
        for o in p["options"]:
            words = set(tokenize(o["id"].replace("_", " ") + " " + o.get("description", "")))
            s = len(q & words) / (1 + len(words)) ** 0.5
            if s > 0 and o["id"] not in set(p.get("excluded", [])):
                scored.append((s, o["id"]))
        scored.sort(key=lambda x: -x[0])
        return LLMResponse(text=json.dumps({"choose": [i for _, i in scored[: p["max"]]]}))


# ================================================================================ selector
@dataclass
class AgentCall:
    level: str  # meta | category:<branch> | tool:<toolkit>
    chosen: list[str]
    input_tokens: int
    output_tokens: int


def parse_choice(text: str, allowed: Sequence[str], limit: int) -> list[str]:
    """The ``choose`` ids of a JSON reply, restricted to ``allowed``, de-duplicated, at most ``limit``."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        ids = obj.get("choose", []) if isinstance(obj, dict) else []
    except ValueError:
        return []
    ok = set(allowed)
    return [i for i in dict.fromkeys(str(x) for x in ids if isinstance(x, (str, int))) if i in ok][:limit]


class AnyToolSelector:
    """B5: hierarchical LLM agents select the tools (see the module docstring)."""

    def __init__(self, registry: Registry, llm, *, max_branches: int = 3, max_toolkits: int = 3, max_tools: int = 3,
                 k_max: int = 10, reflect: bool = True, check: Callable[[list[str]], bool] | None = None,
                 name: str = "B5_anytool"):
        self.registry, self.llm = registry, llm
        self.max_branches, self.max_toolkits, self.max_tools = max_branches, max_toolkits, max_tools
        self.k_max, self.reflect, self.check = k_max, reflect, check
        self.name = name
        self.counter = default_counter()

    def fit(self, tasks):
        return self

    def calibrate(self, tasks):
        return {}

    # -------------------------------------------------------------------- prompts
    def _prompt(self, system: str, limit: int, query: str, history, options: list[dict], excluded=()) -> tuple:
        payload = {"request": query, "history": [{"role": h.role, "content": h.content} for h in history or []],
                   "options": options, "max": limit}
        if excluded:
            payload["excluded"] = sorted(excluded)
        return system.replace("MAX", str(limit)), [{"role": "user", "content": json.dumps(payload)}]

    def _ask_level(self, jobs: list[tuple[str, str, list, list[str], int]], calls: list[AgentCall]) -> list[list[str]]:
        """Ask every agent of one level; raise ExchangePending once for all unanswered prompts."""
        out, pending = [], []
        for level, system, messages, allowed, limit in jobs:
            try:
                r = self.llm.complete(system, messages, [])
            except ExchangePending as e:
                pending += e.keys
                out.append([])
                continue
            n_in = r.input_tokens or (self.counter.count(system) + self.counter.count(messages[-1]["content"]))
            n_out = r.output_tokens or self.counter.count(r.text)
            chosen = parse_choice(r.text, allowed, limit)
            calls.append(AgentCall(level, chosen, n_in, n_out))
            out.append(chosen)
        if pending:
            raise ExchangePending(pending)
        return out

    def _branch_options(self) -> list[dict]:
        return [{"id": b.id, "description": f"{b.name}: {b.summary}"} for b in sorted(self.registry.branches.values(),
                                                                                      key=lambda x: x.id)]

    def _toolkit_options(self, branch: str) -> list[dict]:
        reg = self.registry
        return [{"id": t, "description": f"{reg.toolkits[t].name}: {reg.toolkits[t].summary}"}
                for t in sorted(reg.branch_toolkits(branch)) if reg.toolkit_tools(t)]

    def _tool_options(self, toolkit: str) -> list[dict]:
        reg = self.registry
        return [{"id": reg.tools[t].name, "description": reg.tools[t].description.split(". ")[0][:200]}
                for t in reg.toolkit_tools(toolkit)]

    def _descend(self, query: str, history, excluded: set[str], calls: list[AgentCall]) -> tuple[list[str], list[str]]:
        opts = self._branch_options()
        sysm, msgs = self._prompt(META_SYSTEM, self.max_branches, query, history, opts, excluded)
        (branches,) = self._ask_level([("meta", sysm, msgs, [o["id"] for o in opts], self.max_branches)], calls)
        jobs = []
        for b in branches:
            o = self._toolkit_options(b)
            s, m = self._prompt(CATEGORY_SYSTEM, self.max_toolkits, query, history, o)
            jobs.append((f"category:{b}", s, m, [x["id"] for x in o], self.max_toolkits))
        toolkits = list(dict.fromkeys(tk for chosen in self._ask_level(jobs, calls) for tk in chosen))
        jobs = []
        for tk in toolkits:
            o = self._tool_options(tk)
            s, m = self._prompt(TOOL_SYSTEM, self.max_tools, query, history, o)
            jobs.append((f"tool:{tk}", s, m, [x["id"] for x in o], self.max_tools))
        names = self._ask_level(jobs, calls)
        pool = [f"{tk}.{n}" for tk, chosen in zip(toolkits, names) for n in chosen]
        return branches, [t for t in dict.fromkeys(pool) if t in self.registry.tools]

    def select(self, query: str, history: list[Turn] | None = None) -> Selection:
        calls: list[AgentCall] = []
        branches, pool = self._descend(query, history, set(), calls)
        reflected = False
        if self.reflect and (not pool or (self.check is not None and not self.check(pool))):
            reflected = True
            more_b, more = self._descend(query, history, set(branches), calls)
            branches += [b for b in more_b if b not in branches]
            pool += [t for t in more if t not in pool]
        pool = pool[: self.k_max]
        n_in, n_out = sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls)
        span = TraceSpan(stage="anytool", latency_ms=0.0, tokens_in=n_in, tokens_out=n_out,
                         meta={"calls": [c.level for c in calls], "branches": branches, "reflected": reflected})
        return Selection(strategy=self.name, tool_ids=pool, tool_tokens=self.registry.total_tokens(pool),
                         overhead_tokens=n_in + n_out, router_calls=len(calls), router_kind="llm_hierarchy",
                         spans=[span])


# ============================================================================ exchange CLI
def run_exchange_round(selector: AnyToolSelector, tasks: Sequence[Task]) -> dict[str, Any]:
    """Select every task whose prompts are all answered; collect the next prompts of the others."""
    done, waiting = {}, []
    for t in tasks:
        try:
            done[t.id] = selector.select(t.query, t.history)
        except ExchangePending:
            waiting.append(t.id)
    return {"done": done, "waiting": waiting}


def score_selections(selections: dict[str, Selection], tasks: dict[str, Task]) -> dict[str, Any]:
    from .calibration import slot_miss

    rows = []
    for tid, sel in selections.items():
        t = tasks[tid]
        miss = slot_miss(set(sel.tool_ids), t.gold)
        rows.append({"task": tid, "full_cover": miss == 0, "slot_recall": 1 - miss, "k": len(sel.tool_ids),
                     "tool_tokens": sel.tool_tokens, "agent_tokens": sel.overhead_tokens, "calls": sel.router_calls})
    n = max(1, len(rows))
    return {"n": len(rows), "full_cover": sum(r["full_cover"] for r in rows) / n,
            "slot_recall": sum(r["slot_recall"] for r in rows) / n, "k": sum(r["k"] for r in rows) / n,
            "tool_tokens": sum(r["tool_tokens"] for r in rows) / n,
            "agent_tokens": sum(r["agent_tokens"] for r in rows) / n, "calls": sum(r["calls"] for r in rows) / n,
            "rows": rows}


def _tasks(seed: int, n: int) -> tuple[Registry, list[Task]]:
    import random

    from .bench.data import load_tasks, split
    from .build import DATA, build_registry

    reg = build_registry()
    tasks = [t for t in load_tasks(DATA / "tasks", list(reg.branches)) if all(g in reg.tools for g in t.gold_tools())]
    test = split(tasks, seed=seed)[2]
    random.Random(seed).shuffle(test)
    return reg, test[:n]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cats-anytool", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("round", "score"):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True)
        s.add_argument("--n", type=int, default=40)
        s.add_argument("--seed", type=int, default=2)
    a = ap.parse_args(argv)
    reg, tasks = _tasks(a.seed, a.n)
    llm = ExchangeLLM(a.dir)
    res = run_exchange_round(AnyToolSelector(reg, llm), tasks)
    if a.cmd == "round":
        rounds = len(list(Path(a.dir).glob("requests_*.jsonl")))
        path = llm.flush(rounds)
        print(f"{len(res['done'])} tasks selected, {len(res['waiting'])} waiting"
              + (f"; answer {path}" if path else ""))
    else:
        out = score_selections(res["done"], {t.id: t for t in tasks})
        out["waiting"] = res["waiting"]
        Path(a.dir, "score.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
