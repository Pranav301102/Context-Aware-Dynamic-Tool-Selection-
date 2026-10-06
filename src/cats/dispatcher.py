"""Dispatcher: runs a sub-agent on the selected tools, with request_tools escalation (FR6, FR7).

Each model turn is billed through the prefix-cache simulator, so router calls,
escalations and lost cache discounts all show up in the per-task cost (NFR6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .ledger import PrefixCache, Prices, RequestCost
from .llm import LLM, ToolCall
from .pipeline import CATS
from .tokens import default_counter
from .types import Task


@dataclass
class RunResult:
    task_id: str
    calls: list[str] = field(default_factory=list)  # tool ids actually called
    invalid_calls: list[str] = field(default_factory=list)  # names not in the prompt
    escalations: int = 0
    turns: int = 0
    loaded: list[str] = field(default_factory=list)
    cost: RequestCost = field(default_factory=RequestCost)
    usd: float = 0.0
    success: bool = False


def simulate_tool(tool_id: str, args: dict) -> str:
    """Deterministic stand-in for a tool server (StableToolBench-style virtual API)."""
    return json.dumps({"status": "ok", "tool": tool_id, "echo": args, "result": f"simulated result of {tool_id}"})


class Dispatcher:
    def __init__(self, cats: CATS, llm: LLM, max_escalations: int = 2, max_turns: int = 6,
                 cache: PrefixCache | None = None, prices: Prices | None = None, select: bool = True):
        self.cats, self.llm = cats, llm
        self.max_escalations, self.max_turns = max_escalations, max_turns
        self.cache = cache or PrefixCache()
        self.prices = prices or Prices()
        self.select = select  # False = start with no tools (agent-initiated search, B6)
        self.counter = default_counter()

    def run(self, task: Task) -> RunResult:
        res = RunResult(task_id=task.id)
        reg = self.cats.registry
        sel = self.cats.select(task.query, task.history) if self.select else None
        loaded = list(sel.tool_ids) if sel else []
        res.cost.router_tokens += sel.overhead_tokens if sel else 0
        # Over budget, the assembler drops the lowest scores first; explicitly requested tools rank highest.
        self._score = dict(zip(loaded, sel.scores)) if sel else {}
        order = list(loaded)
        messages = [{"role": m.role, "content": m.content} for m in task.history]
        messages.append({"role": "user", "content": task.query})
        wire = {}
        for _ in range(self.max_turns):
            asm = self.cats.assembler.assemble(loaded, [self._score.get(t, 0.0) for t in loaded], order)
            if asm.dropped:  # dropped tools leave the toolset, so they can be offered again later
                loaded[:] = [t for t in loaded if t not in asm.dropped]
            wire = {reg.tools[t].wire_name: t for t in asm.tool_ids}
            resp = self.llm.complete(asm.system, messages, asm.tools)
            res.turns += 1
            keys = [t["name"] for t in asm.tools]
            suffix = self.counter.count(asm.system) + self.counter.count_json(messages)
            res.cost += self.cache.charge(keys, asm.block_tokens, suffix, resp.output_tokens or 50)
            if not resp.tool_calls:
                break
            messages.append({"role": "assistant", "content": resp.text, "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls]})
            for call in resp.tool_calls:
                messages.append(self._handle(call, wire, loaded, order, res))
        res.loaded = loaded
        res.usd = res.cost.usd(self.prices)
        res.success = all(any(t in res.calls for t in slot) for slot in task.gold) and not res.invalid_calls
        return res

    def _handle(self, call: ToolCall, wire: dict, loaded: list, order: list, res: RunResult) -> dict:
        if call.name == "request_tools":
            if res.escalations >= self.max_escalations:
                return {"role": "tool", "tool_call_id": call.id, "content": "request_tools limit reached."}
            res.escalations += 1
            cap = str(call.arguments.get("capability", ""))
            new = self.cats.escalate(cap, exclude=set(loaded)).tool_ids
            self._score.update({t: 2.0 for t in new})
            loaded.extend(new)
            order.extend(new)
            names = ", ".join(self.cats.registry.tools[t].wire_name for t in new)
            return {"role": "tool", "tool_call_id": call.id, "content": f"Added tools: {names}"}
        tid = wire.get(call.name)
        if tid is None:
            res.invalid_calls.append(call.name)
            return {"role": "tool", "tool_call_id": call.id, "content": f"Unknown tool {call.name}."}
        res.calls.append(tid)
        return {"role": "tool", "tool_call_id": call.id, "content": simulate_tool(tid, call.arguments)}
