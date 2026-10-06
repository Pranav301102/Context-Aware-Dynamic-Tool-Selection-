"""Context assembler: cache-stable prompt within a token budget (FR5, FR12).

Layout of every sub-agent prompt (providers cache by exact prefix, tools first):

    [request_tools meta-tool] [selected tools, in a deterministic order] [system prefix] [task]

Orderings:
  canonical  sort by tool id. Identical selections give byte-identical prompts.
  append     within a session, tools keep the position they were first loaded at and
             new tools go to the end, so earlier turns' prefixes stay cacheable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .registry import Registry
from .tokens import TokenCounter, default_counter

REQUEST_TOOLS = {
    "name": "request_tools",
    "description": (
        "Ask for more tools. Call this only when none of your current tools can do a step the task "
        "needs. Describe the missing capability in plain words (what it must do, not a tool name); "
        "matching tools are added to your toolset. At most 2 calls per task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"capability": {"type": "string", "description": "The capability you are missing."}},
        "required": ["capability"],
    },
}

DEFAULT_SYSTEM = (
    "You are a domain sub-agent. Complete the user's task by calling the tools you have. "
    "Only call tools that are listed. If a needed capability is missing, call request_tools."
)


@dataclass
class Assembled:
    tools: list[dict[str, Any]]
    tool_ids: list[str]
    system: str
    tool_tokens: int  # selected tool definitions only
    meta_tokens: int  # request_tools + system prefix
    dropped: list[str] = field(default_factory=list)
    prefix_hash: str = ""
    block_tokens: list[int] = field(default_factory=list)  # per tool block, in prompt order

    @property
    def prefix_tokens(self) -> int:
        return self.tool_tokens + self.meta_tokens


class ContextAssembler:
    def __init__(self, registry: Registry, counter: TokenCounter | None = None, budget: int = 8000,
                 order: str = "canonical", system: str = DEFAULT_SYSTEM, meta_tool: bool = True):
        self.registry = registry
        self.counter = counter or default_counter()
        self.budget = budget
        self.order = order
        self.system = system
        self.meta_tool = meta_tool

    def assemble(self, tool_ids: list[str], scores: list[float] | None = None,
                 session_order: list[str] | None = None) -> Assembled:
        reg = self.registry
        ids = [t for t in dict.fromkeys(tool_ids) if t in reg.tools]
        # Budget: drop the lowest-scored tools first (input order is the retriever's rank order).
        rank = {t: (scores[i] if scores and i < len(scores) else -i) for i, t in enumerate(ids)}
        dropped: list[str] = []
        total = sum(reg.tools[t].token_cost for t in ids)
        while ids and total > self.budget and len(ids) > 1:
            worst = min(ids, key=lambda t: rank[t])
            ids.remove(worst)
            dropped.append(worst)
            total -= reg.tools[worst].token_cost
        if self.order == "append" and session_order is not None:
            pos = {t: i for i, t in enumerate(session_order)}
            ids.sort(key=lambda t: (pos.get(t, len(pos)), t))
        else:
            ids.sort()
        tools = ([REQUEST_TOOLS] if self.meta_tool else []) + [reg.tools[t].spec() for t in ids]
        blob = json.dumps({"tools": tools, "system": self.system}, sort_keys=True).encode()
        meta_tokens = (self.counter.count_tool(REQUEST_TOOLS) if self.meta_tool else 0) + self.counter.count(self.system)
        block_tokens = ([self.counter.count_tool(REQUEST_TOOLS)] if self.meta_tool else []) + [
            reg.tools[t].token_cost for t in ids
        ]
        return Assembled(
            tools=tools, tool_ids=ids, system=self.system, tool_tokens=total, meta_tokens=meta_tokens,
            dropped=dropped, prefix_hash=hashlib.sha256(blob).hexdigest()[:16], block_tokens=block_tokens,
        )
