"""Hierarchical tool registry: branch -> toolkit -> tool (FR1).

Each tool carries its JSON schema, token cost and a content version. Changes are
published as RegistryEvents so the index can re-sync incrementally (FR11).
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .tokens import TokenCounter, default_counter
from .types import Branch, RegistryEvent, Toolkit, ToolRecord

Listener = Callable[[RegistryEvent], None]


class Registry:
    def __init__(self, counter: TokenCounter | None = None):
        self.counter = counter or default_counter()
        self.branches: dict[str, Branch] = {}
        self.toolkits: dict[str, Toolkit] = {}
        self.tools: dict[str, ToolRecord] = {}
        self._by_toolkit: dict[str, list[str]] = defaultdict(list)
        self.version = 0
        self._listeners: list[Listener] = []

    # ------------------------------------------------------------------ events
    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _emit(self, kind: str, toolkit_id: str, tool_ids: list[str]) -> None:
        if not tool_ids:
            return
        self.version += 1
        event = RegistryEvent(kind=kind, toolkit_id=toolkit_id, tool_ids=tool_ids, registry_version=self.version)
        for listener in self._listeners:
            listener(event)

    # ---------------------------------------------------------------- mutation
    def add_branch(self, branch: Branch) -> None:
        self.branches[branch.id] = branch

    def upsert_toolkit(self, toolkit: Toolkit, tools: Iterable[dict[str, Any]]) -> list[RegistryEvent]:
        """Replace a toolkit's tool list (what an MCP tools/list returns).

        Emits at most one event per kind, so a re-sync that changes nothing emits nothing.
        """
        if toolkit.branch_id not in self.branches:
            raise KeyError(f"unknown branch {toolkit.branch_id!r} for toolkit {toolkit.id!r}")
        self.toolkits[toolkit.id] = toolkit
        incoming: dict[str, ToolRecord] = {}
        for raw in tools:
            rec = ToolRecord(
                id=f"{toolkit.id}.{raw['name']}",
                name=raw["name"],
                toolkit_id=toolkit.id,
                branch_id=toolkit.branch_id,
                description=raw.get("description") or "",
                input_schema=raw.get("input_schema") or raw.get("inputSchema") or {"type": "object"},
            )
            rec.version = rec.content_hash()
            rec.token_cost = self.counter.count_tool(rec.spec())
            incoming[rec.id] = rec

        old_ids = set(self._by_toolkit.get(toolkit.id, []))
        added = [i for i in incoming if i not in old_ids]
        updated = [i for i in incoming if i in old_ids and self.tools[i].version != incoming[i].version]
        removed = [i for i in old_ids if i not in incoming]
        for i in removed:
            del self.tools[i]
        self.tools.update(incoming)
        self._by_toolkit[toolkit.id] = list(incoming)
        events = []
        for kind, ids in (("removed", removed), ("updated", updated), ("added", added)):
            if ids:
                self._emit(kind, toolkit.id, sorted(ids))
                events.append(kind)
        return events

    def remove_toolkit(self, toolkit_id: str) -> None:
        ids = self._by_toolkit.pop(toolkit_id, [])
        for i in ids:
            self.tools.pop(i, None)
        self.toolkits.pop(toolkit_id, None)
        self._emit("removed", toolkit_id, sorted(ids))

    # ----------------------------------------------------------------- queries
    def toolkit_tools(self, toolkit_id: str) -> list[str]:
        return list(self._by_toolkit.get(toolkit_id, []))

    def branch_toolkits(self, branch_id: str) -> list[str]:
        return [t.id for t in self.toolkits.values() if t.branch_id == branch_id]

    def tools_in(self, branch_ids: Iterable[str] | None = None) -> list[str]:
        if branch_ids is None:
            return list(self.tools)
        wanted = set(branch_ids)
        return [i for i, t in self.tools.items() if t.branch_id in wanted]

    def specs(self, tool_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [self.tools[i].spec() for i in tool_ids]

    def total_tokens(self, tool_ids: Iterable[str] | None = None) -> int:
        ids = self.tools if tool_ids is None else tool_ids
        return sum(self.tools[i].token_cost for i in ids)

    def resolve_wire_name(self, wire: str) -> str | None:
        tid = wire.replace("__", ".", 1)
        return tid if tid in self.tools else None

    # --------------------------------------------------------------------- I/O
    def load_branch_file(self, path: Path) -> None:
        doc = json.loads(Path(path).read_text())
        b = doc["branch"]
        self.add_branch(Branch(id=b["id"], name=b.get("name", b["id"]), summary=b.get("summary", "")))
        for tk in doc.get("toolkits", []):
            toolkit = Toolkit(
                id=tk["id"], name=tk.get("name", tk["id"]), branch_id=b["id"],
                server=tk.get("server", ""), summary=tk.get("summary", ""),
            )
            self.upsert_toolkit(toolkit, tk.get("tools", []))

    @classmethod
    def from_corpus(cls, corpus_dir: str | Path, counter: TokenCounter | None = None) -> Registry:
        reg = cls(counter)
        for path in sorted(Path(corpus_dir).glob("*.json")):
            reg.load_branch_file(path)
        return reg

    def to_corpus(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for b in self.branches.values():
            doc = {"branch": b.model_dump(), "toolkits": []}
            for tk_id in self.branch_toolkits(b.id):
                tk = self.toolkits[tk_id]
                doc["toolkits"].append({
                    "id": tk.id, "name": tk.name, "server": tk.server, "summary": tk.summary,
                    "tools": [
                        {"name": self.tools[i].name, "description": self.tools[i].description,
                         "input_schema": self.tools[i].input_schema}
                        for i in self.toolkit_tools(tk_id)
                    ],
                })
            (out / f"{b.id}.json").write_text(json.dumps(doc, indent=1))

    def stats(self) -> dict[str, Any]:
        return {
            "branches": len(self.branches),
            "toolkits": len(self.toolkits),
            "tools": len(self.tools),
            "tool_tokens": self.total_tokens(),
            "tokenizer": self.counter.backend,
        }
