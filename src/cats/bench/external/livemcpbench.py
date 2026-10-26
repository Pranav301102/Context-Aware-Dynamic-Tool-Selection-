"""LiveMCPBench (Mo et al., 2025; icip-cas/LiveMCPBench) -> CATS corpus and tasks.

Source files (data/external/livemcpbench/raw/):
  tools.json            LiveMCPTool: 69 server entries with their MCP ``tools/list`` output and a
                        category (Discovery, Code, File Access, Finance, ...)
  all_annotations.json  95 human-written tasks; each lists the tools an annotator used, by name

Mapping:
  branch   = server category (8)
  toolkit  = MCP server (a config entry may expose several servers; each is a toolkit)
  tool     = one tool from ``tools/list``, schema kept as published
  task     = ``Question``; gold = one slot per distinct annotated tool name, holding every tool
             with that name (a name served by two servers counts on either); kind = the task's
             LiveMCPBench category (Office, Lifestyle, Leisure, Finance, Travel, Shopping)

The annotation is one valid solution path, so other tools could also work; full cover here is
strict. Annotated names that match no tool are dropped and counted in ``meta``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...types import Task
from .common import ExternalBenchmark, branches_of, normalize_name, slots_from_names, snake, tool_name, unique

_ITEM = re.compile(r"^\s*\d+\s*[.)]\s*")


def parse_tool_list(text: str) -> list[str]:
    """'1. get-weread-rank\\n2. generate_word_cloud_chart' -> ['get-weread-rank', ...]."""
    out = []
    for line in str(text or "").splitlines():
        line = _ITEM.sub("", line).strip().strip("`'\"")
        if line:
            out.append(line)
    return out


def convert(tools_json: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> ExternalBenchmark:
    branches: dict[str, dict[str, Any]] = {}
    used_tk: set[str] = set()
    by_name: dict[str, list[str]] = {}
    branch_of: dict[str, str] = {}
    n_tools = 0
    for entry in tools_json:
        cat = entry.get("category") or "Uncategorized"
        bid = snake(cat, "cat")
        doc = branches.setdefault(bid, {"branch": {"id": bid, "name": cat, "summary": ""}, "toolkits": [],
                                        "_servers": []})
        servers = entry.get("tools") or {}
        for key, server in servers.items():
            tid = unique(snake(key, "server"), used_tk)
            name = entry.get("name", key) if len(servers) == 1 else f"{entry.get('name', key)} ({key})"
            tools, seen = [], set()
            for t in server.get("tools", []):
                nm = unique(tool_name(t["name"]), seen)
                tools.append({"name": nm, "description": (t.get("description") or "").strip(),
                              "input_schema": t.get("inputSchema") or t.get("input_schema") or {"type": "object"}})
                tool_id = f"{tid}.{nm}"
                by_name.setdefault(normalize_name(t["name"]), []).append(tool_id)
                branch_of[tool_id] = bid
            if not tools:
                continue
            n_tools += len(tools)
            doc["toolkits"].append({"id": tid, "name": name, "server": f"mcp://{key}",
                                    "summary": (entry.get("description") or "").strip()[:400], "tools": tools})
            doc["_servers"].append(entry.get("name", key))
    corpus = []
    for bid in sorted(branches):
        doc = branches[bid]
        if not doc["toolkits"]:
            continue
        names = sorted(set(doc.pop("_servers")))
        doc["branch"]["summary"] = (f"{doc['branch']['name']} MCP servers: " + "; ".join(names))[:600]
        corpus.append(doc)

    tasks, dropped, missing_names = [], 0, []
    for i, a in enumerate(annotations):
        meta = a.get("Annotator Metadata") or {}
        names = parse_tool_list(meta.get("Tools", ""))
        slots, missing = slots_from_names(names, by_name)
        missing_names += missing
        if not slots:
            dropped += 1
            continue
        tid = str(a.get("task_id") or i)
        tasks.append(Task(id=f"livemcp-{i:03d}-{tid[:8]}", query=str(a["Question"]).strip(), history=[],
                          gold=slots, branches=branches_of(slots, branch_of),
                          kind=snake(a.get("category") or "task")))
    meta = {"name": "livemcpbench", "tools": n_tools, "toolkits": sum(len(d["toolkits"]) for d in corpus),
            "branches": len(corpus), "tasks": len(tasks), "tasks_dropped_no_resolvable_tool": dropped,
            "annotated_names_unresolved": sorted(set(missing_names)),
            "slots_mean": sum(len(t.gold) for t in tasks) / max(1, len(tasks))}
    return ExternalBenchmark("livemcpbench", corpus, tasks, meta)


def load(raw_dir: str | Path) -> ExternalBenchmark:
    d = Path(raw_dir)
    tools = json.loads((d / "tools.json").read_text())
    ann = json.loads((d / "all_annotations.json").read_text())
    return convert(tools, ann)
