"""Berkeley Function Calling Leaderboard (BFCL v4; ShishirPatil/gorilla) -> CATS corpus and tasks.

BFCL gives each question its own short function list (1 to 37 functions) and a ground-truth call.
For tool *selection* we pool every function of every single-turn category into one catalog, so a
request must find its function among thousands, and keep each question's ground-truth function(s)
as the gold slots.

Source files (data/external/bfcl/raw/):
  BFCL_v4_<category>.json                  one JSON object per line: id, question, function
  possible_answer/BFCL_v4_<category>.json  id, ground_truth: [{function_name: {arg: [values]}}]
  BFCL_v4_{irrelevance,live_irrelevance,live_relevance}.json   functions only: distractor tools

Mapping:
  tool     = one distinct (name, description, parameters) function variant; BFCL pseudo-types
             ("dict", "float", Java types) are mapped onto JSON Schema
  toolkit  = the namespace before the first dot (``math.factorial`` -> ``math``); namespace-less
             functions are grouped into small toolkits inside their branch (common.auto_hierarchy)
  branch   = k-means cluster of TF-IDF text (BFCL has no categories)
  task     = last user message (system prompts dropped, earlier turns kept as history); gold = one
             slot per distinct ground-truth function name, holding every variant with that name;
             kind = BFCL category
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...types import Task, Turn
from .common import ExternalBenchmark, auto_hierarchy, branches_of, json_schema, tool_name

CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
              "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
DISTRACTORS = ["irrelevance", "live_irrelevance", "live_relevance"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _variant_key(fn: dict[str, Any]) -> str:
    return json.dumps([fn.get("name"), fn.get("description", ""), fn.get("parameters", {})], sort_keys=True)


def _messages(question: Any) -> list[dict[str, str]]:
    msgs = []
    for turn in question or []:
        for m in turn if isinstance(turn, list) else [turn]:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                msgs.append(m)
    return msgs


def convert(entries: dict[str, list[dict[str, Any]]], answers: dict[str, dict[str, Any]],
            distractors: dict[str, list[dict[str, Any]]] | None = None, n_branches: int = 32,
            toolkit_size: int = 8, seed: int = 0, widen: bool = True) -> ExternalBenchmark:
    """``entries``/``distractors``: category -> rows; ``answers``: entry id -> ground_truth list.

    ``widen`` (default) lets any variant of the gold function name fill its slot; with ``widen=False``
    only the question's own variant counts (exact duplicates are already merged)."""
    variants: dict[str, dict[str, Any]] = {}
    for rows in list(entries.values()) + list((distractors or {}).values()):
        for r in rows:
            for fn in r.get("function") or []:
                if fn.get("name"):
                    variants.setdefault(_variant_key(fn), fn)

    units: dict[str, dict[str, Any]] = {}
    for key, fn in sorted(variants.items()):
        name = fn["name"]
        ns, local = (name.split(".", 1) if "." in name else (None, name))
        ukey = f"ns:{ns}" if ns else f"fn:{name}"
        unit = units.setdefault(ukey, {"namespace": ns, "tools": []})
        unit["tools"].append({"name": tool_name(local), "description": (fn.get("description") or "").strip(),
                              "input_schema": json_schema(fn.get("parameters")), "_key": key, "_fn": name})
    docs = auto_hierarchy(units, n_branches=n_branches, toolkit_size=toolkit_size, seed=seed, branch_prefix="bfcl_c")

    id_of_variant: dict[str, str] = {}
    ids_of_name: dict[str, list[str]] = {}
    branch_of: dict[str, str] = {}
    for doc in docs:
        for tk in doc["toolkits"]:
            for t in tk["tools"]:
                tid = f"{tk['id']}.{t['name']}"
                id_of_variant[t.pop("_key")] = tid
                ids_of_name.setdefault(t.pop("_fn"), []).append(tid)
                branch_of[tid] = doc["branch"]["id"]

    tasks, skipped = [], {"no_answer": 0, "unresolved": 0, "no_query": 0}
    by_kind: dict[str, int] = {}
    for cat, rows in entries.items():
        for r in rows:
            gt = answers.get(r["id"])
            if not gt:
                skipped["no_answer"] += 1
                continue
            own = {fn["name"]: _variant_key(fn) for fn in r.get("function") or [] if fn.get("name")}
            names = []
            for call in gt:
                for fname in call:
                    if fname not in names:
                        names.append(fname)
            if not names or any(n not in own for n in names):
                skipped["unresolved"] += 1
                continue
            # A slot holds the question's own variant plus every other variant of the same function name.
            slots = [sorted(set(ids_of_name[n] if widen else []) | {id_of_variant[own[n]]}) for n in names]
            msgs = [m for m in _messages(r.get("question")) if m.get("role") in ("user", "assistant")]
            users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
            if not users:
                skipped["no_query"] += 1
                continue
            last = users[-1]
            history = [Turn(role=m["role"], content=m["content"]) for m in msgs[:last]]
            tasks.append(Task(id=f"bfcl-{r['id']}", query=msgs[last]["content"].strip(), history=history,
                              gold=slots, branches=branches_of(slots, branch_of), kind=cat))
            by_kind[cat] = by_kind.get(cat, 0) + 1
    n_tools = sum(len(tk["tools"]) for d in docs for tk in d["toolkits"])
    meta = {"name": "bfcl", "tools": n_tools, "toolkits": sum(len(d["toolkits"]) for d in docs),
            "branches": len(docs), "function_names": len(ids_of_name), "tasks": len(tasks), "tasks_by_kind": by_kind,
            "skipped": skipped, "slots_mean": sum(len(t.gold) for t in tasks) / max(1, len(tasks)),
            "hierarchy": {"method": "tfidf-kmeans", "n_branches": n_branches, "toolkit_size": toolkit_size,
                          "seed": seed}, "label_widening": "function name" if widen else "none"}
    return ExternalBenchmark("bfcl", docs, tasks, meta)


def load(raw_dir: str | Path, categories: list[str] | None = None, with_distractors: bool = True,
         n_branches: int = 32, seed: int = 0, widen: bool = True) -> ExternalBenchmark:
    d = Path(raw_dir)
    entries, answers = {}, {}
    for cat in categories or CATEGORIES:
        p = d / f"BFCL_v4_{cat}.json"
        if not p.exists():
            continue
        entries[cat] = read_jsonl(p)
        ap = d / "possible_answer" / f"BFCL_v4_{cat}.json"
        if ap.exists():
            answers.update({row["id"]: row["ground_truth"] for row in read_jsonl(ap)})
    distractors = {}
    if with_distractors:
        for cat in DISTRACTORS:
            p = d / f"BFCL_v4_{cat}.json"
            if p.exists():
                distractors[cat] = read_jsonl(p)
    return convert(entries, answers, distractors, n_branches=n_branches, seed=seed, widen=widen)
