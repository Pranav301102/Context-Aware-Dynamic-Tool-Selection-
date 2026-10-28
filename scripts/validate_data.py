"""Validate the tool corpus and task files against data/SPEC.md.

Usage: python scripts/validate_data.py [branch_id ...]
With no arguments every file under data/ is checked.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
KINDS = {"literal", "paraphrase", "multi", "contextual", "cross"}


def load_corpus(paths):
    tools, toolkit_owner, errors = {}, {}, []
    for path in paths:
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue
        branch = doc.get("branch", {})
        bid = branch.get("id", "")
        if not SNAKE.match(bid) or not branch.get("summary"):
            errors.append(f"{path.name}: bad branch header")
        if path.stem != bid:
            errors.append(f"{path.name}: file name must equal branch id {bid!r}")
        for tk in doc.get("toolkits", []):
            tid = tk.get("id", "")
            if not SNAKE.match(tid):
                errors.append(f"{path.name}: bad toolkit id {tid!r}")
            if tid in toolkit_owner:
                errors.append(f"{path.name}: toolkit {tid} also defined in {toolkit_owner[tid]}")
            toolkit_owner[tid] = path.name
            if not tk.get("summary"):
                errors.append(f"{path.name}: toolkit {tid} has no summary")
            names = set()
            for tool in tk.get("tools", []):
                name = tool.get("name", "")
                if not SNAKE.match(name):
                    errors.append(f"{path.name}: bad tool name {tid}.{name!r}")
                if name in names:
                    errors.append(f"{path.name}: duplicate tool {tid}.{name}")
                names.add(name)
                schema = tool.get("input_schema")
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    errors.append(f"{path.name}: {tid}.{name} input_schema must be an object schema")
                if not tool.get("description"):
                    errors.append(f"{path.name}: {tid}.{name} has no description")
                tools[f"{tid}.{name}"] = bid
    return tools, errors


def check_tasks(paths, tools):
    errors, ids, counts = [], set(), {}
    for path in paths:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON: {exc}")
                continue
            tid = task.get("id")
            if tid in ids:
                errors.append(f"{where}: duplicate task id {tid}")
            ids.add(tid)
            if not task.get("query"):
                errors.append(f"{where}: empty query")
            kind = task.get("kind")
            if kind not in KINDS:
                errors.append(f"{where}: bad kind {kind!r}")
            counts[kind] = counts.get(kind, 0) + 1
            gold = task.get("gold")
            if not gold or not all(isinstance(s, list) and s for s in gold):
                errors.append(f"{where}: gold must be a non-empty list of non-empty slots")
                continue
            gold_branches = set()
            for slot in gold:
                for t in slot:
                    if t not in tools:
                        errors.append(f"{where}: unknown gold tool {t}")
                    else:
                        gold_branches.add(tools[t])
            if gold_branches and set(task.get("branches", [])) != gold_branches:
                errors.append(f"{where}: branches {task.get('branches')} != gold branches {sorted(gold_branches)}")
            for turn in task.get("history", []):
                if turn.get("role") not in {"user", "assistant"} or not turn.get("content"):
                    errors.append(f"{where}: bad history turn")
    return errors, len(ids), counts


def main(argv):
    corpus_paths = sorted((ROOT / "data" / "corpus").glob("*.json"))
    tools, errors = load_corpus(corpus_paths)
    task_paths = sorted((ROOT / "data" / "tasks").glob("*.jsonl"))
    if argv:
        wanted = set(argv)
        task_paths = [p for p in task_paths if p.stem in wanted]
    task_errors, n_tasks, counts = check_tasks(task_paths, tools)
    errors += task_errors
    for e in errors[:200]:
        print("ERROR", e)
    print(f"{len(corpus_paths)} branches, {len(tools)} tools, {n_tasks} tasks, kinds={counts}")
    print("OK" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
