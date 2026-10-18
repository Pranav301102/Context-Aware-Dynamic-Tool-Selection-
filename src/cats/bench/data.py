"""Task loading and deterministic train / calibration / test splits."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..types import Task


def load_tasks(task_dir: str | Path, branches: list[str] | None = None) -> list[Task]:
    """Load every task file; keep tasks whose branches all exist (when ``branches`` is given)."""
    allowed = set(branches) if branches else None
    tasks = []
    for path in sorted(Path(task_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                t = Task.model_validate(json.loads(line))
                if allowed is None or set(t.branches) <= allowed:
                    tasks.append(t)
    return tasks


def split(tasks: list[Task], fractions=(0.4, 0.2, 0.4), seed: int = 0) -> tuple[list[Task], list[Task], list[Task]]:
    """Stratified by (first branch, kind) so every split sees every branch and difficulty."""
    groups: dict[tuple, list[Task]] = defaultdict(list)
    for t in tasks:
        groups[(t.branches[0] if t.branches else "", t.kind)].append(t)
    rng = random.Random(seed)
    train, calib, test = [], [], []
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        n = len(g)
        a = round(n * fractions[0])
        b = a + round(n * fractions[1])
        train += g[:a]
        calib += g[a:b]
        test += g[b:]
    return train, calib, test
