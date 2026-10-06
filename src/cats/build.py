"""One place that wires registry -> index -> analyzer -> CATS and the data splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analyzer import ContextAnalyzer
from .bench.data import load_tasks, split
from .index import ToolIndex, make_embedder
from .pipeline import CATS, CATSConfig
from .registry import Registry
from .types import Task

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass
class System:
    registry: Registry
    index: ToolIndex
    analyzer: ContextAnalyzer
    cats: CATS
    train: list[Task]
    calib: list[Task]
    test: list[Task]


def build_registry(corpus_dirs: list[Path] | None = None) -> Registry:
    reg = Registry()
    for d in corpus_dirs or [DATA / "corpus"]:
        for path in sorted(Path(d).glob("*.json")):
            reg.load_branch_file(path)
    return reg


def build_system(config: CATSConfig | None = None, *, registry: Registry | None = None,
                 tasks: list[Task] | None = None, embedder: str | None = None, seed: int = 0,
                 enrichment: Path | None = None, fit: bool = True) -> System:
    reg = registry or build_registry()
    index = ToolIndex(reg, make_embedder(embedder))
    if enrichment is not None:
        from .enrich import load_enrichment

        index.enrichment = load_enrichment(enrichment, reg)
    index.build()
    analyzer = ContextAnalyzer(reg)
    cats = CATS(reg, index, analyzer, config or CATSConfig())
    all_tasks = tasks if tasks is not None else load_tasks(DATA / "tasks", list(reg.branches))
    all_tasks = [t for t in all_tasks if all(g in reg.tools for g in t.gold_tools())]
    train, calib, test = split(all_tasks, seed=seed)
    if fit:
        cats.fit(train)
        cats.calibrate(calib)
    return System(reg, index, analyzer, cats, train, calib, test)
