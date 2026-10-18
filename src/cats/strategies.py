"""Selection strategies compared in the evaluation (workbook Table 8).

B0  static        every tool in every prompt
B1  bm25          flat BM25 top-k
B2  dense         flat dense top-k (RAG-MCP style)
B3  hybrid        flat BM25+dense fusion with the learned reranker, top-k (ScaleMCP/Toolshed style)
B4  two_stage     top toolkits (servers) first, then tools inside them (MCP-Zero style)
    cats          calibrated branch set + reranked, risk-controlled tool cut-off
O   oracle        exactly one gold tool per slot
B5-B7 need a model in the loop (LLM hierarchy, agent-initiated search, provider tool search);
the subagent harness in cats.agent_cli covers agent-initiated search.
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .analyzer import ContextAnalyzer
from .index import ToolIndex
from .pipeline import CATS
from .registry import Registry
from .router import gather_evidence
from .types import Selection, Task, Turn


class Strategy(Protocol):
    name: str

    def fit(self, tasks: list[Task]): ...

    def calibrate(self, tasks: list[Task]): ...

    def select(self, query: str, history: list[Turn] | None = None) -> Selection: ...


class _Base:
    name = "base"

    def __init__(self, registry: Registry, index: ToolIndex | None = None, analyzer: ContextAnalyzer | None = None):
        self.registry, self.index = registry, index
        self.analyzer = analyzer

    def fit(self, tasks):
        return self

    def calibrate(self, tasks):
        return {}

    def _sel(self, ids: list[str], t0: float, scores=None) -> Selection:
        return Selection(strategy=self.name, tool_ids=ids, tool_tokens=self.registry.total_tokens(ids),
                         scores=[float(x) for x in (scores if scores is not None else [])], spans=[])


class StaticLoading(_Base):
    name = "B0_static"

    def select(self, query, history=None):
        return self._sel(sorted(self.registry.tools), time.perf_counter())


class FlatTopK(_Base):
    def __init__(self, registry, index, analyzer, kind: str = "hybrid", k: int = 5):
        super().__init__(registry, index, analyzer)
        self.kind, self.k = kind, k
        self.name = {"bm25": "B1_bm25", "dense": "B2_dense", "hybrid": "B3_hybrid_rrf"}[kind] + f"@{k}"

    def select(self, query, history=None):
        t0 = time.perf_counter()
        ev = gather_evidence(self.index, self.analyzer, query, history)
        s = {"bm25": ev.bm25, "dense": ev.cos, "hybrid": ev.rrf}[self.kind]
        order = np.argsort(-s, kind="stable")
        ids = [self.index.tools.ids[i] for i in order[: self.k * 2] if self.index.tools.ids[i] is not None][: self.k]
        return self._sel(ids, t0, [float(s[i]) for i in order[: self.k]])


class FlatReranked(_Base):
    """B3 with the same learned reranker CATS uses, but no routing and a fixed k."""

    def __init__(self, registry, index, analyzer, k: int = 5, cats: CATS | None = None):
        super().__init__(registry, index, analyzer)
        self.k = k
        self.cats = cats
        self.name = f"B3_hybrid_rerank@{k}"

    def select(self, query, history=None):
        t0 = time.perf_counter()
        ev = self.cats.evidence(query, history)
        p = self.cats.router.probs(ev)
        ranked, scores = self.cats.retriever.rank(ev, None, p)
        return self._sel(ranked[: self.k], t0, scores[: self.k])


class TwoStage(_Base):
    """Pick the top-m servers by summary match, then the top-k tools inside them."""

    def __init__(self, registry, index, analyzer, m: int = 3, k: int = 5):
        super().__init__(registry, index, analyzer)
        self.m, self.k = m, k
        self.name = f"B4_two_stage@{m}x{k}"

    def select(self, query, history=None):
        t0 = time.perf_counter()
        feats = self.analyzer.analyze(query, history)
        qvec = self.index.embed_query(feats.text)
        servers = [h.id for h in self.index.toolkits.search(feats.text, qvec, k=self.m)]
        allowed = self.index.tools.mask_for(t for s in servers for t in self.registry.toolkit_tools(s))
        hits = self.index.tools.search(feats.text, qvec, allowed=allowed, k=self.k)
        return self._sel([h.id for h in hits], t0, [h.score for h in hits])


class CATSStrategy(_Base):
    def __init__(self, cats: CATS, name: str = "CATS"):
        super().__init__(cats.registry, cats.index, cats.analyzer)
        self.cats = cats
        self.name = name

    def fit(self, tasks):
        self.cats.fit(tasks)
        return self

    def calibrate(self, tasks):
        return self.cats.calibrate(tasks)

    def select(self, query, history=None):
        sel = self.cats.select(query, history)
        sel.strategy = self.name
        return sel


class Oracle(_Base):
    name = "O_oracle"

    def __init__(self, registry, gold_lookup: dict[str, list[list[str]]]):
        super().__init__(registry)
        self.gold = gold_lookup

    def select(self, query, history=None):
        slots = self.gold.get(query, [])
        ids = sorted({slot[0] for slot in slots if slot})
        return self._sel(ids, time.perf_counter())
