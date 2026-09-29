"""Hierarchical router: scores every branch and returns a calibrated set (FR3, FR8).

The scorer is a conditional-logit model with weights shared across branches, so a
branch added later (FR15) is scored without retraining. Features combine top-down
evidence (branch and toolkit summaries) with bottom-up evidence (the best tool
matches inside each branch), all computed outside the model's prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analyzer import ContextAnalyzer
from .calibration import SCORERS, conformal_quantile
from .index import ToolIndex
from .index.hybrid import RRF_K
from .types import Features, RouteDecision, Turn


@dataclass
class Evidence:
    """Everything the router and retriever need about one request, computed once."""
    features: Features
    qvec: np.ndarray
    bm25: np.ndarray  # slot-aligned over the tool index
    cos: np.ndarray
    rrf: np.ndarray  # global reciprocal-rank fusion of bm25 and cosine
    branch_cos: dict[str, float]
    branch_bm25: dict[str, float]
    toolkit_cos: dict[str, float]
    key: str = ""  # request fingerprint, used to look up external (LLM) router scores
    extra: dict = field(default_factory=dict)


def gather_evidence(index: ToolIndex, analyzer: ContextAnalyzer, query: str,
                    history: list[Turn] | None = None) -> Evidence:
    from .llm_router import request_key

    feats = analyzer.analyze(query, history)
    qvec = index.embed_query(feats.text)
    bm, cos = index.score_all(feats.text, qvec)
    n = len(bm)
    r_bm = np.empty(n, dtype=np.int64)
    r_bm[np.argsort(-bm, kind="stable")] = np.arange(n)
    r_bm[bm <= 0] = n
    r_de = np.empty(n, dtype=np.int64)
    r_de[np.argsort(-cos, kind="stable")] = np.arange(n)
    rrf = 1.0 / (RRF_K + r_bm + 1) + 1.0 / (RRF_K + r_de + 1)
    rrf[np.array([i is None for i in index.tools.ids], dtype=bool)] = 0.0
    return Evidence(
        features=feats, qvec=qvec, bm25=bm, cos=cos, rrf=rrf,
        branch_cos=index.level_cosines(index.branches, qvec),
        branch_bm25=index.level_bm25(index.branches, feats.text),
        toolkit_cos=index.level_cosines(index.toolkits, qvec),
        key=request_key(query, history),
    )


BRANCH_FEATURES = [
    "branch_cos", "branch_bm25", "tool_cos_max", "tool_cos_top3", "tool_bm25_max",
    "tool_rrf_max", "toolkit_cos_max", "vendor_hint", "llm_p",
]


class BranchRouter:
    def __init__(self, index: ToolIndex, alpha: float = 0.05, method: str = "aps",
                 t_low: float = 0.0, max_branches: int | None = None, l2: float = 1e-3):
        self.index = index
        self.alpha = alpha
        self.method = method
        self.t_low = t_low
        self.max_branches = max_branches
        self.l2 = l2
        self.w = np.zeros(len(BRANCH_FEATURES))
        self.w[BRANCH_FEATURES.index("tool_rrf_max")] = 1.0
        self.mu = np.zeros(len(BRANCH_FEATURES))
        self.sd = np.ones(len(BRANCH_FEATURES))
        self.threshold = 1.0  # conformal t; 1.0 = keep branches until all mass is covered
        self.llm_scores: dict[str, dict[str, float]] | None = None  # optional LLM router output by request key

    # --------------------------------------------------------------- features
    def branch_features(self, ev: Evidence) -> tuple[list[str], np.ndarray]:
        _, bcode, _, branch_ids, _ = self.index.tool_meta()
        nb = len(branch_ids)
        F = np.zeros((nb, len(BRANCH_FEATURES)))
        bm_max = float(ev.bm25.max()) if ev.bm25.size and ev.bm25.max() > 0 else 1.0
        bb = [ev.branch_bm25.get(b, 0.0) for b in branch_ids]
        bb_max = max(bb) if bb and max(bb) > 0 else 1.0
        hinted = set(ev.features.vendor_hints)
        llm = (self.llm_scores or {}).get(ev.key, {})
        tk_by_branch: dict[int, list[float]] = {}
        for t_id, c in ev.toolkit_cos.items():
            b = self.index.registry.toolkits[t_id].branch_id
            tk_by_branch.setdefault(branch_ids.index(b), []).append(c)
        for j, b in enumerate(branch_ids):
            m = bcode == j
            if not m.any():
                continue
            cos_b = ev.cos[: len(bcode)][m]
            top3 = np.sort(cos_b)[-3:]
            F[j] = [
                ev.branch_cos.get(b, 0.0),
                bb[j] / bb_max,
                cos_b.max(),
                top3.mean(),
                ev.bm25[: len(bcode)][m].max() / bm_max,
                ev.rrf[: len(bcode)][m].max() * RRF_K,
                max(tk_by_branch.get(j, [0.0])),
                1.0 if any(t in self.index.registry.toolkits and self.index.registry.toolkits[t].branch_id == b
                           for t in hinted) else 0.0,
                llm.get(b, 0.0),
            ]
        return branch_ids, F

    def probs(self, ev: Evidence) -> dict[str, float]:
        branch_ids, F = self.branch_features(ev)
        z = ((F - self.mu) / self.sd) @ self.w
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return dict(zip(branch_ids, p.tolist()))

    # ---------------------------------------------------------------- fitting
    def fit(self, examples: list[tuple[Evidence, set[str]]], iters: int = 600, lr: float = 0.2) -> None:
        """Maximise log P(gold branch) under a softmax over branches (shared weights)."""
        data = [self.branch_features(ev) for ev, _ in examples]
        allF = np.concatenate([F for _, F in data])
        self.mu, self.sd = allF.mean(axis=0), allF.std(axis=0) + 1e-6
        Fs = [(F - self.mu) / self.sd for _, F in data]
        G = [np.array([b in gold for b in ids], dtype=float) for (ids, _), (_, gold) in zip(data, examples)]
        w = np.zeros(len(BRANCH_FEATURES))
        m, v = np.zeros_like(w), np.zeros_like(w)
        for it in range(1, iters + 1):
            grad = -self.l2 * w
            for F, g in zip(Fs, G):
                if not g.any():
                    continue
                z = F @ w
                z -= z.max()
                p = np.exp(z)
                p /= p.sum()
                q = p * g
                q /= q.sum()
                grad += (q - p) @ F
            grad /= len(Fs)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            w += lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
        self.w = w

    def calibrate(self, examples: list[tuple[Evidence, set[str]]]) -> float:
        score_fn, _ = SCORERS[self.method]
        scores = [score_fn(self.probs(ev), gold) for ev, gold in examples]
        self.threshold = min(1.0, conformal_quantile(scores, self.alpha))
        return self.threshold

    # ---------------------------------------------------------------- routing
    def route(self, ev: Evidence) -> RouteDecision:
        p = self.probs(ev)
        _, set_fn = SCORERS[self.method]
        selected = set_fn(p, self.threshold)
        fallback = False
        if max(p.values()) < self.t_low or (self.max_branches and len(selected) > self.max_branches):
            selected = sorted(p, key=lambda b: -p[b])
            fallback = True
        return RouteDecision(branch_scores=p, selected=selected, threshold=self.threshold, fallback=fallback)
