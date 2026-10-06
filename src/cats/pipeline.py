"""CATS orchestration: Algorithm 1 (calibrated hierarchical tool selection) for one request."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .analyzer import ContextAnalyzer, decompose
from .assembler import ContextAssembler
from .index import ToolIndex
from .ledger import Tracer
from .registry import Registry
from .retriever import ToolRetriever
from .router import BranchRouter, Evidence, gather_evidence
from .types import Selection, Task, Turn


@dataclass
class CATSConfig:
    alpha_branch: float = 0.03  # router set misses the gold branch at most this often
    alpha_tool: float = 0.05  # end-to-end expected fraction of gold slots missed
    method: str = "lac"  # lac | aps (lac gives much smaller branch sets at equal coverage)
    k_max: int = 10
    k_min: int = 1
    t_low: float = 0.0  # fall back to flat search below this top branch probability
    max_branches: int | None = None  # fall back to flat search when the set is larger
    budget: int = 8000  # tool-definition token budget per prompt
    order: str = "canonical"  # canonical | append
    reranker: str = "logreg"  # logreg | gbt
    calibration: str = "joint"  # joint (end-to-end CRC) | independent (per stage, as in the workbook)
    k_escalation: int = 5
    bypass_tokens: int = 0  # if the whole catalog fits in this many tokens, load it all (static is fine)
    llm_router: str | None = None  # path to LLM router scores (see cats.llm_router)
    decompose: bool = True  # route multi-step requests per sub-request (see analyzer.decompose)
    k_per_extra_part: int = 3  # extra tool slots per additional sub-request
    dedupe: bool = True  # collapse identical tools published by several servers
    split_calibration: bool = True  # disjoint calibration halves for the branch threshold and the cut-off
    # Jev-style typed decision layer (see cats.jev): path to Jev answers, or None.
    jev: str | None = None
    jev_cascade: bool = True  # ask Jev only when the embedding router is unsure
    jev_gate_tau: float = 0.9  # ... i.e. unless its calibrated set is one branch with p >= tau
    jev_steps: bool = True  # Jev's step count sizes k for multi-step requests
    jev_repair: float | None = 0.5  # add tools for a Jev-needed branch (p >= this) the selection missed
    jev_repair_k: int = 2  # tools added per repaired branch


class CATS:
    name = "cats"

    def __init__(self, registry: Registry, index: ToolIndex, analyzer: ContextAnalyzer | None = None,
                 config: CATSConfig | None = None):
        self.cfg = config or CATSConfig()
        self.registry, self.index = registry, index
        self.analyzer = analyzer or ContextAnalyzer(registry)
        self.router = BranchRouter(index, alpha=self.cfg.alpha_branch, method=self.cfg.method,
                                   t_low=self.cfg.t_low, max_branches=self.cfg.max_branches)
        self.retriever = ToolRetriever(index, k_max=self.cfg.k_max, k_min=self.cfg.k_min,
                                       alpha=self.cfg.alpha_tool, model=self.cfg.reranker,
                                       dedupe=self.cfg.dedupe)
        self.assembler = ContextAssembler(registry, budget=self.cfg.budget, order=self.cfg.order)
        self.calibration_report: dict = {}
        self.router_base: BranchRouter | None = None  # embedding-only router gating the Jev cascade
        self.jev = None
        if self.cfg.llm_router:
            from .llm_router import load_scores

            self.router.llm_scores = load_scores(Path(self.cfg.llm_router), registry)
        if self.cfg.jev:
            from .jev import ExchangeJev, branch_probs

            self.jev = ExchangeJev(self.cfg.jev)
            branches = sorted(registry.branches)
            self.router.llm_scores = {k: branch_probs(a, branches) for k, a in self.jev.answers.items()}
            if self.cfg.jev_cascade:
                self.router_base = BranchRouter(index, alpha=self.cfg.alpha_branch, method=self.cfg.method,
                                                t_low=self.cfg.t_low, max_branches=self.cfg.max_branches)

    def evidence(self, query: str, history: list[Turn] | None = None) -> Evidence:
        return gather_evidence(self.index, self.analyzer, query, history)

    # ----------------------------------------------------------- training
    def fit(self, tasks: list[Task], folds: int = 5) -> CATS:
        """Fit the router, then the reranker on cross-fitted router probabilities.

        In-sample router probabilities are over-confident, so a reranker trained on them would
        over-trust the branch_p feature; each training request's probabilities therefore come
        from a router fitted without it.
        """
        evs = [self.evidence(t.query, t.history) for t in tasks]
        labeled = [(ev, set(t.branches)) for ev, t in zip(evs, tasks)]
        oof: list[dict[str, float] | None] = [None] * len(tasks)
        if folds > 1 and len(tasks) >= 2 * folds:
            for f in range(folds):
                self.router.fit([x for i, x in enumerate(labeled) if i % folds != f])
                for i in range(f, len(tasks), folds):
                    oof[i] = self.router.probs(evs[i])
        self.router.fit(labeled)
        if getattr(self, "router_base", None) is not None:
            self.router_base.fit(labeled)
        examples = []
        for i, (ev, t) in enumerate(zip(evs, tasks)):
            p = oof[i] or self.router.probs(ev)
            top = sorted(p, key=lambda b: -p[b])[:2]
            examples.append((ev, p, t.gold_tools(), sorted(set(t.branches) | set(top))))
        self.retriever.fit(examples)
        return self

    def _jev_answers(self, ev: Evidence):
        return self.jev.get(ev.key) if getattr(self, "jev", None) is not None else None

    def _route(self, ev: Evidence):
        """Route one request; returns (route, external_router_called).

        With a Jev cascade, the embedding-only router decides alone when its calibrated set is a
        single branch with probability >= jev_gate_tau; otherwise Jev's typed answers join the
        router's features. Without Jev, an LLM router prior (if configured) is always used.
        """
        base = getattr(self, "router_base", None)
        if base is not None:
            r0 = base.route(ev)
            if len(r0.selected) == 1 and max(r0.branch_scores.values()) >= self.cfg.jev_gate_tau:
                return r0, False
            return self.router.route(ev), self._jev_answers(ev) is not None
        called = bool(self.router.llm_scores) and ev.key in self.router.llm_scores
        return self.router.route(ev), called

    def _rank(self, query: str, history: list[Turn] | None, exclude: set[str] | None = None,
              ev: Evidence | None = None, branches: list[str] | None = None):
        """Route and rank one request. With ``decompose``, each sub-request is routed and ranked
        too and a tool keeps its best score across the parts, so every step gets candidates."""
        ev = ev or self.evidence(query, history)
        route, called = self._route(ev)
        route.conformal_set = list(route.selected)
        ev.extra["router_called"] = called
        ranked, scores = self.retriever.rank(ev, branches or route.selected, route.branch_scores, exclude)
        if not ranked and branches is None:  # routed branches have nothing left: search everything
            ranked, scores = self.retriever.rank(ev, None, route.branch_scores, exclude)
        parts = decompose(query) if self.cfg.decompose else [query]
        if len(parts) > 1:
            # Round-robin over the parts' rankings (then the whole request's), so each step's best
            # tools come early; a tool keeps its best score. Reranker scores are not comparable
            # across queries, so merging by score alone lets one step crowd out the others.
            best = dict(zip(ranked, np.asarray(scores).tolist()))
            lists = []
            selected = list(route.selected)
            for part in parts:
                ev_p = self.evidence(part, history)
                r_p = self.router.route(ev_p)
                rk, sc = self.retriever.rank(ev_p, branches or r_p.selected, r_p.branch_scores, exclude)
                for t, s in zip(rk, np.asarray(sc).tolist()):
                    best[t] = max(best.get(t, 0.0), s)
                lists.append(rk)
                selected += [b for b in r_p.selected if b not in selected]
            lists.append(ranked)
            merged: dict[str, None] = {}
            for r in range(max(len(x) for x in lists)):
                for lst in lists:
                    if r < len(lst):
                        merged.setdefault(lst[r])
            ranked = list(merged)
            scores = np.array([best[t] for t in ranked])
            route = route.model_copy(update={"selected": selected})
        n_steps = len(parts)
        if self.cfg.jev and self.cfg.jev_steps and ev.extra.get("router_called"):
            from .jev import steps_level

            n_steps = max(n_steps, steps_level(self._jev_answers(ev)))
        return ev, route, ranked, scores, n_steps

    def _repair_fn(self, ev: Evidence, route, exclude: set[str] | None = None):
        """Jev-guided branch repair: for every branch Jev says the request needs (p >= jev_repair)
        that the cut-off left without a tool, add that branch's top tools. Deterministic given the
        request, so it is applied inside conformal calibration too and the guarantee covers it."""
        answers = self._jev_answers(ev) if self.cfg.jev_repair is not None else None
        if not answers or not ev.extra.get("router_called"):
            return None
        from .jev import branch_probs

        need = [b for b, p in branch_probs(answers, sorted(self.registry.branches)).items()
                if p >= self.cfg.jev_repair]
        pools = {b: self.retriever.rank(ev, [b], route.branch_scores, exclude)[0][: self.cfg.jev_repair_k]
                 for b in need}
        branch_of = {t: r.branch_id for t, r in self.registry.tools.items()}

        def repair(selected: list[str]) -> list[str]:
            have = {branch_of[t] for t in selected}
            out = list(selected)
            for b in need:
                if b not in have:
                    out += [t for t in pools[b] if t not in out]
            return out
        return repair

    def _k_max(self, n_parts: int) -> int:
        return self.cfg.k_max + self.cfg.k_per_extra_part * max(0, n_parts - 1)

    def calibrate(self, tasks: list[Task]) -> dict:
        """Branch threshold, then the tool cut-off on the routed pipeline's output.

        With ``split_calibration`` the two thresholds use disjoint halves of the calibration
        data, so the cut-off is calibrated on requests the branch threshold never saw and the
        exchangeability CRC needs holds.
        """
        evs = [self.evidence(t.query, t.history) for t in tasks]
        idx = list(range(len(tasks)))
        if self.cfg.split_calibration and len(tasks) >= 20:
            first, second = idx[0::2], idx[1::2]
        else:
            first = second = idx
        t_branch = self.router.calibrate([(evs[i], set(tasks[i].branches)) for i in first])
        if getattr(self, "router_base", None) is not None:
            self.router_base.calibrate([(evs[i], set(tasks[i].branches)) for i in first])
        lists = []
        for ev, t in ((evs[i], tasks[i]) for i in second):
            oracle = list(t.branches) if self.cfg.calibration == "independent" else None
            _, route, ranked, scores, n_parts = self._rank(t.query, t.history, ev=ev, branches=oracle)
            lists.append((ranked, scores, t.gold, self._k_max(n_parts), self._repair_fn(ev, route)))
        lam = self.retriever.calibrate(lists)
        self.calibration_report = {
            "n": len(tasks), "n_branch": len(first), "n_tool": len(second),
            "branch_threshold": t_branch, "lambda": lam,
            # If False, no cut-off reaches alpha_tool: k_max is used and the tool guarantee does not hold.
            "tool_target_feasible": self.retriever.feasible, "min_achievable_alpha_tool": self.retriever.min_alpha,
            **asdict(self.cfg),
        }
        return self.calibration_report

    # ------------------------------------------------------------ inference
    def select(self, query: str, history: list[Turn] | None = None, exclude: set[str] | None = None,
               session_order: list[str] | None = None) -> Selection:
        tr = Tracer()
        if self.cfg.bypass_tokens and self.registry.total_tokens() <= self.cfg.bypass_tokens:
            ids = sorted(set(self.registry.tools) - (exclude or set()))
            return Selection(strategy=self.name, tool_ids=ids, tool_tokens=self.registry.total_tokens(ids))
        with tr.span("analyze"):
            ev = self.evidence(query, history)
        with tr.span("route+retrieve") as sp:
            _, route, ranked, scores, n_parts = self._rank(query, history, exclude, ev=ev)
            sp.meta.update(selected=route.selected, fallback=route.fallback, parts=n_parts)
            tool_ids, kept = self.retriever.cut(ranked, scores, k_max=self._k_max(n_parts))
            repair = self._repair_fn(ev, route, exclude)
            if repair is not None:
                fixed = repair(tool_ids)
                kept = kept + [0.0] * (len(fixed) - len(tool_ids))
                tool_ids = fixed
            sp.meta["k"] = len(tool_ids)
        with tr.span("assemble") as sp:
            asm = self.assembler.assemble(tool_ids, kept, session_order)
            sp.tokens_in = asm.tool_tokens
        overhead, router_kind = 0, ""
        if ev.extra.get("router_called"):
            if getattr(self, "jev", None) is not None:
                from .jev import route_request

                overhead, router_kind = route_request(self.registry, query, history).input_tokens(), "jev"
            else:
                from .llm_router import router_overhead_tokens

                overhead, router_kind = router_overhead_tokens(self.registry, query), "llm"
        score_of = dict(zip(tool_ids, kept))
        return Selection(strategy=self.name, tool_ids=asm.tool_ids, route=route, tool_tokens=asm.tool_tokens,
                         overhead_tokens=overhead, router_calls=int(bool(router_kind)), router_kind=router_kind,
                         spans=tr.spans, scores=[float(score_of[t]) for t in asm.tool_ids])

    def escalate(self, capability: str, exclude: set[str], k: int | None = None) -> Selection:
        """request_tools: re-route on the described capability, never re-offering loaded tools."""
        tr = Tracer()
        with tr.span("escalate"):
            ev = self.evidence(capability)
            route = self.router.route(ev)
            ranked, scores = self.retriever.rank(ev, route.selected, route.branch_scores, exclude)
            if not ranked:  # routed branches exhausted: flat search over everything left
                ranked, scores = self.retriever.rank(ev, None, route.branch_scores, exclude)
            k = k or self.cfg.k_escalation
            tool_ids, kept = ranked[:k], np.asarray(scores[:k]).tolist()
        return Selection(strategy=f"{self.name}:escalate", tool_ids=tool_ids, route=route,
                         tool_tokens=self.registry.total_tokens(tool_ids), spans=tr.spans, scores=kept)
