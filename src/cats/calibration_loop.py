"""Calibration loop: logged outcomes re-tune the router and its thresholds (workbook Figure 3,
"calibration loop: logs re-tune router thresholds"; FR16, "router fine-tuned on logged routing
decisions").

CATS is calibrated once, offline, on labeled requests. In deployment the catalog drifts (servers
are added, tasks change), so the calibrated thresholds stop meaning what they promised. This module
closes the loop with the only labels a deployment has for free, the agent's own behaviour:

    log          LoggedDecision = request, routed branches, tools shown, tools called, request_tools
                 escalations. ``load_agent_sessions`` reads the session files that cats.agent_cli
                 writes (runs/<run>/sessions/<task>.json); the dispatcher's RunResult carries the
                 same fields.
    label        ``implicit_labels``: a tool the agent called successfully is a positive label for
                 the tool and its branch; a tool it had to obtain through request_tools marks a miss
                 of the first selection (weighted up); shown-but-uncalled tools are weak negatives.
                 A record with no successful call gives no label.
    fine-tune    ``finetune_router`` warm-starts the shared-weight conditional logit
                 (cats.router.BranchRouter) from its current weights and takes a few Adam steps on
                 train + logged labels, with an L2 pull towards the old weights so a small log cannot
                 undo the offline fit. ``refit_router`` refits from scratch on the union instead.
    re-calibrate ``AdaptiveConformal`` is adaptive conformal inference (Gibbs & Candès, "Adaptive
                 Conformal Inference Under Distribution Shift", NeurIPS 2021). After each logged
                 outcome, with err_t = 1 if a needed branch was outside the routed set,

                     alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
                     threshold_{t+1} = Quantile_{1 - alpha_{t+1}}(nonconformity scores seen so far)

                 Whatever the drift, the long-run miss rate satisfies
                 |mean(err) - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma * T).
                 ``RollingRiskControl`` does the same for the tool cut-off lambda, whose loss is the
                 fraction of gold slots missed (rolling risk control: Feldman, Ringel, Bates and
                 Romano, "Achieving Risk Control in Online Learning Settings", TMLR 2023):
                 lambda_{t+1} = clip(lambda_t - gamma * (loss_t - alpha)).

``CalibrationLoop`` wires these to a fitted CATS: ``observe`` takes one outcome, updates the
thresholds in place (``cats.router.threshold``, ``cats.retriever.lam``) and fine-tunes the router
every ``refit_every`` labels. scripts/calibration_loop_eval.py measures it under catalog drift.
"""
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import SCORERS, conformal_quantile
from .router import BranchRouter, Evidence
from .types import Task, Turn


# ============================================================================== logged data
@dataclass
class LoggedDecision:
    """One request as the deployment saw it (no gold labels)."""
    source: str  # e.g. "<run>/<task_id>"
    request: str
    history: list[Turn] = field(default_factory=list)
    route: list[str] = field(default_factory=list)  # branches in the routed set, when logged
    shown: list[str] = field(default_factory=list)  # tools in the first prompt
    called: list[str] = field(default_factory=list)  # valid tool calls, in order
    escalations: list[dict[str, Any]] = field(default_factory=list)  # {"capability", "added": [tool ids]}
    invalid: list[str] = field(default_factory=list)  # calls to tools that were not offered
    done: bool = True
    mode: str = ""  # cats | static | search
    task_id: str = ""


def load_agent_sessions(run_dirs: Iterable[str | Path], tasks: Mapping[str, Task] | None = None,
                        modes: Iterable[str] | None = None) -> list[LoggedDecision]:
    """Read cats.agent_cli session files. The request text comes from ``tasks`` (default: data/tasks),
    because a session file stores only the task id; sessions of unknown tasks are skipped, and so are
    files that are being written (invalid JSON)."""
    if tasks is None:
        from .bench.data import load_tasks
        from .build import DATA

        tasks = {t.id: t for t in load_tasks(DATA / "tasks")}
    wanted = set(modes) if modes else None
    out: list[LoggedDecision] = []
    for run in map(Path, run_dirs):
        try:
            meta = json.loads((run / "run.json").read_text())
        except (OSError, ValueError):
            meta = {}
        for f in sorted((run / "sessions").glob("*.json")):
            try:
                s = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            mode = s.get("mode") or meta.get("mode", "")
            t = tasks.get(s.get("task_id", ""))
            if t is None or (wanted and mode not in wanted):
                continue
            esc = [{"capability": e.get("capability", ""), "added": list(e.get("added", []))}
                   for e in s.get("events", []) if e.get("type") == "request_tools"]
            added = {x for e in esc for x in e["added"]}
            out.append(LoggedDecision(
                source=f"{run.name}/{t.id}", request=t.query, history=list(t.history), route=[],
                shown=[x for x in s.get("loaded", []) if x not in added],
                called=[c["tool"] for c in s.get("calls", []) if "tool" in c], escalations=esc,
                invalid=list(s.get("invalid", [])), done=bool(s.get("done")), mode=mode, task_id=t.id))
    return out


@dataclass
class ImplicitLabel:
    """Labels inferred from behaviour: what the agent used, and what the first selection missed."""
    source: str
    request: str
    history: list[Turn]
    tools: list[str]  # called successfully (positives)
    branches: list[str]  # branches of those tools
    missed: list[str]  # called tools that were not in the first prompt (recovered by request_tools)
    negatives: list[str]  # shown and never called (weak negatives)
    weight: float = 1.0
    task_id: str = ""

    @property
    def escalated(self) -> bool:
        return bool(self.missed)


def implicit_labels(records: Iterable[LoggedDecision], registry, escalation_weight: float = 2.0,
                    require_done: bool = True) -> list[ImplicitLabel]:
    """Turn logged outcomes into labels: called tools are positives, escalation-recovered tools mark a
    routing/retrieval miss (weight ``escalation_weight``). Records with no valid call give no label."""
    out = []
    for r in records:
        if require_done and not r.done:
            continue
        tools = list(dict.fromkeys(t for t in r.called if t in registry.tools))
        if not tools:
            continue
        shown = set(r.shown)
        missed = [t for t in tools if t not in shown]
        branches = sorted({registry.tools[t].branch_id for t in tools})
        negatives = [t for t in r.shown if t not in set(tools) and t in registry.tools]
        out.append(ImplicitLabel(r.source, r.request, list(r.history), tools, branches, missed, negatives,
                                 escalation_weight if missed else 1.0, r.task_id))
    return out


def labels_to_tasks(labels: Iterable[ImplicitLabel]) -> list[Task]:
    """Implicit labels as Tasks (one single-tool slot per called tool), usable by CATS.fit."""
    return [Task(id=f"log:{lab.source}", query=lab.request, history=lab.history, gold=[[t] for t in lab.tools],
                 branches=lab.branches, kind="logged") for lab in labels]


def label_quality(labels: Iterable[ImplicitLabel], gold: Mapping[str, Task]) -> dict[str, float]:
    """Agreement of implicit labels with gold labels (for evaluation only)."""
    bp, br, exact, tp, n = [], [], [], [], 0
    for lab in labels:
        t = gold.get(lab.task_id)
        if t is None:
            continue
        n += 1
        got, want = set(lab.branches), set(t.branches)
        bp.append(len(got & want) / max(1, len(got)))
        br.append(len(got & want) / max(1, len(want)))
        exact.append(float(got == want))
        gt = t.gold_tools()
        tp.append(sum(x in gt for x in lab.tools) / max(1, len(lab.tools)))
    m = (lambda x: float(np.mean(x)) if x else float("nan"))
    return {"n": n, "branch_precision": m(bp), "branch_recall": m(br), "branch_exact": m(exact),
            "tool_precision": m(tp)}


# ============================================================================ router tuning
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def router_nll(router: BranchRouter, examples: Sequence[tuple[Evidence, set[str]]],
               weights: Sequence[float] | None = None) -> float:
    """Weighted mean of -log P(some gold branch) under the router (lower is better)."""
    wts = np.ones(len(examples)) if weights is None else np.asarray(weights, dtype=float)
    tot = 0.0
    for (ev, gold), a in zip(examples, wts):
        p = router.probs(ev)
        tot -= a * math.log(max(1e-12, sum(p.get(b, 0.0) for b in gold)))
    return tot / max(1e-12, wts.sum())


def finetune_router(router: BranchRouter, examples: Sequence[tuple[Evidence, set[str]]],
                    weights: Sequence[float] | None = None, iters: int = 150, lr: float = 0.05,
                    anchor: float = 0.01) -> dict[str, float]:
    """Fine-tune the router in place on (evidence, gold branches) examples, e.g. train + logged labels.

    Same objective as BranchRouter.fit (maximise log sum_{g in gold} p(g)), but warm-started from the
    current weights, with the feature standardisation kept fixed and an L2 pull ``anchor`` towards
    the starting weights. Returns the weighted NLL before and after.
    """
    if not examples:
        return {"n": 0, "nll_before": float("nan"), "nll_after": float("nan")}
    data = [router.branch_features(ev) for ev, _ in examples]
    Fs = [(F - router.mu) / router.sd for _, F in data]
    G = [np.array([b in gold for b in ids], dtype=float) for (ids, _), (_, gold) in zip(data, examples)]
    wts = np.ones(len(Fs)) if weights is None else np.asarray(weights, dtype=float)
    before = router_nll(router, examples, wts)
    w0 = router.w.copy()
    w = w0.copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    for it in range(1, iters + 1):
        grad = -anchor * (w - w0)
        for F, g, a in zip(Fs, G, wts):
            if not g.any():
                continue
            p = _softmax(F @ w)
            q = p * g
            q /= q.sum()
            grad += a * ((q - p) @ F) / wts.sum()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        w = w + lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
    router.w = w
    return {"n": len(Fs), "nll_before": before, "nll_after": router_nll(router, examples, wts)}


def refit_router(router: BranchRouter, examples: Sequence[tuple[Evidence, set[str]]],
                 weights: Sequence[float] | None = None) -> None:
    """Refit from scratch on the union (integer weights repeat an example)."""
    reps = [1] * len(examples) if weights is None else [max(1, round(w)) for w in weights]
    router.fit([ex for ex, r in zip(examples, reps) for _ in range(r)])


# ===================================================================== online calibration
class AdaptiveConformal:
    """Adaptive conformal inference (Gibbs & Candès, 2021) for a set {y : score(y) <= threshold}.

    ``scores`` are nonconformity scores of labeled examples (initially the calibration set); new
    scores join as labels arrive (``window`` keeps only the most recent ones). ``update(err)`` moves
    the working miscoverage level alpha_t by gamma * (alpha - err) and returns the new threshold,
    Quantile_{1 - alpha_t}(scores): +inf (the full set) once alpha_t <= 0 and -inf (the smallest set)
    once alpha_t >= 1, capped at ``cap``.
    """

    def __init__(self, scores: Iterable[float], alpha: float, gamma: float = 0.01, window: int | None = None,
                 cap: float = 1.0):
        self.scores: deque[float] = deque((s for s in scores if math.isfinite(s)), maxlen=window)
        self.alpha, self.gamma, self.cap = alpha, gamma, cap
        self.alpha_t = alpha
        self.errs: list[int] = []
        self.alphas: list[float] = []

    def threshold(self) -> float:
        if self.alpha_t >= 1:
            return -math.inf
        if self.alpha_t <= 0 or not self.scores:
            return self.cap
        return min(self.cap, conformal_quantile(list(self.scores), self.alpha_t))

    def update(self, err: bool, score: float | None = None) -> float:
        self.errs.append(int(bool(err)))
        self.alphas.append(self.alpha_t)
        self.alpha_t += self.gamma * (self.alpha - int(bool(err)))
        if score is not None and math.isfinite(score):
            self.scores.append(score)
        return self.threshold()

    def rescore(self, scores: Iterable[float]) -> None:
        """Replace the stored scores, e.g. after the router changed (keeps alpha_t)."""
        self.scores = deque((s for s in scores if math.isfinite(s)), maxlen=self.scores.maxlen)

    @property
    def miss_rate(self) -> float:
        return float(np.mean(self.errs)) if self.errs else float("nan")

    def bound(self) -> float:
        """Gibbs & Candès Prop. 4.1: |miss_rate - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)."""
        t = max(1, len(self.errs))
        return (max(self.alpha, 1 - self.alpha) + self.gamma) / (self.gamma * t)


class RollingRiskControl:
    """Online risk control for a threshold whose loss grows with it (Feldman et al., 2023).

    lambda_{t+1} = clip(lambda_t - gamma * (loss_t - alpha), lo, hi): a loss above the target lowers
    lambda (the cut-off keeps more tools), a loss below it raises lambda. Unclipped, the long-run
    mean loss converges to alpha for any loss sequence bounded in [0, 1].
    """

    def __init__(self, lam: float, alpha: float, gamma: float = 0.02, lo: float = 0.0, hi: float = 1.0):
        self.lam, self.alpha, self.gamma, self.lo, self.hi = lam, alpha, gamma, lo, hi
        self.losses: list[float] = []

    def update(self, loss: float) -> float:
        self.losses.append(float(loss))
        self.lam = float(min(self.hi, max(self.lo, self.lam - self.gamma * (loss - self.alpha))))
        return self.lam


# ======================================================================== loop around CATS
class CalibrationLoop:
    """Keeps a fitted CATS calibrated from logged outcomes (see the module docstring).

    ``calib`` are the labeled calibration requests whose scores seed the online quantile; ``train``
    are the router's training requests, replayed when fine-tuning so the log adds to them rather
    than replacing them. Set ``gamma=0`` to keep the branch threshold fixed, ``refit_every=0`` to
    never fine-tune, ``gamma_tool > 0`` to also adapt the tool cut-off. ``ev_cache`` lets several
    loops over the same index share request evidence (it does not depend on the router).
    """

    def __init__(self, cats, calib: Sequence[Task] = (), train: Sequence[Task] = (), *, gamma: float = 0.01,
                 gamma_tool: float = 0.0, window: int | None = None, refit_every: int = 0,
                 finetune_iters: int = 150, anchor: float = 0.01, max_logged: int | None = None,
                 ev_cache: dict | None = None):
        self.cats = cats
        self.router: BranchRouter = cats.router
        self.score_fn = SCORERS[self.router.method][0]
        self.calib = [(t.query, list(t.history), set(t.branches), 1.0) for t in calib]
        self.train = [(t.query, list(t.history), set(t.branches), 1.0) for t in train]
        self.logged: deque[tuple[str, list[Turn], set[str], float]] = deque(maxlen=max_logged)
        self.gamma, self.refit_every = gamma, refit_every
        self.finetune_iters, self.anchor = finetune_iters, anchor
        self._ev_cache: dict[tuple, Evidence] = {} if ev_cache is None else ev_cache  # may be shared
        self.aci = AdaptiveConformal(self._scores(self.calib), cats.cfg.alpha_branch, gamma, window)
        self.rrc = (RollingRiskControl(cats.retriever.lam, cats.cfg.alpha_tool, gamma_tool)
                    if gamma_tool > 0 else None)
        self.refits: list[dict[str, float]] = []
        self.n_labels = 0  # labels observed so far (``logged`` may be capped by max_logged)
        if gamma > 0:
            self.router.threshold = self.aci.threshold()

    # ------------------------------------------------------------------ helpers
    def evidence(self, query: str, history: list[Turn] | None = None) -> Evidence:
        """Evidence cached per registry version (a catalog change invalidates it)."""
        key = (query, tuple(h.content for h in history or []), self.cats.registry.version)
        ev = self._ev_cache.get(key)
        if ev is None:
            if len(self._ev_cache) > 20000:
                self._ev_cache.clear()
            ev = self._ev_cache[key] = self.cats.evidence(query, history)
        return ev

    def _scores(self, items) -> list[float]:
        return [self.score_fn(self.router.probs(self.evidence(q, h)), g) for q, h, g, _ in items if g]

    # ----------------------------------------------------------------- observing
    def observe(self, request: str, history: list[Turn] | None, routed: Iterable[str], needed_branches: Iterable[str],
                *, failed: bool = False, tool_loss: float | None = None, weight: float = 1.0) -> dict[str, Any]:
        """One logged outcome. ``needed_branches`` are the branches of the tools the agent used
        (empty if it used none); ``failed`` marks a request the agent could not complete, which
        counts as a miss (conservative: the log cannot tell whether routing caused it)."""
        need = set(needed_branches)
        err = failed or not need <= set(routed)
        score = self.score_fn(self.router.probs(self.evidence(request, history)), need) if need else None
        out: dict[str, Any] = {"err": int(err)}
        if self.gamma > 0:
            self.router.threshold = out["threshold"] = self.aci.update(err, score)
        elif score is not None:
            self.aci.update(err, score)  # keeps the score history for a later switch to adaptive mode
        if self.rrc is not None and tool_loss is not None:
            self.cats.retriever.lam = out["lambda"] = self.rrc.update(tool_loss)
        if need:
            self.logged.append((request, list(history or []), need, weight))
            self.n_labels += 1
            if self.refit_every and self.n_labels % self.refit_every == 0:
                out["refit"] = self.refit()
        return out

    def observe_label(self, label: ImplicitLabel, routed: Iterable[str]) -> dict[str, Any]:
        return self.observe(label.request, label.history, routed, label.branches, weight=label.weight)

    def refit(self) -> dict[str, float]:
        """Fine-tune the router on train + logged labels, then re-score the quantile's examples."""
        items = self.train + list(self.logged)
        examples = [(self.evidence(q, h), g) for q, h, g, _ in items]
        rep = finetune_router(self.router, examples, [w for *_, w in items], self.finetune_iters, anchor=self.anchor)
        self.aci.rescore(self._scores(self.calib + list(self.logged)))
        if self.gamma > 0:
            self.router.threshold = self.aci.threshold()
        rep["n_logged"] = len(self.logged)
        self.refits.append(rep)
        return rep
