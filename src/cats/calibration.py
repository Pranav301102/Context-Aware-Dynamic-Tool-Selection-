"""Split-conformal set selection and conformal risk control (workbook Section 6.1).

Branch sets (FR3):
  * APS (Romano et al., 2020): score = probability mass needed before every gold
    branch is in the set; keep the smallest top-scored prefix whose mass reaches t.
  * LAC (least ambiguous classifier): score = 1 - min gold probability; keep every
    branch with p >= 1 - t. Usually smaller sets than APS for the same coverage.
  With t the ceil((n+1)(1-alpha))-th smallest calibration score, the set holds all
  gold branches with probability >= 1 - alpha under exchangeability.

Tool cut-off (FR4): conformal risk control (Angelopoulos et al., 2024) picks the
lowest-risk-safe threshold lambda on reranker scores, with loss = fraction of gold
slots missed (bounded by 1, monotone in lambda).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return 1.0
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])


def aps_score(probs: dict[str, float], gold: set[str]) -> float:
    order = sorted(probs, key=lambda b: -probs[b])
    mass, worst = 0.0, 0.0
    for b in order:
        mass += probs[b]
        if b in gold:
            worst = mass
    return worst if gold & probs.keys() else float("inf")


def lac_score(probs: dict[str, float], gold: set[str]) -> float:
    present = [probs[b] for b in gold if b in probs]
    return 1.0 - min(present) if present else float("inf")


def aps_set(probs: dict[str, float], t: float) -> list[str]:
    order = sorted(probs, key=lambda b: -probs[b])
    out, mass = [], 0.0
    for b in order:
        out.append(b)
        mass += probs[b]
        if mass >= t - 1e-12:
            break
    return out


def lac_set(probs: dict[str, float], t: float) -> list[str]:
    order = sorted(probs, key=lambda b: -probs[b])
    out = [b for b in order if probs[b] >= 1.0 - t - 1e-12]
    return out or order[:1]


SCORERS = {"aps": (aps_score, aps_set), "lac": (lac_score, lac_set)}


def crc_threshold(losses_at: np.ndarray, lambdas: np.ndarray, alpha: float, bound: float = 1.0) -> float:
    """Conformal risk control.

    losses_at[i, j] is calibration example i's loss when the threshold is lambdas[j].
    lambdas must be sorted descending (strict -> lenient) and losses non-increasing along j.
    Returns the strictest lambda whose adjusted risk n/(n+1) * mean + B/(n+1) <= alpha,
    or the most lenient lambda if none qualifies.
    """
    return crc_fit(losses_at, lambdas, alpha, bound)[0]


def crc_fit(losses_at: np.ndarray, lambdas: np.ndarray, alpha: float, bound: float = 1.0) -> tuple[float, bool, float]:
    """Like crc_threshold, also returning whether alpha is achievable and the smallest alpha that is."""
    n = losses_at.shape[0]
    risk = losses_at.mean(axis=0)
    adjusted = n / (n + 1) * risk + bound / (n + 1)
    # Strictest lambda from which every more lenient lambda is also safe (robust to any
    # non-monotone wiggle in the empirical risk curve).
    safe = adjusted <= alpha
    tail_safe = np.flip(np.logical_and.accumulate(np.flip(safe)))
    ok = np.flatnonzero(tail_safe)
    best = float(adjusted.min())
    if ok.size:
        return float(lambdas[ok[0]]), True, best
    return float(lambdas[-1]), False, best


def slot_miss(selected: set[str], gold: Sequence[Sequence[str]]) -> float:
    """Fraction of gold slots with no acceptable tool selected."""
    if not gold:
        return 0.0
    return sum(1 for slot in gold if not selected.intersection(slot)) / len(gold)
