"""Statistics for paired strategy comparisons (workbook §7.2: 95% CIs, paired tests, several baselines).

Every strategy is evaluated on the same tasks, so comparisons are paired:

* ``mcnemar_exact``      exact (binomial) McNemar test for paired binary outcomes, e.g. full cover
                         or end-to-end task success of two strategies on the same tasks;
* ``wilson_interval``    Wilson score interval for one success rate (better than the normal
                         interval near 0 or 1 and for small n, e.g. 36/36);
* ``paired_bootstrap``   percentile bootstrap CI and two-sided p-value for a mean paired difference
                         (any per-task metric: slot recall, tokens, dollars);
* ``holm_bonferroni``    step-down family-wise error control when CATS is compared with several
                         baselines at once.

``compare_paired`` and ``compare_to_reference`` apply them to harness ``Record`` lists (or dicts
with a ``task_id``). Pure Python and numpy; no SciPy needed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from statistics import NormalDist
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------- McNemar
def binom_tail_half(k: int, n: int) -> Fraction:
    """P(X <= k) for X ~ Binomial(n, 1/2), exactly."""
    if k < 0:
        return Fraction(0)
    k = min(k, n)
    return Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)


def mcnemar_exact(b: int, c: int, mid_p: bool = False) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    ``b`` = pairs where A succeeded and B failed, ``c`` = the reverse. Under H0 the discordant pairs
    split 50/50, so p = min(1, 2 P(X <= min(b, c))) with X ~ Bin(b + c, 1/2). ``mid_p`` subtracts
    half the point probability (less conservative, Fagerland et al. 2013).
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = binom_tail_half(k, n)
    if mid_p:
        tail -= Fraction(math.comb(n, k), 2 ** n) / 2
    return float(min(Fraction(1), 2 * tail))


def mcnemar(a: Sequence[bool], b: Sequence[bool], mid_p: bool = False) -> dict[str, Any]:
    """Exact McNemar test on two aligned sequences of binary outcomes."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must have the same length")
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    n10 = int(np.sum(x & ~y))  # A right, B wrong
    n01 = int(np.sum(~x & y))  # A wrong, B right
    n = len(x)
    return {
        "n": n, "a_only": n10, "b_only": n01, "both": int(np.sum(x & y)), "neither": int(np.sum(~x & ~y)),
        "diff": (n10 - n01) / n if n else float("nan"),
        "p_value": mcnemar_exact(n10, n01, mid_p=mid_p),
    }


# ----------------------------------------------------------------------------- Wilson
def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)


# -------------------------------------------------------------------------- bootstrap
def paired_bootstrap(x: Sequence[float], y: Sequence[float], n_boot: int = 10_000, confidence: float = 0.95,
                     seed: int = 0, chunk: int = 500) -> dict[str, Any]:
    """Percentile bootstrap of mean(x - y) over paired observations.

    Resamples task indices (not the two arms separately), so the correlation between strategies on
    the same task is kept. The two-sided p-value is twice the smaller share of bootstrap means on
    either side of zero (capped at 1); its resolution is 1/n_boot, so 0 means "below 1/n_boot".
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError("paired samples must have the same length")
    d = xa - ya
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci": (float("nan"), float("nan")), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        means[start:start + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
    a = (1 - confidence) / 2
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    p = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {"n": n, "mean": float(d.mean()), "ci": (float(lo), float(hi)), "p_value": p}


# ------------------------------------------------------------------------------- Holm
def holm_bonferroni(pvalues: Sequence[float] | Mapping[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """Holm's step-down adjusted p-values and rejections (controls the family-wise error rate).

    Accepts a list or a {name: p} mapping and returns the adjusted values in the same shape.
    """
    keys = list(pvalues.keys()) if isinstance(pvalues, Mapping) else list(range(len(pvalues)))
    ps = [float(pvalues[k]) for k in keys]
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    reject = [a <= alpha for a in adj]
    if isinstance(pvalues, Mapping):
        return {"adjusted": dict(zip(keys, adj)), "reject": dict(zip(keys, reject)), "alpha": alpha}
    return {"adjusted": adj, "reject": reject, "alpha": alpha}


# ------------------------------------------------------------------- record helpers
def _get(r: Any, field: str):
    return r[field] if isinstance(r, Mapping) else getattr(r, field)


def align(a: Sequence[Any], b: Sequence[Any], field: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Values of ``field`` for the tasks both record lists share, in a's order."""
    bm = {_get(r, "task_id"): _get(r, field) for r in b}
    ids = [_get(r, "task_id") for r in a if _get(r, "task_id") in bm]
    av = {_get(r, "task_id"): _get(r, field) for r in a}
    return (np.array([float(av[i]) for i in ids]), np.array([float(bm[i]) for i in ids]), ids)


def compare_paired(a: Sequence[Any], b: Sequence[Any], field: str = "full_cover", n_boot: int = 10_000,
                   seed: int = 0) -> dict[str, Any]:
    """Paired comparison of strategy A against B on one per-task field.

    Binary fields (full cover, success) get the exact McNemar test and Wilson intervals for each arm;
    every field gets the paired bootstrap of the mean difference.
    """
    x, y, ids = align(a, b, field)
    out: dict[str, Any] = {"field": field, "n": len(ids), "mean_a": float(x.mean()) if len(x) else float("nan"),
                           "mean_b": float(y.mean()) if len(y) else float("nan")}
    out["bootstrap"] = paired_bootstrap(x, y, n_boot=n_boot, seed=seed)
    if len(x) and set(np.unique(np.concatenate([x, y]))) <= {0.0, 1.0}:
        out["mcnemar"] = mcnemar(x.astype(bool), y.astype(bool))
        out["wilson_a"] = wilson_interval(int(x.sum()), len(x))
        out["wilson_b"] = wilson_interval(int(y.sum()), len(y))
    return out


def compare_to_reference(records: Mapping[str, Sequence[Any]], reference: str, field: str = "full_cover",
                         alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0,
                         exclude: Sequence[str] = ()) -> dict[str, Any]:
    """Compare ``reference`` with every other strategy, Holm-correcting the family of tests.

    The McNemar p-value is used for binary fields and the bootstrap p-value otherwise.
    """
    rows: dict[str, Any] = {}
    for name, recs in records.items():
        if name == reference or name in exclude:
            continue
        rows[name] = compare_paired(records[reference], recs, field=field, n_boot=n_boot, seed=seed)
    pv = {k: (v["mcnemar"]["p_value"] if "mcnemar" in v else v["bootstrap"]["p_value"]) for k, v in rows.items()}
    holm = holm_bonferroni(pv, alpha=alpha) if pv else {"adjusted": {}, "reject": {}}
    for k, row in rows.items():
        row["p_value"] = pv[k]
        row["p_holm"] = holm["adjusted"][k]
        row["significant_holm"] = holm["reject"][k]
    return {"reference": reference, "field": field, "alpha": alpha, "comparisons": rows}
