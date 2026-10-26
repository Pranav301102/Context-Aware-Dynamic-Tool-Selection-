"""Evaluate CATS and the flat baselines on a converted external benchmark (FR9).

Protocols
  split     the repository's stratified 40/20/40 train / calibration / test split
            (``cats.build.build_system``), for benchmarks with enough tasks (BFCL)
  crossfit  K folds: each fold is the test set once, the rest is split 2:1 into training and
            calibration data, and the per-task records of all folds are pooled, so every task is
            tested exactly once (LiveMCPBench has only 94 usable tasks)

Optional cold-start training (``--pseudo``): one pseudo-request per tool, the first sentence of its
own description, is added to the training data (never to calibration or test). It needs no model
and no labelled requests, which is the situation of a newly connected catalog.

Strategies are the ones in ``cats.bench.experiments.baseline_strategies`` plus flat reranked
retrieval at CATS's mean k (``B3_hybrid_rerank@k=CATS``), so the routing gain is read at equal k.
Paired statistics (exact McNemar on full cover, paired bootstrap, Holm across baselines) come from
``cats.bench.stats``.

  python -m cats.bench.external.run livemcpbench --protocol crossfit --folds 5
  python -m cats.bench.external.run bfcl --protocol split
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ...build import ROOT, System, build_system
from ...calibration import slot_miss
from ...pipeline import CATS, CATSConfig
from ...strategies import CATSStrategy, FlatReranked, FlatTopK, Oracle, StaticLoading, TwoStage
from ...types import Task
from ..harness import evaluate, paired_diff, summarize
from ..stats import compare_to_reference
from . import load_benchmark
from .common import ExternalBenchmark

EXT = ROOT / "data" / "external"
RESULTS = ROOT / "results"
MATCHED = "B3_hybrid_rerank@k=CATS"
K_GRID = (1, 2, 3, 5, 10, 15, 20, 30, 50)


def pseudo_tasks(registry, max_words: int = 40) -> list[Task]:
    """One training request per tool: the first sentence of its description (cold start)."""
    out = []
    for i, (tid, t) in enumerate(sorted(registry.tools.items())):
        first = re.split(r"(?<=[.!?])\s", t.description.strip())[0]
        words = first.split()
        if len(words) < 3:
            continue
        out.append(Task(id=f"pseudo-{i}", query=" ".join(words[:max_words]), gold=[[tid]],
                        branches=[t.branch_id], kind="pseudo"))
    return out


def strategies(s: System, k: int = 5) -> list:
    reg, idx, an = s.registry, s.index, s.analyzer
    gold = {t.query: t.gold for t in s.train + s.calib + s.test}
    return [
        StaticLoading(reg),
        FlatTopK(reg, idx, an, "bm25", k),
        FlatTopK(reg, idx, an, "dense", k),
        FlatTopK(reg, idx, an, "hybrid", k),
        FlatReranked(reg, idx, an, k, s.cats),
        FlatReranked(reg, idx, an, 10, s.cats),
        TwoStage(reg, idx, an, 3, k),
        CATSStrategy(s.cats),
        Oracle(reg, gold),
    ]


def _run_fold(s: System, k: int) -> tuple[dict[str, list], dict]:
    recs: dict[str, list] = {}
    for strat in strategies(s, k):
        recs[strat.name] = evaluate(strat, s.test, s.registry)
    k_cats = max(1, round(float(np.mean([r.k for r in recs["CATS"]]))))
    matched = FlatReranked(s.registry, s.index, s.analyzer, k_cats, s.cats)
    matched.name = MATCHED
    recs[MATCHED] = evaluate(matched, s.test, s.registry)
    cal = dict(s.cats.calibration_report)
    cal["k_matched"] = k_cats
    cal["n_train"], cal["n_calib"], cal["n_test"] = len(s.train), len(s.calib), len(s.test)
    return recs, cal


def _recall_sums(s: System, grid=K_GRID) -> dict[int, list[float]]:
    """Slot recall and full cover of flat reranked retrieval at every k in ``grid`` (sums over the
    fold's test tasks): how large a flat prompt must be to reach a coverage target."""
    flat = FlatReranked(s.registry, s.index, s.analyzer, max(grid), s.cats)
    sums = {k: [0.0, 0.0] for k in grid}
    for t in s.test:
        ids = flat.select(t.query, t.history).tool_ids
        for k in grid:
            miss = slot_miss(set(ids[:k]), t.gold)
            sums[k][0] += 1 - miss
            sums[k][1] += float(miss == 0)
    return sums


def calibration_summary(cals: list[dict[str, Any]]) -> dict[str, Any]:
    """One calibration report shaped like ``bench_*.json``'s (scalars at the top level, as other tools
    reading results/*.json expect); per-fold reports are kept under ``folds``."""
    if len(cals) == 1:
        return dict(cals[0])
    out: dict[str, Any] = {k: v for k, v in cals[0].items() if not isinstance(v, dict | list) and k != "fold"}
    for key in ("branch_threshold", "lambda", "min_achievable_alpha_tool", "k_matched", "n", "n_train", "n_calib",
                "n_test"):
        vals = [c[key] for c in cals if isinstance(c.get(key), int | float)]
        if vals:
            out[key] = float(np.mean(vals))
    out["tool_target_feasible"] = all(bool(c.get("tool_target_feasible")) for c in cals)
    out["folds"] = cals
    return out


def stratified_folds(tasks: list[Task], folds: int, seed: int) -> list[list[Task]]:
    groups: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        groups[t.branches[0] if t.branches else ""].append(t)
    rng = random.Random(seed)
    out: list[list[Task]] = [[] for _ in range(folds)]
    i = 0
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        for t in g:  # deal round-robin, continuing across groups so folds stay balanced
            out[i % folds].append(t)
            i += 1
    return out


def evaluate_benchmark(bench: ExternalBenchmark, cfg: CATSConfig | None = None, protocol: str = "split",
                       folds: int = 5, seed: int = 0, k: int = 5, pseudo: bool = False,
                       per_task: bool = True, n_boot: int = 10_000, variant: str | None = None) -> dict[str, Any]:
    """``variant`` names a robustness or ablation rerun (figures show only main runs)."""
    cfg = cfg or CATSConfig()
    t0 = time.perf_counter()
    reg = bench.registry()
    tasks = [t for t in bench.tasks if all(g in reg.tools for g in t.gold_tools())]
    extra = pseudo_tasks(reg) if pseudo else []
    records: dict[str, list] = defaultdict(list)
    calibrations = []
    curve = {k: [0.0, 0.0] for k in K_GRID}

    def add_curve(sys_: System) -> None:
        for kk, (rec, cov) in _recall_sums(sys_).items():
            curve[kk][0] += rec
            curve[kk][1] += cov

    if protocol == "split":
        s = build_system(cfg, registry=reg, tasks=tasks, seed=seed, fit=False)
        s.cats.fit(s.train + extra)
        s.cats.calibrate(s.calib)
        recs, cal = _run_fold(s, k)
        for name, r in recs.items():
            records[name] += r
        calibrations.append(cal)
        add_curve(s)
        embedder = s.index.embedder.name
    elif protocol == "crossfit":
        base = build_system(cfg, registry=reg, tasks=[], seed=seed, fit=False)  # index built once
        embedder = base.index.embedder.name
        parts = stratified_folds(tasks, folds, seed)
        for f in range(folds):
            test = parts[f]
            rest = [t for j, p in enumerate(parts) if j != f for t in p]
            random.Random(seed * 1000 + f).shuffle(rest)
            n_cal = len(rest) // 3
            calib, train = rest[:n_cal], rest[n_cal:]
            cats = CATS(reg, base.index, base.analyzer, cfg)
            cats.fit(train + extra)
            cats.calibrate(calib)
            s = System(reg, base.index, base.analyzer, cats, train, calib, test)
            recs, cal = _run_fold(s, k)
            for name, r in recs.items():
                records[name] += r
            cal["fold"] = f
            calibrations.append(cal)
            add_curve(s)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    static = reg.total_tokens()
    summaries = [summarize(records[name], static) for name in records]
    for sm, name in zip(summaries, records):
        sm["strategy"] = name
    stats = {f: compare_to_reference(records, "CATS", field=f, n_boot=n_boot, seed=seed,
                                     exclude=("B0_static", "O_oracle"))
             for f in ("full_cover", "slot_recall")}
    out: dict[str, Any] = {
        "benchmark": bench.name, "meta": bench.meta, "registry": reg.stats(), "embedder": embedder,
        "protocol": {"name": protocol, "folds": folds if protocol == "crossfit" else None, "seed": seed, "k": k,
                     "pseudo_train": pseudo, "n_pseudo": len(extra), "n_tasks": len(tasks), "variant": variant},
        "config": asdict(cfg), "calibration": calibration_summary(calibrations), "summaries": summaries,
        "paired_full_cover_vs_cats": {n: paired_diff(records["CATS"], r) for n, r in records.items() if n != "CATS"},
        "stats": stats, "elapsed_s": time.perf_counter() - t0,
        "recall_at_k": {"strategy": "B3_hybrid_rerank", "k": list(K_GRID),
                        "slot_recall": [curve[kk][0] / max(1, len(records["CATS"])) for kk in K_GRID],
                        "full_cover": [curve[kk][1] / max(1, len(records["CATS"])) for kk in K_GRID]},
    }
    if per_task:
        ids = [r.task_id for r in records["CATS"]]
        out["per_task"] = {"task_ids": ids, "kinds": [r.kind for r in records["CATS"]], "strategies": {}}
        for name, recs in records.items():
            by = {r.task_id: r for r in recs}
            out["per_task"]["strategies"][name] = {
                "full_cover": [int(by[i].full_cover) for i in ids],
                "slot_recall": [round(float(by[i].slot_recall), 4) for i in ids],
                "k": [int(by[i].k) for i in ids],
                "tool_tokens": [int(by[i].tool_tokens) for i in ids],
            }
    return out


def headline(res: dict[str, Any]) -> str:
    rows = {s["strategy"]: s for s in res["summaries"]}
    lines = [(f"{res['benchmark']}: {res['registry']['tools']} tools, {res['protocol']['n_tasks']} tasks, "
              f"protocol={res['protocol']['name']} pseudo={res['protocol']['pseudo_train']}")]
    for name, s in rows.items():
        lines.append(f"  {name:<26} cover={s['full_cover']:.3f} recall={s['slot_recall']:.3f} k={s['k_mean']:.1f} "
                     f"tok%={s['token_pct_static']:.2f} p95={s['latency_p95_ms']:.1f}ms")
    for name, c in res["stats"]["full_cover"]["comparisons"].items():
        b = c["bootstrap"]
        lines.append(f"  CATS - {name:<24} {b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] "
                     f"p={c['p_value']:.4f} holm={c['p_holm']:.4f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("benchmark")
    p.add_argument("--raw", default=None, help="raw dir (default data/external/<benchmark>/raw)")
    p.add_argument("--protocol", choices=["split", "crossfit"], default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pseudo", action="store_true", help="add one description pseudo-request per tool to training")
    p.add_argument("--strict", action="store_true", help="BFCL: no label widening across function variants")
    p.add_argument("--no-decompose", dest="decompose", action="store_false", help="CATS without query decomposition")
    p.add_argument("--alpha-tool", dest="alpha_tool", type=float, default=None)
    p.add_argument("--alpha-branch", dest="alpha_branch", type=float, default=None)
    p.add_argument("--k-max", dest="k_max", type=int, default=None)
    p.add_argument("--variant", default=None, help="label for an ablation or robustness rerun")
    p.add_argument("--max-tasks", type=int, default=0, help="subsample tasks (quick runs)")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--name", default=None, help="results/<name>.json (default external_<benchmark>)")
    a = p.parse_args(argv)
    raw = Path(a.raw) if a.raw else EXT / a.benchmark / "raw"
    kwargs = {"widen": not a.strict} if a.benchmark == "bfcl" else {}
    bench = load_benchmark(a.benchmark, raw, **kwargs)
    if a.max_tasks and len(bench.tasks) > a.max_tasks:
        bench.tasks = random.Random(a.seed).sample(bench.tasks, a.max_tasks)
    protocol = a.protocol or ("crossfit" if len(bench.tasks) < 500 else "split")
    cfg = CATSConfig(decompose=a.decompose)
    for key in ("alpha_tool", "alpha_branch", "k_max"):
        if getattr(a, key) is not None:
            setattr(cfg, key, getattr(a, key))
    variant = a.variant or ("strict labels" if a.strict else None) or ("no decomposition" if not a.decompose else None)
    res = evaluate_benchmark(bench, cfg, protocol=protocol, folds=a.folds, seed=a.seed, k=a.k, pseudo=a.pseudo,
                             n_boot=a.n_boot, variant=variant)
    print(headline(res))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{a.name or 'external_' + a.benchmark}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))
    print("saved", path)


if __name__ == "__main__":
    main()


__all__ = ["evaluate_benchmark", "pseudo_tasks", "strategies", "stratified_folds"]
