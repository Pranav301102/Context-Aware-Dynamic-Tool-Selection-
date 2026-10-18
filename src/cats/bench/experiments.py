"""Experiments behind the evaluation plan (workbook Chapter 7): baselines, ablations,
coverage validity, scaling and prompt caching. Each returns plain dicts for JSON."""
from __future__ import annotations

import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..assembler import ContextAssembler
from ..build import DATA, System, build_registry, build_system
from ..ledger import PrefixCache, Prices, RequestCost
from ..pipeline import CATSConfig
from ..strategies import CATSStrategy, FlatReranked, FlatTopK, Oracle, StaticLoading, TwoStage
from ..tokens import default_counter
from .data import load_tasks
from .harness import evaluate, paired_diff, summarize
from .poolgen import build_pool


def baseline_strategies(s: System, k: int = 5) -> list:
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


def run_baselines(s: System, k: int = 5) -> dict:
    static = s.registry.total_tokens()
    out, recs = [], {}
    for strat in baseline_strategies(s, k):
        r = evaluate(strat, s.test, s.registry)
        recs[strat.name] = r
        out.append(summarize(r, static))
    cats_name = "CATS"
    paired = {name: paired_diff(recs[cats_name], r) for name, r in recs.items() if name != cats_name}
    return {"registry": s.registry.stats(), "embedder": s.index.embedder.name,
            "calibration": s.cats.calibration_report, "summaries": out, "paired_full_cover_vs_cats": paired}


def run_variants(variants: dict[str, CATSConfig], seed: int = 0, enrichment: Path | None = None,
                 registry=None) -> list[dict]:
    """Same split, different CATS configurations."""
    rows = []
    for name, cfg in variants.items():
        s = build_system(cfg, seed=seed, enrichment=enrichment, registry=registry)
        summ = summarize(evaluate(CATSStrategy(s.cats, name), s.test, s.registry), s.registry.total_tokens())
        summ["calibration"] = s.cats.calibration_report
        rows.append(summ)
    return rows


def coverage_validity(cfg: CATSConfig, seeds=range(5), enrichment: Path | None = None) -> dict:
    """Conformal guarantees are marginal over calibration draws: average over several splits."""
    reg = build_registry()
    rows = []
    for seed in seeds:
        s = build_system(cfg, seed=seed, enrichment=enrichment, registry=reg)
        summ = summarize(evaluate(CATSStrategy(s.cats), s.test, s.registry), reg.total_tokens())
        miss = 1 - summ["slot_recall"]
        rows.append({"seed": seed, "branch_cover": summ.get("branch_cover"), "slot_miss": miss,
                     "full_cover": summ["full_cover"], "k_mean": summ["k_mean"],
                     "branches_mean": summ.get("branches_mean"), "token_pct_static": summ["token_pct_static"],
                     "lambda": s.cats.calibration_report["lambda"],
                     "t_branch": s.cats.calibration_report["branch_threshold"]})
    return {
        "config": cfg.__dict__, "rows": rows,
        "branch_cover_mean": float(np.mean([r["branch_cover"] for r in rows])),
        "branch_target": 1 - cfg.alpha_branch,
        "slot_miss_mean": float(np.mean([r["slot_miss"] for r in rows])),
        "slot_miss_target": cfg.alpha_tool,
    }


def scaling(sizes: list[int], cfg: CATSConfig, seed: int = 0, enrichment: Path | None = None,
            k: int = 5) -> list[dict]:
    """Tool-pool scaling: real corpus + distractor domains + vendor clones (FR14)."""
    base = build_registry()
    tasks = load_tasks(DATA / "tasks", list(base.branches))
    rows = []
    for n in sizes:
        rng = random.Random(seed)
        sub = tasks
        if n < len(base.tools):
            # Keep a task subset whose gold tools fit in ~60% of the pool.
            pool_tasks, gold = [], set()
            shuffled = tasks[:]
            rng.shuffle(shuffled)
            for t in shuffled:
                g = gold | t.gold_tools()
                if len(g) <= 0.6 * n:
                    pool_tasks.append(t)
                    gold = g
            sub = pool_tasks
        t0 = time.perf_counter()
        pool, ptasks = build_pool(base, sub, n, seed=seed, distractor_dir=DATA / "distractors")
        s = build_system(cfg, registry=pool, tasks=ptasks, seed=seed, enrichment=enrichment)
        build_s = time.perf_counter() - t0
        static = pool.total_tokens()
        row = {"size": len(pool.tools), "branches": len(pool.branches), "tasks_test": len(s.test),
               "static_tokens": static, "build_s": build_s}
        for strat in (FlatTopK(pool, s.index, s.analyzer, "hybrid", k), FlatReranked(pool, s.index, s.analyzer, k, s.cats),
                      FlatReranked(pool, s.index, s.analyzer, 2 * k, s.cats), CATSStrategy(s.cats)):
            summ = summarize(evaluate(strat, s.test, pool), static)
            row[strat.name] = {key: summ.get(key) for key in (
                "full_cover", "slot_recall", "k_mean", "tool_tokens", "token_pct_static", "latency_p50_ms",
                "latency_p95_ms", "branch_cover", "branches_mean")}
        row["cats_calibration"] = {k2: s.cats.calibration_report.get(k2) for k2 in ("branch_threshold", "lambda", "n")}
        rows.append(row)
    return rows


def cache_sessions(s: System, session_len: int = 5, n_sessions: int = 40, seed: int = 0,
                   prices: Prices | None = None) -> dict:
    """Multi-turn sessions of related requests: what tool ordering does to prompt caching.

    canonical  each turn gets its own selection, sorted (cache hits only on identical sets)
    sticky     tools loaded earlier in the session stay, new ones are appended at the end
    static     every tool, every turn (cache-friendly but large)
    """
    prices = prices or Prices()
    counter = default_counter()
    rng = random.Random(seed)
    by_branch: dict[str, list] = {}
    for t in s.test:
        by_branch.setdefault(t.branches[0], []).append(t)
    sessions = []
    for _ in range(n_sessions):
        b = rng.choice(sorted(by_branch))
        pool = by_branch[b]
        sessions.append([rng.choice(pool) for _ in range(session_len)])

    asm = s.cats.assembler
    unbounded = ContextAssembler(s.registry, budget=10**9)  # static loading has no budget
    results = {}
    for policy in ("static", "canonical", "sticky"):
        cache = PrefixCache()
        total, tool_tok, covered, n = RequestCost(), 0, 0, 0
        for sess in sessions:
            loaded: list[str] = []
            history_tokens = 0
            for task in sess:
                if policy == "static":
                    a = unbounded.assemble(sorted(s.registry.tools))
                else:
                    sel = s.cats.select(task.query, task.history)
                    total.router_tokens += sel.overhead_tokens
                    if policy == "sticky":
                        loaded += [t for t in sel.tool_ids if t not in loaded]
                        a = _assemble_append(asm, loaded)
                    else:
                        a = asm.assemble(sel.tool_ids)
                history_tokens += counter.count(task.query) + 80
                keys = [t["name"] for t in a.tools]
                total += cache.charge(keys, a.block_tokens, history_tokens, 80)
                tool_tok += a.tool_tokens
                covered += all(set(a.tool_ids).intersection(slot) for slot in task.gold)
                n += 1
        results[policy] = {
            "usd_per_session": total.usd(prices) / len(sessions),
            "cache_read_share": total.cache_read / max(1, total.cache_read + total.cache_write + total.input_tokens),
            "tool_tokens_per_turn": tool_tok / n,
            "full_cover": covered / n,
        }
    return {"session_len": session_len, "n_sessions": n_sessions, "policies": results}


def _assemble_append(asm, loaded):
    saved = asm.order
    asm.order = "append"
    try:
        # Over budget, evict the tools loaded longest ago (higher score = more recent).
        return asm.assemble(loaded, scores=[float(i) for i in range(len(loaded))], session_order=loaded)
    finally:
        asm.order = saved


def default_variants(base: CATSConfig) -> dict[str, CATSConfig]:
    return {
        "CATS (lac, joint)": replace(base, method="lac"),
        "CATS aps": replace(base, method="aps"),
        "CATS independent-calib": replace(base, calibration="independent"),
        "CATS gbt-reranker": replace(base, reranker="gbt"),
        "CATS k_max=5": replace(base, k_max=5),
        "CATS k_max=20": replace(base, k_max=20),
        "CATS alpha_tool=0.10": replace(base, alpha_tool=0.10),
        "CATS alpha_tool=0.20": replace(base, alpha_tool=0.20),
    }
