"""Calibration loop under catalog drift (FR16, workbook Figure 3): fixed vs. adaptive thresholds.

    python scripts/calibration_loop_eval.py [--runs 3] [--gamma 0.02] [--skip-logged]
        -> results/calibration_loop.json

Drift simulation, per run: two branches are held out of the catalog; CATS is fitted and calibrated on
the other ten (their tasks only). Test requests then arrive as a stream:

    phase 1   half of the held-in test requests
    drift     the two branches are published (registry upsert; the index syncs; the shared-weight router
              scores them without retraining, FR15)
    phase 2   the other held-in test requests mixed with every request that needs a new branch

Variants see the same stream and index; each has its own router copy and thresholds:

    fixed          the offline thresholds, never updated (today's CATS)
    aci_gold       adaptive conformal inference on the branch threshold and rolling risk control on the
                   tool cut-off, fed the true outcome (oracle feedback; an upper bound)
    aci_implicit   the same, fed only what a deployment logs: a simulated agent calls a shown tool for each
                   gold slot, or asks request_tools (at most twice) with the first sentence of a slot
                   tool's description; branches of called tools are the label, a failed request counts as
                   a miss, and the tool loss is the share of slots the first selection did not serve
    finetune       router fine-tuned on train + logged implicit labels every 50 labels; thresholds fixed
    loop           aci_implicit + fine-tuning: the full calibration loop

Reported against gold labels: branch-set coverage (all gold branches in the routed set; target
1 - alpha_branch), branches per set, full cover of the tool selection, tools per prompt, and the agent's
escalation and failure rates, per phase, for new-branch requests, and as trailing-window curves.
Decomposition and Jev are off in this experiment so the variants differ only in routing and thresholds.

A second part reads the real logged sessions in runs/*/sessions (cats.agent_cli), turns them into
implicit labels, measures their agreement with gold labels, and fine-tunes the router on train + logs
of the seed-2 runs, evaluating on seed-2 test requests that are not in the logs.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from cats.analyzer import ContextAnalyzer
from cats.bench.data import load_tasks, split
from cats.build import DATA, ROOT, build_system
from cats.calibration import slot_miss
from cats.calibration_loop import (
    CalibrationLoop,
    finetune_router,
    implicit_labels,
    label_quality,
    load_agent_sessions,
)
from cats.enrich import load_enrichment
from cats.index import ToolIndex, make_embedder
from cats.pipeline import CATS, CATSConfig
from cats.registry import Registry

PAIRS = [("finance", "media"), ("travel_geo", "database"), ("communication", "files_docs"),
         ("cloud_ops", "productivity"), ("search", "crm_commerce")]
VARIANTS = {  # name: (gamma on, feedback, refit)
    "fixed": (False, "implicit", False),
    "aci_gold": (True, "gold", False),
    "aci_implicit": (True, "implicit", False),
    "finetune": (False, "implicit", True),
    "loop": (True, "implicit", True),
}


def build_drift(held: tuple[str, ...], seed: int, cfg: CATSConfig, enrich: bool):
    reg = Registry()
    for p in sorted((DATA / "corpus").glob("*.json")):
        if p.stem not in held:
            reg.load_branch_file(p)
    index = ToolIndex(reg, make_embedder(None))
    if enrich:
        index.enrichment = load_enrichment(DATA / "enrichment")  # all tools, so new ones arrive enriched
    index.build()
    cats = CATS(reg, index, ContextAnalyzer(reg), cfg)
    full = {p.stem for p in (DATA / "corpus").glob("*.json")}
    tasks = load_tasks(DATA / "tasks", sorted(full))
    old = [t for t in tasks if not set(t.branches) & set(held) and all(g in reg.tools for g in t.gold_tools())]
    new = [t for t in tasks if set(t.branches) & set(held)]
    train, calib, test = split(old, seed=seed)
    cats.fit(train)
    cats.calibrate(calib)
    return reg, cats, train, calib, test, new


def clone(cats: CATS) -> CATS:
    c = copy.copy(cats)
    r = copy.copy(cats.router)
    r.w, r.mu, r.sd = r.w.copy(), r.mu.copy(), r.sd.copy()
    c.router = r
    c.retriever = copy.copy(cats.retriever)
    return c


def agent(c: CATS, reg: Registry, sel: list[str], gold: list[list[str]], max_esc: int = 2):
    """Simulated agent: (called tools, escalations, failed, first-selection slot loss)."""
    loaded, called, esc, failed, missed = list(sel), [], 0, False, 0
    for slot in gold:
        hit = next((x for x in loaded if x in slot), None)
        if hit is None:
            missed += 1
        while hit is None and esc < max_esc:
            cap = reg.tools[slot[0]].description.split(". ")[0]
            add = c.escalate(cap, exclude=set(loaded)).tool_ids
            esc += 1
            loaded += add
            hit = next((x for x in add if x in slot), None)
        if hit is None:
            failed = True
        else:
            called.append(hit)
    return called, esc, failed, missed / max(1, len(gold))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    m = (lambda k: float(np.mean([r[k] for r in rows])))
    return {"n": len(rows), "branch_coverage": m("bcov"), "branches_per_set": m("size"), "full_cover": m("full"),
            "k": m("k"), "escalation_rate": m("esc"), "agent_failure": m("failed")}


def run_one(held, seed: int, a, cfg: CATSConfig) -> dict:
    t0 = time.perf_counter()
    reg, base, train, calib, test, new = build_drift(held, seed, cfg, not a.no_enrich)
    t_fit = time.perf_counter() - t0
    rng = random.Random(seed)
    rng.shuffle(test)
    p1, rest = test[: len(test) // 2], test[len(test) // 2:] + new
    rng.shuffle(rest)
    if a.max_stream:  # smoke tests
        p1, rest = p1[: a.max_stream // 2], rest[: a.max_stream - a.max_stream // 2]
    stream = p1 + rest
    drift_at = len(p1)
    shared: dict = {}
    vs = {}
    for name, (adaptive, feedback, refit) in VARIANTS.items():
        c = clone(base)
        # calib[0::2] is the half the offline branch threshold was calibrated on (split_calibration),
        # so every variant starts from the same threshold
        loop = CalibrationLoop(c, calib[0::2], train, gamma=a.gamma if adaptive else 0.0,
                               gamma_tool=a.gamma_tool if adaptive else 0.0,
                               refit_every=a.refit_every if refit else 0, finetune_iters=a.finetune_iters,
                               ev_cache=shared)
        vs[name] = {"c": c, "loop": loop, "feedback": feedback, "rows": [], "trace": []}
    print(f"  run held={held} seed={seed}: fit {t_fit:.0f}s, stream {len(stream)} (drift at {drift_at}, "
          f"{len(new)} new-branch requests), branch t={base.router.threshold:.6f} lambda={base.retriever.lam:.3f}",
          flush=True)
    t1 = time.perf_counter()
    for step, task in enumerate(stream):
        if step == drift_at:
            for b in held:
                reg.load_branch_file(DATA / "corpus" / f"{b}.json")
        ev = vs["fixed"]["loop"].evidence(task.query, task.history)
        is_new = bool(set(task.branches) & set(held))
        for name, v in vs.items():
            c, loop = v["c"], v["loop"]
            route = c.router.route(ev)
            ranked, scores = c.retriever.rank(ev, route.selected, route.branch_scores)
            if not ranked:
                ranked, scores = c.retriever.rank(ev, None, route.branch_scores)
            sel, _ = c.retriever.cut(ranked, scores, k_max=cfg.k_max)
            called, esc, failed, obs_loss = agent(c, reg, sel, task.gold)
            bcov = set(task.branches) <= set(route.selected)
            v["rows"].append({"step": step, "phase": 1 if step < drift_at else 2, "new": is_new, "bcov": float(bcov),
                              "size": len(route.selected), "full": float(slot_miss(set(sel), task.gold) == 0),
                              "k": len(sel), "esc": float(esc > 0), "failed": float(failed)})
            if v["feedback"] == "gold":
                loop.observe(task.query, task.history, route.selected, task.branches,
                             tool_loss=slot_miss(set(sel), task.gold))
            else:
                need = {reg.tools[x].branch_id for x in called}
                loop.observe(task.query, task.history, route.selected, need, failed=failed, tool_loss=obs_loss,
                             weight=2.0 if esc else 1.0)
            v["trace"].append((round(loop.aci.alpha_t, 5), float(c.router.threshold), float(c.retriever.lam)))
        if (step + 1) % 100 == 0:
            print(f"    step {step + 1}/{len(stream)} ({time.perf_counter() - t1:.0f}s) " + " ".join(
                f"{n}={np.mean([r['bcov'] for r in v['rows'][-100:]]):.2f}/{np.mean([r['size'] for r in v['rows'][-100:]]):.1f}"
                for n, v in vs.items()), flush=True)
    out = {"held_out": list(held), "seed": seed, "n_stream": len(stream), "drift_at": drift_at, "n_new": len(new),
           "offline": {"branch_threshold": base.router.threshold, "lambda": base.retriever.lam,
                       "calibration": {k: base.calibration_report.get(k) for k in
                                       ("n_branch", "n_tool", "tool_target_feasible", "min_achievable_alpha_tool")}},
           "seconds": {"fit": round(t_fit, 1), "stream": round(time.perf_counter() - t1, 1)}, "variants": {}}
    w = a.window
    for name, v in vs.items():
        rows = v["rows"]
        curve = []
        for i in range(w, len(rows) + 1, 10):
            win = rows[i - w: i]
            curve.append([i, round(np.mean([r["bcov"] for r in win]), 4), round(np.mean([r["size"] for r in win]), 3),
                          round(np.mean([r["full"] for r in win]), 4), *v["trace"][i - 1]])
        out["variants"][name] = {
            "overall": summarize(rows), "phase1": summarize([r for r in rows if r["phase"] == 1]),
            "phase2": summarize([r for r in rows if r["phase"] == 2]),
            "phase2_new_branch": summarize([r for r in rows if r["phase"] == 2 and r["new"]]),
            "phase2_old_branch": summarize([r for r in rows if r["phase"] == 2 and not r["new"]]),
            "feedback_miss_rate": v["loop"].aci.miss_rate, "final_alpha_t": v["loop"].aci.alpha_t,
            "refits": v["loop"].refits,
            "curve_columns": ["t", "coverage", "branches", "full_cover", "alpha_t", "branch_threshold", "lambda"],
            "curve": curve,
        }
    return out


def logged_sessions(cfg: CATSConfig, enrich: bool) -> dict:
    tasks = {t.id: t for t in load_tasks(DATA / "tasks")}
    runs = sorted(p.parent for p in (ROOT / "runs").glob("*/sessions"))
    recs = load_agent_sessions(runs, tasks)
    s = build_system(cfg, seed=2, enrichment=DATA / "enrichment" if enrich else None)
    labels = implicit_labels(recs, s.registry)
    by_mode = {}
    for mode in sorted({r.mode for r in recs}):
        lab = implicit_labels([r for r in recs if r.mode == mode], s.registry)
        by_mode[mode] = {"records": sum(r.mode == mode for r in recs), "labels": len(lab),
                         "escalation_labels": sum(x.escalated for x in lab), **label_quality(lab, tasks)}
    test_ids = {t.id for t in s.test}
    seed2 = [lab for lab in labels if lab.task_id in test_ids]
    logged_ids = {lab.task_id for lab in seed2}
    held = [t for t in s.test if t.id not in logged_ids]

    def route_stats():
        cov, size, top1 = [], [], []
        for t in held:
            r = s.cats.router.route(s.cats.evidence(t.query, t.history))
            cov.append(set(t.branches) <= set(r.selected))
            size.append(len(r.selected))
            top1.append(max(r.branch_scores, key=r.branch_scores.get) in t.branches)
        return {"n": len(held), "branch_coverage": float(np.mean(cov)), "branches_per_set": float(np.mean(size)),
                "top1_accuracy": float(np.mean(top1))}

    before = route_stats()
    train_ex = [(s.cats.evidence(t.query, t.history), set(t.branches)) for t in s.train]
    log_ex = [(s.cats.evidence(x.request, x.history), set(x.branches)) for x in seed2]
    ft = finetune_router(s.cats.router, train_ex + log_ex, [1.0] * len(train_ex) + [x.weight for x in seed2])
    after = route_stats()
    return {"runs": [p.name for p in runs], "records": len(recs), "labels": len(labels),
            "label_quality": label_quality(labels, tasks), "by_mode": by_mode,
            "finetune_seed2": {"logged_labels": len(seed2), "unique_logged_tasks": len(logged_ids),
                               "nll": ft, "heldout_before": before, "heldout_after": after}}


def mean_over_runs(runs: list[dict]) -> dict:
    out = {}
    for name in VARIANTS:
        out[name] = {}
        for part in ("phase1", "phase2", "phase2_new_branch", "overall"):
            keys = [k for k in runs[0]["variants"][name][part] if k != "n"]
            out[name][part] = {k: float(np.mean([r["variants"][name][part][k] for r in runs])) for k in keys}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--alpha-branch", type=float, default=0.05)
    ap.add_argument("--alpha-tool", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--gamma-tool", type=float, default=0.02)
    ap.add_argument("--refit-every", type=int, default=50)
    ap.add_argument("--finetune-iters", type=int, default=100)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--skip-logged", action="store_true")
    ap.add_argument("--max-stream", type=int, default=0, help="truncate the stream (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "calibration_loop.json"))
    a = ap.parse_args()
    cfg = CATSConfig(alpha_branch=a.alpha_branch, alpha_tool=a.alpha_tool, decompose=False)
    t0 = time.perf_counter()
    runs = []
    for i in range(a.runs):
        runs.append(run_one(PAIRS[i % len(PAIRS)], i, a, cfg))
        Path(a.out).write_text(json.dumps({"partial": True, "runs": runs}, indent=1))
    out = {"config": {k: v for k, v in vars(a).items() if k != "out"}, "variants": VARIANTS,
           "target_branch_coverage": 1 - a.alpha_branch, "mean_over_runs": mean_over_runs(runs), "runs": runs}
    if not a.skip_logged:
        print("logged sessions ...", flush=True)
        out["logged_sessions"] = logged_sessions(CATSConfig(), not a.no_enrich)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name, m in out["mean_over_runs"].items():
        p2, nb = m["phase2"], m["phase2_new_branch"]
        print(f"{name:13s} phase2 cov={p2['branch_coverage']:.3f} size={p2['branches_per_set']:.2f} "
              f"full={p2['full_cover']:.3f} k={p2['k']:.1f} | new-branch cov={nb['branch_coverage']:.3f} "
              f"full={nb['full_cover']:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
