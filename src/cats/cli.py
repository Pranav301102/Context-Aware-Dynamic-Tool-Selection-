"""cats command line.

  cats stats                         registry, tokenizer and embedder in use
  cats route "query" [--history ..]  show the route, tools and token savings for one request (UC7)
  cats bench                         every baseline vs CATS on the held-out split (FR9)
  cats ablate                        CATS configuration variants on the same split
  cats validity                      empirical coverage vs the conformal target over several splits
  cats scale --sizes 100,1000,10000  tool-pool scaling benchmark (FR14)
  cats cache                         prompt-cache cost of tool-ordering policies in multi-turn sessions
  cats run --llm heuristic           end-to-end sub-agent runs with escalation (FR6, FR7)
  cats agent ...                     subagent-as-model harness (see cats.agent_cli)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .build import DATA, ROOT, build_system
from .pipeline import CATSConfig

RESULTS = ROOT / "results"


def _cfg(a) -> CATSConfig:
    cfg = CATSConfig()
    for name in ("alpha_branch", "alpha_tool", "method", "k_max", "calibration", "reranker", "budget",
                 "llm_router", "decompose", "jev"):
        v = getattr(a, name, None)
        if v is not None:
            cfg = replace(cfg, **{name: v})
    return cfg


def _enrich(a) -> Path | None:
    if getattr(a, "enrich", None):
        return Path(a.enrich)
    return None


def _save(name: str, obj) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    return p


def cmd_stats(a):
    s = build_system(_cfg(a), enrichment=_enrich(a), fit=False)
    print(json.dumps({**s.registry.stats(), "embedder": s.index.embedder.name,
                      "tasks": {"train": len(s.train), "calib": len(s.calib), "test": len(s.test)},
                      "vendor_words": len(s.analyzer.vendors)}, indent=1))


def cmd_route(a):
    from .types import Turn

    s = build_system(_cfg(a), enrichment=_enrich(a))
    history = [Turn(role="user", content=h) for h in (a.history or [])]
    sel = s.cats.select(a.query, history)
    static = s.registry.total_tokens()
    p = sel.route.branch_scores
    print("branches:", ", ".join(f"{b}={p[b]:.3f}" for b in sorted(p, key=lambda b: -p[b])[:6]))
    print(f"selected: {sel.route.selected}  (t={sel.route.threshold:.4f}, fallback={sel.route.fallback})")
    for tid, sc in zip(sel.tool_ids, sel.scores):
        print(f"  {sc:6.3f}  {tid}")
    print(f"tool tokens: {sel.tool_tokens} vs static {static} ({100 * sel.tool_tokens / static:.1f}%)")
    print("spans:", ", ".join(f"{sp.stage}={sp.latency_ms:.1f}ms" for sp in sel.spans))


def cmd_bench(a):
    from .bench.experiments import run_baselines
    from .bench.harness import format_table

    s = build_system(_cfg(a), seed=a.seed, enrichment=_enrich(a))
    res = run_baselines(s, k=a.k)
    print(f"registry: {res['registry']}  embedder: {res['embedder']}")
    cal = res["calibration"]
    print(f"calibration: n={cal['n']} t_branch={cal['branch_threshold']:.4f} lambda={cal['lambda']:.3f} "
          f"tool target feasible={cal['tool_target_feasible']} (min alpha {cal['min_achievable_alpha_tool']:.3f})")
    print(format_table(res["summaries"]))
    cats = next(x for x in res["summaries"] if x["strategy"] == "CATS")
    print(f"CATS: branch_cover={cats.get('branch_cover'):.3f} branches={cats.get('branches_mean'):.2f} "
          f"by_kind={json.dumps({k: round(v['full_cover'], 3) for k, v in cats['by_kind'].items()})}")
    print("saved", _save(a.name or "bench", res))


def cmd_ablate(a):
    from .bench.experiments import default_variants, run_variants
    from .bench.harness import format_table

    rows = run_variants(default_variants(_cfg(a)), seed=a.seed, enrichment=_enrich(a))
    print(format_table(rows))
    for r in rows:
        print(f"{r['strategy']:<26} branch_cover={r.get('branch_cover', 0):.3f} "
              f"branches={r.get('branches_mean', 0):.2f} lambda={r['calibration']['lambda']:.3f}")
    print("saved", _save(a.name or "ablate", rows))


def cmd_validity(a):
    from .bench.experiments import coverage_validity

    res = coverage_validity(_cfg(a), seeds=range(a.seeds), enrichment=_enrich(a))
    for r in res["rows"]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    print(f"branch cover {res['branch_cover_mean']:.3f} (target >= {res['branch_target']:.3f}); "
          f"slot miss {res['slot_miss_mean']:.3f} (target <= {res['slot_miss_target']:.3f})")
    print("saved", _save(a.name or "validity", res))


def cmd_scale(a):
    from .bench.experiments import scaling

    sizes = [int(x) for x in a.sizes.split(",")]
    rows = scaling(sizes, _cfg(a), seed=a.seed, enrichment=_enrich(a), k=a.k)
    for r in rows:
        line = f"N={r['size']:>6} branches={r['branches']:>3} test={r['tasks_test']:>4} static={r['static_tokens']:>8}"
        for name in [k for k in r if isinstance(r[k], dict) and "full_cover" in r[k]]:
            m = r[name]
            line += (f" | {name}: cover={m['full_cover']:.3f} k={m['k_mean']:.1f} tok%={m['token_pct_static']:.2f}"
                     f" p95={m['latency_p95_ms']:.1f}ms")
        print(line)
    print("saved", _save(a.name or "scale", rows))


def cmd_cache(a):
    from .bench.experiments import cache_sessions

    s = build_system(_cfg(a), seed=a.seed, enrichment=_enrich(a))
    res = cache_sessions(s, session_len=a.session_len)
    for k, v in res["policies"].items():
        print(f"{k:<10} " + " ".join(f"{m}={x:.4f}" for m, x in v.items()))
    print("saved", _save(a.name or "cache", res))


def cmd_run(a):
    from .dispatcher import Dispatcher
    from .llm import make_llm

    s = build_system(_cfg(a), seed=a.seed, enrichment=_enrich(a))
    d = Dispatcher(s.cats, make_llm(a.llm), select=not a.search_only)
    tasks = s.test[: a.n]
    rs = [d.run(t) for t in tasks]
    n = len(rs)
    print(json.dumps({
        "llm": a.llm, "n": n, "success": sum(r.success for r in rs) / n,
        "escalation_rate": sum(r.escalations > 0 for r in rs) / n,
        "invalid_calls": sum(len(r.invalid_calls) for r in rs),
        "usd_per_task": sum(r.usd for r in rs) / n, "turns_mean": sum(r.turns for r in rs) / n,
    }, indent=1))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "agent":
        from .agent_cli import main as agent_main

        return agent_main(argv[1:])
    p = argparse.ArgumentParser(prog="cats", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        s = sub.add_parser(name)
        s.add_argument("--enrich", default=None, help=f"enrichment dir, e.g. {DATA / 'enrichment'}")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--alpha-branch", dest="alpha_branch", type=float)
        s.add_argument("--alpha-tool", dest="alpha_tool", type=float)
        s.add_argument("--method", choices=["aps", "lac"])
        s.add_argument("--k-max", dest="k_max", type=int)
        s.add_argument("--calibration", choices=["joint", "independent"])
        s.add_argument("--reranker", choices=["logreg", "gbt"])
        s.add_argument("--budget", type=int)
        s.add_argument("--llm-router", dest="llm_router", default=None,
                       help="dir of LLM router scores (runs/llm_router) to add as a router feature")
        s.add_argument("--no-decompose", dest="decompose", action="store_false", default=None)
        s.add_argument("--jev", default=None, help="dir of Jev answers (runs/jev): typed decision layer")
        s.add_argument("--name", default=None, help="results/<name>.json")
        s.set_defaults(fn=fn)
        return s

    add("stats", cmd_stats)
    s = add("route", cmd_route)
    s.add_argument("query")
    s.add_argument("--history", nargs="*")
    s = add("bench", cmd_bench)
    s.add_argument("--k", type=int, default=5)
    add("ablate", cmd_ablate)
    s = add("validity", cmd_validity)
    s.add_argument("--seeds", type=int, default=5)
    s = add("scale", cmd_scale)
    s.add_argument("--sizes", default="100,250,1000,2500,5000,10000")
    s.add_argument("--k", type=int, default=5)
    s = add("cache", cmd_cache)
    s.add_argument("--session-len", dest="session_len", type=int, default=5)
    s = add("run", cmd_run)
    s.add_argument("--llm", default="heuristic")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--search-only", action="store_true")
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
