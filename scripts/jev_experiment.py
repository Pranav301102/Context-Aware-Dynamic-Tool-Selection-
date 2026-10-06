"""Jev-style typed decision layer: what each use of it buys (writes results/jev_experiment.json).

Usage: python scripts/jev_experiment.py [--jev runs/jev] [--keyword runs/jev_keyword]
"""
import argparse
import json
from dataclasses import replace

from cats.bench.harness import evaluate, format_table, summarize
from cats.build import ROOT, build_registry, build_system
from cats.jev import JEV_USD_PER_MTOK
from cats.pipeline import CATSConfig
from cats.strategies import CATSStrategy

LLM_USD_PER_MTOK = 3.0  # the workbook's illustrative main-model input price
ENRICH = ROOT / "data" / "enrichment"


def variants(jev: str, keyword: str | None, llm: str | None):
    base = CATSConfig()
    off = {"jev_cascade": False, "jev_steps": False, "jev_repair": None}
    v = [
        ("improved CATS (no Jev)", base),
        ("+ Jev prior, always called", replace(base, jev=jev, **off)),
        ("+ Jev prior, cascade-gated", replace(base, jev=jev, jev_cascade=True, jev_steps=False, jev_repair=None)),
        ("+ cascade + Jev step count", replace(base, jev=jev, jev_cascade=True, jev_steps=True, jev_repair=None)),
        ("+ cascade + steps + branch repair (CATS-Jev)", replace(base, jev=jev)),
        ("CATS-Jev, Jev always called", replace(base, jev=jev, jev_cascade=False)),
    ]
    if keyword:
        v.append(("CATS-Jev with keyword 'Jev' (heuristic)", replace(base, jev=keyword)))
    if llm:
        v.append(("LLM router prior, always called", replace(base, llm_router=llm)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", default=str(ROOT / "runs" / "jev"))
    ap.add_argument("--keyword", default=str(ROOT / "runs" / "jev_keyword"))
    ap.add_argument("--llm", default=str(ROOT / "runs" / "llm_router"))
    a = ap.parse_args()
    reg = build_registry()
    rows = []
    for name, cfg in variants(a.jev, a.keyword, a.llm):
        s = build_system(cfg, registry=reg, enrichment=ENRICH)
        summ = summarize(evaluate(CATSStrategy(s.cats, name), s.test, reg), reg.total_tokens())
        price = LLM_USD_PER_MTOK if cfg.llm_router else JEV_USD_PER_MTOK
        summ["routing_usd_per_1k_requests"] = summ["router_tokens"] * price / 1e6 * 1000
        summ["calibration"] = s.cats.calibration_report
        rows.append(summ)
        print(f"{name:<46} cover={summ['full_cover']:.3f} miss={1 - summ['slot_recall']:.3f} "
              f"k={summ['k_mean']:.1f} tok%={summ['token_pct_static']:.2f} calls={summ['router_call_rate']:.2f} "
              f"route$/1k={summ['routing_usd_per_1k_requests']:.4f} cross={summ['by_kind'].get('cross', {}).get('full_cover', 0):.2f} "
              f"multi={summ['by_kind'].get('multi', {}).get('full_cover', 0):.2f} "
              f"feasible={summ['calibration']['tool_target_feasible']}", flush=True)
    print(format_table(rows))
    (ROOT / "results" / "jev_experiment.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
