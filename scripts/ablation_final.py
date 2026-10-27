"""Component ablation of the improved CATS configuration (writes results/ablation_final.json).

Usage: python scripts/ablation_final.py
"""
import json
from dataclasses import replace
from pathlib import Path

from cats.bench.harness import evaluate, format_table, summarize
from cats.build import ROOT, build_registry, build_system
from cats.pipeline import CATSConfig
from cats.strategies import CATSStrategy

ENRICH = ROOT / "data" / "enrichment"
LLM = ROOT / "runs" / "llm_router"
BASE = CATSConfig(method="lac")

VARIANTS = [
    # name, config, enrichment
    ("workbook design (APS, no extras)", replace(BASE, method="aps", decompose=False, dedupe=False), None),
    ("+ LAC sets", replace(BASE, decompose=False, dedupe=False), None),
    ("+ enrichment", replace(BASE, decompose=False, dedupe=False), ENRICH),
    ("+ decomposition (= improved CATS)", BASE, ENRICH),
    ("improved CATS + LLM router", replace(BASE, llm_router=str(LLM)), ENRICH),
    ("improved, independent calibration", replace(BASE, calibration="independent"), ENRICH),
    ("improved, alpha_tool=0.10", replace(BASE, alpha_tool=0.10), ENRICH),
    ("improved + LLM router, alpha_tool=0.10", replace(BASE, alpha_tool=0.10, llm_router=str(LLM)), ENRICH),
]


def main():
    reg = build_registry()
    rows = []
    for name, cfg, enrich in VARIANTS:
        if cfg.llm_router and not Path(cfg.llm_router).exists():
            continue
        s = build_system(cfg, registry=reg, enrichment=enrich)
        summ = summarize(evaluate(CATSStrategy(s.cats, name), s.test, reg), reg.total_tokens())
        summ["calibration"] = s.cats.calibration_report
        rows.append(summ)
        print(f"{name:<40} cover={summ['full_cover']:.3f} k={summ['k_mean']:.1f} tok%={summ['token_pct_static']:.2f} "
              f"branches={summ.get('branches_mean', 0):.2f} branch_cover={summ.get('branch_cover', 0):.3f} "
              f"router_tok={summ['router_tokens']:.0f}", flush=True)
    print(format_table(rows))
    (ROOT / "results" / "ablation_final.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
