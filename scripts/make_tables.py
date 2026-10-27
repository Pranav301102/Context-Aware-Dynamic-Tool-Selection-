r"""LaTeX tables (booktabs) from results/*.json into paper/tables/.

  baselines.tex   B0-B4, CATS (+ LLM router) and the oracle on the 477-tool catalog    bench_final*.json
  ablation.tex    what each CATS component buys                                       ablation_final.json
  validity.tex    conformal coverage per split, end-to-end vs per-stage calibration    validity_*.json
  scaling.tex     coverage, tokens and latency from 100 to 10,000 tools               scale_final.json
  e2e.tex         subagent-as-model runs: success (Wilson CI), tokens, cost            e2e_*.json, b2_*.json
  external.tex    CATS vs flat baselines on open benchmarks (McNemar, Holm)            external_*.json

Each file is a complete ``table`` float; the paper needs ``\usepackage{booktabs}``. A table whose
inputs are missing is skipped.

  python scripts/make_tables.py [--results results] [--out paper/tables]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAMES = {
    "B0_static": "B0 static loading", "B1_bm25@5": "B1 BM25 @5", "B2_dense@5": "B2 dense @5",
    "B3_hybrid_rrf@5": "B3 hybrid RRF @5", "B3_hybrid_rerank@5": "B3 hybrid + reranker @5",
    "B3_hybrid_rerank@10": "B3 hybrid + reranker @10", "B3_hybrid_rerank@k=CATS": "B3 hybrid + reranker @$k_{\\mathrm{CATS}}$",
    "B4_two_stage@3x5": "B4 two-stage @3$\\times$5", "CATS": "CATS", "O_oracle": "O oracle",
    "CATS + LLM router": "CATS + LLM router",
}


def esc(text) -> str:
    s = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s


def name(strategy: str) -> str:
    return NAMES.get(strategy, esc(strategy))


def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def ci(pair) -> str:
    if not pair:
        return ""
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def signed_ci(d: dict) -> str:
    lo, hi = d["ci"]
    return f"{d['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def pval(p) -> str:
    if p is None:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def thousands(x) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def table(caption: str, label: str, colspec: str, header: list, rows: list[list[str]],
          notes: str = "", midrules: tuple[int, ...] = (), header_rule: str = "") -> str:
    """``header`` is one row (list of cells) or several rows (list of lists); ``header_rule`` (e.g.
    cmidrules) goes between the first and second header rows."""
    heads = header if header and isinstance(header[0], list) else [header]
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    for i, h in enumerate(heads):
        lines.append(" & ".join(h) + " \\\\")
        if i == 0 and header_rule and len(heads) > 1:
            lines.append(header_rule)
    lines.append("\\midrule")
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {notes}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def load(results: Path, fname: str):
    p = results / fname
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- tables
def t_baselines(results: Path) -> str | None:
    base = load(results, "bench_final.json")
    if not base:
        return None
    llm = load(results, "bench_final_llm.json")
    rows_in = [(s["strategy"], s, base["paired_full_cover_vs_cats"].get(s["strategy"])) for s in base["summaries"]]
    if llm:
        s = next((x for x in llm["summaries"] if x["strategy"] == "CATS"), None)
        if s:
            rows_in.insert([r[0] for r in rows_in].index("CATS") + 1, ("CATS + LLM router", s, None))
    rows = []
    for strat, s, paired in rows_in:
        diff = "--" if strat in ("CATS", "CATS + LLM router") or not paired else signed_ci(
            {"mean": -paired["mean"], "ci": (-paired["ci"][1], -paired["ci"][0])})
        tokens = s["tool_tokens"] + (s.get("router_tokens") or 0)
        lat = "--" if strat in ("B0_static", "O_oracle") else f"{s['latency_p95_ms']:.0f}"
        rows.append([name(strat), f"{f3(s['full_cover'])} {ci(s.get('full_cover_ci'))}", f3(s["slot_recall"]),
                     f"{s['k_mean']:.1f}", thousands(tokens), f"{100 * tokens / max(1, base['registry']['tool_tokens']):.2f}",
                     lat, diff])
    n = base["summaries"][0]["n"]
    reg = base["registry"]
    return table(
        f"Selection quality on the {reg['tools']}-tool catalog ({reg['branches']} branches), held-out test split "
        f"(n = {n}). Tokens include LLM-router prompts.",
        "tab:baselines", "lcccrrrc",
        ["Strategy", "Full cover [95\\% CI]", "Slot recall", "$k$", "Tokens", "\\% static", "p95 ms",
         "Baseline $-$ CATS"],
        rows,
        notes="Full cover: share of requests whose selected tools cover every gold slot. Last column: paired "
              "bootstrap of the baseline's full cover minus CATS's (negative = CATS ahead). CIs: 2,000 bootstrap "
              "resamples of tasks.")


def t_ablation(results: Path) -> str | None:
    rows_in = load(results, "ablation_final.json")
    if not rows_in:
        return None
    rows = []
    for r in rows_in:
        cal = r.get("calibration", {})
        target = cal.get("alpha_tool")
        miss = 1 - r["slot_recall"]
        feas = "" if cal.get("tool_target_feasible", True) else "$^\\dagger$"
        rows.append([esc(r["strategy"]), f"{f3(r['full_cover'])} {ci(r.get('full_cover_ci'))}",
                     f"{miss:.3f}{feas}" + (f" ({target:.2f})" if target is not None else ""),
                     f"{r['k_mean']:.1f}", f"{r.get('branches_mean', float('nan')):.2f}",
                     f3(r.get("branch_cover")), f"{r['token_pct_static']:.2f}"])
    return table(
        f"Ablation: what each change to the workbook design buys (same split, n = {rows_in[0]['n']}).",
        "tab:ablation", "lcccccr",
        ["Configuration", "Full cover [95\\% CI]", "Slot miss (target)", "$k$", "Branches", "Branch cover",
         "\\% static"],
        rows,
        notes="$^\\dagger$ the slot-miss target is infeasible on the calibration data (conformal risk control falls "
              "back to $k_{\\max}$). Branch cover target: 0.97.")


def t_validity(results: Path) -> str | None:
    joint = load(results, "validity_enrich.json")
    indep = load(results, "validity_independent.json")
    if not joint or not indep:
        return None
    rows = []
    for rj, ri in zip(joint["rows"], indep["rows"]):
        rows.append([str(rj["seed"]), f3(rj["branch_cover"]), f3(rj["slot_miss"]), f"{rj['k_mean']:.1f}",
                     f3(ri["branch_cover"]), f3(ri["slot_miss"]), f"{ri['k_mean']:.1f}"])

    def mean(res, f):
        return sum(r[f] for r in res["rows"]) / len(res["rows"])

    def over(res):
        return sum(r["slot_miss"] > res["slot_miss_target"] for r in res["rows"])

    rows.append(["Mean", f3(mean(joint, "branch_cover")), f3(mean(joint, "slot_miss")), f"{mean(joint, 'k_mean'):.1f}",
                 f3(mean(indep, "branch_cover")), f3(mean(indep, "slot_miss")), f"{mean(indep, 'k_mean'):.1f}"])
    rows.append(["Splits over target", "", f"{over(joint)} of {len(joint['rows'])}", "", "",
                 f"{over(indep)} of {len(indep['rows'])}", ""])
    return table(
        f"Conformal validity over {len(joint['rows'])} random splits. Targets: branch cover "
        f"$\\geq$ {joint['branch_target']:.2f}, expected slot miss $\\leq$ {joint['slot_miss_target']:.2f}.",
        "tab:validity", "lcccccc",
        [["Split", "\\multicolumn{3}{c}{End-to-end calibration}", "\\multicolumn{3}{c}{Per-stage (Algorithm 1)}"],
         ["", "Branch cover", "Slot miss", "$k$", "Branch cover", "Slot miss", "$k$"]],
        rows, midrules=(len(rows) - 2,), header_rule="\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        notes="End to end: the tool cut-off is calibrated on the routed pipeline's own output. Per stage: the "
              "cut-off assumes perfect routing, as in the workbook's Algorithm 1.")


def t_scaling(results: Path) -> str | None:
    rows_in = load(results, "scale_final.json")
    if not rows_in:
        return None
    rows = []
    for r in sorted(rows_in, key=lambda x: x["size"]):
        c, f10 = r["CATS"], r.get("B3_hybrid_rerank@10", {})
        delta = c["full_cover"] - f10.get("full_cover", float("nan"))
        rows.append([thousands(r["size"]), str(r["branches"]), str(r["tasks_test"]), thousands(r["static_tokens"]),
                     f3(r.get("B3_hybrid_rrf@5", {}).get("full_cover")),
                     f3(r.get("B3_hybrid_rerank@5", {}).get("full_cover")), f3(f10.get("full_cover")),
                     f3(c["full_cover"]), f"{delta:+.3f}", f"{c['token_pct_static']:.2f}",
                     f"{c['latency_p95_ms']:.0f}"])
    return table(
        "Tool-pool scaling: real catalog plus distractor domains and vendor clones (no enrichment).",
        "tab:scaling", "rrrrcccccrr",
        ["Tools", "Branches", "Test", "Static tokens", "RRF @5", "Rerank @5", "Rerank @10", "CATS",
         "CATS $-$ @10", "CATS \\% static", "p95 ms"],
        rows,
        notes="Pools below 477 tools use task subsets whose gold tools fit, so those rows are not directly "
              "comparable with the larger ones. Columns 5--8: full cover.")


def _e2e(results: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_figures import _e2e_entries, _group

    out = []
    for e in _e2e_entries(results):
        e["group"] = _group(e).replace("\n", ", ")
        out.append(e)
    return out


def t_e2e(results: Path) -> str | None:
    entries = _e2e(results)
    if not entries:
        return None
    from cats.bench.stats import wilson_interval

    paired = {}
    for f in sorted(glob.glob(str(results / "e2e_*.json")) + glob.glob(str(results / "b2_*.json"))):
        try:
            doc = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("comparisons"), dict):
            for c in doc["comparisons"].values():
                paired[(c["a"], c["b"])] = c
    order = {"static": 0, "cats": 1, "catsjev": 2, "search": 3}
    entries.sort(key=lambda e: (e["group"], order.get(e["mode"], 9)))
    mode_name = {"static": "Static loading (B0)", "cats": "CATS", "catsjev": "CATS-Jev",
                 "search": "Agent-initiated search (B6)"}
    rows, rules, last = [], [], None
    for e in entries:
        if last is not None and e["group"] != last:
            rules.append(len(rows))
        last = e["group"]
        n = int(e.get("n") or 0)
        k = round(float(e["success"]) * n)
        lo, hi = wilson_interval(k, n) if n else (float("nan"), float("nan"))
        test = "--"
        static_run = next((x.get("run") for x in entries if x["group"] == e["group"] and x["mode"] == "static"), None)
        c = paired.get((e.get("run"), static_run))
        if c:
            test = pval(c["success"].get("p_holm", c["success"]["p_value"]))
        rows.append([esc(e["group"]), mode_name.get(e["mode"], esc(e["mode"])), f"{k}/{n} [{lo:.2f}, {hi:.2f}]",
                     thousands(e.get("tool_tokens_mean", 0)), thousands(e.get("input_tokens_mean", 0)),
                     f"{e.get('turns_mean', float('nan')):.2f}", f"{100 * e.get('escalation_rate', 0):.0f}\\%",
                     f"{e.get('usd_per_task', float('nan')):.4f}", test])
    return table(
        "End to end with a subagent as the model: the same tasks per model and experiment, only tool selection differs.",
        "tab:e2e", "llcrrrrrc",
        ["Model, run", "Mode", "Success [Wilson 95\\%]", "Tool tok.", "Input tok./task", "Turns", "Escal.",
         "\\$/task", "$p$ vs B0"],
        rows, midrules=tuple(rules),
        notes="Cost: every model turn billed through the prefix-cache simulator at \\$3/M input and \\$15/M output "
              "(cache reads 0.1$\\times$, writes 1.25$\\times$). $p$: exact McNemar test on per-task success against "
              "static loading, Holm-corrected (from scripts/score\\_e2e.py output when present).")


EXT_ORDER = ["B0_static", "B1_bm25@5", "B2_dense@5", "B3_hybrid_rrf@5", "B3_hybrid_rerank@5", "B4_two_stage@3x5",
             "B3_hybrid_rerank@10", "B3_hybrid_rerank@k=CATS", "CATS", "O_oracle"]


def t_external(results: Path) -> str | None:
    docs = []
    for f in sorted(glob.glob(str(results / "external_*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "summaries" in d and "stats" in d:
            docs.append(d)
    docs.sort(key=lambda d: (d["benchmark"], bool((d.get("protocol") or {}).get("variant")
                                                  or (d.get("meta") or {}).get("label_widening") == "none")))
    if not docs:
        return None
    rows, rules = [], []
    for d in docs:
        if rows:
            rules.append(len(rows))
        summ = {s["strategy"]: s for s in d["summaries"]}
        comps = d["stats"]["full_cover"]["comparisons"]
        reg, proto = d["registry"], d["protocol"]
        variant = proto.get("variant") or ("strict labels" if d.get("meta", {}).get("label_widening") == "none"
                                           else None)
        extra = f" {esc(variant)}," if variant else ""
        label = (f"{esc(d['benchmark'])} ({thousands(reg['tools'])} tools, n = {proto['n_tasks']},{extra} "
                 f"{esc(proto['name'])})")
        first = True
        for strat in [s for s in EXT_ORDER if s in summ]:
            s = summ[strat]
            c = comps.get(strat)
            diff = signed_ci({"mean": -c["bootstrap"]["mean"], "ci": (-c["bootstrap"]["ci"][1],
                                                                      -c["bootstrap"]["ci"][0])}) if c else "--"
            p = pval(c["p_holm"]) if c else "--"
            rows.append([label if first else "", name(strat), f"{f3(s['full_cover'])} {ci(s.get('full_cover_ci'))}",
                         f3(s["slot_recall"]), f"{s['k_mean']:.1f}", f"{s['token_pct_static']:.2f}", diff, p])
            first = False
    return table(
        "Open benchmarks that no project model wrote: CATS against static loading and flat retrieval.",
        "tab:external", "p{3.1cm}lcccrcc",
        ["Benchmark", "Strategy", "Full cover [95\\% CI]", "Slot recall", "$k$", "\\% static", "Baseline $-$ CATS",
         "$p_{\\mathrm{Holm}}$"],
        rows, midrules=tuple(rules),
        notes="$p$: exact McNemar test on per-task full cover against CATS, Holm-corrected across the baselines of "
              "each benchmark. Crossfit: every task is tested once, with the router, reranker and conformal "
              "thresholds fitted on the other folds.")


TABLES = {"baselines": t_baselines, "ablation": t_ablation, "validity": t_validity, "scaling": t_scaling,
          "e2e": t_e2e, "external": t_external}


def main(argv=None) -> dict[str, str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default=str(ROOT / "paper" / "tables"))
    p.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(TABLES))
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for key in [x for x in a.only.split(",") if x] or list(TABLES):
        tex = TABLES[key](results)
        if tex is None:
            print(f"{key:<9} skipped (inputs missing in {results})")
            continue
        path = out / f"{key}.tex"
        path.write_text(re.sub(r"[ \t]+\n", "\n", tex))
        written[key] = str(path)
        print(f"{key:<9} -> {path}")
    return written


if __name__ == "__main__":
    main()
