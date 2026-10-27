"""Generate the paper's result tables (and the text around them) from results/ and runs/.

Writes paper/tables/p_{jev,scaling,e2e}.tex and results/e2e_models.json.
Usage: python scripts/paper_tables.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
MODELS = [("haiku", "Haiku 4.5"), ("sonnet", "Sonnet 5.5"), ("opus", "Opus 5.5")]
MODES = [("static", "Static (B0)"), ("search", "Search only (B6)"), ("cats", "\\cats{}"), ("catsjev", "\\catsjev{}")]
# Tasks where this sandbox's permission classifier blocked a simulated write call in at least one run
# (reported by the agents; the harness cannot see blocked commands). Excluded in the sensitivity column.
BLOCKED = {"database-0002", "cross-0012", "cloud_ops-0058", "crm_commerce-0012", "dev_platform-0047", "cross-0005"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def jev_table() -> str:
    path = RES / "jev_experiment.json"
    if not path.exists():
        return "% jev_experiment.json missing\n"
    rows = json.loads(path.read_text())
    names = {
        "improved CATS (no Jev)": "\\cats{} (no typed layer)",
        "+ Jev prior, always called": "+ prior, always asked",
        "+ Jev prior, cascade-gated": "+ prior, gated",
        "+ cascade + Jev step count": "+ gated prior + steps",
        "+ cascade + steps + branch repair (CATS-Jev)": "\\catsjev{} (gated + steps + repair)",
        "CATS-Jev, Jev always called": "\\catsjev{}, always asked",
        "CATS-Jev with keyword 'Jev' (heuristic)": "\\catsjev{}, keyword answers",
        "LLM router prior, always called": "LLM router prior (\\$3/M)",
    }
    lines = [
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{tabular}{lccccccc}\\toprule",
        "Configuration & Cover & Miss & $k$ & Tok.\\ (\\%) & Calls & Route \\$/1k & Cross\\\\\\midrule",
    ]
    for r in rows:
        cross = r["by_kind"].get("cross", {}).get("full_cover", float("nan"))
        lines.append(
            f"{names.get(r['strategy'], r['strategy'])} & {r['full_cover']:.3f} & {1 - r['slot_recall']:.3f} & "
            f"{r['k_mean']:.1f} & {r['token_pct_static']:.2f} & {r['router_call_rate']:.2f} & "
            f"{r['routing_usd_per_1k_requests']:.3f} & {cross:.2f}\\\\")
    lines += [
        "\\bottomrule\\end{tabular}",
        ("\\caption{Typed decision layer (test $n{=}386$). Calls: share of requests that query the layer. "
         "Route \\$/1k: routing cost per thousand requests (typed layer at \\$42 per billion input tokens; "
         "LLM router at \\$3/M). Cross: full cover on cross-domain requests.}"),
        "\\label{tab:jev}\\end{table*}",
    ]
    by = {r["strategy"]: r for r in rows}
    base, cj = by.get("improved CATS (no Jev)"), by.get("+ cascade + steps + branch repair (CATS-Jev)")
    text = ""
    if base and cj:
        text = (
            f"Table~\\ref{{tab:jev}} adds the typed layer one use at a time. The gated prior alone keeps "
            f"coverage and cuts $k$ from {base['k_mean']:.1f} to "
            f"{by['+ Jev prior, cascade-gated']['k_mean']:.1f}, because sharper branch probabilities make the "
            f"risk-controlled cut-off feasible (without the layer no cut-off reaches $\\alpha{{=}}0.05$ on half the "
            f"calibration split). With steps and repair, \\catsjev{{}} reaches {cj['full_cover']:.3f} full cover "
            f"(from {base['full_cover']:.3f}) with {100 * (1 - cj['token_pct_static'] / base['token_pct_static']):.0f}\\% "
            f"fewer tool tokens, while asking the layer on {100 * cj['router_call_rate']:.0f}\\% of requests for "
            f"\\${cj['routing_usd_per_1k_requests']:.3f} per thousand requests.\n")
        kw = by.get("CATS-Jev with keyword 'Jev' (heuristic)")
        if kw:
            text += (f"Keyword answers instead of model judgments give {kw['full_cover']:.3f}, so most of the "
                     f"gain needs a model that reads the request.\n")
    return text + "\n".join(lines) + "\n"


def scaling_table() -> str:
    rows = json.loads((RES / "scale_final.json").read_text())
    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\begin{tabular}{rcccc}\\toprule",
        "Tools & RRF@5 & Rerank@10 & \\cats{} & Tok.\\ (\\%)\\\\\\midrule",
    ]
    for r in rows:
        c = r["CATS"]
        lines.append(f"{r['size']:,} & {r['B3_hybrid_rrf@5']['full_cover']:.3f} & "
                     f"{r['B3_hybrid_rerank@10']['full_cover']:.3f} & {c['full_cover']:.3f} & "
                     f"{c['token_pct_static']:.2f}\\\\")
    lines += ["\\bottomrule\\end{tabular}",
              ("\\caption{Scaling with distractor domains and vendor clones (no enrichment). Pools under 477 tools use a "
               "task subset, so their rows are not comparable with larger ones. Tok.: \\cats{} tool tokens as \\% of static.}"),
              "\\label{tab:scaling}\\end{table}"]
    big = [r for r in rows if r["size"] >= 1000]
    gains = [r["CATS"]["full_cover"] - r["B3_hybrid_rerank@10"]["full_cover"] for r in big]
    text = (f"Table~\\ref{{tab:scaling}} grows the pool to 10{{,}}000 tools. At (nearly) equal $k$, routing gains nothing "
            f"over flat reranked top-10 at 100--250 tools and {100 * min(gains):.1f}--{100 * max(gains):.1f} points from "
            f"1{{,}}000 tools up; p95 selection latency stays under "
            f"{max(r['CATS']['latency_p95_ms'] for r in rows):.0f}\\,ms.\n")
    return text + "\n".join(lines) + "\n"


def e2e_table() -> str:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from cats.agent_cli import score_run

    data, out = {}, []
    for m, _ in MODELS:
        for mode, _ in MODES:
            run = f"b2_{m}_{mode}"
            if not (ROOT / "runs" / run / "run.json").exists():
                continue
            sc = score_run(run)
            open_tasks = {r["task"] for r in sc["rows"] if not r["done"]}
            if open_tasks - BLOCKED:
                continue  # unfinished run
            # Tasks left open because a write was blocked count as failures (score_run already scores them 0).
            data[(m, mode)] = sc
            out.append({**sc["summary"], "model": m, "mode": mode})
    (RES / "e2e_models.json").write_text(json.dumps(out, indent=1))
    lines = [
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{tabular}{llcccccc}\\toprule",
        "Model & Mode & Success & Excl.\\ blocked & Tool tok. & Input tok./task & Turns & \\$/task\\\\\\midrule",
    ]
    for m, mname in MODELS:
        first = True
        for mode, label in MODES:
            if (m, mode) not in data:
                continue
            s = data[(m, mode)]["summary"]
            rows = data[(m, mode)]["rows"]
            kept = [r for r in rows if r["task"] not in BLOCKED]
            excl = sum(r["success"] for r in kept)
            lines.append(f"{mname if first else ''} & {label} & {round(s['success'] * s['n'])}/{s['n']} & "
                         f"{excl}/{len(kept)} & {s['tool_tokens_mean']:,.0f} & {s['input_tokens_mean']:,.0f} & "
                         f"{s['turns_mean']:.2f} & {s['usd_per_task']:.4f}\\\\")
            first = False
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              ("\\caption{Models in the loop: the same 60 held-out tasks per mode, with a Claude model as the agent. "
               "Success = every gold slot called and no invalid call. Excl.\\ blocked: success without the six tasks "
               "where the sandbox's permission classifier blocked a simulated write in some run. Every model is billed "
               "at \\$3/M input and \\$15/M output with simulated prompt caching, so costs compare modes, not models.}"),
              "\\label{tab:e2e}\\end{table*}"]
    # Paired tests: static vs CATS per model.
    tests = []
    for m, mname in MODELS:
        for other in ("cats", "catsjev"):
            if (m, "static") in data and (m, other) in data:
                a = {r["task"]: r["success"] for r in data[(m, "static")]["rows"]}
                b = {r["task"]: r["success"] for r in data[(m, other)]["rows"]}
                common = sorted(set(a) & set(b))
                bc = sum(1 for t in common if b[t] and not a[t])
                cb = sum(1 for t in common if a[t] and not b[t])
                tests.append(f"{mname}: {other} solved {bc} tasks static failed, static solved {cb} that "
                             f"{other} failed (exact McNemar $p={mcnemar_exact(bc, cb):.3f}$)")
    text = ("Table~\\ref{tab:e2e} runs every mode with three Claude models on the same 60 held-out tasks. "
            + ("Paired comparisons against static loading: " + "; ".join(tests) + ".\n" if tests else "\n"))
    return text + "\n".join(lines) + "\n"


def external_table() -> str:
    """Compact view of results/external_*.json: CATS against flat retrieval on benchmarks nobody here wrote."""
    specs = [("bfcl", "BFCL v4"), ("bfcl_nodecomp", "\\quad no decomposition"), ("bfcl_strict", "\\quad strict labels"),
             ("livemcpbench", "LiveMCPBench")]
    lines = ["\\begin{table*}[t]\\centering\\small",
             "\\begin{tabular}{lccccc}\\toprule",
             "Benchmark & BM25@5 & Flat@$k$ & \\cats{} & $\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$\\\\\\midrule"]
    found = {}
    for name, label in specs:
        path = RES / f"external_{name}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        by = {x["strategy"]: x for x in d["summaries"]}
        flat, cats = by["B3_hybrid_rerank@k=CATS"], by["CATS"]
        diff = d["paired_full_cover_vs_cats"]["B3_hybrid_rerank@k=CATS"]
        p = d["stats"]["full_cover"]["comparisons"]["B3_hybrid_rerank@k=CATS"]["p_holm"]
        found[name] = (d, cats, flat)
        lines.append(f"{label} & {by['B1_bm25@5']['full_cover']:.3f} & {flat['full_cover']:.3f} & "
                     f"{cats['full_cover']:.3f} & {diff['mean']:+.3f} [{diff['ci'][0]:+.3f}, {diff['ci'][1]:+.3f}] & "
                     f"{p:.3f}\\\\")
    lines += ["\\bottomrule\\end{tabular}",
              ("\\caption{Full cover on open benchmarks no project model wrote. Flat@$k$: hybrid retrieval with the "
               "learned reranker at the same mean $k$ as \\cats{}. $\\Delta$: \\cats{} minus Flat@$k$ (paired "
               "bootstrap); $p$: exact McNemar, Holm-corrected over each benchmark's baselines. BFCL: 3{,}258 functions, "
               "1{,}011 test requests; LiveMCPBench: 525 tools, 94 tasks, 5-fold cross-fitting.}"),
              "\\label{tab:external}\\end{table*}"]
    if "bfcl" not in found or "livemcpbench" not in found:
        return "\n".join(lines) + "\n"
    bd, bc, _ = found["bfcl"]
    ld, lc, _ = found["livemcpbench"]
    text = (f"Table~\\ref{{tab:external}} runs the same pipeline, unchanged and without enrichment, on two public "
            f"benchmarks. On BFCL~v4 \\citep{{patil2025bfcl}}, grouped into {bd['registry']['branches']} branches by an "
            f"automatic hierarchy, \\cats{{}} covers {bc['full_cover']:.3f} of requests at $k{{=}}{bc['k_mean']:.1f}$ "
            f"({bc['token_pct_static']:.2f}\\% of static tokens). The one-point gain over flat retrieval comes entirely "
            f"from query decomposition: without it the two tie, and with strict labels the gain vanishes. The tool "
            f"cut-off target was infeasible (best achievable miss "
            f"{bd['calibration']['min_achievable_alpha_tool']:.2f}). On LiveMCPBench \\citep{{mo2026livemcpbench}} "
            f"even flat top-50 retrieval covers every needed tool for only half the tasks, and with about 25 "
            f"calibration tasks per fold \\cats{{}} cannot narrow branches; it ends at {lc['full_cover']:.3f}, level "
            f"with flat retrieval. Outside our own catalog the routed pipeline is therefore no worse than a strong "
            f"flat retriever but not clearly better; its gains on our data rely on enrichment and on enough "
            f"calibration data.\n")
    return text + "\n".join(lines) + "\n"


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "p_jev.tex").write_text(jev_table())
    (TAB / "p_scaling.tex").write_text(scaling_table())
    (TAB / "p_e2e.tex").write_text(e2e_table())
    (TAB / "p_external.tex").write_text(external_table())
    print("wrote", [p.name for p in TAB.glob("*.tex")])


if __name__ == "__main__":
    main()
