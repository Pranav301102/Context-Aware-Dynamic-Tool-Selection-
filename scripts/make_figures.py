"""Publication figures from results/*.json (PDF + PNG) into paper/figures/.

  scaling      full cover vs tool-pool size per strategy            results/scale_final.json
  ablation     full cover per CATS configuration, 95% CIs          results/ablation_final.json
  validity     slot miss and branch cover per split vs targets     results/validity_{enrich,independent}.json
  pareto       tool tokens per request vs full cover (frontier)    results/bench_final.json, bench_final_llm.json
  e2e          end-to-end success (Wilson CI) and $ per task       results/e2e_*.json, results/b2_*.json
  external     full cover per strategy on external benchmarks      results/external_*.json
  external_recall  flat recall@k curve vs CATS's operating point   results/external_*.json

Colours follow the strategy, never its rank (the same strategy has the same colour in every figure),
from a palette validated for colour-vision deficiency; every series also has its own marker, line
style or hatch, so the figures stay readable in grayscale. A figure whose inputs are missing is
skipped with a note.

  python scripts/make_figures.py [--results results] [--out paper/figures] [--only scaling,pareto]
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

# Validated categorical palette (CVD-safe adjacent pairs; see docs/components/evaluation.md §10).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# strategy -> (label, colour, marker, linestyle, hatch). Keys are matched by prefix.
STYLE = [
    ("CATS + LLM router", ("CATS + LLM router", VIOLET, "D", "-", "xx")),
    ("CATS", ("CATS", BLUE, "o", "-", "")),
    ("B3_hybrid_rerank@k=CATS", ("Flat reranked, k = CATS", ORANGE, "v", ":", "\\\\")),
    ("B3_hybrid_rerank@10", ("Flat reranked @10", ORANGE, "s", "--", "//")),
    ("B3_hybrid_rerank@5", ("Flat reranked @5", AQUA, "^", "-.", "..")),
    ("B3_hybrid_rrf@5", ("Flat hybrid RRF @5", YELLOW, "P", ":", "--")),
    ("B1_bm25", ("BM25 @5 (B1)", MAGENTA, "X", (0, (5, 1)), "||")),
    ("B2_dense", ("Dense @5 (B2)", GREEN, "d", (0, (3, 1, 1, 1)), "++")),
    ("B4_two_stage", ("Two-stage @3x5 (B4)", RED, "h", (0, (1, 2)), "oo")),
    ("B0_static", ("Static loading (B0)", MUTED, "8", "-", "")),
    ("O_oracle", ("Oracle", INK, "*", "-", "")),
]


def style(name: str):
    for key, st in STYLE:
        if name.startswith(key):
            return st
    return (name, MUTED, "o", "-", "")


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150,
        "hatch.linewidth": 0.6,
    })
    return plt


def load(results: Path, name: str):
    p = results / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- figures
def fig_scaling(plt, results: Path, out: Path):
    rows = load(results, "scale_final.json")
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["size"])
    names = [k for k in rows[0] if isinstance(rows[0][k], dict) and "full_cover" in rows[0][k]]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    xs = [r["size"] for r in rows]
    for name in names:
        label, color, marker, ls, _ = style(name)
        ys = [r[name]["full_cover"] for r in rows]
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, label=label, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3 if name == "CATS" else 2)
    ax.set_xscale("log")
    ax.set_xlabel("Tools in the pool (log scale)")
    ax.set_ylabel("Full cover (test requests)")
    ax.axvspan(min(xs) * 0.8, 477, color=GRID, alpha=0.35, lw=0, zorder=0)
    ymin = min(r[n]["full_cover"] for r in rows for n in names)
    ax.text(math.sqrt(min(xs) * 0.8 * 477), ymin, "task subsets\n(< 477 tools)", fontsize=6, color=MUTED,
            ha="center", va="bottom")
    ax.set_xlim(min(xs) * 0.8, max(xs) * 1.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=2.6, columnspacing=1.2)
    ax.set_title("Coverage as the catalog grows", loc="left")
    return save(fig, out, "scaling")


def fig_ablation(plt, results: Path, out: Path):
    rows = load(results, "ablation_final.json")
    if not rows:
        return None
    import numpy as np

    labels = [r["strategy"] for r in rows]
    cover = np.array([r["full_cover"] for r in rows])
    lo = np.array([r.get("full_cover_ci", [c, c])[0] for r, c in zip(rows, cover)])
    hi = np.array([r.get("full_cover_ci", [c, c])[1] for r, c in zip(rows, cover)])
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(rows) + 0.7))
    y = np.arange(len(rows))[::-1]
    colors = [MUTED if i == 0 else (VIOLET if "LLM" in labels[i] else BLUE) for i in range(len(rows))]
    hatches = ["//" if i == 0 else ("xx" if "LLM" in labels[i] else "") for i in range(len(rows))]
    bars = ax.barh(y, cover, color=colors, height=0.62, edgecolor="white", linewidth=1.0)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.errorbar(cover, y, xerr=[cover - lo, hi - cover], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for yi, c, r in zip(y, cover, rows):
        ax.text(hi[list(y).index(yi)] + 0.008, yi, f"{c:.3f}   k = {r['k_mean']:.1f}", va="center", fontsize=6.5,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(max(0.0, float(lo.min()) - 0.1), 1.08)
    ax.set_xlabel("Full cover on the held-out split (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What each CATS component buys", loc="left")
    return save(fig, out, "ablation")


def fig_validity(plt, results: Path, out: Path):
    joint = load(results, "validity_enrich.json")
    indep = load(results, "validity_independent.json")
    if not joint or not indep:
        return None
    import numpy as np

    seeds = [r["seed"] for r in joint["rows"]]
    w = 0.38
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    series = [("End-to-end calibration (default)", joint, BLUE, ""),
              ("Per-stage calibration (Algorithm 1)", indep, ORANGE, "//")]
    panels = [("slot_miss", "slot_miss_target", "Slot miss (lower is better)", "≤"),
              ("branch_cover", "branch_target", "Branch cover (higher is better)", "≥")]
    for ax, (field, tkey, ylabel, sym) in zip(axes, panels):
        for i, (label, res, color, hatch) in enumerate(series):
            vals = [r[field] for r in res["rows"]]
            ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, hatch=hatch, edgecolor="white", linewidth=1.0,
                   label=f"{label}: mean {np.mean(vals):.3f}")
        target = joint[tkey]
        ax.axhline(target, color=INK, linestyle="--", linewidth=1.0)
        ax.text(-0.55, target, f"target {sym} {target:.2f}", va="bottom", ha="left", fontsize=6.5, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        ax.set_xticks(x)
        ax.set_xticklabels([f"split {s}" for s in seeds])
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        if field == "branch_cover":
            lo = min(min(r[field] for r in res["rows"]) for _, res, _, _ in series)
            ax.set_ylim(min(lo, target) - 0.02, 1.005)
        else:
            ax.set_ylim(0, max(max(r[field] for r in res["rows"]) for _, res, _, _ in series) * 1.25)
    axes[0].legend(loc="upper left", fontsize=6.5)
    axes[0].set_title("Conformal validity over five random splits", loc="left")
    fig.tight_layout(w_pad=2.0)
    return save(fig, out, "validity")


def _tokens(s: dict) -> float:
    return float(s["tool_tokens"]) + float(s.get("router_tokens") or 0.0)


def fig_pareto(plt, results: Path, out: Path):
    base = load(results, "bench_final.json")
    if not base:
        return None
    llm = load(results, "bench_final_llm.json")
    pts = [(s["strategy"], s) for s in base["summaries"]]
    if llm:
        cats_llm = next((s for s in llm["summaries"] if s["strategy"] == "CATS"), None)
        if cats_llm:
            pts.append(("CATS + LLM router", cats_llm))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    frontier_pts = []
    for name, s in pts:
        label, color, marker, _, _ = style(name)
        x, y = _tokens(s), s["full_cover"]
        size = 46 if name.startswith("CATS") else 30
        face = "white" if name == "O_oracle" else color
        ax.scatter([x], [y], s=size, marker=marker, color=face, edgecolors=INK if name == "O_oracle" else "white",
                   linewidths=0.8, zorder=3, label=label)
        if name != "O_oracle":
            frontier_pts.append((x, y))
        if name.startswith("CATS + LLM") and s.get("router_tokens"):
            ax.scatter([s["tool_tokens"]], [y], s=size, marker=marker, facecolors="none", edgecolors=color,
                       linewidths=0.9, zorder=3)
            ax.annotate("", xy=(x, y), xytext=(s["tool_tokens"], y),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8})
    # Pareto frontier: fewest tokens for at least this coverage.
    frontier_pts.sort()
    best, front = -1.0, []
    for x, y in frontier_pts:
        if y > best:
            front.append((x, y))
            best = y
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.step(fx, fy, where="post", color=MUTED, linewidth=0.9, linestyle="--", zorder=1, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Tokens per request (tool definitions + router, log)")
    ax.set_ylabel(f"Full cover (test, n = {base['summaries'][0]['n']})")
    for name, s in pts:  # direct labels for the points the text discusses
        if name in ("CATS", "CATS + LLM router", "B0_static", "O_oracle", "B3_hybrid_rerank@10"):
            ha, dx = ("right", -6) if name in ("B0_static", "CATS + LLM router", "B3_hybrid_rerank@10") else ("left", 6)
            dy = -10 if name in ("CATS", "B3_hybrid_rerank@10") else 3
            ax.annotate(style(name)[0], (_tokens(s), s["full_cover"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6, color=INK, ha=ha)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6, ncol=2, handletextpad=0.3,
              columnspacing=1.0)
    ax.set_title("Cost-coverage trade-off (477 tools)", loc="left")
    return save(fig, out, "pareto")


def _e2e_entries(results: Path) -> list[dict]:
    """Collect end-to-end run summaries from e2e_*.json and b2_*.json in any of the shapes the
    harness writes: a list of summaries, {"summary": ...}, or {"runs"/"summaries": [...]}."""
    out = []
    files = sorted(set(glob.glob(str(results / "e2e_*.json")) + glob.glob(str(results / "b2_*.json"))))
    for f in files:
        try:
            doc = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        items = doc if isinstance(doc, list) else (
            doc.get("runs") or doc.get("summaries") or ([doc["summary"]] if "summary" in doc else [doc]))
        for it in items:
            if isinstance(it, dict) and "summary" in it and isinstance(it["summary"], dict):
                it = {**it["summary"], **{k: v for k, v in it.items() if k not in ("summary", "rows")}}
            if not isinstance(it, dict) or "success" not in it or "mode" not in it:
                continue
            run = str(it.get("run", ""))
            model = it.get("model") or _model_from_run(run)
            out.append({**it, "model": model, "source": Path(f).name})
    seen, uniq = set(), []
    for e in out:  # the same run may be summarised in several files
        key = (e.get("run"), e["mode"]) if e.get("run") else (id(e), e["mode"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


def _model_from_run(run: str) -> str:
    m = re.match(r"(?:e2e|b2)_(?:(\w+?)_)?(static|cats|search)$", run)
    return (m.group(1) or "strong") if m else (run or "model")


def _group(e: dict) -> str:
    """x-axis group: model and experiment (e2e = first batch, b2 = second batch, ...)."""
    model = re.sub(r"\s*subagent$", "", str(e["model"]), flags=re.IGNORECASE).strip().capitalize()
    exp = str(e.get("run", "")).split("_")[0] or Path(e.get("source", "")).stem
    return f"{model}\n{exp}"


MODE_STYLE = {"static": ("Static loading (B0)", MUTED, "//"), "cats": ("CATS", BLUE, ""),
              "search": ("Agent-initiated search (B6)", VIOLET, "xx")}


def fig_e2e(plt, results: Path, out: Path):
    entries = _e2e_entries(results)
    if not entries:
        return None
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from cats.bench.stats import wilson_interval

    for e in entries:
        e["group"] = _group(e)
    models = list(dict.fromkeys(e["group"] for e in entries))
    modes = [m for m in ("static", "cats", "search") if any(e["mode"] == m for e in entries)]
    w = 0.8 / len(modes)
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for j, mode in enumerate(modes):
        label, color, hatch = MODE_STYLE.get(mode, (mode, MUTED, ""))
        succ, lo, hi, usd = [], [], [], []
        for m in models:
            e = next((e for e in entries if e["group"] == m and e["mode"] == mode), None)
            if e is None:
                succ.append(math.nan)
                lo.append(math.nan)
                hi.append(math.nan)
                usd.append(math.nan)
                continue
            n = int(e.get("n") or 0)
            k = round(float(e["success"]) * n)
            a, b = wilson_interval(k, n) if n else (math.nan, math.nan)
            succ.append(float(e["success"]))
            lo.append(a)
            hi.append(b)
            usd.append(float(e.get("usd_per_task", math.nan)))
        pos = x + (j - (len(modes) - 1) / 2) * w
        s, a, b = np.array(succ), np.array(lo), np.array(hi)
        axes[0].bar(pos, s, width=w, color=color, hatch=hatch, edgecolor="white", linewidth=1.0, label=label)
        axes[0].errorbar(pos, s, yerr=[s - a, b - s], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        cents = [100 * v for v in usd]
        axes[1].bar(pos, cents, width=w, color=color, hatch=hatch, edgecolor="white", linewidth=1.0, label=label)
        for p, v in zip(pos, cents):
            if not math.isnan(v):
                axes[1].text(p, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=INK)
    ns = {m: max(int(e.get("n") or 0) for e in entries if e["group"] == m) for m in models}
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}, n = {ns[m]}" for m in models], fontsize=7 if len(models) <= 3 else 5.8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Task success (95% Wilson CI)")
    axes[1].set_ylabel("US cents per task (prefix-cache simulation)")
    axes[0].set_title("End to end, subagent as the model", loc="left")
    fig.tight_layout(w_pad=2.0, rect=(0, 0.08, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(modes), fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    return save(fig, out, "e2e")


def _variant(d: dict) -> str | None:
    """Label of an ablation or robustness rerun of an external benchmark (None for the main run)."""
    v = (d.get("protocol") or {}).get("variant")
    if not v and (d.get("meta") or {}).get("label_widening") == "none":
        v = "strict labels"
    return v


EXT_ORDER = ["B1_bm25@5", "B2_dense@5", "B3_hybrid_rrf@5", "B3_hybrid_rerank@5", "B4_two_stage@3x5",
             "B3_hybrid_rerank@10", "B3_hybrid_rerank@k=CATS", "CATS"]


def fig_external(plt, results: Path, out: Path):
    files = sorted(glob.glob(str(results / "external_*.json")))
    docs = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "summaries" in d and not _variant(d):
            docs.append((Path(f).stem.replace("external_", ""), d))  # ablation/robustness reruns: table only
    if not docs:
        return None
    import numpy as np

    fig, axes = plt.subplots(1, len(docs), figsize=(3.4 * len(docs), 2.6), squeeze=False)
    for ax, (name, d) in zip(axes[0], docs):
        rows = {s["strategy"]: s for s in d["summaries"]}
        names = [n for n in EXT_ORDER if n in rows]
        y = np.arange(len(names))[::-1]
        for yi, n in zip(y, names):
            _, color, _, _, hatch = style(n)
            s = rows[n]
            lo, hi = s.get("full_cover_ci", (s["full_cover"], s["full_cover"]))
            ax.barh([yi], [s["full_cover"]], color=color, hatch=hatch, edgecolor="white", linewidth=1.0, height=0.66)
            ax.errorbar([s["full_cover"]], [yi], xerr=[[s["full_cover"] - lo], [hi - s["full_cover"]]], fmt="none",
                        ecolor=INK, elinewidth=0.8, capsize=2)
            ax.text(hi + 0.01, yi, f"{s['full_cover']:.3f} (k {s['k_mean']:.1f})", va="center", fontsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels([style(n)[0] for n in names])
        ax.set_xlim(0, min(1.15, max(rows[n].get("full_cover_ci", (0, rows[n]["full_cover"]))[1] for n in names) + 0.22))
        reg = d.get("registry", {})
        ax.set_title(f"{name}: {reg.get('tools', 0):,} tools, n = {rows['CATS']['n']:,}", loc="left")
        ax.set_xlabel("Full cover (95% bootstrap CI)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=2.0)
    return save(fig, out, "external")


def fig_external_recall(plt, results: Path, out: Path):
    """Slot recall of flat reranked retrieval as k grows, with CATS's operating point and NFR2's 0.95."""
    docs = []
    for f in sorted(glob.glob(str(results / "external_*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("recall_at_k") and "summaries" in d and not _variant(d):
            docs.append((Path(f).stem.replace("external_", ""), d))
    if not docs:
        return None
    fig, axes = plt.subplots(1, len(docs), figsize=(3.0 * len(docs), 2.3), squeeze=False, sharey=True)
    for ax, (name, d) in zip(axes[0], docs):
        cur = d["recall_at_k"]
        _, color, marker, ls, _ = style("B3_hybrid_rerank@10")
        ax.plot(cur["k"], cur["slot_recall"], color=color, marker=marker, linestyle=ls, markersize=4,
                label="Flat reranked, slot recall @k")
        ax.plot(cur["k"], cur["full_cover"], color=color, marker=marker, linestyle=":", markersize=3.5,
                markerfacecolor="white", label="Flat reranked, full cover @k")
        cats = next(s for s in d["summaries"] if s["strategy"] == "CATS")
        _, c_color, c_marker, _, _ = style("CATS")
        ax.scatter([cats["k_mean"]], [cats["slot_recall"]], s=40, marker=c_marker, color=c_color, zorder=4,
                   edgecolors="white", linewidths=0.8, label="CATS, slot recall")
        ax.scatter([cats["k_mean"]], [cats["full_cover"]], s=34, marker="s", facecolors="white", edgecolors=c_color,
                   linewidths=1.1, zorder=4, label="CATS, full cover")
        ax.axhline(0.95, color=INK, linestyle="--", linewidth=0.9)
        ax.text(cur["k"][0], 0.955, "NFR2 recall target 0.95", fontsize=6, color=INK, va="bottom")
        ax.set_xscale("log")
        ax.set_xticks(cur["k"])
        ax.set_xticklabels([str(k) for k in cur["k"]])
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Tools in the prompt (k, log)")
        ax.set_title(f"{name}: {d['registry']['tools']:,} tools", loc="left")
    axes[0][0].set_ylabel("Share of test requests / slots")
    fig.tight_layout(rect=(0, 0.1 if len(docs) > 1 else 0.16, 1, 1))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4 if len(docs) > 1 else 2, fontsize=6.3,
               bbox_to_anchor=(0.5, 0.0))
    return save(fig, out, "external_recall")


FIGURES = {"scaling": fig_scaling, "ablation": fig_ablation, "validity": fig_validity, "pareto": fig_pareto,
           "e2e": fig_e2e, "external": fig_external, "external_recall": fig_external_recall}


def main(argv=None) -> dict[str, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    p.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(FIGURES))
    a = p.parse_args(argv)
    plt = setup()
    results, out = Path(a.results), Path(a.out)
    wanted = [x for x in a.only.split(",") if x] or list(FIGURES)
    written: dict[str, list[str]] = {}
    for name in wanted:
        paths = FIGURES[name](plt, results, out)
        if paths:
            written[name] = [str(x) for x in paths]
            print(f"{name:<9} -> {', '.join(Path(x).name for x in paths)}")
        else:
            print(f"{name:<9} skipped (inputs missing in {results})")
    return written


if __name__ == "__main__":
    main()
