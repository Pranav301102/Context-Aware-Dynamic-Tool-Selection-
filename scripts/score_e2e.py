"""Score subagent-as-model runs (runs/<run>/) and test the paired differences between modes.

Each run is re-billed with ``cats.agent_cli.score_run`` (prefix-cache simulation). Runs of the same
model and experiment share their task ids, so modes are compared per task: exact McNemar on task
success, paired bootstrap on $ per task and input tokens, Holm-corrected across all comparisons.

  python scripts/score_e2e.py e2e_static e2e_cats e2e_search e2e_haiku_static e2e_haiku_cats
  python scripts/score_e2e.py --all --out results/e2e_paired.json
  python scripts/score_e2e.py b2_haiku_static b2_haiku_cats --label b2_haiku_static=Haiku ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cats.bench.stats import holm_bonferroni, mcnemar, paired_bootstrap, wilson_interval

RUN_RE = re.compile(r"^(?P<exp>[a-z0-9]+)_(?:(?P<model>[a-z0-9]+)_)?(?P<mode>static|catsjev|cats|search)$")


def infer(run: str) -> tuple[str, str, str | None]:
    """(experiment, model, mode) from a run name such as e2e_haiku_cats or e2e_static (strong model).
    The mode comes from the name because CATS-Jev runs record mode "cats" in their run.json."""
    m = RUN_RE.match(run)
    if not m:
        return ("run", run, None)
    return (m.group("exp"), (m.group("model") or "strong").capitalize(), m.group("mode"))


def score(runs: list[str], labels: dict[str, str], n_boot: int = 10_000) -> dict:
    from cats.agent_cli import score_run

    scored = {}
    for run in runs:
        res = score_run(run)
        exp, model, mode = infer(run)
        model = labels.get(run, model)
        s = res["summary"]
        s["mode"] = mode or s["mode"]
        k, n = round(s["success"] * s["n"]), s["n"]
        s.update({"experiment": exp, "model": model, "success_count": k,
                  "success_wilson": wilson_interval(k, n) if n else None})
        scored[run] = {"summary": s, "rows": res["rows"]}
    comparisons = {}
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for run, r in scored.items():
        s = r["summary"]
        groups.setdefault((s["experiment"], s["model"]), {})[s["mode"]] = run
    for (exp, model), modes in groups.items():
        for m1, m2 in combinations([m for m in ("cats", "catsjev", "search", "static") if m in modes], 2):
            a = {row["task"]: row for row in scored[modes[m1]]["rows"]}
            b = {row["task"]: row for row in scored[modes[m2]]["rows"]}
            ids = [t for t in a if t in b]
            if not ids:
                continue
            key = f"{exp}/{model}: {m1} vs {m2}"
            comparisons[key] = {
                "experiment": exp, "model": model, "a": modes[m1], "b": modes[m2], "n": len(ids),
                "success": mcnemar([a[t]["success"] for t in ids], [b[t]["success"] for t in ids]),
                "usd": paired_bootstrap([a[t]["usd"] for t in ids], [b[t]["usd"] for t in ids], n_boot=n_boot),
                "input_tokens": paired_bootstrap([a[t]["input_tokens"] for t in ids],
                                                 [b[t]["input_tokens"] for t in ids], n_boot=n_boot),
            }
    if comparisons:
        holm = holm_bonferroni({k: v["success"]["p_value"] for k, v in comparisons.items()})
        for k, v in comparisons.items():
            v["success"]["p_holm"] = holm["adjusted"][k]
    return {"runs": [dict(r["summary"], rows=r["rows"]) for r in scored.values()], "comparisons": comparisons}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--all", action="store_true", help="every runs/<run> with a run.json and sessions/")
    p.add_argument("--label", action="append", default=[], help="run=Model label override")
    p.add_argument("--out", default=None, help="write JSON here (default: print only)")
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args(argv)
    runs = list(a.runs)
    if a.all:
        runs += sorted(d.name for d in (ROOT / "runs").iterdir()
                       if (d / "run.json").exists() and (d / "sessions").is_dir() and RUN_RE.match(d.name))
    runs = list(dict.fromkeys(runs))
    if not runs:
        p.error("no runs given")
    labels = dict(x.split("=", 1) for x in a.label)
    res = score(runs, labels, a.n_boot)
    for r in res["runs"]:
        print(f"{r['run']:<22} {r['model']:<8} {r['mode']:<7} success {r['success_count']}/{r['n']} "
              f"${r['usd_per_task']:.4f}/task  {r['input_tokens_mean']:.0f} input tokens/task")
    for k, c in res["comparisons"].items():
        s = c["success"]
        print(f"{k:<38} success {s['a_only']}:{s['b_only']} discordant, McNemar p={s['p_value']:.4g} "
              f"(Holm {s['p_holm']:.4g}); $ diff {c['usd']['mean']:+.4f} [{c['usd']['ci'][0]:+.4f}, "
              f"{c['usd']['ci'][1]:+.4f}]")
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        print("saved", out)


if __name__ == "__main__":
    main()
