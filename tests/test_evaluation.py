"""Evaluation area: statistics, external-benchmark adapters, figures and tables (all offline)."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from cats.bench.external import ExternalBenchmark, auto_hierarchy, registry_from_docs
from cats.bench.external import bfcl as bfcl_adapter
from cats.bench.external import livemcpbench as live_adapter
from cats.bench.external.common import json_schema, snake
from cats.bench.external.run import evaluate_benchmark, pseudo_tasks, stratified_folds
from cats.bench.stats import (
    compare_paired,
    compare_to_reference,
    holm_bonferroni,
    mcnemar,
    mcnemar_exact,
    paired_bootstrap,
    wilson_interval,
)
from cats.tokens import TokenCounter
from cats.types import Task

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ statistics
def test_mcnemar_exact_known_values():
    assert mcnemar_exact(1, 6) == pytest.approx(0.125)  # 2 * (1 + 7) / 2**7
    assert mcnemar_exact(6, 1) == pytest.approx(0.125)  # symmetric
    assert mcnemar_exact(2, 10) == pytest.approx(158 / 4096)  # 2 * (1 + 12 + 66) / 2**12
    assert mcnemar_exact(0, 11) == pytest.approx(2 / 2048)  # the Haiku result: 11 flips, all one way
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 6, mid_p=True) == pytest.approx(9 / 128)  # 2 * (8/128 - 7/256)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_mcnemar_counts_discordant_pairs():
    a = [True] * 25 + [True] * 11
    b = [True] * 25 + [False] * 11
    r = mcnemar(a, b)
    assert (r["a_only"], r["b_only"], r["both"], r["neither"]) == (11, 0, 25, 0)
    assert r["diff"] == pytest.approx(11 / 36)
    assert r["p_value"] == pytest.approx(0.0009765625)
    big = mcnemar([True] * 600 + [False] * 400, [False] * 600 + [True] * 400)  # exact arithmetic at large n
    assert 0 < big["p_value"] < 1e-9


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(81, 263)  # Newcombe (1998), example (Wilson score: 0.2553 to 0.3662)
    assert lo == pytest.approx(0.2553, abs=5e-4) and hi == pytest.approx(0.3662, abs=5e-4)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1.959964 ** 2 / (10 + 1.959964 ** 2), abs=1e-6)  # 0.2775
    lo, hi = wilson_interval(36, 36)
    assert hi == 1.0 and lo == pytest.approx(36 / (36 + 1.959964 ** 2), abs=1e-6)  # 0.9036
    lo, hi = wilson_interval(25, 36)
    assert (round(lo, 3), round(hi, 3)) == (0.531, 0.820)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo90, hi90 = wilson_interval(25, 36, confidence=0.90)
    assert lo < lo90 < hi90 < hi


def test_paired_bootstrap_degenerate_and_shifted():
    x = np.array([0.2, 0.5, 0.9, 0.1, 0.4])
    same = paired_bootstrap(x, x, n_boot=500)
    assert same["mean"] == 0 and same["ci"] == (0.0, 0.0) and same["p_value"] == 1.0
    shifted = paired_bootstrap(x + 1, x, n_boot=500)
    assert shifted["ci"] == pytest.approx((1.0, 1.0)) and shifted["p_value"] == 0.0
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.3, 1, 400), rng.normal(0.0, 1, 400)
    r1 = paired_bootstrap(a, b, n_boot=2000, seed=3)
    r2 = paired_bootstrap(a, b, n_boot=2000, seed=3)
    assert r1 == r2  # deterministic for a seed
    se = np.std(a - b, ddof=1) / math.sqrt(400)
    assert r1["ci"][0] < r1["mean"] < r1["ci"][1]
    assert r1["ci"][1] - r1["ci"][0] == pytest.approx(2 * 1.96 * se, rel=0.15)  # close to normal theory
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])


def test_holm_bonferroni_known_values():
    r = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    assert r["adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert r["reject"] == [True, False, False, True]
    m = holm_bonferroni({"a": 0.5, "b": 0.001})
    assert m["adjusted"] == {"a": 0.5, "b": 0.002}
    assert holm_bonferroni([0.02, 0.025])["adjusted"] == pytest.approx([0.04, 0.04])  # step-down stays monotone
    assert holm_bonferroni([0.9, 0.8])["adjusted"] == [1.0, 1.0]  # capped at 1


def test_compare_to_reference_pairs_by_task_id_and_corrects():
    ids = [f"t{i}" for i in range(40)]
    ref = [{"task_id": i, "full_cover": 1} for i in ids]
    worse = [{"task_id": i, "full_cover": int(n % 4 != 0)} for n, i in enumerate(reversed(ids))]
    equal = [{"task_id": i, "full_cover": 1} for i in ids]
    res = compare_to_reference({"CATS": ref, "worse": worse, "equal": equal}, "CATS", n_boot=500)
    w, e = res["comparisons"]["worse"], res["comparisons"]["equal"]
    assert w["mcnemar"]["a_only"] == 10 and w["mcnemar"]["b_only"] == 0
    assert w["p_holm"] == pytest.approx(min(1.0, 2 * w["p_value"]))
    assert e["p_value"] == 1.0 and not e["significant_holm"]
    cont = compare_paired([{"task_id": "a", "x": 0.5}], [{"task_id": "a", "x": 0.25}], field="x", n_boot=10)
    assert "mcnemar" not in cont and cont["bootstrap"]["mean"] == 0.25


# ---------------------------------------------------------------------------- adapters
LIVE_TOOLS = [
    {"name": "Fetch Server", "category": "Discovery", "description": "Fetch web pages.",
     "tools": {"fetch": {"tools": [
         {"name": "fetch", "description": "Fetch a URL and return markdown.",
          "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}},
         {"name": "read_file", "description": "Read a cached page from disk.", "inputSchema": {"type": "object"}},
     ]}}},
    {"name": "Desktop", "category": "File Access", "description": "Local files.",
     "tools": {"desktop-commander": {"tools": [
         {"name": "read_file", "description": "Read a local file.", "inputSchema": {"type": "object"}},
         {"name": "write-file", "description": "Write a local file.", "inputSchema": {"type": "object"}},
     ]}, "empty-server": {"tools": []}}},
]
LIVE_ANN = [
    {"task_id": "aaaaaaaa-1", "Question": "Save the page example.com to notes.md", "category": "Office",
     "Annotator Metadata": {"Tools": "1. fetch\n2. write-file\n3. write-file\n4. no_such_tool"}},
    {"task_id": "bbbbbbbb-2", "Question": "Show me notes.md", "category": "Lifestyle",
     "Annotator Metadata": {"Tools": "1) read_file"}},
    {"task_id": "cccccccc-3", "Question": "Unanswerable", "category": "Travel",
     "Annotator Metadata": {"Tools": "1. unknown_tool"}},
]


def test_livemcpbench_conversion():
    b = live_adapter.convert(LIVE_TOOLS, LIVE_ANN)
    assert b.meta["tools"] == 4 and b.meta["branches"] == 2 and b.meta["toolkits"] == 2  # empty server dropped
    assert b.meta["tasks"] == 2 and b.meta["tasks_dropped_no_resolvable_tool"] == 1
    assert b.meta["annotated_names_unresolved"] == ["no_such_tool", "unknown_tool"]
    t0, t1 = b.tasks
    assert t0.gold == [["fetch.fetch"], ["desktop_commander.write-file"]]  # duplicate step merged
    assert t0.branches == ["discovery", "file_access"] and t0.kind == "office"
    assert t1.gold == [["desktop_commander.read_file", "fetch.read_file"]]  # same name on two servers
    reg = b.registry(TokenCounter("approx"))
    assert set(reg.tools) == b.tool_ids()
    assert reg.toolkits["desktop_commander"].name == "Desktop (desktop-commander)"
    assert all(g in reg.tools for t in b.tasks for g in t.gold_tools())


def _bfcl_fixture():
    def fn(name, desc, props=None):
        return {"name": name, "description": desc,
                "parameters": {"type": "dict", "properties": props or {"x": {"type": "float", "description": "x"}},
                               "required": []}}

    entries = {
        "simple_python": [
            {"id": "simple_python_0", "question": [[{"role": "user", "content": "factorial of 5"}]],
             "function": [fn("math.factorial", "Compute the factorial of a number.")]},
            {"id": "simple_python_1", "question": [[{"role": "user", "content": "weather in Paris"}]],
             "function": [fn("get_weather", "Get the current weather for a city.")]},
        ],
        "live_multiple": [
            {"id": "live_multiple_0-0-0",
             "question": [[{"role": "system", "content": "You are a helpful bot."},
                           {"role": "user", "content": "first turn"}, {"role": "assistant", "content": "ok"},
                           {"role": "user", "content": "weather in Rome and factorial of 3"}]],
             "function": [fn("get_weather", "Weather for a city, with units."), fn("math.factorial", "Factorial.")]},
            {"id": "live_multiple_1-0-0", "question": [[{"role": "user", "content": "no answer here"}]],
             "function": [fn("noop", "Does nothing.")]},
        ],
    }
    answers = {"simple_python_0": [{"math.factorial": {"x": [5]}}],
               "simple_python_1": [{"get_weather": {"x": [1]}}],
               "live_multiple_0-0-0": [{"get_weather": {}}, {"math.factorial": {}}, {"get_weather": {}}]}
    distractors = {"irrelevance": [{"id": "irrelevance_0", "question": [], "function": [
        fn("stock.price", "Get a stock price."), fn("geo.distance", "Distance between two places.")]}]}
    return entries, answers, distractors


def test_bfcl_conversion_widening_and_history():
    entries, answers, distractors = _bfcl_fixture()
    b = bfcl_adapter.convert(entries, answers, distractors, n_branches=2, seed=0)
    reg = b.registry(TokenCounter("approx"))
    assert len(reg.tools) == b.meta["tools"] == 7  # 2 + 2 + 1 noop + 2 distractors, variants kept apart
    assert b.meta["tasks"] == 3 and b.meta["skipped"]["no_answer"] == 1
    by_id = {t.id: t for t in b.tasks}
    live = by_id["bfcl-live_multiple_0-0-0"]
    assert live.query == "weather in Rome and factorial of 3"
    assert [h.role for h in live.history] == ["user", "assistant"]  # system prompt dropped
    assert len(live.gold) == 2 and all(len(s) == 2 for s in live.gold)  # both variants of each name count
    fact = next(s for s in live.gold if any(".factorial" in x for x in s))
    assert all(x.startswith("math.") for x in fact)  # namespace became the toolkit
    assert reg.tools[fact[0]].input_schema["properties"]["x"]["type"] == "number"
    assert reg.tools[fact[0]].input_schema["type"] == "object"
    strict = bfcl_adapter.convert(entries, answers, distractors, n_branches=2, seed=0, widen=False)
    assert all(len(s) == 1 for t in strict.tasks for s in t.gold)
    again = bfcl_adapter.convert(entries, answers, distractors, n_branches=2, seed=0)
    assert again.corpus == b.corpus  # deterministic hierarchy for a seed


def test_auto_hierarchy_and_roundtrip(tmp_path):
    units = {f"fn:t{i}": {"namespace": None, "tools": [{"name": f"t{i}", "description": d, "input_schema": {}}]}
             for i, d in enumerate(["send email to a person", "send an email message", "read the inbox email",
                                    "weather forecast rain", "weather temperature now", "storm weather alert"])}
    units["ns:sql"] = {"namespace": "sql", "tools": [
        {"name": "query", "description": "run a database query", "input_schema": {}},
        {"name": "query", "description": "run a database query, read only", "input_schema": {}}]}
    docs = auto_hierarchy(units, n_branches=3, toolkit_size=2, seed=0)
    ids = [f"{tk['id']}.{t['name']}" for d in docs for tk in d["toolkits"] for t in tk["tools"]]
    assert len(ids) == len(set(ids)) == 8  # names made unique inside a toolkit
    assert {"sql.query", "sql.query_2"} <= set(ids)
    assert all(d["branch"]["summary"] for d in docs)
    tasks = [Task(id="x1", query="email bob", gold=[[ids[0]]], branches=[docs[0]["branch"]["id"]])]
    bench = ExternalBenchmark("toy", docs, tasks, {"name": "toy"})
    back = ExternalBenchmark.read(bench.write(tmp_path / "converted"))
    assert back.corpus == docs and back.tasks == tasks
    assert set(registry_from_docs(back.corpus, TokenCounter("approx")).tools) == set(ids)


def test_snake_and_schema_helpers():
    assert snake("File Access") == "file_access" and snake("2nd-server") == "x_2nd_server" and snake("!!") == "x"
    s = json_schema({"type": "dict", "properties": {"a": {"type": "tuple", "items": {"type": "String"}}}})
    assert s == {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}}}}
    assert json_schema(None) == {"type": "object", "properties": {}}


# ------------------------------------------------------------------------- end to end
def _toy_benchmark() -> ExternalBenchmark:
    def obj(**p):
        return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in p.items()}}

    corpus = [
        {"branch": {"id": "weather", "name": "Weather", "summary": "Forecasts and weather alerts."},
         "toolkits": [{"id": "ow", "name": "OpenWeather", "summary": "Weather API.", "tools": [
             {"name": "forecast", "description": "Multi-day weather forecast with rain chance.",
              "input_schema": obj(city="City")},
             {"name": "current", "description": "Current temperature and wind for a city.",
              "input_schema": obj(city="City")},
             {"name": "alerts", "description": "Active severe storm and flood warnings.",
              "input_schema": obj(region="Region")}]}]},
        {"branch": {"id": "chat", "name": "Chat", "summary": "Team chat messages."},
         "toolkits": [{"id": "slack", "name": "Slack", "summary": "Slack messaging.", "tools": [
             {"name": "post", "description": "Post a message to a Slack channel.", "input_schema": obj(text="Text")},
             {"name": "search", "description": "Search Slack message history.", "input_schema": obj(q="Query")}]}]},
        {"branch": {"id": "code", "name": "Code", "summary": "Run code and tests."},
         "toolkits": [{"id": "box", "name": "Sandbox", "summary": "Python sandbox.", "tools": [
             {"name": "run_python", "description": "Execute a Python snippet.", "input_schema": obj(code="Code")},
             {"name": "run_tests", "description": "Run the pytest suite.", "input_schema": obj(path="Dir")}]}]},
    ]
    qs = [("weather", "ow.forecast", "will it rain in {} tomorrow"), ("weather", "ow.current", "how hot is {} now"),
          ("weather", "ow.alerts", "storm warnings near {}"), ("chat", "slack.post", "tell the {} channel hello"),
          ("chat", "slack.search", "find slack messages about {}"), ("code", "box.run_python", "run python for {}"),
          ("code", "box.run_tests", "do the {} unit tests pass")]
    words = ["paris", "oslo", "lima", "rome", "kyiv"]
    tasks = [Task(id=f"q{i}-{j}", query=q.format(w), gold=[[g]], branches=[b], kind="literal")
             for i, (b, g, q) in enumerate(qs) for j, w in enumerate(words)]
    return ExternalBenchmark("toy", corpus, tasks, {"name": "toy"})


def test_stratified_folds_partition_every_task_once():
    tasks = _toy_benchmark().tasks
    folds = stratified_folds(tasks, 4, seed=1)
    ids = [t.id for f in folds for t in f]
    assert sorted(ids) == sorted(t.id for t in tasks)
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


@pytest.mark.parametrize("protocol", ["crossfit", "split"])
def test_evaluate_benchmark_end_to_end(monkeypatch, protocol):
    monkeypatch.setenv("CATS_EMBEDDER", "lsa")
    bench = _toy_benchmark()
    res = evaluate_benchmark(bench, protocol=protocol, folds=3, k=2, n_boot=200)
    names = {s["strategy"] for s in res["summaries"]}
    assert {"B0_static", "CATS", "O_oracle", "B3_hybrid_rerank@k=CATS", "B4_two_stage@3x2"} <= names
    n = res["protocol"]["n_tasks"]
    cats = next(s for s in res["summaries"] if s["strategy"] == "CATS")
    assert isinstance(res["calibration"], dict) and "branch_threshold" in res["calibration"]  # bench_*.json shape
    if protocol == "crossfit":
        assert cats["n"] == n == len(bench.tasks) and len(res["calibration"]["folds"]) == 3
    curve = res["recall_at_k"]
    assert len(curve["k"]) == len(curve["slot_recall"]) and curve["slot_recall"] == sorted(curve["slot_recall"])
    static = next(s for s in res["summaries"] if s["strategy"] == "B0_static")
    assert static["full_cover"] == 1.0 and static["token_pct_static"] == pytest.approx(100.0)
    comps = res["stats"]["full_cover"]["comparisons"]
    assert "B0_static" not in comps and all(0 <= c["p_holm"] <= 1 for c in comps.values())
    pt = res["per_task"]
    assert len(pt["task_ids"]) == cats["n"] and set(pt["strategies"]) == names
    json.dumps(res, default=str)  # serialisable


def test_pseudo_tasks_one_per_described_tool():
    reg = _toy_benchmark().registry(TokenCounter("approx"))
    ps = pseudo_tasks(reg)
    assert len(ps) == len(reg.tools)
    assert all(p.gold == [[tid]] and p.branches == [reg.tools[tid].branch_id]
               for p, tid in zip(ps, sorted(reg.tools)))


# ----------------------------------------------------------------- figures and tables
def _mini_results(d: Path) -> None:
    summ = [{"strategy": s, "n": 10, "full_cover": c, "full_cover_ci": [c - 0.1, min(1, c + 0.1)], "slot_recall": c,
             "k_mean": k, "tool_tokens": 100.0 * k, "token_pct_static": k, "router_tokens": 0.0,
             "latency_p50_ms": 1.0, "latency_p95_ms": 2.0, "precision": 0.5}
            for s, c, k in [("B0_static", 1.0, 100), ("B1_bm25@5", 0.7, 5), ("CATS", 0.9, 8), ("O_oracle", 1.0, 1)]]
    paired = {s["strategy"]: {"mean": 0.9 - s["full_cover"], "ci": [-0.1, 0.1], "n": 10} for s in summ}
    (d / "bench_final.json").write_text(json.dumps({"registry": {"tools": 100, "branches": 4, "tool_tokens": 10000},
                                                     "summaries": summ, "paired_full_cover_vs_cats": paired}))
    (d / "e2e_x.json").write_text(json.dumps([
        {"run": "e2e_static", "mode": "static", "n": 10, "success": 0.7, "usd_per_task": 0.05, "model": "Small"},
        {"run": "e2e_cats", "mode": "cats", "n": 10, "success": 1.0, "usd_per_task": 0.01, "model": "Small"}]))
    (d / "b2_y.json").write_text(json.dumps({"summary": {"run": "b2_big_cats", "mode": "cats", "n": 5,
                                                         "success": 0.8, "usd_per_task": 0.02}, "rows": []}))
    for name, variant in (("external_toy.json", None), ("external_toy_strict.json", "strict labels")):
        (d / name).write_text(json.dumps({
            "benchmark": "toy", "registry": {"tools": 100}, "meta": {},
            "protocol": {"name": "split", "n_tasks": 10, "variant": variant},
            "summaries": [s for s in summ if s["strategy"] in ("B1_bm25@5", "CATS")],
            "stats": {"full_cover": {"comparisons": {"B1_bm25@5": {"bootstrap": {"mean": 0.2, "ci": [0.1, 0.3]},
                                                                   "p_holm": 0.004}}}},
            "recall_at_k": {"k": [1, 5, 10], "slot_recall": [0.5, 0.8, 0.9], "full_cover": [0.4, 0.7, 0.85]}}))


def test_figures_and_tables_from_results(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _mini_results(res)
    figs = _script("make_figures")
    entries = figs._e2e_entries(res)
    assert {(e["model"], e["mode"]) for e in entries} == {("Small", "static"), ("Small", "cats"), ("big", "cats")}
    written = figs.main(["--results", str(res), "--out", str(tmp_path / "fig"),
                         "--only", "pareto,e2e,scaling,external,external_recall"])
    assert set(written) == {"pareto", "e2e", "external", "external_recall"}  # scaling skipped: no input
    assert all(Path(p).stat().st_size > 1000 for ps in written.values() for p in ps)
    tables = _script("make_tables")
    out = tables.main(["--results", str(res), "--out", str(tmp_path / "tab")])
    assert set(out) == {"baselines", "e2e", "external"}
    ext = Path(out["external"]).read_text()
    assert ext.count("toy (100 tools") == 2 and "strict labels" in ext and "$<$0.001" not in ext
    assert "-0.200 [-0.300, -0.100] & 0.004" in ext  # baseline minus CATS, Holm p
    tex = Path(out["baselines"]).read_text()
    assert "\\toprule" in tex and "\\bottomrule" in tex and "B1 BM25 @5" in tex
    assert "7/10 [0.40, 0.89]" in Path(out["e2e"]).read_text()


def test_score_e2e_run_names():
    mod = _script("score_e2e")
    assert mod.infer("e2e_static") == ("e2e", "Strong", "static")
    assert mod.infer("e2e_haiku_cats") == ("e2e", "Haiku", "cats")
    assert mod.infer("b2_sonnet_search") == ("b2", "Sonnet", "search")
    assert mod.infer("b2_opus_catsjev") == ("b2", "Opus", "catsjev")


def test_fetch_external_pins_commit_and_checks_sha256(tmp_path, monkeypatch):
    mod = _script("fetch_external")
    payload = {"v": b"version one"}

    class _Resp:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload["v"]

    seen = []
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda url, timeout=0: seen.append(url) or _Resp(url))
    monkeypatch.setattr(mod, "EXT", tmp_path)
    monkeypatch.setattr(mod, "SOURCES", {"toy": {"repo": "o/r", "ref": "abc123", "licence": "MIT",
                                                 "files": {"data.json": "dir/data.json"}}})
    man = mod.fetch("toy")
    assert seen == ["https://raw.githubusercontent.com/o/r/abc123/dir/data.json"]  # pinned to a commit
    assert man["files"]["data.json"]["sha256"] == mod.sha256(tmp_path / "toy" / "raw" / "data.json")
    mod.fetch("toy")  # present and unchanged: no download, same manifest
    assert len(seen) == 1
    payload["v"] = b"upstream changed"
    with pytest.raises(SystemExit):
        mod.fetch("toy", force=True)
