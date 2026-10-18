"""Platform layer: REST/SSE gateway, OpenTelemetry export, Parquet metrics, run manifests, Hydra
runner and deployment files. Everything runs offline on a tiny fitted CATS (9 tools)."""
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("pyarrow")
pytest.importorskip("hydra")

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cats import api as api_mod
from cats.analyzer import ContextAnalyzer
from cats.api import Settings, create_app, stage_events
from cats.bench.harness import evaluate, summarize
from cats.build import ROOT, System
from cats.dispatcher import Dispatcher
from cats.index import LSAEmbedder, ToolIndex
from cats.llm import HeuristicLLM
from cats.manifest import _git_from_files, git_info, load_manifest, run_manifest, verify_data
from cats.metrics_store import MetricsStore, convert_results, detect_kind
from cats.pipeline import CATS, CATSConfig
from cats.registry import Registry
from cats.strategies import CATSStrategy
from cats.tokens import TokenCounter
from cats.tracing import TraceExporter, attr_value, langfuse_processor, make_provider
from cats.types import Branch, Task, Toolkit


# ------------------------------------------------------------------ tiny corpus (as in conftest)
def _obj(**props):
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props)[:1]}


CORPUS = [
    (Branch(id="weather", name="Weather", summary="Forecasts, current conditions and severe weather alerts."), [
        (Toolkit(id="openweather", name="OpenWeather", branch_id="weather", summary="Weather data API."), [
            ("current_weather", "Get the current temperature, wind and conditions for a city.", _obj(city="City")),
            ("forecast", "Get a multi-day weather forecast with rain probability for a location.", _obj(location="P")),
            ("weather_alerts", "List active severe weather warnings such as storms or floods.", _obj(region="R")),
        ]),
    ]),
    (Branch(id="chat", name="Team chat", summary="Send and read messages in team chat channels."), [
        (Toolkit(id="slack", name="Slack", branch_id="chat", summary="Slack workspace messaging."), [
            ("post_message", "Post a message to a Slack channel.", _obj(channel="Channel", text="Text")),
            ("search_messages", "Search Slack message history.", _obj(query="Search terms")),
        ]),
        (Toolkit(id="teams", name="Microsoft Teams", branch_id="chat", summary="Teams channel messaging."), [
            ("send_channel_message", "Send a message to a Microsoft Teams channel.", _obj(team="Team", text="T")),
        ]),
    ]),
    (Branch(id="code", name="Code execution", summary="Run code, shell commands and tests in a sandbox."), [
        (Toolkit(id="sandbox", name="Python sandbox", branch_id="code", summary="Isolated Python runtime."), [
            ("run_python", "Execute a Python snippet and return stdout.", _obj(code="Python source")),
            ("run_tests", "Run the pytest suite in a directory and report failures.", _obj(path="Directory")),
            ("install_package", "Install a pip package into the sandbox.", _obj(name="Package name")),
        ]),
    ]),
]

QUERIES = [
    ("weather", ["openweather.forecast"], "will it rain in paris tomorrow"),
    ("weather", ["openweather.forecast"], "forecast for berlin this weekend"),
    ("weather", ["openweather.current_weather"], "how hot is it in madrid right now"),
    ("weather", ["openweather.current_weather"], "current temperature in oslo"),
    ("weather", ["openweather.weather_alerts"], "any storm warnings for florida"),
    ("weather", ["openweather.weather_alerts"], "are there flood alerts near the coast"),
    ("chat", ["slack.post_message", "teams.send_channel_message"], "tell the team the deploy is done"),
    ("chat", ["slack.post_message"], "post in the slack channel that lunch is here"),
    ("chat", ["slack.search_messages"], "find the slack thread where we discussed pricing"),
    ("chat", ["teams.send_channel_message"], "send a teams message to the design team"),
    ("chat", ["slack.search_messages"], "search messages about the outage"),
    ("chat", ["slack.post_message", "teams.send_channel_message"], "let everyone in the channel know I'm late"),
    ("code", ["sandbox.run_python"], "run this python snippet print(2+2)"),
    ("code", ["sandbox.run_python"], "execute some code to compute primes"),
    ("code", ["sandbox.run_tests"], "do my unit tests pass"),
    ("code", ["sandbox.run_tests"], "run pytest on the repo"),
    ("code", ["sandbox.install_package"], "install numpy please"),
    ("code", ["sandbox.install_package"], "pip install requests in the sandbox"),
]


def _registry() -> Registry:
    reg = Registry(TokenCounter("approx"))
    for branch, toolkits in CORPUS:
        reg.add_branch(branch)
        for tk, tools in toolkits:
            reg.upsert_toolkit(tk, [{"name": n, "description": d, "input_schema": s} for n, d, s in tools])
    return reg


def _tasks() -> list[Task]:
    return [Task(id=f"t{i}", query=q, gold=[g], branches=[b], kind="literal") for i, (b, g, q) in enumerate(QUERIES)]


def _system(**cfg) -> System:
    reg = _registry()
    index = ToolIndex(reg, LSAEmbedder(dim=32)).build()
    analyzer = ContextAnalyzer(reg)
    cats = CATS(reg, index, analyzer, CATSConfig(alpha_branch=0.2, alpha_tool=0.2, k_max=3, **cfg))
    tasks = _tasks()
    cats.fit(tasks)
    cats.calibrate(tasks)
    return System(reg, index, analyzer, cats, tasks, tasks, tasks)


@pytest.fixture(scope="module")
def tiny() -> System:
    return _system()


@pytest.fixture(scope="module")
def client(tiny):
    return TestClient(create_app(tiny, Settings(max_escalations=2)))


def _memory_tracer():
    mem = InMemorySpanExporter()
    return TraceExporter(make_provider(span_exporter=mem)), mem


def _sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((fields["event"], json.loads(fields["data"])))
    return out


# ------------------------------------------------------------------------------- REST gateway
def test_healthz_stats_and_openapi(client, tiny):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ready"] and r.json()["tools"] == 9
    s = client.get("/v1/stats").json()
    assert s["registry"]["tools"] == 9 and s["embedder"].startswith("lsa")
    assert s["calibration"]["n"] == len(tiny.calib) and s["split"]["test"] == len(tiny.test)
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/select", "/v1/escalate", "/v1/stats", "/v1/route/stream", "/healthz"} <= set(paths)


def test_select_returns_route_tools_savings_and_spans(client, tiny):
    q = "will it rain in paris tomorrow"
    body = client.post("/v1/select", json={"query": q, "include_specs": True}).json()
    expected = tiny.cats.select(q)
    assert body["tool_ids"] == expected.tool_ids and body["k"] == len(expected.tool_ids)
    assert "openweather.forecast" in body["tool_ids"]
    assert "weather" in body["route"]["selected"] and body["route"]["branch_scores"]
    tool = body["tools"][0]
    assert tool["spec"]["name"] == tool["wire_name"] and tool["branch_id"] in {"weather", "chat", "code"}
    sv = body["savings"]
    assert sv["static_tokens"] == tiny.registry.total_tokens()
    assert sv["tokens_saved"] == sv["static_tokens"] - sv["tool_tokens"] > 0
    assert sv["usd_saved"] == pytest.approx(sv["usd_static"] - sv["usd_selected"] - sv["usd_router"])
    assert [sp["stage"] for sp in body["spans"]] == ["analyze", "route+retrieve", "assemble"]
    stats = client.get("/v1/stats").json()["served"]
    assert stats["selections"] >= 1 and stats["tokens_saved_total"] > 0 and stats["latency_p95_ms"] is not None


def test_select_validates_and_excludes(client):
    assert client.post("/v1/select", json={"query": ""}).status_code == 422
    assert client.post("/v1/select", json={}).status_code == 422
    body = client.post("/v1/select", json={"query": "will it rain in paris tomorrow",
                                           "exclude": ["openweather.forecast"]}).json()
    assert "openweather.forecast" not in body["tool_ids"]


def test_escalate_session_excludes_loaded_tools_and_enforces_limit(client):
    first = client.post("/v1/select", json={"query": "post in the slack channel", "session_id": "s-1"}).json()
    loaded = set(first["tool_ids"])
    e1 = client.post("/v1/escalate", json={"capability": "run a python snippet", "session_id": "s-1"})
    assert e1.status_code == 200
    e1 = e1.json()
    assert e1["escalations"] == 1 and e1["remaining"] == 1 and e1["request_id"] == first["request_id"]
    assert e1["tool_ids"] and not loaded & set(e1["tool_ids"])
    e2 = client.post("/v1/escalate", json={"capability": "weather forecast", "session_id": "s-1", "k": 2}).json()
    assert len(e2["tool_ids"]) <= 2 and not (loaded | set(e1["tool_ids"])) & set(e2["tool_ids"])
    e3 = client.post("/v1/escalate", json={"capability": "anything else", "session_id": "s-1"})
    assert e3.status_code == 409
    assert client.post("/v1/escalate", json={"capability": "x", "session_id": "nope"}).status_code == 404
    free = client.post("/v1/escalate", json={"capability": "run tests", "exclude": ["sandbox.run_tests"]}).json()
    assert "sandbox.run_tests" not in free["tool_ids"] and free["escalations"] is None
    assert client.get("/v1/stats").json()["served"]["escalations_refused"] >= 1


def test_concurrent_escalations_respect_the_session_limit(tiny):
    from concurrent.futures import ThreadPoolExecutor

    c = TestClient(create_app(tiny, Settings(max_escalations=2)))
    c.post("/v1/select", json={"query": "tell the team the deploy is done", "session_id": "race"})
    caps = ["run python", "weather forecast", "run tests", "install a package", "storm alerts", "search slack"]
    with ThreadPoolExecutor(6) as pool:
        codes = list(pool.map(lambda cap: c.post("/v1/escalate", json={"capability": cap, "session_id": "race"})
                              .status_code, caps))
    assert sorted(codes) == [200, 200, 409, 409, 409, 409]


@pytest.mark.parametrize("query", ["will it rain in paris tomorrow",
                                   "check the weather forecast then post a message in slack"])
def test_route_stream_one_event_per_stage_and_same_result(client, tiny, query):
    r = client.get("/v1/route/stream", params={"query": query, "history": ["hello"]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert [e for e, _ in events] == ["analyze", "route", "retrieve", "assemble", "result"]
    analyze, route = events[0][1], events[1][1]
    assert analyze["sub_requests"] and route["selected"] and route["branch_scores"]
    from cats.types import Turn

    expected = tiny.cats.select(query, [Turn(role="user", content="hello")])
    result = events[-1][1]
    assert result["tool_ids"] == expected.tool_ids
    assert result["route"]["selected"] == expected.route.selected
    assert [sp["stage"] for sp in result["spans"]] == [sp.stage for sp in expected.spans]


def test_stage_events_mirror_select_including_bypass(tiny):
    for _, _, q in QUERIES[::3]:
        sel = next(p for s, p in stage_events(tiny.cats, q) if s == "selection")["selection"]
        ref = tiny.cats.select(q)
        assert sel.tool_ids == ref.tool_ids and sel.scores == pytest.approx(ref.scores)
        assert sel.tool_tokens == ref.tool_tokens and sel.route.selected == ref.route.selected
    bypass = _system(bypass_tokens=10**6)
    stages = list(stage_events(bypass.cats, "anything", exclude={"slack.post_message"}))
    assert [s for s, _ in stages] == ["bypass", "selection"]
    assert stages[-1][1]["selection"].tool_ids == bypass.cats.select("anything", exclude={"slack.post_message"}).tool_ids


def test_api_exports_otel_traces_and_logs_routes(tiny):
    tracer, mem = _memory_tracer()

    class FakeStore:
        def __init__(self):
            self.rows = []

        def log_route(self, selection, *, request_id=None, query=None, history=None):
            self.rows.append((request_id, query, selection.tool_ids))

    store = FakeStore()
    c = TestClient(create_app(tiny, Settings(), tracer=tracer, store=store))
    body = c.post("/v1/select", json={"query": "run pytest on the repo"}).json()
    spans = mem.get_finished_spans()
    root = next(s for s in spans if s.name == "cats.select")
    assert format(root.context.trace_id, "032x") == body["trace_id"]
    assert root.attributes["cats.request_id"] == body["request_id"]
    assert sorted(s.name for s in spans if s.parent is not None) == sorted(
        f"cats.stage.{sp['stage']}" for sp in body["spans"])
    assert store.rows == [(body["request_id"], "run pytest on the repo", body["tool_ids"])]
    events = _sse_events(c.get("/v1/route/stream", params={"query": "install numpy please"}).text)
    assert store.rows[-1] == (events[-1][1]["request_id"], "install numpy please", events[-1][1]["tool_ids"])
    assert c.get("/v1/stats").json()["served"]["selections"] == 2


def test_api_logs_to_registry_store(tiny, tmp_path):
    store_mod = pytest.importorskip("cats.store")
    if not hasattr(store_mod, "RegistryStore"):
        pytest.skip("cats.store has no RegistryStore")
    db = tmp_path / "routes.sqlite"
    c = TestClient(create_app(tiny, Settings(store=str(db))))
    sel = c.post("/v1/select", json={"query": "post in the slack channel", "session_id": "s"}).json()
    c.post("/v1/escalate", json={"capability": "run python", "session_id": "s"})
    assert c.get("/v1/stats").json()["served"]["store_errors"] == 0
    import sqlite3

    rows = sqlite3.connect(db).execute("SELECT request_id, strategy FROM route_decision ORDER BY ts").fetchall()
    assert rows == [(sel["request_id"], "cats"), (sel["request_id"], "cats:escalate")]


def test_lifespan_builds_system_once(monkeypatch, tiny):
    calls = []

    def fake_load(settings):
        calls.append(settings)
        return tiny

    monkeypatch.setattr(api_mod, "load_system", fake_load)
    app = create_app(settings=Settings(seed=4))
    with TestClient(app) as c:
        assert c.get("/healthz").json()["ready"]
        assert c.post("/v1/select", json={"query": "install numpy please"}).status_code == 200
        assert c.get("/v1/stats").json()["service"]["build_s"] is not None
    assert len(calls) == 1 and calls[0].seed == 4
    assert TestClient(create_app(settings=Settings())).get("/healthz").status_code == 503  # not started


def test_settings_from_env_and_flags():
    s = Settings.from_env({"CATS_SEED": "3", "CATS_K_MAX": "7", "CATS_ALPHA_TOOL": "0.1", "CATS_OTEL": "console",
                           "CATS_ENRICH": "data/enrichment"}, port=9000, seed=None)
    assert (s.seed, s.k_max, s.alpha_tool, s.otel, s.port) == (3, 7, 0.1, "console", 9000)
    cfg = s.cats_config()
    assert cfg.k_max == 7 and cfg.alpha_tool == 0.1 and cfg.method == "lac"
    assert Path(api_mod._resolve("data/enrichment")).resolve() == (ROOT / "data/enrichment").resolve()


def test_json_safe_handles_non_finite():
    out = api_mod.json_safe({"a": float("inf"), "b": float("nan"), "c": [1.5, -float("inf")]})
    assert out == {"a": "inf", "b": None, "c": [1.5, "-inf"]}
    json.dumps(out, allow_nan=False)


# ----------------------------------------------------------------------------- OpenTelemetry
def test_export_selection_one_span_per_stage(tiny):
    tracer, mem = _memory_tracer()
    sel = tiny.cats.select("will it rain in paris tomorrow")
    trace_id = tracer.export_selection(sel, query="will it rain in paris tomorrow", start_ns=1_000_000_000,
                                       static_tokens=tiny.registry.total_tokens())
    spans = mem.get_finished_spans()
    root = next(s for s in spans if s.name == "cats.select")
    kids = sorted((s for s in spans if s.parent is not None), key=lambda s: s.start_time)
    assert [k.name for k in kids] == [f"cats.stage.{sp.stage}" for sp in sel.spans]
    assert all(k.parent.span_id == root.context.span_id for k in kids)
    assert format(root.context.trace_id, "032x") == trace_id
    assert root.start_time == 1_000_000_000 and root.end_time >= kids[-1].end_time
    assert all(a.end_time <= b.start_time for a, b in itertools.pairwise(kids))  # stages laid end to end
    ra = root.attributes
    assert ra["cats.k"] == len(sel.tool_ids) and list(ra["cats.tool_ids"]) == sel.tool_ids
    assert list(ra["cats.route.selected"]) == sel.route.selected and ra["cats.route.top_branch"] == "weather"
    assert ra["cats.tokens_saved"] == tiny.registry.total_tokens() - sel.tool_tokens
    route = next(k for k in kids if k.name == "cats.stage.route+retrieve").attributes
    assert route["cats.k"] == len(sel.tool_ids) and "cats.route.selected" in route and route["cats.latency_ms"] >= 0
    assemble = next(k for k in kids if k.name == "cats.stage.assemble").attributes
    assert assemble["cats.tokens_in"] == sel.tool_tokens


def test_export_dispatcher_run(tiny):
    tracer, mem = _memory_tracer()
    task = tiny.test[0]
    run = Dispatcher(tiny.cats, HeuristicLLM()).run(task)
    sel = tiny.cats.select(task.query, task.history)
    esc = tiny.cats.escalate("run python", exclude=set(sel.tool_ids))
    tracer.export_run(run, selection=sel, escalations=[esc], duration_ms=50.0)
    spans = mem.get_finished_spans()
    root = next(s for s in spans if s.name == "cats.dispatch")
    assert root.attributes["cats.task_id"] == task.id and root.attributes["cats.turns"] == run.turns
    assert root.attributes["cats.cost.cache_write"] == run.cost.cache_write
    assert len([e for e in root.events if e.name == "tool_call"]) == len(run.calls) + len(run.invalid_calls)
    children = [s for s in spans if s.parent is not None and s.parent.span_id == root.context.span_id]
    assert sorted(s.name for s in children) == ["cats.escalate", "cats.select"]
    assert (root.end_time - root.start_time) >= 50_000_000


def test_attr_value_coercion():
    assert attr_value(None) is None and attr_value(3) == 3 and attr_value(True) is True
    assert attr_value(["a", "b"]) == ["a", "b"] and attr_value((1, 2)) == [1, 2]
    assert json.loads(attr_value({"x": [1, 2]})) == {"x": [1, 2]}
    assert json.loads(attr_value([1, "a"])) == [1, "a"]


def test_langfuse_processor_keeps_cats_spans(tiny):
    mem = InMemorySpanExporter()
    proc = langfuse_processor("pk-lf-test", "sk-lf-test", "http://localhost:3000", span_exporter=mem)
    provider = make_provider("none")
    provider.add_span_processor(proc)
    TraceExporter(provider).export_selection(tiny.cats.select("do my unit tests pass"))
    provider.force_flush()
    assert {s.name for s in mem.get_finished_spans()} >= {"cats.select", "cats.stage.assemble"}
    provider.shutdown()


def test_console_and_otlp_providers(tiny, capsys):
    tx = TraceExporter(make_provider("console"))
    tx.export_selection(tiny.cats.select("install numpy please"))
    tx.force_flush()
    assert '"cats.select"' in capsys.readouterr().out
    otlp = make_provider("otlp", endpoint="http://127.0.0.1:9/v1/traces")  # constructed only, nothing sent
    otlp.shutdown()
    with pytest.raises(ValueError):
        make_provider("zipkin")


# ------------------------------------------------------------------------------ metrics store
def test_metrics_store_appends_and_summarizes_records(tiny, tmp_path):
    recs = evaluate(CATSStrategy(tiny.cats), tiny.test, tiny.registry)
    store = MetricsStore(tmp_path)
    store.append_records(recs, run_id="bench-s0", experiment="bench", seed=0, config=CATSConfig(), git_rev="abc")
    store.append_records(recs[:5], run_id="bench-s1", experiment="bench", seed=1, tag="extra")
    tbl = store.read("records")
    assert tbl.num_rows == len(recs) + 5 and str(tbl.schema.field("created_at").type).startswith("timestamp")
    assert {"run_id", "seed", "git_rev", "config", "tag", "branch_cover"} <= set(tbl.column_names)
    assert len(store.query("records", where={"run_id": "bench-s1"})) == 5
    assert [r["run_id"] for r in store.runs()] == ["bench-s0", "bench-s1"]
    summ = {r["run_id"]: r for r in store.summarize_records(by=("run_id",))}
    ref = summarize(recs, tiny.registry.total_tokens())
    assert summ["bench-s0"]["full_cover_mean"] == pytest.approx(ref["full_cover"])
    assert summ["bench-s0"]["k_mean"] == pytest.approx(ref["k_mean"]) and summ["bench-s0"]["n"] == len(recs)
    assert store.tables() == ["records"]


def test_metrics_store_agent_runs(tiny, tmp_path):
    runs = [Dispatcher(tiny.cats, HeuristicLLM()).run(t) for t in tiny.test[:3]]
    store = MetricsStore(tmp_path)
    store.append_agent_runs(runs, run_id="heur", mode="cats", model="heuristic")
    store.append_agent_runs([{"task": "x", "success": True, "usd": 0.01, "turns": 2}], run_id="sub",
                            mode="static", model="haiku")
    rows = store.query("agent_runs")
    assert len(rows) == 4 and {r["mode"] for r in rows} == {"cats", "static"}
    assert {r["task"] for r in rows if r["run_id"] == "heur"} == {t.id for t in tiny.test[:3]}


def _write(p: Path, obj) -> None:
    p.write_text(json.dumps(obj))


def test_convert_results_every_shape(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    summ = {"strategy": "CATS", "n": 10, "full_cover": 0.9, "full_cover_ci": [0.8, 0.95], "k_mean": 4.0,
            "by_kind": {"literal": {"n": 5, "full_cover": 1.0, "slot_recall": 1.0, "k_mean": 3.0}}}
    _write(res / "bench_x.json", {"registry": {"tools": 9}, "embedder": "lsa", "calibration": {"lambda": 0.4},
                                  "summaries": [summ, {**summ, "strategy": "B1_bm25@5"}],
                                  "paired_full_cover_vs_cats": {"B1_bm25@5": {"mean": 0.1, "ci": [0.0, 0.2], "n": 10}}})
    _write(res / "ablate_x.json", [summ])
    _write(res / "validity_x.json", {"config": {"alpha_tool": 0.05}, "rows": [{"seed": 0, "slot_miss": 0.04}],
                                     "branch_cover_mean": 0.98, "branch_target": 0.97, "slot_miss_mean": 0.04,
                                     "slot_miss_target": 0.05})
    _write(res / "scale_x.json", [{"size": 100, "branches": 3, "tasks_test": 5, "static_tokens": 900, "build_s": 1.0,
                                   "CATS": {"full_cover": 0.8, "k_mean": 4.0}, "cats_calibration": {"lambda": 0.2}}])
    _write(res / "cache_x.json", {"session_len": 5, "n_sessions": 2, "policies": {"static": {"usd_per_session": 0.1}}})
    _write(res / "e2e_x.json", [{"run": "r", "mode": "cats", "success": 1.0, "n": 3}])
    _write(res / "odd.json", {"something": 1})
    (res / "broken.json").write_text("{not json")
    _write(res / "misread.json", {"summaries": "not a list of summaries"})  # recorded, does not abort
    counts = convert_results(res, tmp_path / "pq")
    assert counts == {"files": 9, "summaries": 3, "by_kind": 3, "paired": 1, "validity": 1, "scale": 1,
                      "cache": 1, "agent": 1}
    store = MetricsStore(tmp_path / "pq")
    kinds = {r["file"]: r["kind"] for r in store.query("files", columns=["file", "kind"])}
    assert kinds["odd.json"] == "unknown" and kinds["broken.json"] == "invalid" and kinds["e2e_x.json"] == "agent"
    assert "AttributeError" in store.query("files", where={"file": "misread.json"})[0]["error"]
    row = store.query("summaries", where={"file": "bench_x.json", "strategy": "CATS"})[0]
    assert (row["full_cover_ci_lo"], row["full_cover_ci_hi"], row["cal_lambda"]) == (0.8, 0.95, 0.4)
    assert store.query("scale")[0]["strategy"] == "CATS" and store.query("scale")[0]["cal_lambda"] == 0.2


def test_convert_real_results_read_only(tmp_path):
    src = ROOT / "results"
    files = sorted(src.glob("*.json"))
    if not files:
        pytest.skip("no results/*.json in this checkout")
    before = {p: p.stat().st_mtime_ns for p in files}
    counts = convert_results(src, tmp_path)
    assert counts["files"] >= len(files) and counts.get("summaries", 0) > 0
    assert all(p.stat().st_mtime_ns == t for p, t in before.items() if p.exists())
    assert detect_kind(json.loads((src / files[0].name).read_text())) != "" and not (tmp_path / "records").exists()


# ----------------------------------------------------------------------------------- manifests
def test_run_manifest_records_everything(tiny, tmp_path, monkeypatch):
    monkeypatch.setenv("CATS_API_KEY", "secret-value")
    monkeypatch.setenv("CATS_SEED_NOTE", "visible")
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.json").write_text('{"x": 1}')
    out = tmp_path / "result.json"
    with run_manifest("unit run/1", out_dir=tmp_path / "m", config=CATSConfig(k_max=7), seed=3, data=[data],
                      models={"agent": "heuristic"}, experiment="bench") as m:
        out.write_text("{}")
        m.add_output(out)
        m.set_system(tiny)
        m.note(full_cover=0.9)
        assert load_manifest(m.path).status == "running"
    man = load_manifest(tmp_path / "m" / "unit_run_1.json")
    assert man.status == "ok" and man.seed == 3 and man.config["k_max"] == 7 and man.duration_s >= 0
    sha = hashlib.sha256(b'{"x": 1}').hexdigest()
    assert list(man.data.values()) == [sha] and man.data_digest
    assert list(man.outputs.values()) == [hashlib.sha256(b"{}").hexdigest()]
    assert man.embedder.startswith("lsa") and man.tokenizer == "approx-regex"
    assert man.system["split"]["test"] == len(tiny.test) and man.notes == {"full_cover": 0.9}
    assert man.packages["numpy"] and man.packages["pyarrow"] and "nonexistent" not in man.packages
    assert man.models == {"agent": "heuristic"} and man.python["version"]
    assert man.env["CATS_API_KEY"] == "<redacted>" and man.env["CATS_SEED_NOTE"] == "visible"
    assert "rev" in man.git and man.created_at <= man.finished_at
    (data / "a.json").write_text('{"x": 2}')
    (data / "b.json").write_text("{}")
    rep = verify_data(man, root=ROOT)
    assert len(rep["changed"]) == 1 and rep["changed"][0].endswith("a.json") and rep["added"][0].endswith("b.json")


def test_run_manifest_error_status(tmp_path):
    with pytest.raises(RuntimeError, match="boom"), run_manifest("failing", out_dir=tmp_path, data=[]):
        raise RuntimeError("boom")
    man = load_manifest("failing", tmp_path)
    assert man.status == "error" and "boom" in man.error and man.finished_at


def test_git_info_is_read_only_and_has_file_fallback(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert _git_from_files(tmp_path) == {"rev": "a" * 40, "branch": "main"}
    info = git_info(tmp_path)  # not a real repository: git rev-parse fails, the files are read
    assert info["rev"] == "a" * 40 and info["source"] == ".git files"
    assert sorted(p.name for p in git.rglob("*")) == ["HEAD", "heads", "main", "refs"]  # nothing written


# --------------------------------------------------------------------------------- Hydra runner
def _tiny_data(root: Path) -> tuple[Path, Path]:
    _registry().to_corpus(root / "corpus")
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"id": f"t{i}", "query": q, "gold": [gold], "branches": [b], "kind": "literal"})
             for i, (b, gold, q) in enumerate(QUERIES * 2)]
    (root / "tasks" / "tiny.jsonl").write_text("\n".join(lines) + "\n")
    return root / "corpus", root / "tasks"


def _tiny_overrides(tmp_path: Path) -> list[str]:
    corpus, tasks = _tiny_data(tmp_path / "data")
    return ["data.enrich=false", "data.embedder=lsa", f"data.corpus_dir={corpus}", f"data.tasks_dir={tasks}",
            f"results_dir={tmp_path / 'results'}"]


def test_runner_config_groups_compose():
    from cats.runner import cats_config, compose, run_name

    cfg = compose(["strategy=all", "calibration=aps", "data=raw", "experiment=validity", "seed=2"])
    assert cfg.strategy.baselines == "all" and cfg.strategy.cats.k_max == 10 and cfg.experiment.n_seeds == 5
    c = cats_config(cfg)
    assert (c.method, c.alpha_branch, c.alpha_tool, c.calibration) == ("aps", 0.03, 0.05, "joint")
    assert run_name(cfg) == "validity-all-raw-aps-s2"
    cfg = compose(["strategy=cats_gbt", "calibration=independent", "data.jev=true", "calibration.alpha_tool=0.1"])
    c = cats_config(cfg)
    assert c.reranker == "gbt" and c.calibration == "independent" and c.alpha_tool == 0.1
    assert c.jev and c.jev.endswith("runs/jev") and run_name(cfg) == "bench-cats_gbt-enrich-jev-lac-indep-at0.1-s0"


def test_runner_bench_writes_results_manifest_and_records(tmp_path):
    from cats.runner import compose, run

    cfg = compose(["experiment=bench", "strategy=flat_reranked", *_tiny_overrides(tmp_path)])
    out = run(cfg)
    res = json.loads(Path(out["result_path"]).read_text())
    names = [s["strategy"] for s in res["summaries"]]
    assert names == ["B3_hybrid_rerank@5", "B3_hybrid_rerank@10", "CATS"] and set(res["paired_full_cover_vs_cats"])
    man = load_manifest(out["manifest_path"])
    assert man.status == "ok" and man.experiment == "bench" and man.system["registry"]["tools"] == 9
    assert any(k.endswith(".json") for k in man.outputs) and any(k.endswith(".parquet") for k in man.outputs)
    assert man.config["cats"]["k_max"] == 10 and man.config["hydra"]["strategy"]["name"] == "flat_reranked"
    rows = MetricsStore(tmp_path / "results" / "parquet").query("records", where={"run_id": out["name"]})
    assert len(rows) == sum(s["n"] for s in res["summaries"]) and {r["strategy"] for r in rows} == set(names)


def test_runner_dispatches_to_existing_experiments(tmp_path, monkeypatch):
    from cats.bench import experiments
    from cats.runner import compose, run

    seen = {}
    monkeypatch.setattr(experiments, "coverage_validity",
                        lambda cfg, seeds, enrichment: seen.setdefault("validity", (cfg, list(seeds), enrichment)))
    monkeypatch.setattr(experiments, "scaling",
                        lambda sizes, cfg, seed, enrichment, k: seen.setdefault("scale", (sizes, seed, k)) and [])
    monkeypatch.setattr(experiments, "run_variants",
                        lambda variants, seed, enrichment, registry: seen.setdefault("ablate", sorted(variants)) and [])
    common = ["data.enrich=false", f"results_dir={tmp_path}", "seed=3"]
    run(compose(["experiment=validity", "experiment.n_seeds=2", "calibration=independent", *common]))
    cfg, seeds, enrich = seen["validity"]
    assert seeds == [3, 4] and cfg.calibration == "independent" and enrich is None
    run(compose(["experiment=scale", "experiment.sizes=[100,250]", *common]))
    assert seen["scale"] == ([100, 250], 3, 5)
    run(compose(["experiment=ablate", *common]))
    assert "CATS aps" in seen["ablate"]
    assert {p.stem for p in (tmp_path / "manifests").glob("*.json")} == {
        "validity-cats-raw-lac-indep-s3", "scale-cats-raw-lac-s3", "ablate-cats-raw-lac-s3"}


def test_runner_multirun_seed_sweep_cli(tmp_path):
    env = {**os.environ, "CATS_HYDRA_DIR": str(tmp_path / "hydra"), "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, "-m", "cats.runner", "-m", "seed=0,1", "experiment=bench", "strategy=cats",
           "manifest=true", *_tiny_overrides(tmp_path)]
    p = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert p.returncode == 0, p.stderr[-2000:]
    names = {json.loads(line)["name"] for line in p.stdout.splitlines() if line.startswith("{")}
    assert names == {"bench-cats-raw-lac-s0", "bench-cats-raw-lac-s1"}
    for n in names:
        man = load_manifest(n, tmp_path / "results" / "manifests")
        assert man.status == "ok" and man.notes["hydra_mode"].endswith("MULTIRUN")
        assert (tmp_path / "results" / f"{n}.json").exists()
    assert len(list((tmp_path / "hydra" / "multirun").glob("*/*/.hydra/config.yaml"))) == 2
    single = [sys.executable, "-m", "cats.runner", "seed=5", "name=single", *_tiny_overrides(tmp_path)]
    p = subprocess.run(single, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert p.returncode == 0 and "Error" not in p.stderr, p.stderr[-2000:]
    man = load_manifest("single", tmp_path / "results" / "manifests")
    assert man.status == "ok" and man.seed == 5 and man.notes["hydra_job_num"] is None


# ------------------------------------------------------------------------------ deployment files
def test_deployment_files_are_consistent():
    yaml = pytest.importorskip("yaml")
    compose_doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose_doc["services"]
    assert services["postgres"]["image"].startswith("postgres:16")
    assert services["qdrant"]["image"].startswith("qdrant/qdrant")
    assert {"langfuse-web", "langfuse-worker", "clickhouse", "redis", "minio", "cats-api"} <= set(services)
    api = services["cats-api"]
    assert api["build"]["context"] == "." and api["environment"]["CATS_OTEL"] == "langfuse"
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.12-slim" in dockerfile and "uv sync" in dockerfile and "cats.api" in dockerfile
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    triggers = ci.get("on", ci.get(True))  # YAML 1.1 reads the key "on" as True
    assert {"push", "pull_request"} <= set(triggers)
    steps = " ".join(str(s.get("run", "")) for s in ci["jobs"]["test"]["steps"])
    assert all(x in steps for x in ("uv sync", "ruff check", "pyright", "pytest"))
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = py["project"]["optional-dependencies"]
    assert {"api", "tracing", "metrics", "runner", "platform"} <= set(extras) and "pyright" in py["tool"]
