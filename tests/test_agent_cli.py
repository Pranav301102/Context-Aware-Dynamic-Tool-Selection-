"""Scoring in the subagent-as-model harness (cats.agent_cli)."""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import pytest
from tiny_cats import build_tiny_cats

from cats import agent_cli
from cats.jev import JEV_USD_PER_MTOK
from cats.ledger import Prices
from cats.types import Task


def _write_run(root, name: str, cats, task: Task, jev: str, router_tokens: int) -> None:
    d = root / name
    (d / "sessions").mkdir(parents=True)
    sysm = SimpleNamespace(registry=cats.registry, cats=cats, test=[task])
    with open(d / "system.pkl", "wb") as f:
        pickle.dump(sysm, f)
    (d / "run.json").write_text(json.dumps({"run": name, "mode": "cats", "jev": jev, "llm_router": "",
                                            "tasks": [task.id]}))
    sel = cats.select(task.query)
    loaded = list(sel.tool_ids)
    session = {"task_id": task.id, "mode": "cats", "loaded": loaded, "order": loaded,
               "scores": dict(zip(loaded, sel.scores)), "router_tokens": router_tokens, "escalations": 0,
               "calls": [{"tool": task.gold[0][0], "args": {}}], "invalid": [], "done": True,
               "events": [{"type": "show", "t": 0.0}, {"type": "call", "tool": task.gold[0][0], "t": 1.0}]}
    (d / "sessions" / f"{task.id}.json").write_text(json.dumps(session))


def test_typed_layer_tokens_billed_at_its_own_price(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_cli, "RUNS", tmp_path)
    cats = build_tiny_cats()
    task = Task(id="t0", query="what is the weather forecast for Paris this week",
                gold=[["openweather.forecast"]], branches=["weather"], kind="literal")
    _write_run(tmp_path, "plain", cats, task, jev="", router_tokens=1450)
    _write_run(tmp_path, "jev", cats, task, jev="runs/jev", router_tokens=1450)
    plain, jev = agent_cli.score_run("plain"), agent_cli.score_run("jev")
    assert plain["summary"]["success"] == jev["summary"]["success"] == 1.0
    diff = plain["rows"][0]["usd"] - jev["rows"][0]["usd"]
    # Without a typed layer the overhead is an LLM router at the main model's input price.
    assert diff == pytest.approx(1450 * (Prices().input - JEV_USD_PER_MTOK) / 1e6)
