"""Persistent registry store (workbook Section 6.2 data model): snapshot, write-through, logs."""
import os
import sqlite3

import numpy as np
import pytest

from cats.index import LSAEmbedder, ToolIndex
from cats.registry import Registry
from cats.store import CachedEmbedder, RegistryStore
from cats.types import Toolkit


def _same_catalog(a: Registry, b: Registry) -> None:
    assert list(a.branches) == list(b.branches)
    assert list(a.toolkits) == list(b.toolkits)
    assert list(a.tools) == list(b.tools)
    for i, t in a.tools.items():
        u = b.tools[i]
        assert (u.version, u.token_cost, u.input_schema, u.description) == (
            t.version, t.token_cost, t.input_schema, t.description)
    assert a.total_tokens() == b.total_tokens()


def test_snapshot_round_trip_keeps_versions_tokens_and_vectors(registry, index, tmp_path):
    path = tmp_path / "reg.sqlite"
    with RegistryStore(path) as st:
        counts = st.save_registry(registry, index)
        assert counts["tool"] == len(registry.tools) and counts["branch"] == 3
        docs = st.bm25_documents()
        assert docs["slack.post_message"] == index.tool_text("slack.post_message")
        vecs = st.embeddings("tool")
        s = index.tools.slot["openweather.forecast"]
        assert np.allclose(vecs["openweather.forecast"], index.tools.vecs[s])
        assert set(st.embeddings("branch")) == set(registry.branches)
    with RegistryStore(path) as st:  # reopened from disk
        loaded = st.load_registry(registry.counter)
        _same_catalog(registry, loaded)
        assert not any(st.verify(loaded).values())
        assert loaded.version >= registry.version
        assert st.info()["meta"]["embedder"] == index.embedder.name


def test_store_follows_registry_events(registry, index):
    st = RegistryStore().attach(registry, index)
    pd = Toolkit(id="pagerduty", name="PagerDuty", branch_id="chat", summary="Paging.")
    registry.upsert_toolkit(pd, [{"name": "trigger_incident", "description": "Page the on-call engineer."}])
    registry.upsert_toolkit(pd, [{"name": "trigger_incident", "description": "Page whoever is on call now."}])
    registry.remove_toolkit("teams")
    assert [e.kind for e in st.events()] == ["added", "updated", "removed"]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        st.conn.execute("DELETE FROM registry_event")
    loaded = st.load_registry(registry.counter)
    _same_catalog(registry, loaded)
    assert "teams.send_channel_message" not in loaded.tools
    assert loaded.tools["pagerduty.trigger_incident"].description == "Page whoever is on call now."
    # the changed tool's stored vector is the index's new one
    s = index.tools.slot["pagerduty.trigger_incident"]
    assert np.allclose(st.embeddings("tool")["pagerduty.trigger_incident"], index.tools.vecs[s])


def test_save_without_index_drops_only_stale_vectors(registry, index):
    st = RegistryStore()
    st.save_registry(registry, index)
    tk = registry.toolkits["sandbox"]
    tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
             for t in (registry.tools[i] for i in registry.toolkit_tools("sandbox"))]
    tools[0]["description"] = "Run a Python snippet in an isolated runtime."
    registry.upsert_toolkit(tk, tools)
    st.save_registry(registry)  # no index: nothing to re-embed with
    vecs = st.embeddings("tool")
    assert "sandbox.run_python" not in vecs  # changed: its old vector would be wrong
    assert "sandbox.run_tests" in vecs  # unchanged: kept


def test_route_log_is_append_only_and_feeds_usage_stats(cats):
    st = RegistryStore()
    st.attach(cats.registry, cats.index)
    q = "will it rain in paris tomorrow"
    sel = cats.select(q)
    rid = st.log_route(sel, query=q)
    esc = cats.escalate("send a message to the team channel", exclude=set(sel.tool_ids))
    st.log_route(esc, request_id=rid)
    st.record_outcome(rid, success=True, called=[sel.tool_ids[0]])

    recs = st.routes(rid)
    assert [r.strategy for r in recs] == ["cats", "cats:escalate"]
    assert recs[0].route.selected == sel.route.selected
    assert recs[0].route.branch_scores == pytest.approx(sel.route.branch_scores)
    assert recs[0].tool_ids == sel.tool_ids and recs[0].query == q and recs[0].request_key
    assert [s.stage for s in st.spans(rid)] == [s.stage for s in sel.spans + esc.spans]
    usage = st.usage()
    assert all(usage[t].selections == 1 for t in sel.tool_ids)
    assert usage[sel.tool_ids[0]].calls == 1 and usage[sel.tool_ids[0]].success_rate == 1.0
    assert st.outcomes(rid)[0]["called"] == [sel.tool_ids[0]]
    for sql in ("UPDATE route_decision SET strategy = 'x'", "DELETE FROM trace_span",
                "UPDATE request_outcome SET success = 0"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            st.conn.execute(sql)


def test_cached_embedder_reuses_stored_vectors(registry):
    class Counting:
        name, dim = "fake:hash", 8

        def __init__(self):
            self.calls = 0

        def fit(self, corpus):
            pass

        def embed(self, texts):
            self.calls += len(texts)
            return np.stack([np.random.default_rng(abs(hash(t)) % 2**32).normal(size=8) for t in texts])

    st = RegistryStore()
    base = Counting()
    first = ToolIndex(registry, CachedEmbedder(base, st)).build()
    n = base.calls
    second = ToolIndex(registry, CachedEmbedder(base, st)).build()
    assert base.calls == n  # every vector came from the store
    assert np.allclose(first.tools.vecs[: len(registry.tools)], second.tools.vecs[: len(registry.tools)])
    assert not CachedEmbedder(LSAEmbedder(), st).stateless  # fitted embedders are never cached


@pytest.mark.skipif(not os.environ.get("CATS_TEST_PG_DSN"), reason="set CATS_TEST_PG_DSN to test PostgreSQL")
def test_postgres_round_trip(registry, index, cats):
    import psycopg

    st = RegistryStore(os.environ["CATS_TEST_PG_DSN"])
    for t in ("tool", "toolkit", "branch"):
        st.conn.execute(f"DELETE FROM {t}")
    st.attach(registry, index)
    registry.remove_toolkit("teams")
    _same_catalog(registry, st.load_registry(registry.counter))
    rid = st.log_route(cats.select("run pytest on the repo"), query="run pytest on the repo")
    assert st.routes(rid)[0].strategy == "cats"
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        st.conn.execute("DELETE FROM route_decision")
