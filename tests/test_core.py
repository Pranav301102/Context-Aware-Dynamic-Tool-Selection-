import itertools

import numpy as np

from cats.assembler import REQUEST_TOOLS, ContextAssembler
from cats.calibration import aps_set, conformal_quantile, crc_threshold, lac_set, slot_miss
from cats.dispatcher import Dispatcher
from cats.index import BM25
from cats.ledger import PrefixCache
from cats.llm import HeuristicLLM
from cats.types import Task, Toolkit


# ------------------------------------------------------------------ registry (FR1)
def test_registry_hierarchy_and_token_costs(registry):
    assert set(registry.branches) == {"weather", "chat", "code"}
    assert registry.tools["slack.post_message"].branch_id == "chat"
    assert all(t.token_cost > 0 for t in registry.tools.values())
    assert registry.total_tokens() == sum(t.token_cost for t in registry.tools.values())
    assert registry.resolve_wire_name("slack__post_message") == "slack.post_message"


def test_resync_emits_only_real_changes(registry):
    events = []
    registry.subscribe(events.append)
    tk = registry.toolkits["teams"]
    same = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in (registry.tools[i] for i in registry.toolkit_tools("teams"))]
    registry.upsert_toolkit(tk, same)
    assert events == []
    registry.upsert_toolkit(tk, same + [{"name": "list_channels", "description": "List Teams channels."}])
    assert [e.kind for e in events] == ["added"]
    registry.upsert_toolkit(tk, same[:0] + [{"name": "list_channels", "description": "List Teams channels."}])
    assert [e.kind for e in events] == ["added", "removed"]


# --------------------------------------------------------------------- index (FR2, FR11)
def test_bm25_add_remove():
    bm = BM25()
    bm.add(0, "send a slack message")
    bm.add(1, "weather forecast for a city")
    s = bm.scores("forecast", 2)
    assert s[1] > 0 and s[0] == 0
    bm.remove(1)
    assert bm.scores("forecast", 2).sum() == 0


def test_changed_tool_is_searchable_after_sync(registry, index):
    registry.upsert_toolkit(Toolkit(id="pagerduty", name="PagerDuty", branch_id="chat", summary="Paging."),
                            [{"name": "trigger_incident", "description": "Page the on-call engineer about an incident."}])
    assert "pagerduty.trigger_incident" in index.tools.slot
    q = index.embed_query("page the on-call engineer")
    hits = index.tools.search("page the on-call engineer incident", q, k=3)
    assert hits[0].id == "pagerduty.trigger_incident"
    assert index.sync_log[-1]["ms"] < 60_000  # NFR5: searchable within 60 s


# ----------------------------------------------------------------- calibration (FR3, FR4)
def test_conformal_quantile_and_sets():
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.5) == 0.3
    assert conformal_quantile([0.1, 0.2], 0.01) == float("inf")
    p = {"a": 0.6, "b": 0.3, "c": 0.1}
    assert aps_set(p, 0.85) == ["a", "b"]
    assert lac_set(p, 0.75) == ["a", "b"]


def test_aps_coverage_holds_on_synthetic_data():
    rng = np.random.default_rng(0)
    from cats.calibration import aps_score

    def draw(n):
        out = []
        for _ in range(n):
            logits = rng.normal(size=6)
            p = np.exp(logits) / np.exp(logits).sum()
            gold = rng.choice(6, p=p)
            out.append(({str(i): float(x) for i, x in enumerate(p)}, {str(gold)}))
        return out

    cal, test = draw(500), draw(2000)
    t = conformal_quantile([aps_score(p, g) for p, g in cal], 0.1)
    cover = np.mean([bool(g & set(aps_set(p, t))) for p, g in test])
    assert cover >= 0.87


def test_crc_picks_strictest_safe_threshold():
    lambdas = np.array([0.9, 0.5, 0.1])
    losses = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    assert crc_threshold(losses, lambdas, alpha=0.5) == 0.5
    assert crc_threshold(losses, lambdas, alpha=0.01) == 0.1  # infeasible -> most lenient


def test_slot_miss():
    assert slot_miss({"a"}, [["a", "b"], ["c"]]) == 0.5


# ------------------------------------------------------------------- assembler (FR5, FR12)
def test_identical_selection_gives_identical_prompt(registry):
    asm = ContextAssembler(registry)
    a = asm.assemble(["slack.post_message", "openweather.forecast"])
    b = asm.assemble(["openweather.forecast", "slack.post_message"])
    assert a.prefix_hash == b.prefix_hash and a.tools == b.tools
    assert a.tools[0] == REQUEST_TOOLS


def test_budget_drops_lowest_scored(registry):
    ids = ["slack.post_message", "openweather.forecast", "sandbox.run_python"]
    budget = registry.total_tokens(ids[:2])
    asm = ContextAssembler(registry, budget=budget)
    out = asm.assemble(ids, scores=[0.9, 0.8, 0.1])
    assert out.dropped == ["sandbox.run_python"] and out.tool_tokens <= budget


# -------------------------------------------------------------------- pipeline (FR3-FR8)
def test_select_routes_and_never_returns_zero_tools(cats, tasks):
    sel = cats.select("will it rain in rome tomorrow")
    assert sel.tool_ids and "openweather.forecast" in sel.tool_ids
    assert "weather" in sel.route.selected
    assert {sp.stage for sp in sel.spans} == {"analyze", "route+retrieve", "assemble"}
    covered = np.mean([all(set(cats.select(t.query).tool_ids).intersection(s) for s in t.gold) for t in tasks])
    assert covered >= 0.8


def test_low_confidence_falls_back_to_all_branches(cats):
    cats.router.t_low = 1.01
    sel = cats.select("something vague")
    assert sel.route.fallback and set(sel.route.selected) == set(cats.registry.branches)


def test_escalation_excludes_loaded_tools(cats):
    sel = cats.escalate("send a message to a chat channel", exclude={"slack.post_message"})
    assert "slack.post_message" not in sel.tool_ids
    assert "teams.send_channel_message" in sel.tool_ids


def test_bypass_loads_everything_for_small_catalogs(cats):
    cats.cfg.bypass_tokens = 10**9
    assert set(cats.select("anything").tool_ids) == set(cats.registry.tools)


# ------------------------------------------------------------------ dispatcher (FR6, FR7)
def test_request_tools_recovers_a_routing_miss(cats):
    """Fault injection: the first selection cannot contain the needed tool."""
    task = Task(id="f1", query="install the numpy package", gold=[["sandbox.install_package"]], branches=["code"])
    original = cats.select
    cats.select = lambda q, h=None, **kw: original("weather forecast", h)
    try:
        res = Dispatcher(cats, HeuristicLLM()).run(task)
    finally:
        cats.select = original
    assert res.escalations >= 1 and res.success
    assert res.escalations <= 2


def test_escalations_are_capped(cats):
    class AlwaysAsk:
        name = "always-ask"

        def complete(self, system, messages, tools):
            from cats.llm import LLMResponse, ToolCall
            return LLMResponse(tool_calls=[ToolCall("request_tools", {"capability": "teleport"})])

    res = Dispatcher(cats, AlwaysAsk(), max_turns=6).run(
        Task(id="x", query="teleport me", gold=[["slack.post_message"]], branches=["chat"]))
    assert res.escalations == 2 and not res.success


# ---------------------------------------------------------------------- ledger (NFR6)
def test_prefix_cache_reads_repeated_prefix():
    cache = PrefixCache(min_cacheable=10)
    first = cache.charge(["a", "b"], [600, 600], 50)
    second = cache.charge(["a", "b", "c"], [600, 600, 300], 50)
    assert first.cache_read == 0 and first.cache_write == 1200
    assert second.cache_read == 1200 and second.cache_write == 300


# ------------------------------------------------------- regressions from the code review
def test_select_survives_removed_toolkit(cats):
    cats.registry.remove_toolkit("slack")
    sel = cats.select("post the release notes in slack #general")
    assert sel.tool_ids and not any(t.startswith("slack.") for t in sel.tool_ids)


def test_cut_is_monotone_for_unsorted_rankings(cats):
    r = cats.retriever
    ranked, scores = ["A", "B", "C"], np.array([0.30, 0.90, 0.10])  # interleaved, not score-sorted
    sizes = [len(r.cut(ranked, scores, lam, 3)[0]) for lam in (1.0, 0.9, 0.3, 0.0)]
    kept = [set(r.cut(ranked, scores, lam, 3)[0]) for lam in (1.0, 0.9, 0.3, 0.0)]
    assert sizes == sorted(sizes) and all(a <= b for a, b in itertools.pairwise(kept))


def test_crc_ignores_non_monotone_dips():
    lambdas = np.array([0.9, 0.6, 0.3, 0.0])
    # Risk dips at 0.6, rises at 0.3: 0.6 is not safe to return.
    losses = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
    assert crc_threshold(losses, lambdas, alpha=0.5) == 0.0


def test_requested_tools_survive_the_budget(cats):
    """Over budget, the oldest low-scored tools go, not the ones the model just asked for."""
    from cats.dispatcher import Dispatcher

    budget = max(t.token_cost for t in cats.registry.tools.values()) * 2
    cats.assembler.budget = budget
    task = Task(id="b1", query="install the numpy package", gold=[["sandbox.install_package"]], branches=["code"])
    original = cats.select
    cats.select = lambda q, h=None, **kw: original("weather forecast and storm alerts", h)
    try:
        res = Dispatcher(cats, HeuristicLLM()).run(task)
    finally:
        cats.select = original
    assert res.success and not res.invalid_calls
