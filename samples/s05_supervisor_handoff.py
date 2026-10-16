"""S5 - A supervisor with two domain sub-agents and one hand-off (LangGraph, or plain Python).

Shows
  The orchestration pattern of the workbook's sub-agent design on a two-step request ("check the
  forecast for Paris tomorrow, then post it in the Slack channel"):

    supervisor     splits the request into steps (cats.analyzer.decompose) and routes each step to a
                   branch with the calibrated CATS router; starts the first step's sub-agent
    weather_agent  sees only the tools CATS selects inside its branch; calls one; then HANDS OFF to the
                   chat agent with its result, because the next step belongs to the chat branch
    chat_agent     same, in the chat branch; uses the handed-off result in its message; ends the run

  Each sub-agent's model is cats.llm.HeuristicLLM (no key needed; ``--llm anthropic:<model>`` etc. to
  use a real one). Tool results come from the dispatcher's deterministic simulator.

Needs
  LangGraph (``pip install langgraph``; tested with 1.2); falls back to a plain-Python loop with the
  same transitions when it is missing or with ``--plain``. Uses the 9-tool sample catalog (_mini.py).

  python samples/s05_supervisor_handoff.py ["request"] [--plain] [--llm heuristic]
"""
from __future__ import annotations

import argparse
import json
import operator
from typing import Annotated, Any, TypedDict

from _mini import mini_cats

from cats.analyzer import decompose
from cats.dispatcher import simulate_tool
from cats.llm import HeuristicLLM, make_llm

REG, CATS_, _ = mini_cats()


class State(TypedDict):
    request: str
    plan: list[dict]  # [{"step", "branch"}]
    i: int  # index of the step being worked on
    log: Annotated[list[str], operator.add]
    results: Annotated[list[dict], operator.add]
    handoffs: int


def supervisor(state: State) -> dict:
    plan = []
    for step in decompose(state["request"]):
        r = CATS_.router.route(CATS_.evidence(step))
        plan.append({"step": step, "branch": max(r.branch_scores, key=r.branch_scores.get)})
    return {"plan": plan, "i": 0, "handoffs": 0,
            "log": [f"supervisor: plan = {json.dumps(plan)}"]}


def run_subagent(branch: str, state: State, llm) -> dict[str, Any]:
    """One sub-agent step: CATS-selected tools of this branch only, one model turn, one tool call."""
    step = state["plan"][state["i"]]["step"]
    context = "; ".join(r["result"] for r in state["results"])
    sel = CATS_.select(step)
    tools = [REG.tools[t] for t in sel.tool_ids if REG.tools[t].branch_id == branch]
    specs = [t.spec() for t in tools]
    messages = ([{"role": "user", "content": f"Context from the previous agent: {context}"},
                 {"role": "assistant", "content": "Noted."}] if context else []) + [{"role": "user", "content": step}]
    resp = llm.complete(f"You are the {branch} sub-agent.", messages, specs)
    call = resp.tool_calls[0] if resp.tool_calls else None
    by_wire = {t.wire_name: t.id for t in tools}
    if call is None or call.name not in by_wire:
        return {"log": [f"{branch}_agent: no tool called ({resp.text[:60]!r})"], "results": []}
    args = dict(call.arguments)
    if branch == "chat" and context:
        args.setdefault("text", f"FYI: {context}")
    out = json.loads(simulate_tool(by_wire[call.name], args))
    result = f"{by_wire[call.name]} -> {out['result']}"
    return {"log": [f"{branch}_agent: tools offered {[t.id for t in tools]}; called {by_wire[call.name]}({args})"],
            "results": [{"agent": branch, "result": result}]}


def next_hop(state: State, branch: str) -> str | None:
    nxt = state["i"] + 1
    return state["plan"][nxt]["branch"] if nxt < len(state["plan"]) else None


def build_langgraph(llm):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    def make_agent(branch: str):
        def agent(state: State):
            upd = run_subagent(branch, state, llm)
            target = next_hop(state, branch)
            if target is None:
                return Command(goto=END, update=upd)
            upd["log"] = upd["log"] + [f"{branch}_agent: hand-off -> {target}_agent"]
            return Command(goto=f"{target}_agent", update={**upd, "i": state["i"] + 1,
                                                            "handoffs": state["handoffs"] + 1})
        return agent

    def route_first(state: State) -> str:
        return f"{state['plan'][0]['branch']}_agent"

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    for b in sorted(REG.branches):
        g.add_node(f"{b}_agent", make_agent(b))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_first, [f"{b}_agent" for b in sorted(REG.branches)])
    return g.compile()


def run_plain(request: str, llm) -> State:
    state: State = {"request": request, "plan": [], "i": 0, "log": [], "results": [], "handoffs": 0}
    upd = supervisor(state)
    state.update({**upd, "log": state["log"] + upd["log"]})
    branch = state["plan"][0]["branch"]
    while branch is not None:
        upd = run_subagent(branch, state, llm)
        state["log"] += upd["log"]
        state["results"] += upd["results"]
        target = next_hop(state, branch)
        if target is not None:
            state["log"].append(f"{branch}_agent: hand-off -> {target}_agent")
            state["i"] += 1
            state["handoffs"] += 1
        branch = target
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("request", nargs="?", default="check the forecast for paris tomorrow, then post it in the slack channel")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--llm", default="heuristic")
    a = ap.parse_args()
    # the stand-in model needs less word overlap for short sub-requests than for whole tasks
    llm = HeuristicLLM(min_overlap=0.15) if a.llm == "heuristic" else make_llm(a.llm)
    engine = "plain"
    if not a.plain:
        try:
            graph = build_langgraph(llm)
            engine = "langgraph"
        except ImportError:
            pass
    if engine == "langgraph":
        final = graph.invoke({"request": a.request, "plan": [], "i": 0, "log": [], "results": [], "handoffs": 0})
    else:
        final = run_plain(a.request, llm)
    print(f"engine: {engine}\nrequest: {a.request}")
    for line in final["log"]:
        print("  " + line)
    print(f"hand-offs: {final['handoffs']}; results: {json.dumps(final['results'])}")


if __name__ == "__main__":
    main()
