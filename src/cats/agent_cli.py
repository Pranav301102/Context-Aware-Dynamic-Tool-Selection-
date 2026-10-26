"""Subagent harness: an external agent plays the LLM, CATS plays everything else.

No model API is needed: a coding agent (e.g. a Claude Code subagent) is the token
provider. It reads the prompt CATS assembled for a task, asks for tools with
``request-tools`` when it lacks one, and records the tool calls it would make with
``call``. The harness logs exactly what the model saw, so tokens, escalations and
task success can be scored per strategy afterwards.

Modes (same tasks, same model, only tool selection differs):
  cats    CATS-selected tools + request_tools (max 2)
  static  every tool definition in the prompt (B0); the catalog is printed once per run
  search  no tools at first, only request_tools (max 3): agent-initiated search (B6)

  python -m cats.agent_cli prepare --run R --mode cats --n 40
  python -m cats.agent_cli show --run R --task ID
  python -m cats.agent_cli request-tools --run R --task ID --capability "..."
  python -m cats.agent_cli call --run R --task ID --tool NAME --args '{...}'
  python -m cats.agent_cli done --run R --task ID
  python -m cats.agent_cli score --run R [R2 ...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

from .assembler import ContextAssembler
from .build import ROOT, build_system
from .dispatcher import simulate_tool
from .jev import jev_cost_usd
from .ledger import PrefixCache, Prices, RequestCost
from .tokens import default_counter

RUNS = ROOT / "runs"
LIMITS = {"cats": 2, "static": 0, "search": 3}


def _run_dir(run: str) -> Path:
    return RUNS / run


def _load_system(run: str):
    with open(_run_dir(run) / "system.pkl", "rb") as f:
        return pickle.load(f)


def _session_path(run: str, task_id: str) -> Path:
    return _run_dir(run) / "sessions" / f"{task_id}.json"


def _load_session(run: str, task_id: str) -> dict:
    p = _session_path(run, task_id)
    if not p.exists():
        sys.exit(f"unknown task {task_id} in run {run}")
    return json.loads(p.read_text())


def _save_session(run: str, s: dict) -> None:
    _session_path(run, s["task_id"]).write_text(json.dumps(s, indent=1))


def _assemble(sysm, s: dict, loaded: list[str] | None = None):
    """The prompt the model sees: over budget, the lowest-scored tools go first and tools the
    model explicitly requested (score 2.0) stay."""
    loaded = s["loaded"] if loaded is None else loaded
    scores = s.get("scores", {})
    return sysm.cats.assembler.assemble(loaded, [scores.get(t, 0.0) for t in loaded], s["order"])


def _require_shown(s: dict, action: str) -> None:
    """Guard against blind automation: a task must be shown before the model acts on it."""
    if not any(e["type"] == "show" for e in s["events"]):
        sys.exit(f"refused: run `show` for task {s['task_id']} before `{action}`")


def _print_tools(specs: list[dict]) -> None:
    for spec in specs:
        print(json.dumps(spec, separators=(",", ":")))


# ------------------------------------------------------------------- commands
def cmd_prepare(a) -> None:
    d = _run_dir(a.run)
    (d / "sessions").mkdir(parents=True, exist_ok=True)
    from .pipeline import CATSConfig

    enrich = Path(a.enrich) if a.enrich else None
    cfg = CATSConfig(jev=a.jev or None, llm_router=a.llm_router or None)
    sysm = build_system(cfg, seed=a.seed, enrichment=enrich)
    test = sysm.test[:]
    random.Random(a.seed).shuffle(test)
    if a.kinds:
        test = [t for t in test if t.kind in a.kinds.split(",")]
    test = test[: a.n]
    with open(d / "system.pkl", "wb") as f:
        pickle.dump(sysm, f)
    meta = {"run": a.run, "mode": a.mode, "seed": a.seed, "enrich": a.enrich, "jev": a.jev,
            "llm_router": a.llm_router, "tasks": [t.id for t in test],
            "static_tokens": sysm.registry.total_tokens(), "calibration": sysm.cats.calibration_report}
    (d / "run.json").write_text(json.dumps(meta, indent=1))
    for t in test:
        sel = sysm.cats.select(t.query, t.history) if a.mode == "cats" else None
        loaded = list(sel.tool_ids) if sel else []
        if a.mode == "static":
            loaded = sorted(sysm.registry.tools)
        _save_session(a.run, {"task_id": t.id, "mode": a.mode, "loaded": loaded, "order": list(loaded),
                              "scores": dict(zip(loaded, sel.scores)) if sel else {},
                              "router_tokens": sel.overhead_tokens if sel else 0,
                              "escalations": 0, "calls": [], "invalid": [], "events": [], "done": False})
    print(f"prepared {len(test)} tasks in {d} (mode={a.mode})")
    print("task ids:", " ".join(t.id for t in test))


def cmd_catalog(a) -> None:
    meta = json.loads((_run_dir(a.run) / "run.json").read_text())
    if meta["mode"] != "static":
        sys.exit("catalog is only shown in static mode")
    sysm = _load_system(a.run)
    print(f"# ALL {len(sysm.registry.tools)} TOOLS (static loading). These stay in your context for every task.")
    _print_tools(sysm.registry.specs(sorted(sysm.registry.tools)))


def cmd_show(a) -> None:
    sysm = _load_system(a.run)
    s = _load_session(a.run, a.task)
    task = next(t for t in sysm.test if t.id == a.task)
    reg = sysm.registry
    asm = _assemble(sysm, s)
    if s["mode"] != "static":
        print(f"# SYSTEM\n{asm.system}\n")
        print(f"# TOOLS ({len(asm.tools)})")
        _print_tools(asm.tools)
    else:
        print("# TOOLS: the full catalog you were shown (static loading); request_tools is not available.")
    print("\n# CONVERSATION")
    for turn in task.history:
        print(f"[{turn.role}] {turn.content}")
    print(f"[user] {task.query}")
    tool_tokens = reg.total_tokens(s["loaded"])
    s["events"].append({"type": "show", "n_tools": len(s["loaded"]), "tool_tokens": tool_tokens,
                        "t": time.time()})
    _save_session(a.run, s)


def cmd_request_tools(a) -> None:
    sysm = _load_system(a.run)
    s = _load_session(a.run, a.task)
    _require_shown(s, "request-tools")
    limit = LIMITS[s["mode"]]
    if s["escalations"] >= limit:
        print(f"request_tools refused: limit of {limit} reached.")
        return
    s["escalations"] += 1
    sel = sysm.cats.escalate(a.capability, exclude=set(s["loaded"]))
    s["loaded"] += sel.tool_ids
    s["order"] += sel.tool_ids
    s.setdefault("scores", {}).update({t: 2.0 for t in sel.tool_ids})
    s["events"].append({"type": "request_tools", "capability": a.capability, "added": sel.tool_ids,
                        "tool_tokens": sysm.registry.total_tokens(s["loaded"]), "t": time.time()})
    _save_session(a.run, s)
    print(f"# ADDED {len(sel.tool_ids)} TOOLS")
    _print_tools(sysm.registry.specs(sel.tool_ids))


def cmd_call(a) -> None:
    sysm = _load_system(a.run)
    s = _load_session(a.run, a.task)
    _require_shown(s, "call")
    reg = sysm.registry
    visible = s["loaded"] if s["mode"] == "static" else _assemble(sysm, s).tool_ids
    wire = {reg.tools[t].wire_name: t for t in visible}
    tid = wire.get(a.tool)
    args = json.loads(a.args or "{}")
    if tid is None:
        s["invalid"].append(a.tool)
        print(f"ERROR: tool {a.tool} is not in your toolset.")
    else:
        s["calls"].append({"tool": tid, "args": args})
        print(simulate_tool(tid, args))
    s["events"].append({"type": "call", "tool": a.tool, "valid": tid is not None, "t": time.time()})
    _save_session(a.run, s)


def cmd_done(a) -> None:
    s = _load_session(a.run, a.task)
    _require_shown(s, "done")
    s["done"] = True
    s["t_done"] = time.time()
    s["final"] = a.answer or ""
    _save_session(a.run, s)
    print("ok")


def score_run(run: str, prices: Prices | None = None) -> dict:
    prices = prices or Prices()
    sysm = _load_system(run)
    meta = json.loads((_run_dir(run) / "run.json").read_text())
    reg, counter = sysm.registry, default_counter()
    tasks = {t.id: t for t in sysm.test}
    cache = PrefixCache()
    # Bill what the model was shown: static loading had the whole catalog, with no budget.
    assembler = sysm.cats.assembler
    if meta["mode"] == "static":
        assembler = ContextAssembler(reg, budget=10**9, meta_tool=False)
    # Typed-layer (Jev) calls are billed at the layer's own price; LLM-router calls at the main model's.
    jev_priced = bool(meta.get("jev")) and not meta.get("llm_router")
    rows = []
    for tid in meta["tasks"]:
        s = _load_session(run, tid)
        t = tasks[tid]
        called = {c["tool"] for c in s["calls"]}
        success = all(called.intersection(slot) for slot in t.gold) and not s["invalid"]
        selected_ok = all(set(s["loaded"]).intersection(slot) for slot in t.gold)
        # Re-bill every model turn: one per show/request_tools/call event, with the tools loaded then.
        cost = RequestCost()
        added = {y for e in s["events"] if e["type"] == "request_tools" for y in e["added"]}
        loaded = [x for x in s["loaded"] if x not in added]
        msg_tokens = counter.count(t.query) + sum(counter.count(h.content) for h in t.history) + 60
        turns = 0
        for e in s["events"]:
            if e["type"] == "request_tools":
                loaded = loaded + e["added"]
            if e["type"] in ("show", "request_tools", "call"):
                turns += 1
                asm = (assembler.assemble(loaded, session_order=loaded) if meta["mode"] == "static"
                       else _assemble(sysm, s, loaded))
                keys = [t["name"] for t in asm.tools]
                cost += cache.charge(keys, asm.block_tokens, msg_tokens + 40 * turns, 60)
        router_tokens = s.get("router_tokens", 0)
        router_usd = jev_cost_usd(router_tokens) if jev_priced else 0.0
        if not jev_priced:
            cost.router_tokens += router_tokens
        rows.append({"task": tid, "kind": t.kind, "success": success, "selected_ok": selected_ok,
                     "escalations": s["escalations"], "invalid": len(s["invalid"]), "turns": turns,
                     "tool_tokens": reg.total_tokens(s["loaded"]), "usd": cost.usd(prices) + router_usd,
                     "router_tokens": router_tokens,
                     "input_tokens": cost.input_tokens + cost.cache_write + cost.cache_read,
                     "done": s["done"]})
    n = max(1, len(rows))
    agg = {
        "run": run, "mode": meta["mode"], "n": len(rows),
        "done": sum(r["done"] for r in rows),
        "success": sum(r["success"] for r in rows) / n,
        "tools_present": sum(r["selected_ok"] for r in rows) / n,
        "escalation_rate": sum(r["escalations"] > 0 for r in rows) / n,
        "escalations_mean": sum(r["escalations"] for r in rows) / n,
        "invalid_calls": sum(r["invalid"] for r in rows),
        "turns_mean": sum(r["turns"] for r in rows) / n,
        "tool_tokens_mean": sum(r["tool_tokens"] for r in rows) / n,
        "input_tokens_mean": sum(r["input_tokens"] for r in rows) / n,
        "usd_per_task": sum(r["usd"] for r in rows) / n,
    }
    return {"summary": agg, "rows": rows}


def cmd_score(a) -> None:
    out = [score_run(r) for r in a.runs]
    for o in out:
        print(json.dumps(o["summary"]))
        if a.verbose:
            for r in o["rows"]:
                if not r["success"]:
                    print("  FAIL", json.dumps(r))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="cats-agent", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--run", required=True)
    s.add_argument("--mode", choices=list(LIMITS), default="cats")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--kinds", default="")
    s.add_argument("--enrich", default="")
    s.add_argument("--jev", default="", help="Jev answers dir: CATS-Jev selection")
    s.add_argument("--llm-router", dest="llm_router", default="")
    s.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("catalog")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_catalog)
    for name, fn in (("show", cmd_show), ("done", cmd_done)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--task", required=True)
        if name == "done":
            s.add_argument("--answer", default="")
        s.set_defaults(fn=fn)
    s = sub.add_parser("request-tools")
    s.add_argument("--run", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--capability", required=True)
    s.set_defaults(fn=cmd_request_tools)
    s = sub.add_parser("call")
    s.add_argument("--run", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--tool", required=True)
    s.add_argument("--args", default="{}")
    s.set_defaults(fn=cmd_call)
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_score)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
