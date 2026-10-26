"""Tool-pool scaling generator, 10 to 10,000 tools (FR14).

Pools are built from three sources, in order:
  1. the gold tools of the evaluated tasks (always present);
  2. the rest of the real corpus and the distractor domains (data/distractors);
  3. vendor clones: copies of real toolkits under new vendor names, with lightly
     reworded descriptions, which is how large public MCP registries actually grow
     (many servers wrapping the same service).

A clone is functionally equivalent to its source, so labels are widened: a clone
satisfies a gold slot when its source does, unless the task names the source
vendor (then only that vendor is correct).
"""
from __future__ import annotations

import copy
import random
import re
from pathlib import Path

from ..registry import Registry
from ..types import Branch, Task, Toolkit

VENDOR_PREFIXES = """acme nimbus vertex orbit quanta helix lumen apex nova zenith pulse cobalt ember atlas
sigma delta prism vector harbor summit cedar maple aurora polar onyx terra fluent swift kite relay""".split()
VENDOR_SUFFIXES = "hq io labs cloud hub works ly stack base kit".split()

SYNONYMS = {
    "retrieve": "fetch", "fetch": "retrieve", "get": "retrieve", "list": "enumerate", "create": "make",
    "delete": "remove", "remove": "delete", "update": "modify", "search": "look up", "send": "dispatch",
    "returns": "gives back", "specified": "given", "information": "details", "details": "information",
}


def _reword(text: str, rng: random.Random) -> str:
    def swap(m: re.Match) -> str:
        w = m.group(0)
        rep = SYNONYMS.get(w.lower())
        if rep and rng.random() < 0.5:
            return rep.capitalize() if w[0].isupper() else rep
        return w
    return re.sub(r"[A-Za-z]+", swap, text)


def _vendor(rng: random.Random, used: set[str]) -> str:
    for attempt in range(10_000):
        v = f"{rng.choice(VENDOR_PREFIXES)}{rng.choice(VENDOR_SUFFIXES)}"
        if attempt > 50:  # the 300 base names run out at large pools; number them
            v += str(rng.randint(2, 999))
        if v not in used:
            used.add(v)
            return v
    raise RuntimeError("could not find an unused vendor name")


def load_distractors(reg: Registry, distractor_dir: Path) -> None:
    for path in sorted(Path(distractor_dir).glob("*.json")):
        try:
            reg.load_branch_file(path)
        except (KeyError, ValueError):  # a malformed distractor file must not break the benchmark
            continue


def build_pool(base: Registry, tasks: list[Task], size: int, seed: int = 0,
               distractor_dir: Path | None = None) -> tuple[Registry, list[Task]]:
    """Return a registry with exactly ``size`` tools and the tasks with widened gold labels."""
    rng = random.Random(seed)
    src = Registry(base.counter)
    for b in base.branches.values():
        src.add_branch(b)
    for tk in base.toolkits.values():
        src.upsert_toolkit(tk, [_raw(base, i) for i in base.toolkit_tools(tk.id)])
    if distractor_dir:
        load_distractors(src, distractor_dir)

    gold = sorted({g for t in tasks for g in t.gold_tools() if g in src.tools})
    others = [i for i in src.tools if i not in set(gold)]
    rng.shuffle(others)
    chosen = gold + others[: max(0, size - len(gold))]
    clones: dict[str, str] = {}  # clone tool id -> source tool id
    vendors = vendor_words(src)
    if len(chosen) < size:
        chosen, clones = _add_clones(src, chosen, size, rng, vendors)

    pool = Registry(base.counter)
    keep = set(chosen)
    for tk_id, tk in src.toolkits.items():
        ids = [i for i in src.toolkit_tools(tk_id) if i in keep]
        if not ids:
            continue
        if tk.branch_id not in pool.branches:
            pool.add_branch(src.branches[tk.branch_id])
        pool.upsert_toolkit(tk, [_raw(src, i) for i in ids])
    return pool, _widen(tasks, clones, src, pool, vendors)


def _raw(reg: Registry, tool_id: str) -> dict:
    t = reg.tools[tool_id]
    return {"name": t.name, "description": t.description, "input_schema": t.input_schema}


def vendor_words(reg: Registry) -> dict[str, set[str]]:
    """toolkit id -> the words that name its vendor, as CATS's analyzer defines them (words of the
    toolkit id/name that rarely occur in other servers' descriptions). Generic first words such as
    "PDF Tools" or "Package Manager" yield none, so they never block label widening."""
    from ..analyzer import ContextAnalyzer

    out: dict[str, set[str]] = {tk: set() for tk in reg.toolkits}
    for word, tks in ContextAnalyzer(reg).vendors.items():
        for tk in tks:
            out[tk].add(word)
    return out


def _swap_vendor(text: str, words: set[str], vendor: str) -> str:
    for w in words:
        text = re.sub(rf"\b{re.escape(w)}\b", vendor.capitalize(), text, flags=re.IGNORECASE)
    return text


def _add_clones(src: Registry, chosen: list[str], size: int, rng: random.Random,
                vendors: dict[str, set[str]]):
    used = {tk.split("_")[0] for tk in src.toolkits}
    clones: dict[str, str] = {}
    toolkits = list(src.toolkits.values())
    chosen = list(chosen)
    while len(chosen) < size:
        tk = rng.choice(toolkits)
        vendor = _vendor(rng, used)
        words = vendors.get(tk.id, set())
        # The clone must not carry the source vendor's name anywhere (id, name, descriptions), or a
        # user naming that vendor would match every clone. Tool names stay: servers wrapping the same
        # API usually keep them, which is what duplicate collapse keys on.
        rest = [p for p in tk.id.split("_") if p not in words]
        new_id = "_".join([vendor] + rest)
        while new_id in src.toolkits:
            new_id += "x"
        name = _swap_vendor(tk.name, words, vendor)
        if name == tk.name:
            name = f"{vendor.capitalize()} {tk.name}"
        new_tk = Toolkit(id=new_id, name=name, branch_id=tk.branch_id, server=f"mcp://{vendor}",
                         summary=_swap_vendor(_reword(tk.summary, rng), words, vendor))
        tools = []
        for tid in src.toolkit_tools(tk.id):
            raw = copy.deepcopy(_raw(src, tid))
            raw["description"] = _swap_vendor(_reword(raw["description"], rng), words, vendor)
            tools.append(raw)
            clones[f"{new_tk.id}.{raw['name']}"] = clones.get(tid, tid)
        vendors[new_tk.id] = {vendor}
        src.upsert_toolkit(new_tk, tools)
        chosen.extend(f"{new_tk.id}.{r['name']}" for r in tools)
    return chosen[:size], clones


def _widen(tasks: list[Task], clones: dict[str, str], src: Registry, pool: Registry,
           vendors: dict[str, set[str]]) -> list[Task]:
    by_source: dict[str, list[str]] = {}
    for c, s in clones.items():
        if c in pool.tools:
            by_source.setdefault(s, []).append(c)
    out = []
    for t in tasks:
        text = " ".join([t.query] + [h.content for h in t.history]).lower()
        words_in_text = set(re.findall(r"[a-z0-9]+", text))
        slots = []
        for slot in t.gold:
            new = [g for g in slot if g in pool.tools]
            for g in slot:
                named = vendors.get(src.tools[g].toolkit_id, set()) if g in src.tools else set()
                if named & words_in_text:
                    continue  # the user named this vendor; clones do not count
                new += by_source.get(g, [])
            slots.append(sorted(set(new)))
        if all(slots):
            out.append(t.model_copy(update={"gold": slots}))
    return out


def ensure_branches(reg: Registry, branches: list[Branch]) -> None:
    for b in branches:
        if b.id not in reg.branches:
            reg.add_branch(b)
