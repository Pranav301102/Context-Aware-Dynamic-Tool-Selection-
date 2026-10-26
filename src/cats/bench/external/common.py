"""Shared pieces of the external-benchmark adapters.

An adapter turns a public benchmark into the two things the rest of CATS already understands
(data/SPEC.md): branch documents (branch -> toolkit -> tool) and ``Task`` objects whose gold labels
are *slots* of equivalent tool ids. ``ExternalBenchmark`` holds both, builds a ``Registry`` and can
write them out in the same file layout as data/corpus and data/tasks.

Benchmarks that ship no hierarchy (BFCL) get one from ``auto_hierarchy``: tools are clustered into
branches by their text (TF-IDF + k-means), namespaced functions (``math.factorial``) keep their
namespace as the toolkit, and the remaining functions of a branch are grouped into small toolkits.
This is what a deployment would have to do for an uncategorised catalog, and it is deterministic
for a given seed.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...registry import Registry
from ...tokens import TokenCounter
from ...types import Branch, Task, Toolkit

_NON_ID = re.compile(r"[^a-z0-9]+")


def snake(text: str, fallback: str = "x") -> str:
    """ASCII snake_case id (data/SPEC.md); never empty, never starts with a digit."""
    s = _NON_ID.sub("_", str(text).lower()).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"{fallback}_{s}"
    return s


def unique(base: str, used: set[str]) -> str:
    out, i = base, 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


def tool_name(name: str) -> str:
    """Tool names keep their original spelling except dots, which would break ``toolkit.name`` ids."""
    return str(name).replace(".", "_").strip() or "tool"


_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "list": "array", "int": "integer",
             "bool": "boolean", "str": "string", "String": "string", "any": "string", "Any": "string",
             "long": "integer", "double": "number", "char": "string", "byte": "integer", "short": "integer",
             "HashMap": "object", "ArrayList": "array", "Array": "array", "Boolean": "boolean"}


def json_schema(schema: Any) -> dict[str, Any]:
    """Map BFCL-style pseudo types ("dict", "float", "tuple", Java types) onto JSON Schema types,
    recursively, so every converted tool carries a valid JSON-Schema-shaped ``input_schema``."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    def fix(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    out[k] = _TYPE_MAP.get(v, v)
                elif k == "properties" and isinstance(v, dict):
                    out[k] = {pk: fix(pv) for pk, pv in v.items()}
                else:
                    out[k] = fix(v)
            return out
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    out = fix(schema)
    out.setdefault("type", "object")
    return out


@dataclass
class ExternalBenchmark:
    """A converted benchmark: branch documents in the data/SPEC.md format plus tasks."""

    name: str
    corpus: list[dict[str, Any]]
    tasks: list[Task]
    meta: dict[str, Any] = field(default_factory=dict)

    def registry(self, counter: TokenCounter | None = None) -> Registry:
        return registry_from_docs(self.corpus, counter)

    def tool_ids(self) -> set[str]:
        return {f"{tk['id']}.{t['name']}" for doc in self.corpus for tk in doc["toolkits"] for t in tk["tools"]}

    def write(self, out_dir: str | Path) -> Path:
        """Write converted/corpus/<branch>.json, converted/tasks/tasks.jsonl and meta.json."""
        out = Path(out_dir)
        (out / "corpus").mkdir(parents=True, exist_ok=True)
        (out / "tasks").mkdir(parents=True, exist_ok=True)
        for old in (out / "corpus").glob("*.json"):
            old.unlink()
        for doc in self.corpus:
            (out / "corpus" / f"{doc['branch']['id']}.json").write_text(json.dumps(doc, indent=1))
        (out / "tasks" / "tasks.jsonl").write_text(
            "".join(json.dumps(t.model_dump(), ensure_ascii=False) + "\n" for t in self.tasks))
        (out / "meta.json").write_text(json.dumps(self.meta, indent=1) + "\n")
        return out

    @classmethod
    def read(cls, out_dir: str | Path, name: str | None = None) -> ExternalBenchmark:
        d = Path(out_dir)
        corpus = [json.loads(p.read_text()) for p in sorted((d / "corpus").glob("*.json"))]
        tasks = [Task.model_validate(json.loads(line)) for line in
                 (d / "tasks" / "tasks.jsonl").read_text().splitlines() if line.strip()]
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        return cls(name or meta.get("name", d.parent.name), corpus, tasks, meta)


def registry_from_docs(docs: Iterable[dict[str, Any]], counter: TokenCounter | None = None) -> Registry:
    """Same as Registry.load_branch_file, from in-memory documents."""
    reg = Registry(counter)
    for doc in docs:
        b = doc["branch"]
        reg.add_branch(Branch(id=b["id"], name=b.get("name", b["id"]), summary=b.get("summary", "")))
        for tk in doc.get("toolkits", []):
            toolkit = Toolkit(id=tk["id"], name=tk.get("name", tk["id"]), branch_id=b["id"],
                              server=tk.get("server", ""), summary=tk.get("summary", ""))
            reg.upsert_toolkit(toolkit, tk.get("tools", []))
    return reg


def slots_from_names(names: Iterable[str], by_name: dict[str, list[str]]) -> tuple[list[list[str]], list[str]]:
    """One slot per distinct gold function name; a slot holds every tool id with that name
    (servers or variants exposing the same function count as equivalent). Unknown names are
    returned separately so the caller can report them."""
    slots, missing, seen = [], [], set()
    for n in names:
        key = normalize_name(n)
        if not key or key in seen:
            continue
        seen.add(key)
        ids = by_name.get(key)
        if ids:
            slots.append(sorted(ids))
        else:
            missing.append(n)
    return slots, missing


def normalize_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def branches_of(slots: list[list[str]], branch_of_tool: dict[str, str]) -> list[str]:
    """Every branch a gold tool belongs to; the first slot's branch first (splits stratify on it)."""
    out: list[str] = []
    for slot in slots:
        for t in slot:
            b = branch_of_tool[t]
            if b not in out:
                out.append(b)
    return out


# ------------------------------------------------------------------ auto hierarchy
def _top_terms(centroid, vocab, n: int = 10) -> list[str]:
    import numpy as np

    order = np.argsort(-np.asarray(centroid).ravel(), kind="stable")[:n]  # stable: ties keep vocabulary order
    return [vocab[i] for i in order if centroid[i] > 0]


def _canonical_labels(raw) -> tuple[list[int], list[int]]:
    """Relabel clusters 0, 1, ... in order of first appearance; also return the original id of each."""
    mapping: dict[int, int] = {}
    for lab in raw:
        mapping.setdefault(int(lab), len(mapping))
    first = sorted(mapping, key=mapping.get)
    return [mapping[int(lab)] for lab in raw], first


def auto_hierarchy(units: dict[str, dict[str, Any]], n_branches: int = 32, toolkit_size: int = 8, seed: int = 0,
                   branch_prefix: str = "c") -> list[dict[str, Any]]:
    """Cluster an uncategorised catalog into branch documents.

    ``units`` maps a unit key to ``{"namespace": str | None, "tools": [tool dicts]}``. A unit with a
    namespace becomes one toolkit (all its functions stay together); namespace-less units are single
    functions, grouped inside their branch into toolkits of about ``toolkit_size``. Branch and
    toolkit summaries list the cluster's top TF-IDF terms, which is what the router reads.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    keys = sorted(units)
    texts = [" ".join([k.split(":", 1)[-1].replace("_", " ").replace(".", " ")] +
                      [f"{t['name'].replace('_', ' ')} {t.get('description', '')}" for t in units[k]["tools"]])
             for k in keys]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1, token_pattern=r"[A-Za-z][A-Za-z]+")
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    k = max(1, min(n_branches, len(keys)))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
    # Restarts with near-equal inertia can win in either order (thread-dependent float sums), which
    # permutes cluster ids; number clusters by first appearance so ids depend only on membership.
    labels, first = _canonical_labels(km.labels_)
    centers = km.cluster_centers_[first]
    width = max(2, len(str(k - 1)))
    docs = []
    used_tk: set[str] = set()
    for c in range(k):
        members = [i for i in range(len(keys)) if labels[i] == c]
        if not members:
            continue
        terms = _top_terms(centers[c], vocab, 12)
        bid = f"{branch_prefix}{c:0{width}d}"
        branch = {"id": bid, "name": " / ".join(terms[:3]) or bid,
                  "summary": "Functions about " + ", ".join(terms) + "." if terms else "Miscellaneous functions."}
        toolkits = []
        loose = []
        for i in members:
            u = units[keys[i]]
            if u.get("namespace"):
                tid = unique(snake(u["namespace"], "ns"), used_tk)
                toolkits.append({"id": tid, "name": u["namespace"], "server": f"bfcl://{u['namespace']}",
                                 "summary": _summary(u["tools"]), "tools": u["tools"]})
            else:
                loose.append(i)
        if loose:
            g = max(1, math.ceil(len(loose) / toolkit_size))
            if g > 1:
                sub = KMeans(n_clusters=g, n_init=2, random_state=seed).fit(X[loose])
                groups = defaultdict(list)
                for i, lab in zip(loose, _canonical_labels(sub.labels_)[0]):
                    groups[lab].append(i)
            else:
                groups = {0: loose}
            for j, idx in sorted(groups.items()):
                tools = [t for i in idx for t in units[keys[i]]["tools"]]
                sub_terms = _top_terms(np.asarray(X[idx].mean(axis=0)).ravel(), vocab, 6)
                tid = unique(f"{bid}_g{j}", used_tk)
                toolkits.append({"id": tid, "name": " ".join(sub_terms[:3]) or tid, "server": f"bfcl://{tid}",
                                 "summary": "Functions for " + ", ".join(sub_terms) + "." if sub_terms
                                 else _summary(tools), "tools": tools})
        for tk in toolkits:  # tool names must be unique inside a toolkit
            seen: set[str] = set()
            for t in tk["tools"]:
                t["name"] = unique(t["name"], seen)
        docs.append({"branch": branch, "toolkits": toolkits})
    return docs


def _summary(tools: list[dict[str, Any]], n: int = 2) -> str:
    first = [re.split(r"(?<=[.!?])\s", t.get("description", "").strip())[0] for t in tools[:n]]
    return " ".join(x for x in first if x)[:300]
