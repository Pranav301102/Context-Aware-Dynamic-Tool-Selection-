"""Jev-style typed decision layer ("System One") for CATS.

Jev (TypeSafe AI, 2026) answers typed questions about a state with probabilities instead of
generating text, and agents act when it is confident and fall back to a strong LLM when it is
not. Its three question types:

    noul    yes/no               -> probability of "yes"
    choice  one of N options     -> probability per option
    score   ordered levels       -> probability per level

CATS asks Jev three kinds of question:

    route   one noul per branch: "does this request need a tool from <branch>?"  (multi-label, so
            cross-branch requests can say yes to several)
    steps   score 1..3: how many separate steps the request has (sizes decomposition and k)
    enough  noul: "can the request be completed with only these tools?"  (pre-escalation gate)

Backends implement ``ask(requests) -> answers``. ``ExchangeJev`` reads answers produced offline by
an external source (here offline answer files act as Jev); ``LLMJev`` prompts any cats.llm backend to
return the typed JSON. Neither is TypeSafe's hosted service, whose API is not used here.

Cost model (public launch figures): $42 per billion input tokens, about 150 ms per call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .llm_router import request_key
from .registry import Registry
from .tokens import default_counter
from .types import Turn

JEV_USD_PER_MTOK = 0.042  # $42 per billion input tokens
JEV_LATENCY_MS = 150.0

Kind = Literal["noul", "choice", "score"]


@dataclass
class Question:
    id: str
    kind: Kind
    text: str
    options: list[str] = field(default_factory=list)  # choice options or score levels (low -> high)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.kind, "text": self.text}
        if self.options:
            d["options" if self.kind == "choice" else "levels"] = self.options
        return d


@dataclass
class JevRequest:
    key: str
    state: str
    questions: list[Question]

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "state": self.state, "questions": [q.to_json() for q in self.questions]}

    def input_tokens(self) -> int:
        return default_counter().count_json(self.to_json())


@dataclass
class Answer:
    id: str
    kind: Kind
    p: float | None = None  # noul
    dist: dict[str, float] = field(default_factory=dict)  # choice / score

    @property
    def confidence(self) -> float:
        if self.kind == "noul":
            return max(self.p or 0.0, 1 - (self.p or 0.0))
        return max(self.dist.values()) if self.dist else 0.0

    def expected_level(self) -> float:
        """Score answers: expected level index (0-based)."""
        if not self.dist:
            return 0.0
        return sum(i * p for i, p in enumerate(self.dist.values())) / max(1e-9, sum(self.dist.values()))


class Jev(Protocol):
    def ask(self, requests: list[JevRequest]) -> dict[str, dict[str, Answer]]: ...


# ----------------------------------------------------------------------- question builders
def state_text(query: str, history: list[Turn] | None = None) -> str:
    turns = [f"[{t.role}] {t.content}" for t in (history or [])]
    return "\n".join(turns + [f"[user] {query}"])


def route_questions(reg: Registry) -> list[Question]:
    qs = []
    for b in sorted(reg.branches.values(), key=lambda x: x.id):
        servers = ", ".join(reg.toolkits[t].name for t in reg.branch_toolkits(b.id))
        qs.append(Question(f"needs:{b.id}", "noul",
                           f"Does completing the user's latest request need a tool from '{b.id}' "
                           f"({b.summary} Servers: {servers})?"))
    qs.append(Question("steps", "score", "How many separate tool-using steps does the latest request need?",
                       ["1", "2", "3+"]))
    return qs


def route_request(reg: Registry, query: str, history: list[Turn] | None = None) -> JevRequest:
    return JevRequest(request_key(query, history), state_text(query, history), route_questions(reg))


def enough_request(reg: Registry, query: str, history: list[Turn] | None, tool_ids: list[str]) -> JevRequest:
    lines = []
    for t in tool_ids:
        rec = reg.tools[t]
        lines.append(f"- {rec.wire_name}: {rec.description.split('. ')[0][:160]}")
    state = state_text(query, history) + "\n\nTools available:\n" + "\n".join(lines)
    q = Question("enough", "noul", "Can every step of the latest request be done with only the tools listed?")
    return JevRequest(request_key(query, history) + ":enough", state, [q])


# ------------------------------------------------------------------------------ backends
def parse_answers(rec: dict[str, Any]) -> dict[str, Answer]:
    """Accepts the typed form {"answers": [...]} or the compact route form
    {"needs": {branch: p}, "steps": {"1": p, "2": p, "3+": p}}."""
    if "needs" in rec or "steps" in rec:
        answers = [{"id": f"needs:{b}", "type": "noul", "p": p} for b, p in (rec.get("needs") or {}).items()]
        if rec.get("steps"):
            answers.append({"id": "steps", "type": "score", "dist": rec["steps"]})
        if "enough" in rec:
            answers.append({"id": "enough", "type": "noul", "p": rec["enough"]})
        rec = {"answers": answers}
    out = {}
    for a in rec.get("answers", []):
        kind = a.get("type", "noul")
        if kind == "noul":
            out[a["id"]] = Answer(a["id"], "noul", p=float(min(1.0, max(0.0, a.get("p", 0.0)))))
        else:
            dist = {str(k): float(v) for k, v in (a.get("dist") or {}).items()}
            s = sum(dist.values()) or 1.0
            out[a["id"]] = Answer(a["id"], kind, dist={k: v / s for k, v in dist.items()})
    return out


class ExchangeJev:
    """Answers written offline by an external model, as JSONL: {"key", "answers": [...]}."""

    def __init__(self, path: str | Path):
        path = Path(path)
        files = sorted(path.glob("answers_*.jsonl")) if path.is_dir() else [path]
        self.answers: dict[str, dict[str, Answer]] = {}
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.answers[rec["key"]] = parse_answers(rec)

    def ask(self, requests: list[JevRequest]) -> dict[str, dict[str, Answer]]:
        return {r.key: self.answers[r.key] for r in requests if r.key in self.answers}

    def get(self, key: str) -> dict[str, Answer] | None:
        return self.answers.get(key)


JEV_SYSTEM = (
    "You are a System One decision model. You never write prose. For the given state, answer every "
    "typed question and return JSON: {\"answers\": [{\"id\": ..., \"type\": \"noul\", \"p\": <prob of yes>} "
    "| {\"id\": ..., \"type\": \"choice\"|\"score\", \"dist\": {option: prob}}]}. Probabilities must be "
    "calibrated: 0.9 means right 9 times in 10."
)


class LLMJev:
    """Any cats.llm backend prompted to act as Jev (one call per request)."""

    def __init__(self, llm):
        self.llm = llm

    def ask(self, requests: list[JevRequest]) -> dict[str, dict[str, Answer]]:
        out = {}
        for r in requests:
            resp = self.llm.complete(JEV_SYSTEM, [{"role": "user", "content": json.dumps(r.to_json())}], [])
            try:
                text = resp.text[resp.text.index("{"): resp.text.rindex("}") + 1]
                out[r.key] = parse_answers(json.loads(text))
            except (ValueError, KeyError):
                continue
        return out


# ------------------------------------------------------------------------------- export
def export_requests(requests: list[JevRequest], out_dir: Path, n_batches: int, prefix: str = "requests") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_batches):
        p = out_dir / f"{prefix}_{i}.jsonl"
        with p.open("w") as f:
            for r in requests[i::n_batches]:
                f.write(json.dumps(r.to_json()) + "\n")
        paths.append(p)
    return paths


def branch_probs(answers: dict[str, Answer] | None, branches: list[str]) -> dict[str, float]:
    if not answers:
        return {}
    return {b: answers[f"needs:{b}"].p or 0.0 for b in branches if f"needs:{b}" in answers}


def steps_level(answers: dict[str, Answer] | None) -> int:
    """Most likely number of steps (1, 2 or 3) from the 'steps' score answer; 1 if absent."""
    if not answers or "steps" not in answers or not answers["steps"].dist:
        return 1
    d = answers["steps"].dist
    best = max(d, key=d.get)
    return {"1": 1, "2": 2, "3+": 3}.get(best, 1)


def jev_cost_usd(input_tokens: int) -> float:
    return input_tokens * JEV_USD_PER_MTOK / 1e6
