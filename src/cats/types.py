"""Typed contracts between CATS components (workbook Section 9.3).

RouteDecision     orchestration -> sub-agents
RetrievalRequest  orchestration -> retrieval
RetrievalResult   retrieval -> orchestration
ToolRecord        registry -> everyone
RegistryEvent     registry -> index / ingestor
TraceSpan         everyone -> evaluation
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class Branch(BaseModel):
    id: str
    name: str
    summary: str


class Toolkit(BaseModel):
    id: str
    name: str
    branch_id: str
    server: str = ""
    summary: str = ""


class ToolRecord(BaseModel):
    id: str  # "<toolkit_id>.<name>"
    name: str
    toolkit_id: str
    branch_id: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    token_cost: int = 0
    version: str = ""

    def spec(self) -> dict[str, Any]:
        """The definition a model sees (Anthropic-style tool block)."""
        return {"name": self.wire_name, "description": self.description, "input_schema": self.input_schema}

    @property
    def wire_name(self) -> str:
        # Tool names on the wire may not contain dots for most providers.
        return self.id.replace(".", "__")

    def content_hash(self) -> str:
        blob = json.dumps({**self.spec(), "toolkit": self.toolkit_id, "branch": self.branch_id},
                          sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class RegistryEvent(BaseModel):
    kind: Literal["added", "updated", "removed"]
    toolkit_id: str
    tool_ids: list[str]
    registry_version: int


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Task(BaseModel):
    id: str
    query: str
    history: list[Turn] = Field(default_factory=list)
    gold: list[list[str]]  # slots; each slot lists equivalent tool ids
    branches: list[str] = Field(default_factory=list)
    kind: str = "literal"

    def gold_tools(self) -> set[str]:
        return {t for slot in self.gold for t in slot}


class Features(BaseModel):
    """Output of the context analyzer."""
    text: str  # the effective retrieval query (query + weighted history)
    query: str
    vendor_hints: list[str] = Field(default_factory=list)  # toolkit ids named in the request
    model_config = {"arbitrary_types_allowed": True}


class RouteDecision(BaseModel):
    branch_scores: dict[str, float]
    selected: list[str]  # branches searched (for decomposed requests, the union over the parts)
    conformal_set: list[str] = Field(default_factory=list)  # the whole request's calibrated set
    threshold: float
    fallback: bool = False
    escalations: int = 0


class RetrievalRequest(BaseModel):
    query: str
    branch_ids: list[str] | None = None  # None = search every branch
    k_max: int = 10
    vendor_hints: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    tool_ids: list[str]
    scores: list[float]
    candidates_considered: int = 0


class TraceSpan(BaseModel):
    stage: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class Selection(BaseModel):
    """What a selection strategy hands to the sub-agent for one request."""
    strategy: str
    tool_ids: list[str]
    route: RouteDecision | None = None
    tool_tokens: int = 0  # tokens of the tool definitions placed in the prompt
    overhead_tokens: int = 0  # router LLM calls etc.
    router_calls: int = 0  # external router calls made for this request (0 or 1)
    router_kind: str = ""  # "llm" | "jev" | ""
    spans: list[TraceSpan] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)  # aligned with tool_ids
