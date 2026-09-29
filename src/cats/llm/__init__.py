"""Model backends behind one small interface.

AnthropicLLM / OpenAICompatLLM call real models (API or vLLM). HeuristicLLM is a
deterministic stand-in for tests and offline runs. The "subagent" provider in
cats.agent_cli instead lets an external agent act as
the model through the command line.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..text import tokenize


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class HeuristicLLM:
    """Calls the listed tool whose name/description best overlaps the request.

    If no tool clears ``min_overlap``, it calls request_tools with the request text,
    which exercises escalation without a real model.
    """

    name = "heuristic"

    def __init__(self, min_overlap: float = 0.34, max_calls: int = 1):
        self.min_overlap = min_overlap
        self.max_calls = max_calls

    def complete(self, system, messages, tools):
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"
                     and isinstance(m["content"], str)), "")
        last = messages[-1] if messages else {}
        if last.get("role") == "tool" and not str(last.get("content", "")).startswith("Added tools"):
            return LLMResponse(text="done")  # a real tool answered; finish
        q = set(tokenize(user))
        best, best_score = None, 0.0
        for t in tools:
            if t["name"] == "request_tools":
                continue
            words = set(tokenize(t["name"] + " " + t.get("description", "")))
            score = len(q & words) / max(1, len(q))
            if score > best_score:
                best, best_score = t, score
        if best is None or best_score < self.min_overlap:
            if any(t["name"] == "request_tools" for t in tools):
                return LLMResponse(tool_calls=[ToolCall("request_tools", {"capability": user})])
            return LLMResponse(text="I cannot do this with the tools I have.")
        return LLMResponse(tool_calls=[ToolCall(best["name"], {})])


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system, messages, tools):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=_to_anthropic(messages), tools=tools)
        calls = [ToolCall(b.name, dict(b.input), b.id) for b in r.content if b.type == "tool_use"]
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResponse(text, calls, r.usage.input_tokens, r.usage.output_tokens)


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint, e.g. vLLM serving an open-weight model."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        import httpx

        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.http = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', 'x')}"})
        self.name = f"openai-compat:{model}"

    def complete(self, system, messages, tools):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"]}} for t in tools],
        }
        r = self.http.post(f"{self.base_url}/chat/completions", json=body).json()
        msg = r["choices"][0]["message"]
        calls = [ToolCall(c["function"]["name"], json.loads(c["function"].get("arguments") or "{}"), c.get("id", ""))
                 for c in msg.get("tool_calls") or []]
        u = r.get("usage", {})
        return LLMResponse(msg.get("content") or "", calls, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def _to_anthropic(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                                                      "content": m["content"]}]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in m["tool_calls"]]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def make_llm(spec: str) -> LLM:
    kind, _, model = spec.partition(":")
    if kind == "heuristic":
        return HeuristicLLM()
    if kind == "anthropic":
        return AnthropicLLM(model or "claude-sonnet-5-5")
    if kind in ("openai", "vllm"):
        return OpenAICompatLLM(model)
    raise ValueError(f"unknown LLM spec {spec!r}")
