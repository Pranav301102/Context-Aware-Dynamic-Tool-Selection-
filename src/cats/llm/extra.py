"""Extra model backends: LiteLLM (one client for 100+ providers), plus Gemini and OpenAI through it.

``LiteLLMBackend`` implements the same ``complete(system, messages, tools) -> LLMResponse`` interface
as the backends in cats.llm, so the dispatcher, LLMJev, the LLM router and the B5 baseline can run on
any provider LiteLLM supports (``pip install litellm``; https://docs.litellm.ai/docs/completion).
Model strings use LiteLLM's "<provider>/<model>" form and read the provider's usual key from the
environment:

    litellm:openai/gpt-5.4               OPENAI_API_KEY
    litellm:gemini/gemini-3-pro          GEMINI_API_KEY (Google AI Studio); vertex_ai/... uses ADC
    litellm:anthropic/claude-sonnet-5-5  ANTHROPIC_API_KEY
    litellm:hosted_vllm/Qwen/Qwen3-8B    an OpenAI-compatible vLLM server (api_base=..., see below)
    litellm:ollama/qwen3:8b              a local Ollama server

``make_extra_llm`` also accepts the shorthands ``gemini:<model>`` and ``openai:<model>``.

Note on Gemini and OpenAI. Both are reached through LiteLLM's OpenAI-style chat-completions
interface: tools go out as {"type": "function", "function": {...}} and calls come back as
``message.tool_calls`` whatever the provider, which is all this interface needs. Provider-specific
features are not exposed through it: OpenAI's tool search with namespaces lives in the Responses API
and Anthropic's tool search tool in the Messages API (both are built as raw requests in
cats.provider_search, baseline B7); Gemini's function-calling modes (AUTO / ANY / NONE) map to
``tool_choice``. Gemini rejects some JSON-schema keywords in function parameters (e.g.
``additionalProperties``, ``$ref``); LiteLLM strips the common ones, so schemas with deeper
features may need flattening first.

Offline, pass ``mock_response="..."`` (and ``mock_tool_calls=[...]``) through ``**kwargs``: LiteLLM
returns them without a network call, which is how samples and tests exercise this backend without a
key. Tests can also inject ``completion_fn`` to avoid importing LiteLLM at all.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from . import LLMResponse, ToolCall


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """cats.llm messages -> OpenAI chat format (system first; tool calls with JSON-string arguments)."""
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": str(m.get("content", ""))})
        elif role == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": [
                {"id": c.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
                for i, c in enumerate(m["tool_calls"])]})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic-style tool blocks (name, description, input_schema) -> OpenAI function tools."""
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
            for t in tools]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_completion(resp: Any) -> LLMResponse:
    """A chat-completions response (LiteLLM ModelResponse, OpenAI object or plain dict) -> LLMResponse."""
    msg = _get(_get(resp, "choices")[0], "message")
    calls = []
    for c in _get(msg, "tool_calls") or []:
        fn = _get(c, "function")
        raw = _get(fn, "arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except ValueError:
            args = {"_raw": raw}
        calls.append(ToolCall(_get(fn, "name"), args, _get(c, "id") or ""))
    usage = _get(resp, "usage") or {}
    return LLMResponse(_get(msg, "content") or "", calls, int(_get(usage, "prompt_tokens", 0) or 0),
                       int(_get(usage, "completion_tokens", 0) or 0))


class LiteLLMBackend:
    """Any LiteLLM model behind the cats.llm interface (see the module docstring)."""

    def __init__(self, model: str, completion_fn: Callable[..., Any] | None = None, max_tokens: int = 1024,
                 **kwargs: Any):
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs  # e.g. api_base, api_key, temperature, mock_response, tool_choice
        self._completion = completion_fn
        self.name = f"litellm:{model}"
        self.last_request: dict[str, Any] = {}

    def _fn(self) -> Callable[..., Any]:
        if self._completion is None:
            import litellm  # optional dependency: pip install litellm

            self._completion = litellm.completion
        return self._completion

    def request(self, system: str, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        """The keyword arguments of the completion call (without sending it)."""
        req: dict[str, Any] = {"model": self.model, "messages": to_openai_messages(system, messages),
                               "max_tokens": self.max_tokens, **self.kwargs}
        if tools:
            req["tools"] = to_openai_tools(tools)
        return req

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.last_request = self.request(system, messages, tools)
        return parse_completion(self._fn()(**self.last_request))


def make_extra_llm(spec: str, **kwargs: Any) -> LiteLLMBackend:
    """``litellm:<provider>/<model>``, ``gemini:<model>`` or ``openai:<model>`` -> LiteLLMBackend."""
    kind, _, model = spec.partition(":")
    if kind == "litellm":
        return LiteLLMBackend(model, **kwargs)
    if kind == "gemini":
        return LiteLLMBackend(f"gemini/{model}", **kwargs)
    if kind == "openai":
        return LiteLLMBackend(f"openai/{model}", **kwargs)
    raise ValueError(f"unknown extra LLM spec {spec!r} (use litellm:<provider>/<model>, gemini:<m> or openai:<m>)")
