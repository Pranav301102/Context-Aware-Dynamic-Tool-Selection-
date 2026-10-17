"""Send function tools to a vLLM OpenAI-compatible endpoint.

Without --base-url, print the request body for inspection. With a running
vLLM server, for example one serving Qwen/Qwen3-8B with automatic tool choice
and the hermes parser, pass --base-url http://localhost:8000/v1.
"""
from __future__ import annotations

import argparse
import json
import os
from urllib.request import Request, urlopen


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get a multi-day weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                },
                "required": ["city"],
            },
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", nargs="?", default="Will it rain in Paris tomorrow?")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"), help="vLLM /v1 base URL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    options = parser.parse_args()

    body = {
        "model": options.model,
        "messages": [
            {"role": "system", "content": "Use the available tools to answer weather questions."},
            {"role": "user", "content": options.request},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if not options.base_url:
        print("Dry run: POST <base-url>/chat/completions")
        print(json.dumps(body, indent=2))
        return

    endpoint = options.base_url.rstrip("/") + "/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'x')}",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    if message.get("content"):
        print("Model:", message["content"])
    offered = {tool["function"]["name"] for tool in TOOLS}
    for call in message.get("tool_calls") or []:
        function = call["function"]
        name = function["name"]
        arguments = json.loads(function.get("arguments") or "{}")
        print(f"Tool call: {name}({json.dumps(arguments)})")
        if name not in offered:
            raise ValueError(f"Model requested an unoffered tool: {name}")
    usage = result.get("usage", {})
    print(f"Usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion tokens")


if __name__ == "__main__":
    main()
