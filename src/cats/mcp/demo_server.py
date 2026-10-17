"""A small MCP tool server for tests and samples (stdio).

Tools: get_weather, echo, calculate and enable_extra_tools. Calling enable_extra_tools reveals
get_forecast at runtime and announces notifications/tools/list_changed, so a client can watch a
server's tool list change (FR11). tools/list is paginated (--page-size, default 2) to exercise
cursor handling in ingestors.

Depends only on the MCP SDK, so any host can launch it by path:

    python src/cats/mcp/demo_server.py [--page-size N]     (or: python -m cats.mcp.demo_server)
"""
from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from typing import Any

import anyio
from mcp import types
from mcp.server import NotificationOptions, Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ToolsListChanged
from mcp.shared.exceptions import MCPError


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] = ()) -> types.Tool:
    schema = {"type": "object", "properties": properties, "required": list(required)}
    return types.Tool(name=name, description=description, input_schema=schema)


_CITY = {"type": "string", "description": "City name, e.g. 'Paris'."}
BASE_TOOLS = [
    _tool("get_weather", "Get the current weather (temperature and conditions) for a city.",
          {"city": _CITY, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    _tool("echo", "Echo the given text back unchanged.", {"text": {"type": "string"}}, ["text"]),
    _tool("calculate", "Apply an arithmetic operation to two numbers.",
          {"a": {"type": "number"}, "b": {"type": "number"},
           "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]}}, ["a", "b", "op"]),
    _tool("enable_extra_tools", "Unlock additional weather tools on this server.", {}),
]
HIDDEN_TOOLS = [
    _tool("get_forecast", "Get a multi-day weather forecast for a city.",
          {"city": _CITY, "days": {"type": "integer", "minimum": 1, "maximum": 7}}, ["city"]),
]

_CONDITIONS = ["sunny", "partly cloudy", "overcast", "light rain", "windy"]


def _fake_weather(city: str, day: int = 0) -> tuple[int, str]:
    h = int(hashlib.sha256(f"{city.lower()}:{day}".encode()).hexdigest(), 16)
    return 5 + h % 25, _CONDITIONS[h % len(_CONDITIONS)]


def get_weather(city: str, units: str = "celsius") -> str:
    t, cond = _fake_weather(city)
    return f"Weather in {city}: {t if units == 'celsius' else round(t * 9 / 5 + 32)}°{units[0].upper()}, {cond}"


def get_forecast(city: str, days: int = 3) -> str:
    return f"Forecast for {city}: " + "; ".join(
        "day {}: {}°C {}".format(d + 1, *_fake_weather(city, d + 1)) for d in range(days))


def calculate(a: float, b: float, op: str) -> str:
    ops: dict[str, Callable[[float, float], float]] = {
        "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "div": lambda x, y: x / y,
    }
    return str(ops[op](a, b))


HANDLERS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather, "get_forecast": get_forecast, "calculate": calculate, "echo": lambda text: text,
}


def _text(text: str, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=error)


def build_server(page_size: int = 2) -> Server:
    """A fresh demo server. Each instance has its own (initially base) tool list."""
    tools = {t.name: t for t in BASE_TOOLS}
    bus = InMemorySubscriptionBus()  # 2026-07-28 change events go out on subscriptions/listen streams

    async def list_tools(ctx: ServerRequestContext, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        cursor = params.cursor if params else None
        if cursor is not None and not cursor.isdigit():
            raise MCPError(types.INVALID_PARAMS, f"invalid cursor {cursor!r}")
        start = int(cursor or 0)
        end = start + page_size
        return types.ListToolsResult(tools=list(tools.values())[start:end],
                                     next_cursor=str(end) if end < len(tools) else None)

    async def call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
        name, args = params.name, params.arguments or {}
        if name not in tools:
            return _text(f"Unknown tool: {name}", error=True)
        if name == "enable_extra_tools":
            added = [t.name for t in HIDDEN_TOOLS if t.name not in tools]
            tools.update({t.name: t for t in HIDDEN_TOOLS})
            if added:  # announce in both protocol eras; each path is a no-op in the other one
                await ctx.session.send_tool_list_changed()
                await bus.publish(ToolsListChanged())
            return _text(f"Enabled: {', '.join(added)}" if added else "Extra tools were already enabled.")
        try:
            return _text(HANDLERS[name](**args))
        except Exception as exc:  # noqa: BLE001 - bad arguments surface to the model, not as a protocol error
            return _text(f"{type(exc).__name__}: {exc}", error=True)

    return Server("cats-demo", version="0.1.0", instructions="Demo weather, echo and calculator tools.",
                  on_list_tools=list_tools, on_call_tool=call_tool, on_subscriptions_listen=ListenHandler(bus))


async def serve(page_size: int = 2) -> None:
    server = build_server(page_size)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options(NotificationOptions(tools_changed=True)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page-size", type=int, default=2, help="tools per tools/list page")
    anyio.run(serve, ap.parse_args(argv).page_size)


if __name__ == "__main__":
    main()
