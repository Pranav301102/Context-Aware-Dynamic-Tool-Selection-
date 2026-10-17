"""S2 - An MCP server and client: list tools (paginated), call them, follow a runtime tool change.

Shows
  The demo tool server (cats.mcp.demo_server) launched over stdio and driven by an MCP client: the
  handshake / 2026-07-28 negotiation, paging through tools/list, tools/call, and
  notifications/tools/list_changed when the server unlocks a new tool at runtime (the event CATS's
  ingestor turns into incremental index updates, FR11).

Needs
  The MCP Python SDK >= 2.2 (``pip install -e ".[mcp]"``). No network, no key.

  python samples/s02_mcp_server_client.py [--mode auto|legacy]
"""
from __future__ import annotations

import argparse
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters, types
from mcp.types.version import MODERN_PROTOCOL_VERSIONS

DEMO = Path(__file__).resolve().parents[1] / "src" / "cats" / "mcp" / "demo_server.py"


async def list_all(client: Client) -> list[types.Tool]:
    tools, cursor, pages = [], None, 0
    while True:
        page = await client.list_tools(cursor=cursor)
        tools += page.tools
        pages += 1
        cursor = page.next_cursor
        if not cursor:
            print(f"  tools/list: {len(tools)} tools in {pages} pages")
            return tools


async def main(mode: str) -> None:
    changed = anyio.Event()

    async def on_message(msg) -> None:
        if isinstance(msg, types.ToolListChangedNotification):
            changed.set()

    params = StdioServerParameters(command=sys.executable, args=[str(DEMO), "--page-size", "2"])
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(Client(params, mode=mode, message_handler=on_message))
        info = client.server_info
        print(f"connected to {info.name if info else '?'} (protocol {client.protocol_version})")
        if client.protocol_version in MODERN_PROTOCOL_VERSIONS:  # change events arrive on a listen stream
            sub = await stack.enter_async_context(client.listen(tools_list_changed=True))

            async def pump():
                async for _ in sub:
                    changed.set()
            tg = await stack.enter_async_context(anyio.create_task_group())
            stack.callback(tg.cancel_scope.cancel)
            tg.start_soon(pump)
        for t in await list_all(client):
            print(f"  - {t.name}: {t.description}")
        r = await client.call_tool("calculate", {"a": 6, "b": 7, "op": "mul"})
        print("call calculate(6, 7, mul) ->", r.content[0].text)
        r = await client.call_tool("get_weather", {"city": "Paris"})
        print("call get_weather(Paris) ->", r.content[0].text)
        r = await client.call_tool("enable_extra_tools", {})
        print("call enable_extra_tools ->", r.content[0].text)
        with anyio.fail_after(5):
            await changed.wait()
        print("received tools/list_changed; re-listing")
        names = [t.name for t in await list_all(client)]
        print("  now:", ", ".join(names))
        r = await client.call_tool("get_forecast", {"city": "Lima", "days": 2})
        print("call get_forecast(Lima, 2) ->", r.content[0].text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="auto", choices=["auto", "legacy"])
    anyio.run(main, ap.parse_args().mode)
