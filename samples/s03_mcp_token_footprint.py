"""Measure the prompt footprint of tools exposed by an MCP server.

Run with the bundled demo server or pass a different stdio command:

    python samples/s03_mcp_token_footprint.py
    python samples/s03_mcp_token_footprint.py --command npx --args -y @modelcontextprotocol/server-filesystem /tmp

The default count is a regex estimate. Install tiktoken and pass --encoding
o200k_base to count with a specific model encoding instead.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

DEMO = Path(__file__).resolve().parents[1] / "src" / "cats" / "mcp" / "demo_server.py"


def count_tokens(text: str, encoding: str | None) -> int:
    if encoding:
        import tiktoken

        return len(tiktoken.get_encoding(encoding).encode(text))
    return len(re.findall(r"\w+|[^\w\s]", text))


async def server_footprint(command: str, args: list[str], encoding: str | None) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    cursor = None
    async with Client(StdioServerParameters(command=command, args=args)) as client:
        while True:
            page = await client.list_tools(cursor=cursor)
            for tool in page.tools:
                definition = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.input_schema,
                }
                serialized = json.dumps(definition, sort_keys=True, separators=(",", ":"))
                rows.append((tool.name, count_tokens(serialized, encoding)))
            cursor = page.next_cursor
            if not cursor:
                return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--command", default=sys.executable)
    parser.add_argument("--encoding", help="tiktoken encoding, for example o200k_base")
    parser.add_argument("--args", nargs=argparse.REMAINDER, default=[str(DEMO)])
    options = parser.parse_args()
    rows = anyio.run(server_footprint, options.command, options.args, options.encoding)
    method = options.encoding or "regex estimate"
    print(f"Token count: {method}")
    print(f"Server: {' '.join([options.command, *options.args])}")
    for name, count in sorted(rows, key=lambda row: -row[1]):
        print(f"  {name:28s} {count:6d}")
    print(f"  {'TOTAL':28s} {sum(count for _, count in rows):6d} for {len(rows)} tools")


if __name__ == "__main__":
    main()
