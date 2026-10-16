"""S9 - Provider-side tool search with deferred tools (baseline B7): Anthropic and OpenAI requests.

Shows
  How the 477-tool corpus is handed to a provider's own tool search instead of to CATS: every tool is
  sent with ``defer_loading: true`` and only the search tool (plus any always-loaded tools) enters the
  model's context; the model searches and the API loads the matches (cats.provider_search). Prints
  the request shape and its footprint (tokens the model sees up front vs. tokens only shipped for
  search), the OpenAI Responses variant (one namespace per server, functions deferred inside), and
  what an offline BM25 proxy of the hosted search would return for the request.

  With ANTHROPIC_API_KEY set and the SDK installed (``pip install anthropic``) it sends the request and
  prints the server_tool_use / tool_search_tool_result / tool_use blocks.

Needs
  An Anthropic API key for the live call (none was available in the build container, so only the
  dry run ran). Docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
  and https://developers.openai.com/api/docs/guides/tools-tool-search .

  python samples/s09_provider_tool_search.py ["request"] [--variant bm25|regex] [--model claude-opus-5-5]
"""
from __future__ import annotations

import argparse
import json
import os

from cats.build import build_registry
from cats.provider_search import (
    ProviderBM25Search,
    anthropic_tool_search_request,
    openai_tool_search_request,
    request_footprint,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("request", nargs="?", default="is anything slow in the prod database right now?")
    ap.add_argument("--variant", default="bm25", choices=["bm25", "regex"])
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--out", help="also write the Anthropic request JSON here")
    a = ap.parse_args()
    reg = build_registry()
    req = anthropic_tool_search_request(reg, a.request, variant=a.variant, model=a.model)
    fp = request_footprint(req)
    print(f"Anthropic request: {len(req['tools'])} tools; first entries:")
    for t in req["tools"][:3]:
        print("  " + json.dumps({k: (v if k != "input_schema" else "...") for k, v in t.items()})[:160])
    print(f"  footprint: {json.dumps(fp)}")
    print(f"  static loading would put {reg.total_tokens():,} tool tokens in context; deferred loading puts "
          f"{fp['upfront_tool_tokens']} up front and loads ~5 tools per search")
    oai = openai_tool_search_request(reg, a.request)
    print(f"OpenAI request: {len(oai['tools']) - 1} namespaces + {json.dumps(oai['tools'][-1])}; "
          f"footprint {json.dumps(request_footprint(oai))}")
    print("offline proxy of the hosted BM25 search (request text as the query):",
          ProviderBM25Search(reg).search(a.request))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(req, f, indent=1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("no ANTHROPIC_API_KEY: dry run only")
        return
    import anthropic

    resp = anthropic.Anthropic().messages.create(**req)
    for block in resp.content:
        if block.type == "server_tool_use":
            print("search:", json.dumps(block.input))
        elif block.type == "tool_search_tool_result":
            print("found:", [r.tool_name for r in getattr(block.content, "tool_references", [])])
        elif block.type == "tool_use":
            print("call:", block.name, json.dumps(block.input))
        elif block.type == "text":
            print("text:", block.text[:200])
    print("usage:", resp.usage)


if __name__ == "__main__":
    main()
