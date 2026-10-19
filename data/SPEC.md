# CATS data format

## Tool corpus: `data/corpus/<branch_id>.json`

One file per branch (a domain such as Search or Database). A branch holds
toolkits (one toolkit is one MCP server); a toolkit holds tools.

```json
{
  "branch": {
    "id": "search",
    "name": "Search",
    "summary": "One or two sentences on what requests this branch serves. Used by the router."
  },
  "toolkits": [
    {
      "id": "brave_search",
      "name": "Brave Search",
      "server": "mcp://brave-search",
      "summary": "One sentence on what this server does.",
      "tools": [
        {
          "name": "web_search",
          "description": "Realistic MCP-style description, 1-3 sentences, like a real server would ship.",
          "input_schema": {
            "type": "object",
            "properties": {
              "query": {"type": "string", "description": "Search terms."},
              "count": {"type": "integer", "description": "Number of results (1-20).", "default": 10}
            },
            "required": ["query"]
          }
        }
      ]
    }
  ]
}
```

Rules:
- `branch.id`, `toolkit.id` and tool `name` are `snake_case`, ASCII.
- Toolkit ids are globally unique across all branches (prefix with the vendor, e.g. `github`, `gitlab`).
- Tool names are unique within a toolkit. A tool's global id is `<toolkit_id>.<name>`.
- Schemas are valid JSON Schema objects, with realistic parameters (enums, defaults,
  nested objects where a real API has them). Schema size should vary the way real
  MCP servers do: some tools tiny, some large.

## Tasks: `data/tasks/<branch_id>.jsonl`

One JSON object per line:

```json
{"id": "search-0001",
 "query": "what are people saying about the new EU AI act enforcement this week?",
 "history": [],
 "gold": [["brave_search.news_search", "tavily.search_news"]],
 "branches": ["search"],
 "kind": "paraphrase"}
```

- `gold` is a list of *slots*. Each slot is the list of tool ids (`toolkit_id.name`)
  that would each satisfy that step of the task equally well. A task that needs two
  steps (e.g. look something up, then save it) has two slots.
- `branches` lists every branch a gold tool belongs to.
- `kind` is one of:
  - `literal`: the query shares obvious words with the tool description;
  - `paraphrase`: the query describes the need in different words than the tool
    (non-literal match; this is the hard and important case);
  - `multi`: needs two or three slots;
  - `contextual`: the need only becomes clear from `history` (prior turns, a list of
    `{"role": "user"|"assistant", "content": "..."}`), e.g. "ok now do the same for
    the staging cluster";
  - `cross`: gold slots span two or more branches.
- `history` is usually empty except for `contextual` tasks.
