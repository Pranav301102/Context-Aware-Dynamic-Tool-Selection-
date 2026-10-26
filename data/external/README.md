# External benchmarks

Every task in `data/tasks/` was written by Claude subagents, so the project's own numbers may share
phrasing with the enrichment and the LLM router (docs/RESULTS.md §4.1, §8). This directory holds
open benchmarks that no project model wrote, pinned to a commit and converted to the CATS corpus
and task format (`data/SPEC.md`).

| Benchmark | Status | Tools | Tasks | Source (pinned) | Licence |
|---|---|---|---|---|---|
| [LiveMCPBench](livemcpbench/README.md) | obtained | 525 (69 MCP servers, 8 categories) | 94 of 95 | `icip-cas/LiveMCPBench@36a51a1` | Apache-2.0 |
| [BFCL v4](bfcl/README.md) | obtained | 3,258 function variants (1,853 names) | 2,501 | `ShishirPatil/gorilla@f7cf735` | Apache-2.0 |
| MCP-Zero (MCP-tools) | not obtained | (308 servers, 2,797 tools) | none shipped | `xfey/MCP-Zero` | MIT |
| MCP-Bench | not obtained | (28 servers) | 104 | `Accenture/mcp-bench` | Apache-2.0 |
| ToolRet | not obtained | (43k tools) | (7.6k) | Hugging Face `mangopy/ToolRet-*` | see dataset card |

Layout per obtained benchmark:

```
<name>/raw/          the upstream files, byte-identical, with LICENSE
<name>/manifest.json URL, commit, SHA-256 and size of every raw file, retrieval date
<name>/converted/    corpus/<branch>.json + tasks/tasks.jsonl + meta.json (our format)
<name>/README.md     source, licence, mapping and caveats
```

Re-download and verify (fails if an upstream file no longer matches its recorded SHA-256):

```bash
python scripts/fetch_external.py --convert
python -m cats.bench.external.run livemcpbench      # -> results/external_livemcpbench.json
python -m cats.bench.external.run bfcl              # -> results/external_bfcl.json
python -m cats.bench.external.run bfcl --strict --name external_bfcl_strict          # no label widening
python -m cats.bench.external.run bfcl --no-decompose --name external_bfcl_nodecomp  # ablation
```

Headline (details in docs/components/evaluation.md §9): full cover of CATS vs flat reranked
retrieval at the same k is 0.893 vs 0.883 on BFCL (McNemar p = 0.031, 0.062 after Holm; the gain
disappears without query decomposition and with strict labels) and 0.266 vs 0.255 on LiveMCPBench
(p = 1.0), where even k = 50 covers every needed tool for only half the tasks.

## What was tried for the benchmarks not obtained (2026-10-06)

The build container could reach `raw.githubusercontent.com` and PyPI only; `huggingface.co`,
`api.github.com`, `github.com`, `codeload.github.com`, `cdn.jsdelivr.net`, `data.jsdelivr.com` and
Google Drive were blocked or unreachable. Repository trees were listed with a web fetcher outside the
container and files were then fetched from `raw.githubusercontent.com`.

* **MCP-Zero.** The tool dataset (`MCP-tools/mcp_tools_with_embedding.json`) is a Git LFS object;
  the README says the author's LFS bandwidth quota is exhausted and points to a Google Drive file
  (`1RjBGU-AGdHdhUABoeYSztbfQlD0hjUBn`). Tried: `raw.githubusercontent.com/xfey/MCP-Zero/master/...`
  (404, LFS pointer only), `media.githubusercontent.com/media/...` (404),
  `drive.google.com/uc?export=download&id=...` and `drive.usercontent.google.com/download?id=...`
  (connection refused). MCP-Zero also ships no query set: its "needle" experiment generates queries
  with an LLM at run time (`MCP-zero/experiment_mcptools.py`), and `MCP-zero/test_cases.jsonl` is a
  single example. Fallback: none needed for the claim, since LiveMCPBench covers real MCP servers.
* **MCP-Bench.** The tasks are on GitHub (`tasks/mcpbench_tasks_*_runner_format.json`, 56 single-,
  30 two- and 18 three-server tasks, each with a `fuzzy_description` written for users), but the
  tool definitions are not: they are collected live from 28 running MCP servers
  (`utils/collect_mcp_info.py`), several of which need API keys (NPS, NASA, Google Maps, NCI, HF), and
  gold tools appear only as free-text `Server:tool` mentions in `task_description`. Converting it
  needs the servers running, or parsing the source of 28 servers in three languages. Not done.
* **ToolRet.** Evaluation data is only on Hugging Face (`mangopy/ToolRet-Queries`,
  `mangopy/ToolRet-Tools`); the GitHub repository (`mangopy/tool-retrieval-benchmark`) holds code
  only. No GitHub or PyPI mirror was found (searched; the `toolrank` PyPI package is a retriever, not
  a data mirror). With Hugging Face access: `datasets.load_dataset("mangopy/ToolRet-Queries")`.

## Caveats

* Gold labels are the benchmarks' own; slots are widened only across tools that share the gold
  function name (LiveMCPBench servers exposing the same tool; BFCL variants of one function).
  `--strict` turns the BFCL widening off.
* BFCL has no categories or servers. Its hierarchy is generated (TF-IDF + k-means, seed 0, 32
  branches; namespaces such as `math.` become toolkits), so routing is evaluated on a machine-made
  taxonomy, not a curated one.
* Neither benchmark has enrichment (synthetic requests per tool), which was the largest single gain
  on the project's own catalog. That is deliberate: these runs measure CATS without any
  model-written text.
