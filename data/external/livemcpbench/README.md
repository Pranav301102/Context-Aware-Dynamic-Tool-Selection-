# LiveMCPBench (LiveMCPTool + annotated tasks)

* **Paper:** Mo et al., *LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools?*, arXiv:2508.01780 (2025).
* **Source:** https://github.com/icip-cas/LiveMCPBench
* **Pinned commit:** `36a51a1065dda49c1e503b87edf33a1e0b331a74` (main, 2025-12-18, "fix(tools): remove
  error bing tools"); SHA-256 of every file in `manifest.json`.
* **Files:** `raw/tools.json` = `tools/LiveMCPTool/tools.json`; `raw/all_annotations.json` =
  `annotated_data/all_annotations.json`; `raw/LICENSE`.
* **Licence:** Apache-2.0 (`raw/LICENSE`). Redistributed unmodified; the converted files are a
  derivative work under the same licence.
* **Retrieved:** 2026-10-06 via `raw.githubusercontent.com` (`scripts/fetch_external.py`).

## Contents

`tools.json` lists 69 server entries (70 MCP servers in the paper; one was removed upstream) with
their `tools/list` output: 525 tools, 508 distinct names. Each entry has a category: Discovery (24
entries), Code (11), Miscellaneous (8), Visualization (7), File Access (7), Finance (6),
Entertainment (5), Location (1). `all_annotations.json` holds 95 human-written tasks in six domains
(Office 31, Lifestyle 15, Leisure 14, Finance 14, Travel 12, Shopping 9), each with the annotator's
step list and the tools used, by name.

## Conversion (`cats.bench.external.livemcpbench`)

| CATS | LiveMCPBench |
|---|---|
| branch | server category (8) |
| toolkit | MCP server (config key, e.g. `yahoo-finance` -> `yahoo_finance`) |
| tool | one `tools/list` entry; name and `inputSchema` kept |
| task query | `Question` |
| gold slots | one per distinct annotated tool name; a slot holds every tool with that name |
| task kind | the task's domain (office, lifestyle, ...) |

* 94 of 95 tasks are kept. 5 annotated names match no tool in the pinned `tools.json`
  (`ddg-search`, `execute_command`, `get_maven_latest_version`, `open_document`,
  `trial_references`, from servers removed upstream); those slots are dropped, and one task whose
  only tool was among them is dropped. Mean 2.67 slots per task.
* The annotation is one valid solution path. Another server's tool could also work, so full cover
  is strict (a lower bound on how often the needed tools are present).
* 94 tasks cannot support a 40/20/40 split, so results use 5-fold cross-fitting: each task is
  tested once, with the router, reranker and conformal thresholds fitted on the other folds
  (about 50 training and 25 calibration tasks per fold).
