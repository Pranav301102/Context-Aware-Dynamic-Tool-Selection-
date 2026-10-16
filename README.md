# Context-Aware Dynamic Tool Selection

The system sits between an LLM application and its tools. For every request it decides which few tool
definitions the model should see, using calibrated hierarchical routing (branch → toolkit → tool)
with conformal coverage guarantees, then hands the task to a sub-agent that works only with those
tools and can ask for more with `request_tools`. This repository is the working vertical slice of
the design in *Project Workbook 1* (CMPE 295A).

Results and the list of improvements found while building it: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Quick start

```bash
pip install -e ".[dev,tokenize,embed,mcp]"
python scripts/fetch_tokenizer.py          # offline BPE tokenizer for token counts (optional)
python scripts/validate_data.py            # 12 branches, 477 tools, 960 tasks

cats stats
cats route "is anything slow in prod db right now" --enrich data/enrichment
cats bench --enrich data/enrichment        # every baseline vs the full system on the held-out split
pytest -q                                  # 30 tests (core + MCP end-to-end)
```

Every experiment writes `results/<name>.json`. Common flags: `--enrich data/enrichment`,
`--llm-router runs/llm_router`, `--method lac|aps`, `--alpha-branch`, `--alpha-tool`, `--k-max`,
`--calibration joint|independent`, `--seed`.

| Command | What it measures | Requirement |
|---|---|---|
| `cats bench` | B0–B4 baselines, the full system and the oracle: coverage, recall, k, tokens, latency, paired CIs | FR9, NFR1/2/4 |
| `cats ablate` / `scripts/ablation_final.py` | System variants on one split | Ch. 7 ablations |
| `cats validity --seeds 5` | empirical coverage vs. the conformal targets over several splits | NFR3 |
| `cats scale --sizes 100,...,10000` | tool-pool scaling with distractor domains and vendor clones | FR14, NFR5 |
| `cats cache` | prompt-cache cost of tool-ordering policies over multi-turn sessions | FR12, NFR6 |
| `cats run --llm heuristic\|anthropic:<model>\|vllm:<model>` | sub-agent loop with escalation and cost ledger | FR6, FR7 |
| `cats agent ...` | an external agent plays the model | FR9 end-to-end |

## Layout (maps to workbook Figure 3)

```
src/cats/
  types.py         typed contracts: ToolRecord, RouteDecision, RetrievalRequest/Result, RegistryEvent, TraceSpan
  registry.py      hierarchical registry branch → toolkit → tool, token costs, change events      FR1, FR11
  index/           BM25 + dense (BGE-M3 | WordLlama | LSA) hybrid index per level, auto-sync     FR2, FR11
  analyzer.py      ① context analyzer: query + history, vendor hints, multi-step decomposition
  router.py        ② hierarchical router: shared-weight conditional logit, conformal set (LAC/APS) FR3, FR8
  retriever.py     ③ tool retriever: hybrid candidates, learned reranker, CRC adaptive k, dedupe FR4
  assembler.py     ④ context assembler: request_tools first, canonical/append order, budget      FR5, FR12
  dispatcher.py    ⑤ dispatcher: sub-agent loop, request_tools escalation (max 2), cost ledger    FR6, FR7
  pipeline.py      Algorithm 1 end to end (CATS.fit / calibrate / select / escalate)
  calibration.py   split conformal (APS, LAC) and conformal risk control
  ledger.py        token/cost ledger, prefix-cache simulator, trace spans                         NFR6, NFR8
  enrich.py        tool-document enrichment with synthetic requests (Re-Invoke style)
  llm_router.py    LLM branch router as an extra router feature
  llm/             model backends: Anthropic, OpenAI-compatible (vLLM), heuristic stand-in       NFR9
  mcp/             MCP ingestor (tools/list + list_changed), the project's MCP gateway, demo server         FR2, FR10, FR11
  strategies.py    baselines B0–B4, the full system, oracle                                                   Table 8
  bench/           data splits, harness + bootstrap CIs, pool generator, experiments               FR9, FR14
  agent_cli.py     subagent-as-model harness (show / request-tools / call / done / score)
data/
  corpus/          12 branches, 70 toolkits, 477 tools modeled on real MCP servers
  tasks/           960 labeled requests (literal, paraphrase, multi, contextual, cross)
  distractors/     24 extra domains, 684 tools, for the scaling benchmark
  enrichment/      6 synthetic user requests per tool (generated from tool definitions only)
  SPEC.md          data format; gold labels are slots of equivalent tools
```

## No API key? Offline answer files stand in for the model

The build environment had no model API key, and Hugging Face and the tiktoken downloads were blocked.
Every model-shaped job was therefore done offline, through files and CLI commands
the harness controls:

* **Data**: the tool corpus, tasks, distractor domains and enrichment were written offline
  following `data/SPEC.md`. Enrichment never saw `data/tasks/`.
* **LLM router**: `cats.llm_router.export_batches` writes the routing prompt and queries;
  `runs/llm_router/scores_*.jsonl` were filled offline, which `--llm-router` loads as a router feature.
* **Model in the loop**: `python -m cats.agent_cli prepare --run R --mode cats|static|search`,
  then `show`, `request-tools`, `call` and `done` are run per task, and `score` bills every turn
  through the prefix-cache simulator.

With an API key, `cats run --llm anthropic:<model>` (or `vllm:<model>`) runs the same loop directly.

## Status against the workbook

Built: FR1–FR9, FR11–FR15 and an FR10 gateway tested in-process against a demo MCP server.
FR16 (fine-tuned router) is approximated by the learned conditional-logit router and the reranker.
Not built: LangGraph sub-agents, PostgreSQL/Qdrant storage (in-memory + JSON instead), Langfuse
tracing (spans are recorded in-process), BFCL/LiveMCPBench/ToolRet adapters (the corpus here is
synthetic), and the demo web app.
