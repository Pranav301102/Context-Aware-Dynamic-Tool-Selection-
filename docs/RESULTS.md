# CATS build: results and improvements

This report covers what was built from *Project Workbook 1*, what it measured on a 477-tool
catalog and on pools up to 10,000 tools, and what to change in the design. Every number comes
from a file under `results/` and can be regenerated with the commands in §8. All numbers are
after the fixes from an independent code review (listed in §7).

## 1. Setup

| Item | Value |
|---|---|
| Catalog | 12 branches, 70 toolkits, 477 tools modeled on real MCP servers; 96,774 tool-definition tokens under static loading |
| Tasks | 960 labeled requests: 216 literal, 312 paraphrase, 168 multi-step, 144 contextual (need history), 120 cross-branch |
| Split | stratified 40 / 20 / 40: 383 train (router, reranker), 191 calibration (half for the branch threshold, half for the tool cut-off), 386 test |
| Gold labels | *slots*: each step lists every tool that would serve it equally well (e.g. any of three web-search tools) |
| Scaling pools | real corpus + 24 distractor domains (684 tools) + vendor clones, 100 to 10,000 tools |
| Tokenizer | offline BPE (Anthropic SDK 0.34 `tokenizer.json`) as a proxy; tiktoken files were unreachable |
| Embedder | WordLlama static embeddings (bundled in its wheel); BGE-M3 was unreachable, so dense scores are a lower bound |
| Model in the loop | Claude Code subagents (the "token provider"); see §6 |

Main metric: **full cover**, the share of test requests whose selected tools cover every gold slot.
A request with full cover can be completed with the tools in the prompt. Also reported: **slot miss**
(the expected share of slots missed, which conformal risk control bounds), **k** (tools in the
prompt), tool tokens as % of static loading, and selection latency.

## 2. Headline

| Strategy (test n = 386) | Full cover [95% CI] | k | Tool tokens (% static) | p95 latency |
|---|---|---|---|---|
| B0 static loading | 1.000 | 477 | 100% | — |
| Workbook design as specified (APS sets, no enrichment) | 0.754 | 10.0 | 2.29% | 4 ms |
| B3 hybrid + learned reranker, top-10 (enriched index) | 0.920 | 10.0 | 2.18% | 3 ms |
| **Improved CATS** (LAC, enrichment, decomposition, end-to-end CRC) | **0.930** [0.904, 0.953] | 10.6 | 2.33% | 8 ms |
| **Improved CATS + LLM router feature** | **0.938** [0.912, 0.961] | 7.7 | **1.72%** | 8 ms |
| O oracle | 1.000 | 1.3 | 0.31% | — |

* Against flat reranked top-10 on the same enriched index, improved CATS gains +0.010
  (paired 95% CI [−0.003, 0.026]); this is **not significant** at 477 tools. With the LLM-router
  feature it gains **+0.021 [0.005, 0.036]** using 21% fewer tool tokens. CATS's routing pays off
  as the pool grows (§5).
* End to end with a subagent as the model on 36 held-out tasks (§6): a strong model completed 36/36
  with CATS and with static loading, but CATS used **70× fewer input tokens** (3.1K vs 215K per task)
  and cost **8.8× less**, counting static loading's cache discount. A smaller model (Haiku) completed
  **25/36 with static loading and 36/36 with CATS**.

## 3. What each change bought (`results/ablation_final.json`)

| Configuration | Full cover | Slot miss (target 0.05) | k | Branches in set | Branch cover (target 0.97) |
|---|---|---|---|---|---|
| Workbook design: APS sets, CRC cut-off | 0.754 | 0.200 (target infeasible; best 0.151) | 10.0 | 10.1 of 12 | 1.000 |
| + LAC sets instead of APS | 0.751 | 0.202 | 10.0 | **7.4** | 0.982 |
| + tool-document enrichment | **0.925** | 0.047 | 10.0 | 3.6 | 0.987 |
| + query decomposition (= improved CATS) | 0.930 | 0.044 | 10.6 | 3.6 | 0.987 |
| + LLM router as a feature | 0.938 | 0.038 | **7.7** | **1.25** | 0.982 |
| improved, alpha_tool = 0.10 | 0.881 | 0.082 | 4.5 | 3.6 | 0.987 |
| improved + LLM router, alpha_tool = 0.10 | 0.873 | 0.085 | 3.9 | 1.25 | 0.982 |

By request type (full cover), workbook design → improved CATS: paraphrase 0.61 → 0.95,
contextual 0.87 → 0.98, multi-step 0.70 → 0.87, cross-branch 0.64 → 0.78, literal 1.00 → 1.00.

**Coverage validity over five random splits** (`results/validity_enrich.json`,
`results/validity_independent.json`; conformal guarantees hold on average over calibration draws):

| Tool cut-off calibrated on | Mean branch cover (≥ 0.97) | Mean slot miss (≤ 0.05) | Splits over target | Mean k |
|---|---|---|---|---|
| the routed pipeline's own output (end to end, new default) | 0.984 | **0.052** | 2 of 5 | 8.0 |
| each stage separately, assuming perfect routing (workbook Algorithm 1) | 0.984 | **0.075** | 4 of 5 | 5.3 |

## 4. Improvements, in priority order

### 4.1 Enrich tool documents before anything else (largest gain)
Static embeddings and BM25 cannot match "do I need a brolly tomorrow" to a forecast tool.
Subagents wrote six user-style requests per tool, from the tool definition only (never from the
tasks). The index appends them to each tool's BM25 document and scores each tool by its
best-matching request (`cats/enrich.py`, `ToolIndex.score_all`).
**Full cover 0.751 → 0.925; paraphrase requests 0.61 → 0.95.** Every flat baseline gains too
(BM25@5 0.547 → 0.811), so enrichment is a property of the index, not of CATS.
*Plan change:* move tool-document enrichment from Tanisha's S6 sprint to S2/S3, and generate it for
every tool at ingestion (one LLM call per new or changed tool).
*Caveat:* the same model family wrote the tasks and the enrichment, so part of the gain may be
shared phrasing. Re-measure on ToolRet or LiveMCPBench queries, which no model wrote.

### 4.2 Calibrate the tool cut-off end to end, not per stage
Algorithm 1 calibrates the branch set and the tool cut-off separately; the cut-off then assumes
routing was perfect. Over five splits, that **misses 7.5% of slots against a 5% target** (4 of 5
splits over). Calibrating the cut-off on the routed pipeline's real output gives 5.2%. Use disjoint
calibration halves for the two thresholds (now the default). With ~95 requests per half, the
finite-sample term (1/(n+1) ≈ 0.01) is a large share of α = 0.05. Grow the calibration set to 500+
before claiming NFR3 in the paper.

### 4.3 Use LAC, not APS, for the branch set
The router is right about 85% of the time at top-1, but its softmax is sharp: when wrong it gives the
gold branch about 1e-4. APS then needs cumulative mass of about 0.9998 and keeps almost every branch
(10.1 of 12), so routing does nothing. LAC (keep branches with p ≥ 1 − q̂) meets the same coverage
target with 7.4 branches, and 3.6 once enrichment is on. Now the default (`method="lac"`).
Mortier et al.'s hierarchical conformal sets remain worth comparing in Workbook 2.

### 4.4 Report when the coverage target is infeasible
With the workbook's design, no cut-off reaches α = 0.05 (best achievable 0.151): conformal risk
control silently falls back to k_max, and NFR2 cannot hold. Even improved CATS sits at 0.062 achievable
with half the calibration set. The calibration report now carries `tool_target_feasible` and
`min_achievable_alpha_tool`, and `cats bench` prints them. Treat "infeasible" as a failing test.

### 4.5 Route multi-step requests per step
A rule-based splitter (`analyzer.decompose`) routes and ranks each step of "check the api pods and
open a jira ticket", interleaves the rankings round-robin, and gives 3 extra slots per extra step.
Merging by score alone did not help, because reranker scores are not comparable across queries.
Multi-step requests 0.83 → 0.87; cross-branch requests stay at 0.78, the weakest class. Next step:
plan-then-retrieve with one cheap LLM call that names the steps.

### 4.6 Collapse duplicate servers at scale
Real registries hold many servers wrapping the same API, and at 10,000 tools copies of one tool fill
the top-k. Moving identical tools (same name and parameters) behind the best-ranked copy
(`ToolRetriever._collapse_duplicates`) is part of every CATS number in §5. The first scaling run,
without it and with a generator bug, scored 0.43 at 10,000 tools; the final one scores 0.64.

### 4.7 Use an LLM router as a feature, and only when it pays
Subagents acting as the router put a gold branch first 99.6% of the time, against about 85% for the
embedding router. That is an upper bound, because the same model family wrote the tasks. Fed into
the conditional-logit router as one feature, it shrinks sets from 3.6 to 1.25 branches and tools from
10.6 to 7.7, with a significant gain over flat retrieval (§2); conformal calibration still applies.
The routing prompt costs about **1,070 tokens per request**, more than the ~600 tool tokens it
saves, and is now billed (`Prices.router_mult`). It pays only with a much cheaper router model and
a cached router prompt (the prompt is static). Better still, call it only when the embedding router
is unsure, a cascade that is not built yet.

### 4.8 Rethink the cache argument (FR12)
In 5-turn sessions of related requests (`results/cache_enrich.json`, $3/M input, cache reads 0.1×):

| Policy | $ per session | Cache-read share | Tool tokens per turn | Full cover |
|---|---|---|---|---|
| Static loading | 0.164 | 99% | 96,774 | 1.000 |
| CATS, canonical order (per-turn selection) | **0.045** | 24% | 2,324 | 0.925 |
| CATS, sticky (append-only within session) | 0.066 | 46% | 5,278 | 0.920 |

Canonical ordering makes identical selections byte-identical, but identical selections are rare.
Keeping earlier tools to protect the cache costs more than the cache saves. Even with a 99% cache
hit rate, static loading costs 3.7× more. *Recommendation:* keep per-turn selection, and put the cache
breakpoint after the stable part (system prompt + `request_tools`), not after the tools.

### 4.9 The MCP gateway only sees `request_tools`
MCP hosts do not send the user's request to tool servers. A gateway can therefore only select on
the capability the model describes in `request_tools`, which is baseline B6 (agent-initiated
search), not CATS routing. End to end, that mode took **47% more turns and cost 32% more per task**
than CATS in the 36-task round, and 47–49% more turns and 10–15% more cost in the three-model round
(§6b), for similar success. To get CATS behavior through MCP, pass the request in: a host-side
hook, a `set_task_context` tool the system prompt tells the model to call first (now in
`cats.mcp.launch`), or MCP sampling.

### 4.10 Scope the accuracy claim (Chapter 2) by model size
A strong model with all 477 tools in context completed 36/36 tasks; Haiku completed 25/36, and 36/36
with CATS. For frontier models the case for CATS is cost and context; for small and open-weight
models (the Qwen3-8B / vLLM path in the plan) it is also accuracy. The 60-task, three-model round
(§6b) confirms it: Haiku 44/60 static vs 53/60 with CATS (McNemar p = 0.035); Sonnet and Opus show
no significant difference.

### 4.11 Evaluation design
* Keep slot-based gold labels. Single-gold labels would turn correct choices among overlapping
  vendors into false failures.
* In scaling pools, a vendor clone counts as correct only when the user did not name the vendor.
  Vendors are identified the same way CATS's analyzer identifies them, matched on word boundaries,
  and removed from clone ids and descriptions (`bench/poolgen.py`).
* Use different model families to generate the data and to run it (style leakage, §4.1, §4.7).

## 5. Scaling (`results/scale_final.json`; no enrichment, because distractors have none)

| Tools | Branches | Test tasks | Static tokens | Flat RRF @5 | Flat reranked @5 | Flat reranked @10 | CATS | CATS − flat@10 | CATS tokens (% static) | CATS p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 27 | 62 | 21,316 | 0.823 | 0.839 | 0.887 | 0.887 | +0.000 | 12.61% | 6 ms |
| 250 | 33 | 127 | 49,272 | 0.827 | 0.827 | 0.898 | 0.890 | -0.008 | 5.16% | 7 ms |
| 1,000 | 36 | 386 | 176,380 | 0.567 | 0.580 | 0.712 | 0.728 | +0.016 | 1.21% | 10 ms |
| 2,500 | 36 | 386 | 432,312 | 0.487 | 0.562 | 0.676 | 0.707 | +0.031 | 0.48% | 15 ms |
| 5,000 | 36 | 386 | 857,655 | 0.443 | 0.549 | 0.663 | 0.679 | +0.016 | 0.24% | 24 ms |
| 10,000 | 36 | 386 | 1,732,578 | 0.409 | 0.482 | 0.617 | 0.642 | +0.026 | 0.12% | 30 ms |

Pools below 477 tools use a task subset whose gold tools fit, so those rows are not directly
comparable with the larger ones.

* **The workbook's hypothesis holds at scale, and only there.** At (nearly) equal k, CATS gains
  nothing over flat reranked top-10 at 100–250 tools. It leads by 1.6 to 3.1 points from 1,000 tools
  up. For small catalogs, load everything when it fits (`bypass_tokens`) or use flat retrieval.
* Latency stays far inside NFR4 (p95 ≤ 30 ms against 400 ms). Building, indexing, fitting and
  calibrating a 10,000-tool pool takes about 30 s on 4 CPU cores, and a changed tool is searchable in
  milliseconds (NFR5).
* Absolute coverage at 10,000 tools (0.64) is low because distractors are near-duplicates and nothing
  is enriched. Closing that gap needs a stronger embedder (BGE-M3) and enrichment for every tool,
  not more routing.

## 6. End to end with a subagent as the model (`results/e2e_subagent.json`)

36 held-out tasks (seed 1), the same tasks in every mode. Each turn is billed through the
prefix-cache simulator at $3/M input and $15/M output.

| Model | Mode | Success | Tool tokens per prompt | Input tokens per task | Turns | Escalation rate | $ per task |
|---|---|---|---|---|---|---|---|
| Strong subagent | Static loading (B0) | 36/36 | 96,774 | 215,393 | 2.22 | 0% | 0.0768 |
| Strong subagent | Improved CATS | 36/36 | 1,061 | 3,073 | 2.28 | 6% | 0.0087 |
| Strong subagent | Agent-initiated search only (B6) | 36/36 | 1,187 | 3,729 | 3.36 | 100% | 0.0115 |
| Haiku subagent | Static loading (B0) | 25/36 | 96,774 | 188,463 | 1.94 | 0% | 0.0683 |
| Haiku subagent | Improved CATS | 36/36 | 1,148 | 3,437 | 2.39 | 11% | 0.0096 |

* **Smaller model: CATS fixes accuracy, as the workbook claims.** Haiku with the full catalog
  completed 25/36 tasks. In 7 of the 11 failures it called no tool at all; in the other 4 it called
  the wrong tool or skipped a step. With CATS it completed 36/36, including all 11. n is small (36
  tasks), but every flip went the same way.
* **Strong model: same success, 8.8× cheaper.** Static loading re-sends the catalog every turn;
  prompt caching discounts it but does not remove it.
* **Agent-initiated search (B6)** also reached 36/36, but needed a search on every task (§4.9).
* CATS routing misses were recovered with `request_tools` (2 tasks for the strong model, 4 for Haiku),
  and no run produced an invalid tool call.

## 6b. Three models, 60 tasks, four modes (`results/e2e_models.json`, `paper/tables/p_e2e.tex`)

The follow-up the workbook plan asks for: the same 60 held-out tasks (stratified by domain and kind)
in four modes, with Haiku 4.5, Sonnet 5.5 and Opus 5.5 each playing the agent, 720 sessions in all.
Every model is billed at $3/M input and $15/M output through the cache simulator, so the dollar
column compares modes, not models. "Excl. blocked" drops six tasks where the sandbox's permission
classifier refused a simulated write (a PagerDuty resolve, a Drive share, a bulk SQL update) in at
least one run; the agent could not finish those through no fault of the selection.

| Model | Mode | Success | Excl. blocked | Input tokens per task | Turns | $ per task |
|---|---|---|---|---|---|---|
| Haiku 4.5 | Static (B0) | 44/60 | 40/54 | 232,644 | 2.40 | 0.0786 |
| Haiku 4.5 | Search only (B6) | 53/60 | 50/54 | 4,911 | 3.78 | 0.0134 |
| Haiku 4.5 | CATS | 53/60 | 51/54 | 5,439 | 2.57 | 0.0121 |
| Haiku 4.5 | CATS-Jev | 54/60 | 51/54 | 5,386 | 2.62 | 0.0120 |
| Sonnet 5.5 | Static (B0) | 57/60 | 51/54 | 242,336 | 2.50 | 0.0816 |
| Sonnet 5.5 | Search only (B6) | 58/60 | 53/54 | 5,009 | 3.90 | 0.0142 |
| Sonnet 5.5 | CATS | 57/60 | 52/54 | 5,555 | 2.63 | 0.0125 |
| Sonnet 5.5 | CATS-Jev | 55/60 | 51/54 | 5,016 | 2.55 | 0.0118 |
| Opus 5.5 | Static (B0) | 57/60 | 51/54 | 245,569 | 2.53 | 0.0826 |
| Opus 5.5 | Search only (B6) | 58/60 | 52/54 | 4,906 | 3.85 | 0.0138 |
| Opus 5.5 | CATS | 56/60 | 50/54 | 5,447 | 2.58 | 0.0121 |
| Opus 5.5 | CATS-Jev | 55/60 | 50/54 | 5,172 | 2.58 | 0.0117 |

Paired against static loading (exact McNemar on discordant tasks):

| Model | CATS wins / static wins | p | CATS-Jev wins / static wins | p |
|---|---|---|---|---|
| Haiku 4.5 | 12 / 3 | **0.035** | 14 / 4 | **0.031** |
| Sonnet 5.5 | 1 / 1 | 1.000 | 1 / 3 | 0.625 |
| Opus 5.5 | 0 / 1 | 1.000 | 0 / 2 | 0.500 |

* **The small model gains, significantly.** Haiku with all 477 tools failed 16 tasks: in 7 it
  stopped after part of the request, in 5 it called no tool, in 4 it picked a wrong tool. CATS cut
  that to 7 failures, 3 of them blocked writes. This replicates the 36-task round (25/36 vs 36/36)
  with a significance test.
* **Larger models do not care about the mode.** Sonnet and Opus solve 55–58 of 60 in every mode;
  no difference is significant. For them CATS is a cost and context-window argument.
* **Selection is 6–7× cheaper.** CATS sends 43–45× fewer input tokens per task and costs 6.5–6.8×
  less than static loading; across all selective modes the saving is 5.8–7.1×.
* **Search only (B6) is a strong baseline.** It matches or beats CATS by one or two tasks because
  the model writes a capability query per step, but takes about 1.3 more turns per task (47–49% more)
  and costs 10–15% more. In a deployed agent each extra turn is a full model round trip.
* **The typed layer did not carry over end to end.** CATS-Jev improved offline coverage
  (0.930 → 0.943, §10) but was +1 task with Haiku and −2 / −1 with Sonnet / Opus, all within noise. It is 1–6%
  cheaper per task than CATS: Jev calls (about 1,450 tokens on 67% of requests) are billed at
  $42 per billion tokens. An earlier version of the scorer billed them at the main model's $3/M,
  which made CATS-Jev look 10–16% more expensive; fixed in `agent_cli.score_run`
  (`tests/test_agent_cli.py`).
* **Where the selective modes still fail.** Two tasks account for most CATS / CATS-Jev misses with
  the larger models. In both, the selection left out the gold tool but kept a close substitute
  (`postgres.execute_sql` on `information_schema` instead of `postgres.get_object_details`;
  `google_maps.maps_search_places` instead of `brave_search.brave_local_search`), and the model used
  the substitute rather than calling `request_tools`. Escalation only helps when the model notices
  a missing capability. Sonnet and Opus chose the same maps substitute with every tool loaded, so
  part of this is gold slots that are narrower than what a reasonable agent would accept.
* **Audit.** The harness refuses `call`, `request-tools` and `done` before `show`. No session lacked
  a `show`; median gaps between actions are 6.5–11.8 s; Haiku runs were redone after an earlier
  attempt used automation scripts (discarded). Tool results are opaque strings, so agents filled IDs
  with placeholders and took conditional steps without checking the condition; success measures
  tool choice, not task outcome.

## 10. Jev-style typed decision layer (`results/jev_experiment.json`)

Jev (TypeSafe AI, 2026) is a "System One" model. Instead of writing text, it answers typed questions
(`noul` yes/no, `choice`, `score`) with calibrated probabilities, in about 150 ms at $42 per billion
input tokens. CATS asks it, for each request:

* 12 `noul` questions, "does this request need a tool from branch X?";
* one `score` question, "how many steps?" (1, 2, 3+).

The answers are used in three ways: as a router prior that is consulted only when the embedding
router is unsure, to size k by the step count, and to repair branches the cut-off missed
(`cats/jev.py`, `pipeline._route`, `pipeline._repair_fn`; see docs/components/orchestration_and_routing.md).

**Claude played Jev.** Sonnet subagents answered all 960 requests, with a rule against generating
answers with code; their answers have 24–30 distinct graded values per batch. A first attempt with
Haiku answered with keyword rules (one batch used only the values 0.0 and 0.9). It is kept as the
"keyword" baseline below.

| Configuration | Full cover | Slot miss | k | Tool tokens (% static) | Calls | Routing $ per 1k requests | Cross-domain |
|---|---|---|---|---|---|---|---|
| improved CATS (no Jev) | 0.930 | 0.044 | 10.6 | 2.33% | 0.00 | 0.000 | 0.78 |
| + Jev prior, always called | 0.935 | 0.042 | 7.8 | 1.73% | 1.00 | 0.061 | 0.76 |
| + Jev prior, cascade-gated | 0.935 | 0.042 | 7.8 | 1.73% | 0.67 | 0.041 | 0.76 |
| + cascade + Jev step count | 0.938 | 0.041 | 8.0 | 1.78% | 0.67 | 0.041 | 0.78 |
| + cascade + steps + branch repair (CATS-Jev) | 0.943 | 0.037 | 8.1 | 1.79% | 0.67 | 0.041 | 0.80 |
| CATS-Jev, Jev always called | 0.946 | 0.036 | 8.1 | 1.80% | 1.00 | 0.061 | 0.82 |
| CATS-Jev with keyword 'Jev' (heuristic) | 0.935 | 0.040 | 11.5 | 2.52% | 0.67 | 0.041 | 0.80 |
| LLM router prior, always called | 0.938 | 0.038 | 7.7 | 1.72% | 1.00 | 3.216 | 0.74 |

* **CATS-Jev (gated prior + steps + repair): 0.943 full cover against 0.930**, with 23% fewer tool
  tokens. Jev is asked on 67% of requests, at $0.041 per thousand requests.
* **The LLM router prior** (an Opus-class subagent, billed at $3/M) reaches 0.938 at $3.22 per
  thousand requests, so Jev is better and about 79× cheaper.
* **The cut-off becomes feasible.** Sharper branch probabilities let the risk-controlled cut-off reach
  α = 0.05 on the half-size calibration split; without Jev it cannot.
* **Model judgment matters.** Keyword answers reach 0.935, but with k = 11.5, and the cut-off
  target stays infeasible.
* **Latency is not comparable across rows.** These runs shared 4 cores with about 20 concurrent
  agents. Jev's 150 ms call time is a published figure, not measured here.

## 11. Public benchmarks (`results/external_*.json`; docs/components/evaluation.md)

The same pipeline, unchanged and without enrichment, on benchmarks no project model wrote
(pinned commits, SHA-256 checked). Flat@k is hybrid retrieval + reranker at CATS's mean k.

| Benchmark | Tools / tasks | BM25@5 | Flat@k | CATS | CATS − Flat@k [95% CI] | McNemar p (Holm) |
|---|---|---|---|---|---|---|
| BFCL v4, single-turn pooled | 3,258 / 1,011 test | 0.822 | 0.883 | 0.893 | +0.010 [+0.002, +0.019] | 0.062 |
| BFCL v4, no decomposition | same | 0.822 | 0.883 | 0.883 | +0.000 | 1.000 |
| BFCL v4, strict labels | same | 0.644 | 0.765 | 0.766 | +0.001 | 1.000 |
| LiveMCPBench (5-fold cross-fit) | 525 / 94 | 0.191 | 0.255 | 0.266 | +0.011 [−0.043, +0.064] | 1.000 |

* **Outside our catalog CATS ties a strong flat retriever.** On BFCL the whole gain is query
  decomposition; the slot-miss target is infeasible on its calibration split (best 0.11).
* **LiveMCPBench is retrieval-limited:** flat top-50 covers every needed tool for 51% of tasks, and
  25 calibration tasks per fold are too few for CATS to narrow branches.
* MCP-Bench, MCP-Zero's query set and ToolRet could not be downloaded here (URLs tried are in
  `data/external/README.md`).

## 12. Tool poisoning (`results/security_eval.json`; docs/components/security_and_learning.md)

20 poisoned look-alike tools in 8 attack families, 78 affected test requests.

| Setting | Poisoned tool in prompt | Benign cover |
|---|---|---|
| CATS, no scanner | 37.2% of affected requests | 0.872 |
| Flat top-10, no scanner | 32.1% | 0.821 |
| CATS + ingestion scanner (18/20 quarantined) | 7.7% (evasive family only) | 0.872 |

* The scanner flagged 0 of 1,161 benign tools; it catches 6 families fully, keyword stuffing 96%,
  and the plain-language "evasive" family never.
* Without the scanner, `request_tools` surfaces the poisoned twin for 18 of 20 targets.

## 13. Calibration under drift (`results/calibration_loop.json`)

Two branches withheld, then added mid-stream; three runs; branch-coverage target 0.95.

| Thresholds | Coverage after drift | Branches per set |
|---|---|---|
| Fixed | 0.942 (0.907 worst run) | 1.89 |
| Adaptive (ACI) | 0.950 (all runs within 0.01) | 2.92 |
| Router fine-tuning only | 0.973 (over-covers) | 2.95 |
| Full loop: ACI + fine-tuning | 0.949 | 2.36 |

Labels inferred from 882 logged sessions match gold branches with 0.99 precision, so the loop can
learn from production logs without annotation.

## 7. Code-review fixes behind these numbers

An independent review found nine issues; all are fixed and covered by tests (`tests/test_core.py`):

* a crash when a request names a vendor whose toolkit was removed;
* LLM-router tokens that were never billed;
* a non-monotone tool cut-off for interleaved rankings, which conformal risk control assumes is
  monotone (CRC now also requires every more lenient threshold to be safe);
* newly requested tools evicted first when over the token budget;
* the reranker trained on in-sample router probabilities (now cross-fitted);
* both thresholds calibrated on the same data (now disjoint halves);
* branch coverage measured on the union over sub-requests instead of the conformal set;
* a tool's version that ignored its place in the hierarchy;
* in the scaling generator, vendor detection by substring and clone ids or descriptions that kept the
  source vendor's name.

## 8. Threats to validity

* Tools, tasks, labels, enrichment and LLM-router scores were all generated by Claude subagents.
  Absolute numbers are optimistic; the comparisons between strategies are the useful part.
* Dense retrieval used static WordLlama embeddings because the BGE-M3 download was blocked; absolute
  retrieval quality is a lower bound.
* Token counts use a proxy BPE tokenizer. Cost comes from a simulated prefix cache, not a provider bill.
* Latencies were measured on a shared 4-core container while other jobs ran, so treat them as upper bounds.
* The end-to-end runs use 36 (first round) and 60 (three-model round) tasks per mode: enough for
  the small model's gain, not for one- or two-task differences. Tool results are simulated.

## 9. Reproduce

```bash
cats bench --enrich data/enrichment --name bench_final
cats bench --enrich data/enrichment --llm-router runs/llm_router --name bench_final_llm
python scripts/ablation_final.py
cats validity --enrich data/enrichment --seeds 5
cats validity --enrich data/enrichment --seeds 5 --calibration independent --name validity_independent
cats scale --sizes 100,250,1000,2500,5000,10000 --name scale_final
cats cache --enrich data/enrichment --session-len 5
python -m cats.agent_cli score e2e_static e2e_cats e2e_search e2e_haiku_static e2e_haiku_cats
python scripts/paper_tables.py   # scores the 12 b2_<model>_<mode> runs -> results/e2e_models.json
```
