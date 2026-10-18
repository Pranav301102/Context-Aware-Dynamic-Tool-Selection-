# Evaluation, benchmarking and metrics

Owner in the team plan: **Dhruv Sachin Jain** (evaluation, benchmarking and metrics). Workbook
requirements: FR9, FR14, NFR1–NFR4, NFR6, NFR8; deliverables DL5 (evaluation harness: benchmark
adapters, cost ledger, one-command reproduction) and DL6 (scaling benchmark); risks D3 (benchmarks)
and D6 (model drift).

This component decides what a number in the paper means and makes every number reproducible. All
strategies run through one harness on the same tasks with the same token counter, so only tool
selection differs (workbook §7.2).

```
 own catalog (477 tools, 960 tasks) ─┐
 scaling pools (100 … 10,000 tools) ─┼─► split: train / calibration / test     bench/data.py, external/run.py
 external benchmarks (LiveMCPBench,  ┘   (or K-fold cross-fitting)
   BFCL v4; pinned, converted)
                │
                ▼
 strategies B0–B4, CATS, oracle ──► harness.evaluate: one Record per task     bench/harness.py, strategies.py
                │                     (cover, recall, k, tokens, latency, route)
                ▼
 summaries + bootstrap CIs ──► paired tests (McNemar, bootstrap, Holm)          bench/stats.py
                │
                ▼
 results/*.json ──► scripts/make_figures.py ──► paper/figures/*.pdf|png
                └─► scripts/make_tables.py  ──► paper/tables/*.tex

 subagent-as-model sessions (runs/) ──► agent_cli.score_run (prefix-cache billing)
                                    ──► scripts/score_e2e.py (paired McNemar per model)
```

## 1. Code map

| File | What it does |
|---|---|
| `src/cats/bench/harness.py` | `evaluate` (one `Record` per task), `summarize` (means, 2,000-resample bootstrap CIs, per-kind breakdown), `paired_diff` |
| `src/cats/bench/stats.py` | exact McNemar, Wilson intervals, paired bootstrap with p-values, Holm–Bonferroni, `compare_to_reference` |
| `src/cats/bench/data.py` | task loading, stratified 40/20/40 split |
| `src/cats/bench/experiments.py` | baselines, ablation variants, coverage validity, scaling, prompt-cache sessions |
| `src/cats/bench/poolgen.py` | scaling pools with distractor domains, vendor clones and label widening |
| `src/cats/bench/external/` | adapters (LiveMCPBench, BFCL), auto-hierarchy, cross-fitting runner |
| `src/cats/strategies.py` | B0–B4, CATS, oracle |
| `src/cats/ledger.py` | token/cost ledger, prefix-cache simulator, trace spans |
| `src/cats/agent_cli.py` | subagent-as-model harness and per-turn billing |
| `scripts/fetch_external.py` | downloads pinned benchmark files, checks SHA-256, converts |
| `scripts/score_e2e.py` | re-scores end-to-end runs and tests modes against each other per task |
| `scripts/make_figures.py`, `scripts/make_tables.py` | paper figures (PDF + PNG) and booktabs tables |
| `scripts/reproduce_all.sh` | one command for every results file, figure and table (`--quick` for a smoke run) |
| `tests/test_evaluation.py` | offline tests for all of the above |

## 2. Metrics

### 2.1 Selection level (no model in the loop)

Gold labels are **slots**: each step of a task lists every tool that would serve it equally well
(`data/SPEC.md`). For a selected set *S* and gold slots *G₁ … Gₘ*:

| Metric | Definition | Why |
|---|---|---|
| **Full cover** | 1 if every slot has a tool in *S* (∀ i: S ∩ Gᵢ ≠ ∅), else 0; reported as the mean over tasks | the request can be completed with the tools in the prompt; the headline metric |
| **Slot recall** / **slot miss** | share of slots covered / 1 − slot recall | the loss conformal risk control bounds (`calibration.slot_miss`); recall@k of NFR2 |
| Precision | \|S ∩ ∪Gᵢ\| / \|S\| | how much of the prompt is useful |
| **k** | \|S\| | prompt size in tools |
| **Tool tokens**, % static | tokens of the selected definitions (`Registry.total_tokens`), and as a share of loading every tool | NFR1 |
| Router tokens | prompt tokens of LLM routing calls (`Selection.overhead_tokens`), billed at `Prices.router_mult` | NFR1 says "router overhead included" |
| Branch cover | share of tasks whose calibrated branch set contains every gold branch (measured on the conformal set, not the union over sub-requests) | the first conformal guarantee |
| Branches in set, fallback rate | mean conformal-set size; share of requests that fell back to flat search | how much routing narrows |
| Latency p50 / p95 | wall time of `select()` per request, `time.perf_counter` | NFR4 |

Every metric is also broken down by task kind (literal, paraphrase, multi, contextual, cross; or the
external benchmark's own categories).

Token counts use one counter for every strategy (`cats.tokens`): tiktoken `o200k_base` when cached,
otherwise a Hugging Face BPE file (the Anthropic SDK 0.34 tokenizer in this build), otherwise a regex
approximation. A tool's cost is the JSON of its definition as the model sees it
(`ToolRecord.spec()`), the same measure as workbook Listing 8.1.

### 2.2 Cost: the prefix-cache simulation (NFR6)

`ledger.PrefixCache` models provider prompt caching: a request re-uses the longest prefix, cut at
tool-block boundaries, that an earlier request already sent, if that prefix is at least 1,024 tokens.
Cached reads cost 0.1× the input price, first writes 1.25×, the uncached suffix (system prompt,
conversation) 1×, output 5× ($3/M input and $15/M output by default, `ledger.Prices`). The assembler
puts `request_tools` and the tools first in a canonical order (FR12), so identical selections are
byte-identical and cacheable. This is a simulation for comparing strategies, not a provider bill.
It is used in two places:

* `cats cache` (`experiments.cache_sessions`): 40 five-turn sessions of related requests, comparing
  static loading, per-turn canonical selection and "sticky" append-only selection;
* `agent_cli.score_run`: every model turn of a subagent run (each `show`, `request-tools` and `call`
  event) is re-billed with the tools loaded at that moment, plus the router tokens.

### 2.3 End to end (model in the loop)

| Metric | Definition (`agent_cli.score_run`) |
|---|---|
| **Success** | the model called a tool from every gold slot and made no invalid call (a tool not in its prompt) |
| Tools present | the final loaded set covered every slot (selection was sufficient) |
| Escalation rate | share of tasks with at least one `request_tools` call |
| Turns | model turns billed (shows + escalations + calls) |
| Input tokens per task | uncached + cache-write + cache-read tokens over all turns |
| $ per task | the prefix-cache bill above, router calls included |

Tools are simulated (`dispatcher.simulate_tool` echoes the arguments), so success measures tool
choice and argument intent, not the tool's real effect.

## 3. Strategies (workbook Table 8)

| ID | Strategy | Implementation |
|---|---|---|
| B0 | Static loading: every tool, every prompt | `StaticLoading` |
| B1 | Flat BM25 top-k (k = 5) | `FlatTopK(kind="bm25")` |
| B2 | Flat dense top-k (RAG-MCP style), WordLlama embeddings | `FlatTopK(kind="dense")` |
| B3 | Hybrid BM25 + dense with reciprocal-rank fusion, top-5; and with the learned reranker CATS uses, top-5 and top-10 | `FlatTopK(kind="hybrid")`, `FlatReranked` |
| B3 @ k = CATS | flat reranked at CATS's mean k (external runs), so routing is compared at equal prompt size | `run.py`, `B3_hybrid_rerank@k=CATS` |
| B4 | Two-stage: top-3 servers by summary match, then top-5 tools inside them (MCP-Zero / LiveMCPBench retrieval agent) | `TwoStage` |
| B5 | Hierarchy of LLM agents (AnyTool) | not built (needs paid model calls; workbook plans a fixed subset) |
| B6 | Agent-initiated search: no tools at first, only `request_tools` | subagent harness `--mode search` |
| B7 | Provider-hosted tool search | not built (needs provider API access) |
| CATS | calibrated branch set + reranked, risk-controlled cut-off (+ optional LLM-router feature) | `CATSStrategy` |
| O | Oracle: one gold tool per slot | `Oracle` |

B3 uses the CATS reranker deliberately: the difference between B3 and CATS is then routing and the
calibrated cut-off, not the ranking model. Static loading and the oracle bound every metric from
above (cover 1.0); the oracle also bounds tokens from below.

## 4. Data and splits

* **Own catalog.** 12 branches, 70 toolkits, 477 tools; 960 tasks (216 literal, 312 paraphrase, 168
  multi, 144 contextual, 120 cross). Enrichment (six synthetic requests per tool) was generated from
  tool definitions only; its writers never saw `data/tasks/`.
* **Split** (`bench/data.split`): stratified by (first gold branch, kind), 40 % train (router,
  reranker), 20 % calibration, 40 % test; seed 0 for the headline numbers. Calibration is halved
  (even/odd) so the branch threshold and the tool cut-off use disjoint data
  (`CATSConfig.split_calibration`), which the exchangeability argument of conformal risk control
  needs.
* **Cross-fitting** (`external/run.py --protocol crossfit`): for benchmarks too small to split, K
  folds stratified by first gold branch; each fold is the test set once, the rest is split 2:1 into
  training and calibration data; records of all folds are pooled so every task is tested exactly once.
  Thresholds differ per fold, so the pooled result is an average over K calibrations.
* **Train-only cold start** (`--pseudo`): one pseudo-request per tool (the first sentence of its own
  description) can be added to the training data, never to calibration or test. It did not help on
  LiveMCPBench (full cover 0.245 vs 0.266 without), so it is off by default.

## 5. Conformal validity protocol (NFR3)

Conformal guarantees are **marginal over calibration draws**: a single split can miss the target
by chance. `cats validity --seeds 5` therefore rebuilds the split with five seeds, re-fits and
re-calibrates, and reports per split and on average:

* branch cover on the test set vs the target 1 − α_branch = 0.97;
* slot miss (expected fraction of gold slots missed) vs the target α_tool = 0.05;
* mean k, branch-set size, the calibrated thresholds, and whether the tool target was feasible on
  the calibration data (`tool_target_feasible`, `min_achievable_alpha_tool`).

Two calibration modes are compared (`results/validity_enrich.json`, `validity_independent.json`):

| Tool cut-off calibrated on | Mean branch cover (≥ 0.97) | Mean slot miss (≤ 0.05) | Splits over target | Mean k |
|---|---|---|---|---|
| the routed pipeline's own output (end to end, default) | 0.984 | **0.052** | 2 of 5 | 8.0 |
| each stage separately, assuming perfect routing (Algorithm 1) | 0.984 | **0.075** | 4 of 5 | 5.3 |

Reading: the end-to-end mean is within sampling error of the target; per-stage calibration is not.
With about 95 requests per calibration half the finite-sample term 1/(n + 1) ≈ 0.01 is a fifth of α,
and split-to-split spread is large (0.035 to 0.069). Before NFR3 is claimed in the paper, grow the
calibration set to 500+ and report the mean over at least 20 splits with its standard error. An
infeasible target (CRC falls back to k_max) must be reported as a failure, never averaged away.

## 6. Scaling benchmark (FR14, NFR5, DL6)

`cats scale --sizes 100,250,1000,2500,5000,10000` builds pools with `bench/poolgen.build_pool`
from three sources, in order: (1) the gold tools of the evaluated tasks, (2) the rest of the real
catalog and 24 distractor domains (684 tools in `data/distractors/`), (3) **vendor clones**: copies
of real toolkits under new vendor names with lightly reworded descriptions, which is how public MCP
registries grow (many servers wrapping one API).

**Label widening.** A clone is functionally equivalent to its source, so it fills any slot its
source fills, *unless the user named the source vendor* (then only that vendor is correct). Vendors
are detected exactly as CATS's analyzer detects them (rare words of toolkit ids and names, matched on
word boundaries), and the source vendor's name is scrubbed from clone ids, names and descriptions,
so a request naming a vendor cannot match its clones. Below 477 tools, pools use a task subset whose
gold tools fit in 60 % of the pool, so those rows are not comparable with the larger ones.

Results (`results/scale_final.json`, no enrichment because distractors have none; figure
`paper/figures/scaling.pdf`):

| Tools | Flat reranked @10 | CATS | CATS − flat | CATS tokens (% static) | CATS p95 |
|---|---|---|---|---|---|
| 100 | 0.887 | 0.887 | +0.000 | 12.61 % | 6 ms |
| 1,000 | 0.712 | 0.728 | +0.016 | 1.21 % | 10 ms |
| 10,000 | 0.617 | 0.642 | +0.026 | 0.12 % | 30 ms |

For the DL6 public release, the pool generator, the distractor domains and a datasheet (sources,
generation procedure, label-widening rule, known biases from §12) are what must ship; the generator
is deterministic for a seed, so pools need not be stored.

## 7. Subagent-as-model harness (FR9 end to end)

No model API was available, so a Claude Code subagent plays the model and CATS plays everything else
(`src/cats/agent_cli.py`):

```bash
python -m cats.agent_cli prepare --run R --mode cats|static|search --n 36 --seed 1 --enrich data/enrichment
python -m cats.agent_cli show --run R --task ID              # the exact prompt the model sees
python -m cats.agent_cli request-tools --run R --task ID --capability "..."   # ≤ 2 (cats), ≤ 3 (search)
python -m cats.agent_cli call --run R --task ID --tool NAME --args '{...}'
python -m cats.agent_cli done --run R --task ID
python scripts/score_e2e.py R1 R2 ... --out results/e2e_paired.json
```

Same tasks, same model, same prompts; only tool selection differs. `static` prints the whole catalog
once per run (it stays in the subagent's context) and bills it on every turn; `search` starts with no
tools and only `request_tools` (B6). `score_e2e.py` pairs runs of the same model and experiment by
task id and reports exact McNemar on success (Holm-corrected) and paired bootstrap CIs on $ and input
tokens. On the first batch (36 tasks, seed 1): the strong model succeeded 36/36 in every mode, with
CATS 8.8× cheaper than static loading; Haiku succeeded 25/36 with static loading and 36/36 with CATS —
11 discordant tasks, all in CATS's favour, **exact McNemar p = 0.00098** (0.0039 after Holm over the
four mode comparisons; Wilson 95 % intervals [0.53, 0.82] vs [0.90, 1.00]); with CATS, Haiku cost
$0.059 less per task (paired bootstrap 95 % CI $0.044 to $0.082). `score_e2e.py --all` also scores
the second batch (`runs/b2_*`, 60 tasks, seed 2, three model sizes); at the time of writing, Haiku
CATS vs static was 12 : 3 discordant (McNemar p = 0.035 unadjusted, 0.42 after Holm over all 13 mode
comparisons), and Sonnet and Opus were tied with static loading (1 : 1 and 0 : 1) at 6.5 to 7× lower
cost. A small-model accuracy claim therefore rests on one significant batch so far. Output: `results/e2e_paired.json`.

## 8. Statistics (`src/cats/bench/stats.py`)

| Question | Method | Notes |
|---|---|---|
| Uncertainty of a mean (cover, recall, tokens) | percentile bootstrap over tasks, 2,000 resamples (`harness.summarize`) | tasks are the sampling unit |
| Uncertainty of a success rate with small n | **Wilson score interval** | correct near 0 and 1 (36/36 → [0.904, 1]) where the normal interval collapses |
| Is strategy A's binary outcome better than B's on the same tasks? | **exact McNemar** (binomial on discordant pairs), two-sided; mid-p optional | only discordant tasks carry information; exact for any n (big-integer arithmetic) |
| Is the mean paired difference non-zero (recall, $, tokens)? | **paired bootstrap** of mean(A − B), 10,000 resamples, CI and two-sided p | resamples tasks, keeping the A/B correlation |
| Several baselines at once | **Holm–Bonferroni** step-down on the family of p-values | controls the family-wise error rate; uniformly more powerful than Bonferroni |

`compare_to_reference(records, "CATS")` runs all of these against CATS and is what the external
benchmark results use. Power: on the 386-task test split a 1-point difference in full cover is
undetectable unless nearly all discordant tasks go one way (CATS vs flat@10 was +0.010, CI
[−0.003, 0.026]); claims of "no difference" need equivalence margins, not non-significance.

## 9. External benchmarks (FR9, risk D3)

The project's weakest point is that every task was written by Claude subagents. Two open benchmarks
were obtained, pinned to a commit with SHA-256 per file, and converted (`data/external/`):

| | LiveMCPBench | BFCL v4 (single-turn, pooled) |
|---|---|---|
| Source | `icip-cas/LiveMCPBench@36a51a1` | `ShishirPatil/gorilla@f7cf735` |
| Tools | 525 MCP tools, 69 servers, 8 categories | 3,258 function variants (1,853 names) incl. distractor functions |
| Hierarchy | branch = category, toolkit = server | generated: k-means branches (32), namespaces as toolkits |
| Tasks | 94 human-written multi-step tasks (2.67 slots) | 2,501 questions (1.13 slots), incl. 1,351 live user queries |
| Gold | annotator's tool list; slot = all tools with that name | ground-truth call; slot = all variants of that name |
| Protocol | 5-fold cross-fitting | 40/20/40 split, seed 0 |

Results (`results/external_livemcpbench.json`, `results/external_bfcl.json`; figure
`paper/figures/external.pdf`, table `paper/tables/external.tex`):

| Strategy | LiveMCPBench full cover [95 % CI] | recall | k | BFCL full cover [95 % CI] | recall | k | BFCL tokens (% static) |
|---|---|---|---|---|---|---|---|
| B0 static loading | 1.000 | 1.000 | 525 | 1.000 | 1.000 | 3,258 | 592,381 (100 %) |
| B1 BM25 @5 | 0.191 [0.106, 0.277] | 0.384 | 5 | 0.822 [0.799, 0.844] | 0.843 | 5 | 965 (0.16 %) |
| B3 hybrid + reranker @5 | 0.181 [0.106, 0.255] | 0.394 | 5 | 0.829 [0.806, 0.850] | 0.852 | 5 | 971 (0.16 %) |
| B4 two-stage @3×5 | 0.181 [0.106, 0.255] | 0.377 | 5 | 0.534 [0.503, 0.565] | 0.555 | 4.9 | 835 (0.14 %) |
| B3 hybrid + reranker @10 | 0.245 [0.160, 0.330] | 0.483 | 10 | 0.883 [0.864, 0.902] | 0.903 | 10 | 1,976 (0.33 %) |
| B3 hybrid + reranker @ k = CATS | 0.255 [0.170, 0.351] | 0.491 | 10.8 | 0.883 [0.864, 0.902] | 0.903 | 10 | 1,976 (0.33 %) |
| **CATS** | **0.266** [0.181, 0.351] | 0.497 | 11.0 | **0.893** [0.874, 0.912] | 0.911 | 10.3 | 2,007 (0.34 %) |
| O oracle | 1.000 | 1.000 | 2.7 | 0.999 | 0.999 | 1.1 | 210 (0.04 %) |

Paired tests of CATS against each baseline (exact McNemar on full cover, Holm across the seven
baselines of a benchmark):

| CATS vs | LiveMCPBench: discordant (CATS : baseline), p, p_Holm | BFCL: discordant, p, p_Holm |
|---|---|---|
| flat reranked, same k | 4 : 3, p = 1.0, 1.0 | 14 : 4, p = 0.031, 0.062 |
| flat reranked @5 | 8 : 0, p = 0.008, 0.055 | 66 : 1, p < 10⁻⁶, < 10⁻⁶ |
| BM25 @5 | 7 : 0, p = 0.016, 0.094 | 78 : 6, p < 10⁻⁶, < 10⁻⁶ |
| two-stage @3×5 | 10 : 2, p = 0.039, 0.154 | 370 : 7, p < 10⁻⁶, < 10⁻⁶ |

What this says:

1. **At equal prompt size the routing gain is small, on open data as on our own.** On BFCL
   (1,011 test questions, 3,258 tools) CATS beats flat reranked retrieval at the same k by
   +0.010 [+0.002, +0.019] (14 vs 4 discordant questions; McNemar p = 0.031, 0.062 after Holm), the
   same size as on the 477-tool catalog (+0.010). The whole gain is query decomposition: most of it
   (7 of 10 net questions) is on multi-function requests (parallel_multiple 0.634 vs 0.549), and
   CATS without decomposition ties flat retrieval exactly (0.883 vs 0.883, below). Branch routing
   and the calibrated cut-off add nothing at equal k here. The large gains over the k = 5 baselines
   mostly buy a bigger k.
2. **The conformal targets are not met out of the box.** On BFCL the calibrated branch sets
   (2.3 of 32 branches) contain every gold branch for 0.951 of test questions against a 0.97 target,
   and no cut-off reaches α_tool = 0.05 on the calibration data (best achievable 0.11), so CRC falls
   back to k_max and the slot miss is 0.089. A generated taxonomy, 246 calibration questions for the
   branch threshold and no enrichment are the likely causes; NFR2 and NFR3 must be re-tested here.
   For scale: flat reranked retrieval needs k = 30 to reach NFR2's slot recall of 0.95 on BFCL
   (0.903 at k = 10, 0.951 at 30, 0.962 at 50; `recall_at_k`), still only 1 % of static tokens.
3. **LiveMCPBench is out of reach for single-shot retrieval.** Its human-written tasks state goals
   ("a PDF report on WeChat Reading trends with a word cloud") whose steps (create a document, add a
   picture, convert to PDF) are implicit. The best selection covers every needed tool for 27 % of
   tasks; flat reranked retrieval reaches only 0.72 slot recall and 0.51 full cover even at k = 50
   (`recall_at_k` in the results file; figure `external_recall`). Plan-then-retrieve (one LLM call
   that lists the steps, §4.5 of RESULTS.md) or `request_tools` escalation end to end is needed; better
   ranking alone will not close this.
4. **CATS needs labelled requests to calibrate.** With about 50 training and 25 calibration tasks
   per LiveMCPBench fold, the branch threshold is 1.0 (all 8 branches) and the tool target is
   infeasible in every fold, so CATS reduces to flat reranked retrieval with k_max and matches it
   (4 : 3 discordant). Description-based pseudo-requests for training did not help (0.245).
5. **Two-stage server routing (B4) depends on meaningful servers.** It matches the other k = 5
   baselines on LiveMCPBench's real servers but collapses on BFCL's generated toolkits (0.534).
6. Absolute coverage is far below the own catalog's 0.93 (no enrichment here; human or user-written
   requests), which supports the caveat that subagent-written tasks flatter every strategy.

**Robustness checks on BFCL** (same split; `results/external_bfcl_strict.json`,
`results/external_bfcl_nodecomp.json`; rows of `paper/tables/external.tex`):

* *Strict labels* (`--strict`: only the question's own function variant counts, no widening across
  the 1,405 extra variants of reused names): every selector loses about 12 points (CATS 0.766, flat
  reranked @10 0.765, BM25 @5 0.644) and CATS's edge at equal k vanishes (+0.001 [−0.007, +0.009],
  McNemar p = 1.0). Near-duplicate variants crowd each other out of a top-10, which is the
  duplicate-server problem of §4.6 of RESULTS.md; CATS's duplicate collapse only merges tools with
  identical names and parameters, so it does not help here.
* *No decomposition* (`--no-decompose`): CATS 0.883 [k = 10.0], identical to flat reranked @10
  (+0.000 [−0.006, +0.006]); parallel_multiple falls from 0.634 to 0.537 and live_parallel_multiple
  from 0.455 to 0.364. On BFCL, routing a request to 2.3 of 32 machine-made branches neither helps
  nor hurts coverage at equal k. On the scaling pools (§6) CATS led flat retrieval by 1.6–3.1 points
  from 1,000 tools up, but with decomposition on; whether routing or decomposition carries that lead
  needs the same ablation there (`scaling()` with `decompose=False`) before the paper credits routing.

**Adding a benchmark.** Write `load(raw_dir) -> ExternalBenchmark` in
`src/cats/bench/external/<name>.py` (branch documents in the `data/SPEC.md` format plus `Task`s with
gold slots; `common.auto_hierarchy` if the benchmark has no categories), register it in
`external/__init__.load_benchmark`, add its files with a commit SHA to `SOURCES` in
`scripts/fetch_external.py`, add an inline-fixture test to `tests/test_evaluation.py`, then run
`python -m cats.bench.external.run <name>`.

Not obtained: MCP-Zero (dataset only on Google Drive / exhausted Git LFS; no query set), MCP-Bench
(tool definitions only from 28 live servers, several needing API keys), ToolRet (Hugging Face only).
What was tried is in `data/external/README.md`. With Hugging Face access ToolRet is the next
benchmark to add: it is retrieval-only and large (7.6k queries), which suits the selection harness.

## 10. Figures and tables

`python scripts/make_figures.py` and `python scripts/make_tables.py` read `results/*.json` and
skip anything whose input is missing:

| Figure (`paper/figures/`) | Table (`paper/tables/`) | Input |
|---|---|---|
| `scaling` — full cover vs pool size per strategy | `scaling.tex` | `scale_final.json` |
| `ablation` — full cover per configuration, 95 % CIs, k annotated | `ablation.tex` | `ablation_final.json` |
| `validity` — slot miss and branch cover per split vs targets | `validity.tex` | `validity_enrich.json`, `validity_independent.json` |
| `pareto` — tokens per request (tools + router) vs full cover, Pareto frontier | `baselines.tex` | `bench_final.json`, `bench_final_llm.json` |
| `e2e` — success (Wilson CI) and cents per task per model | `e2e.tex` | `e2e_*.json`, `b2_*.json` |
| `external` — full cover per strategy per benchmark (main runs) | `external.tex` (main runs and variants) | `external_*.json` |
| `external_recall` — flat recall@k curve vs CATS's operating point, NFR2 line | — | `external_*.json` (`recall_at_k`) |

Design rules: a strategy has the same colour in every figure (colour follows the entity, not its
rank), from a categorical palette validated for colour-vision deficiency (adjacent-pair ΔE ≥ 9 under
protan/deutan simulation); every series also has its own marker, line style or hatch, so figures read
in grayscale; values are labelled directly where the text cites them; one y-axis per panel; fonts
embedded as TrueType (`pdf.fonttype = 42`). Tables use booktabs and need only `\usepackage{booktabs}`;
they compile with pdflatex.

## 11. Reproducibility (NFR8, DL5, risk D6)

```bash
scripts/reproduce_all.sh            # every results file, figure and table (hours: the 10k-tool scale run dominates)
scripts/reproduce_all.sh --quick    # smoke run into results/quick/, paper/{figures,tables}/quick/ (2 seeds, 2 pool sizes)
scripts/reproduce_all.sh --fetch    # also re-download external benchmarks and verify SHA-256
```

Requirements beyond `pip install -e ".[dev,tokenize,embed]"`: `matplotlib` for the figures
(`pip install matplotlib`), and `pdflatex` with booktabs only to typeset the tables. Statistics use
numpy and the standard library (no SciPy).

* **Seeds.** Splits, pools, folds, bootstrap and k-means take explicit seeds (default 0; e2e runs
  seed 1/2). Re-running gives identical coverage numbers; latency varies with machine load.
* **Recorded configuration.** Every results file stores the full `CATSConfig`, the calibration
  report (thresholds, feasibility), the tokenizer backend and the embedder; external results also
  store the benchmark's commit, conversion metadata and protocol; end-to-end runs store the task ids,
  seed and mode in `runs/<run>/run.json` and the exact tools shown per turn in each session.
* **Pinned data.** External files are fetched from a commit SHA, not a branch, and checked against
  `manifest.json`; a changed upstream file stops the script.
* **Per-request traces.** Every CATS selection carries `TraceSpan`s (analyze, route+retrieve,
  assemble) with latency and tokens (NFR8 "every request traced with tokens, latency, route and tools").
* **Model drift (D6).** The subagent runs name the model family only ("strong", "Haiku"); before the
  paper, record dated model IDs in `run.json` and repeat the headline run on an open-weight model.

## 12. Mapping to the workbook

| Item | Requirement (target) | Status and evidence | Gap |
|---|---|---|---|
| FR9 | harness runs static loading, the baselines and CATS on BFCL, LiveMCPBench and ToolRet with identical prompts and models, recording tokens, latency, cost, accuracy | selection-level harness on the own catalog, LiveMCPBench and BFCL (§9); subagent harness end to end on the own catalog (§7) | ToolRet not obtainable here; no model-in-the-loop run on the external benchmarks yet |
| FR14 | public scaling benchmark, 10–10,000 tools, realistic distractors | generator + distractors + label widening (§6), 100–10,000 tools | datasheet and public release (DL6) |
| NFR1 | tool tokens ≤ 10 % of static at ≥ 100 tools, router included | 2.33 % at 477 tools (2.82 % with LLM router); 0.12–5.2 % from 250 to 10,000 tools | **fails at 100 tools (12.6 %)**: use `bypass_tokens` or a smaller k there |
| NFR2 | accuracy within 1 pp of static loading; recall@k ≥ 0.95 | end to end: strong model 36/36 = static; Haiku 36/36 vs 25/36; slot recall 0.956 (own catalog) | recall far below 0.95 on LiveMCPBench and below on BFCL (§9); n = 36 cannot show "within 1 pp" |
| NFR3 | coverage ≥ 1 − α on held-out tasks; never zero tools | branch cover 0.984 ≥ 0.97; slot miss 0.052 vs 0.05 over 5 splits (§5); k ≥ 1 always | needs a larger calibration set and more splits |
| NFR4 | routing ≤ 400 ms median, ≤ 1 s p95 at 10,000 tools | p50 10 ms, p95 30 ms at 10,000 tools, shared 4-core container | — |
| NFR6 | cost per task below static at ≥ 100 tools, router and escalation included | 8.8× cheaper end to end (strong), 7.1× (Haiku), with cache discounts for static | provider-reported usage at list prices |
| NFR8 | every request traced; every run records configuration, seed, model, benchmark version | §11 | dated model IDs |
| DL5 | benchmark adapters, cost ledger, one-command reproduction | `bench/external/`, `ledger.py`, `reproduce_all.sh` | StableToolBench adapter (planned in the team table) |
| DL6 | 10–10,000-tool benchmark with gold labels and datasheet; public release | generator and labels (§6) | datasheet, release |
| D3 | pinned, mirrored benchmarks; two benchmarks per claim | pinned + SHA-256 (§9, §11) | the routing claim now has two external benchmarks, the accuracy claim none |

## 13. Threats to validity

* **Construct.** Full cover is necessary, not sufficient, for task success; it ignores argument
  correctness. Slot labels can be incomplete: a tool that would work but is not in a slot counts as a
  miss (LiveMCPBench's single annotated path makes this strict). Tools in end-to-end runs are
  simulated.
* **Internal.** On the own catalog, the same model family wrote tools, tasks, labels, enrichment and
  LLM-router scores, so absolute numbers are optimistic and the enrichment and LLM-router gains may
  include shared phrasing. Hyperparameters (k_max, α) were chosen on the same catalog that reports
  them. External benchmarks remove the task-authoring confound but use a generated taxonomy (BFCL)
  or tiny training sets (LiveMCPBench: ~50 per fold), which handicaps routing in a different way.
* **External.** Dense retrieval used static WordLlama embeddings (BGE-M3 was unreachable); token
  counts use a proxy BPE; costs are a cache simulation. Scaling distractors are near-duplicates by
  construction. End-to-end runs cover two model sizes of one family.
* **Statistical.** One split for the headline baselines; 36 tasks per end-to-end mode; five
  calibration splits for validity. Comparisons across many baselines are Holm-corrected in the
  external results but not in the older `bench_*.json` files (those report paired bootstrap CIs only).
  Latencies were measured while other jobs ran on the same 4 cores.
