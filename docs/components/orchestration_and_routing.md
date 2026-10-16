# Orchestration and routing

Owner in the team plan: **Pranav Jitendra Trivedi** (agent orchestration and decision-tree
routing). Workbook requirements: FR3, FR5, FR7, FR8, FR12, NFR1–NFR4; deliverable DL1 (core library,
shared with retrieval).

This component turns one request into the few tool definitions a sub-agent sees, and recovers
when that set turns out to be incomplete. It is Algorithm 1 of the workbook, with the changes the
measurements called for (docs/RESULTS.md §4).

```
request + history
   │
   ▼
① ContextAnalyzer ─ effective query, vendor hints, sub-requests          analyzer.py
   │
   ▼
② Router ─ branch probabilities → calibrated branch set                   router.py
   │        (Jev cascade: ask the typed decision layer only when unsure)   pipeline._route, jev.py
   ▼
③ Retriever ─ hybrid candidates → learned reranker → round-robin merge    retriever.py, pipeline._rank
   │          of sub-requests → risk-controlled cut-off → duplicate collapse
   │          (Jev repair: add tools for a needed branch the cut missed)   pipeline._repair_fn
   ▼
④ Assembler ─ request_tools + tools in canonical order, token budget      assembler.py
   │
   ▼
⑤ Dispatcher ─ sub-agent loop; request_tools re-routes (≤ 2 times)        dispatcher.py
   │
   ▼
⑥ Fallback ─ empty candidate set → flat search over every branch         pipeline._rank / escalate
```

## 1. Context analyzer (`cats/analyzer.py`)

* **Effective query.** The request is repeated twice and followed by up to three earlier turns (at
  most 400 characters), so BM25 term frequencies favour the request while "do the same for
  staging" can still be resolved from history.
* **Vendor hints.** Words from toolkit ids and names that rarely occur in other servers'
  descriptions ("slack", "hubspot") mark a named vendor. A small list of generic words
  ("package", "maps", "finance", …) is excluded. The map rebuilds lazily after every registry change.
* **Decomposition.** `decompose()` splits multi-step requests on sequencing words ("then", "after
  that", ";") and on "and"/commas that introduce an action verb. Pieces of fewer than two words are
  dropped and at most three parts are kept. It is a cheap stand-in for an LLM planner.

## 2. Router (`cats/router.py`)

**Scorer.** A conditional-logit model with weights shared across branches:
`p(b | x) = softmax_b(w · f(x, b))`. Because the weights are shared, a new branch is scored without
retraining (FR15). The per-branch features are:

| Feature | Meaning |
|---|---|
| `branch_cos`, `branch_bm25` | match against the branch summary + server summaries (top-down) |
| `tool_cos_max`, `tool_cos_top3`, `tool_bm25_max`, `tool_rrf_max` | best tool matches inside the branch (bottom-up) |
| `toolkit_cos_max` | best server-summary match in the branch |
| `vendor_hint` | the request names a vendor in this branch |
| `llm_p` | external router prior: an LLM router's or Jev's probability for the branch (0 when not called) |

Training maximises `log Σ_{g∈gold} p(g)` with Adam and L2 regularisation (`BranchRouter.fit`).
Probabilities for training the reranker are cross-fitted (5 folds), so the reranker does not
learn from over-confident in-sample router scores.

**Calibrated set (FR3).** Split conformal prediction on the first calibration half:

* **LAC** (default): score = 1 − min gold probability; keep every branch with `p ≥ 1 − t`.
* **APS**: score = mass needed before every gold branch enters; keep the top branches until their
  mass reaches `t`. With a sharp softmax, APS keeps almost every branch (10.1 of 12) to reach 97%
  coverage, while LAC keeps 3.6, so LAC is the default.

`t` is the ⌈(n+1)(1−α)⌉-th smallest calibration score (`calibration.conformal_quantile`).

**Fallback (FR8).** If the top probability is below `t_low`, or the set has more than
`max_branches` branches, the router searches every branch. If the routed branches have no
candidates left, `_rank` also searches everything.

## 3. Jev-style typed decision layer (`cats/jev.py`)

Jev (TypeSafe AI, 2026) is a "System One" model. It answers typed questions about a state
(`noul` yes/no, `choice`, `score`) with calibrated probabilities instead of text, in about 150 ms at
$42 per billion input tokens. The system asks it:

| Question | Type | Used for |
|---|---|---|
| `needs:<branch>` for each of the 12 branches | noul | router prior (`llm_p` feature); multi-label, so cross-branch requests can say yes to several |
| `steps` ("how many separate tool-using steps") with levels `1`, `2`, `3+` | score | adds `k_per_extra_part` slots per extra step |
| `enough` ("can every step be done with only these tools?") | noul | pre-escalation gate (offline evaluation) |

It is used in three places:

1. **Cascade (`CATS._route`).** The embedding-only router (`router_base`) decides alone when its
   calibrated set is a single branch with `p ≥ jev_gate_tau` (default 0.9). Otherwise Jev is
   asked and its answers join the full router's features. Only the requests that need it pay for
   the call.
2. **Step count.** When Jev was asked, `k_max` grows with `max(rule-based parts, Jev steps)`.
3. **Branch repair (`CATS._repair_fn`).** For every branch with Jev `p ≥ jev_repair` (default 0.5)
   that the cut-off left without a tool, the branch's top `jev_repair_k` tools (default 2) are
   added. This targets cross-branch requests, the weakest class.

The repair is a deterministic function of the request, so it runs inside conformal calibration as
well (`ToolRetriever.calibrate` accepts a post-cut step), and the end-to-end guarantee covers it.

**Backends.**
* `ExchangeJev` reads answers produced offline; in this project, offline answer files play
  Jev (see §7).
* `LLMJev` prompts any `cats.llm` backend to emit the typed JSON.

TypeSafe's hosted API is not used; its interface was not available here.

**Billing.**
* Each Jev call is billed with its real input size (`JevRequest.input_tokens()`, about 1,450
  tokens with all 13 questions) at the Jev price (`JEV_USD_PER_MTOK`).
* An LLM router call is billed at the main model's price (`Prices.router_mult`).
* `Selection.router_calls` and `router_kind` record which one ran.

## 4. Retriever cut-off and merging (`cats/retriever.py`, `CATS._rank`)

* **Candidates.** Tools in the routed branches, pre-filtered to the top 100 by reciprocal-rank
  fusion of BM25 and dense scores.
* **Reranker.** Logistic regression over 10 features (cosine, relative BM25, local RRF, toolkit
  cosine, branch probability, vendor match / mismatch, name overlap, cosine gap, top-BM25 flag).
* **Decomposed requests.** Each part is routed and ranked on its own, and the rankings are
  interleaved round-robin (parts first, then the whole request). Merging by score alone let one
  step crowd out the others, because reranker scores are not comparable across queries.
* **Cut-off (FR4).** Keep the first `k_max` candidates whose score is at least `λ`; the first
  `k_min` are always kept, so the kept set only grows as λ falls. `k_max` grows by
  `k_per_extra_part` per extra step.
* **Choosing λ.** Conformal risk control on the second calibration half, with
  loss = the fraction of gold slots missed (`calibration.crc_fit`). The pipeline is run end to end,
  routing and repair included ("joint" calibration). The chosen λ is the strictest value from which
  every more lenient one is also safe. If no λ reaches `alpha_tool`, the report sets
  `tool_target_feasible = false` and gives `min_achievable_alpha_tool`.
* **Duplicate collapse.** Tools with the same name and parameter set (the same API republished
  by several servers) are moved behind the best-ranked copy.

## 5. Assembler (`cats/assembler.py`, FR5, FR12)

* **Prompt order.** `request_tools` first, then the selected tools in canonical order (sorted by
  id), so identical selections give byte-identical prompts. The `append` order keeps a session's
  earlier tools in place for cache reuse.
* **Budget.** Over the token budget, the lowest-scored tools are dropped first. Tools the model
  asked for (score 2.0) are kept.
* **Cost record.** The exact cost of every tool block is recorded for the cost ledger.

## 6. Dispatcher and escalation (`cats/dispatcher.py`, FR6, FR7)

The sub-agent loop calls the model, executes tool calls and handles `request_tools(capability)`:

* the capability text is re-routed and ranked, excluding loaded tools, and up to `k_escalation`
  (5) tools are added;
* at most 2 escalations per task;
* every model turn is billed through the prefix-cache simulator (`ledger.PrefixCache`), with
  router or Jev overhead included.

## 7. Offline answers as Jev

No Jev endpoint was reachable, so Jev's answers were produced offline:

* `runs/jev/route_questions.json` holds the 13 questions; `runs/jev/states_*.jsonl` holds the 960
  conversations;
* each answer line is `{"key", "needs": {branch: p}, "steps": {"1": p, "2": p, "3+": p}}`.

A first answer batch used keyword rules instead of judgment: one batch
used only two distinct probability values. It is kept in `runs/jev_keyword/` as a heuristic
baseline. The reported Jev results use the higher-quality answer batch, produced under a rule that
forbids generating answers with code.

## 8. Commands

```bash
cats route "check the api pods and open a jira ticket" --enrich data/enrichment --jev runs/jev
cats bench --enrich data/enrichment --jev runs/jev --name bench_jev
python scripts/jev_experiment.py                      # what each Jev use buys
cats validity --enrich data/enrichment --seeds 5      # NFR3 over five splits
```

## 9. Limitations

* The decomposition is rule-based, and cross-branch requests are still the weakest class.
* Jev's answers come from the offline answer files, not TypeSafe's model, and its latency (150 ms)
  is taken from public figures, not measured.
* The calibration halves hold about 95 requests each; the finite-sample term (≈ 0.01) is a large
  share of α = 0.05. Use 500 or more calibration requests before relying on NFR3.
