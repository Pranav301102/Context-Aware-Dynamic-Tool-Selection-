# Berkeley Function Calling Leaderboard (BFCL v4), pooled for tool selection

* **Source:** https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data
* **Pinned commit:** `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000` (2025-12-17, "[BFCL] Multiple Dataset
  Fix and New Model Support (#1266)", the last change to `bfcl_eval/data`); SHA-256 of every file
  in `manifest.json`.
* **Files:** `raw/BFCL_v4_<category>.json` and `raw/possible_answer/BFCL_v4_<category>.json` for the
  ten single-turn categories with ground truth (simple_python, simple_java, simple_javascript,
  multiple, parallel, parallel_multiple, live_simple, live_multiple, live_parallel,
  live_parallel_multiple), plus `BFCL_v4_{irrelevance,live_irrelevance,live_relevance}.json`, whose
  functions serve only as distractor tools. `raw/LICENSE` is the repository licence.
* **Licence:** Apache-2.0 (repository `LICENSE`; the data README's card says `license: apache-2.0`).
* **Retrieved:** 2026-10-06 via `raw.githubusercontent.com` (`scripts/fetch_external.py`).
* Not used: multi-turn, memory, web-search and format-sensitivity categories (they test state
  and agent behaviour rather than selecting from a catalog).

## Conversion (`cats.bench.external.bfcl`)

BFCL gives each question its own list of 1 to 37 functions. Here every function of every category
is pooled into one catalog, and each question keeps its ground-truth function(s) as gold, so a request
must find its function among 3,258.

| CATS | BFCL |
|---|---|
| tool | one distinct (name, description, parameters) variant; 1,853 names, 3,258 variants |
| toolkit | the namespace before the first dot (`math.factorial` -> toolkit `math`, tool `factorial`); namespace-less functions are grouped into toolkits of about 8 inside their branch |
| branch | k-means cluster (32, seed 0) of TF-IDF text; BFCL has no categories |
| task query | the last user message; system prompts dropped, earlier user/assistant turns kept as history |
| gold slots | one per distinct ground-truth function name (parallel calls of one function = one slot) |
| task kind | BFCL category |

* All 2,501 questions with ground truth convert (none unresolved). Mean 1.13 slots per task.
* **Label widening:** a slot holds every variant with the gold function name (BFCL reuses functions
  with reworded descriptions across categories, and live entries repeat the same user API).
  `python -m cats.bench.external.run bfcl --strict` keeps only the question's own variant.
* BFCL pseudo-types (`dict`, `float`, `tuple`, Java types) are mapped to JSON Schema types so every
  `input_schema` is valid; token counts are taken on the converted schemas.
* The generated taxonomy is a threat to validity for routing (and for B4 two-stage): a curated
  taxonomy would likely route better. The flat baselines do not depend on it.
