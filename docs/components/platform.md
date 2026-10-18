# Platform and infrastructure

The CATS core (analyzer, router, retriever, assembler, dispatcher; see
[orchestration_and_routing.md](orchestration_and_routing.md)) was built as an in-process library.
This component adds the platform around it that *Project Workbook 1* lists in its architecture
(Figure 3) and environment (Table 9): a REST/SSE gateway, OpenTelemetry tracing with a Langfuse
sink, a Parquet metrics store, NFR8 run manifests, a Hydra experiment runner, and the deployment
files (Dockerfile, Docker Compose, CI, type-checker config, lock file).

Every piece is new code; no existing module under `src/cats/` was changed. The workbook PDF is not
in this repository, so the mapping below uses the component names from the team's task list and
the README's layout.

```
                    clients (agent hosts, demo UI, curl)
                                 │  HTTP / SSE
                                 ▼
  ┌─────────────────────── cats.api (FastAPI) ───────────────────────┐
  │ POST /v1/select   POST /v1/escalate   GET /v1/route/stream (SSE) │
  │ GET /v1/stats     GET /healthz        sessions, counters, lock   │
  └───────┬────────────────────┬───────────────────────┬─────────────┘
          │ CATS.select /      │ Selection.spans       │ RouteDecision + spans
          │ escalate           ▼                       ▼
          │            cats.tracing ── OTLP ──▶ Langfuse 4 (ClickHouse, Redis, MinIO)
          ▼                                      cats.store ──▶ PostgreSQL 16
   CATS core (in-process index)                  cats.index.vector_store ──▶ Qdrant
          ▲
          │ build_system / experiment functions
  cats.runner (Hydra: conf/) ──▶ results/<name>.json
          │                   ──▶ results/manifests/<name>.json   (cats.manifest, NFR8)
          └──────────────────────▶ results/parquet/records/*.parquet (cats.metrics_store)
  results/*.json ── convert_results ──▶ results/parquet/<table>.parquet
```

## 1. Map to the workbook

| Workbook item | Where | Status in this repository |
|---|---|---|
| Figure 3, API layer: REST + streaming gateway (FastAPI, Pydantic contracts) | `src/cats/api.py` | built; tested with TestClient and against a live uvicorn server on the full corpus |
| Figure 3, observability: per-stage trace spans to Langfuse (OpenTelemetry) | `src/cats/tracing.py` | built; tested with in-memory and console exporters; Langfuse export tested through the SDK's processor with an in-memory exporter, never against a running Langfuse |
| Figure 3, storage: PostgreSQL registry/route log, Qdrant vectors | `cats.store`, `cats.index.vector_store` (other workstream); wired by `docker-compose.yml` and `CATS_STORE` | gateway logging tested with a fake store and a SQLite `cats.store`; Postgres and Qdrant containers not started here |
| Table 9: evaluation metrics in Parquet | `src/cats/metrics_store.py` | built and tested; `results/*.json` converted to `results/parquet/` |
| NFR8: reproducible runs (config, seed, revision, versions, data hashes) | `src/cats/manifest.py` | built and tested; written by every runner job |
| Table 9: Hydra configuration and sweeps | `conf/`, `src/cats/runner.py` | built and tested (compose API, single runs, a 2-seed multirun through the CLI) |
| Table 9: Python 3.12, uv, Docker, Docker Compose | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `deploy/postgres/`, `uv.lock` | `uv lock` and a locked `uv sync` run here; `docker compose config` validates; no image built and no container started (no Docker daemon in this container) |
| Table 9: CI with ruff, pyright, pytest | `.github/workflows/ci.yml`, `[tool.pyright]` in `pyproject.toml` | written, not run on GitHub (nothing was pushed); its steps were run locally |

## 2. REST/SSE gateway (`cats.api`)

The process builds the system once at start-up: `cats.build.build_system` with the configured
`CATSConfig`, enrichment, Jev / LLM-router answers, embedder and split seed, so it fits and
calibrates exactly as `cats bench` does. `--system runs/<run>/system.pkl` serves a pickled
`System` (as written by `python -m cats.agent_cli prepare`) instead; pickles execute code on load,
so only use files you produced. Until the system is ready, `/healthz` answers 503.

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/select` | `{query, history: [{role, content}], exclude: [tool ids], session_id?, session_order?, include_specs?}` | `request_id`, `route` (the `RouteDecision`: branch probabilities, selected branches, conformal set, threshold, fallback), `tools` (id, wire name, toolkit, branch, score, token cost, optional definition), `tool_ids`, `k`, `savings`, `spans` (the pipeline's `TraceSpan`s), `latency_ms`, `registry_version`, `trace_id` |
| `POST /v1/escalate` | `{capability, exclude, session_id?, k?}` | the added tools, route and spans. With a `session_id` from `/v1/select`, the session's loaded tools are excluded and at most `max_escalations` (2, FR7) calls are allowed; the third gets 409, an unknown session 404 |
| `GET /v1/route/stream?query=...&history=...&exclude=...` | query parameters | `text/event-stream`: events `analyze`, `route`, `retrieve`, `assemble`, then `result` (same body as `/v1/select`); `bypass` replaces the stages when the catalog fits `bypass_tokens`; `error` if a stage fails |
| `GET /v1/stats` | | registry stats and version, embedder, `CATSConfig`, calibration report, split sizes, served counters (selections, escalations and refusals, mean k, tokens saved, router tokens, USD saved, fallback rate, p50/p95 latency) |
| `GET /healthz` | | `{status, ready, tools, registry_version, uptime_s}`; 503 while building |

Responses are strict JSON: a non-finite number (e.g. a conformal threshold of +∞ when the
calibration set is too small for the target) is sent as the string `"inf"`, NaN as `null`.

**Savings.** `savings` compares the selected tool definitions with static loading of the whole
catalog: `tokens_saved = static_tokens − tool_tokens`, priced at `CATS_PRICE_INPUT` USD per
million input tokens (default $3, as in RESULTS.md), minus the router overhead (Jev calls at
Jev's price, LLM-router calls at the main model's price). It is a per-request, uncached estimate;
the cached, multi-turn cost model is `cats cache` and the dispatcher's ledger.

**Stream stages.** `stage_events` runs `CATS.select` one stage at a time, calling the same
methods in the same order (`evidence`, `_rank`, `retriever.cut`, `_repair_fn`, `assembler.assemble`)
and yielding after each, so a client can watch the branch probabilities, the conformal set, the
candidate ranking, λ and the cut, the Jev repair and the assembled prompt. Tests check that its
final selection equals `CATS.select` for single- and multi-step requests and for the bypass path.
Because it calls private pipeline methods, a change to `CATS.select` must be mirrored there; the
equality test fails if it is not.

**Configuration** (flag > `CATS_*` variable > default): `--enrich` / `CATS_ENRICH`, `--jev`,
`--llm-router`, `--embedder` / `CATS_EMBEDDER`, `--seed`, `--system`, `--alpha-branch`,
`--alpha-tool`, `--method`, `--k-max`, `--budget` (plus `CATS_CALIBRATION`, `CATS_RERANKER`),
`CATS_MAX_ESCALATIONS`, `CATS_PRICE_INPUT`, `--otel` / `CATS_OTEL`
(`none|console|otlp|otlp-grpc|langfuse`), `CATS_OTEL_ENDPOINT`, `--store` / `CATS_STORE`,
`--host`, `--port`. Relative paths resolve against the working directory, then the repo root.

**Concurrency.** Endpoints are synchronous and run in FastAPI's thread pool; calls into the
pipeline hold one lock because the analyzer and caches are not written for concurrent use. A
selection takes milliseconds, so one process serves on the order of a hundred selections per
second; run more replicas behind a load balancer for more. Sessions live in process memory
(bounded LRU), so escalation sessions need sticky routing across replicas.

**Hooks.** With `CATS_OTEL` set, every selection, escalation and streamed route becomes a trace
(§3). With `CATS_STORE` set (a SQLite path or PostgreSQL DSN), every route decision and its spans
(selections, streamed routes, escalations) are appended through
`cats.store.RegistryStore.log_route`, escalations under the session's request id. A store that fails to open or write is logged and counted (`store_errors`); selection
keeps working.

```bash
pip install -e ".[platform,tokenize,embed]"
python -m cats.api --port 8000 --enrich data/enrichment          # about 1-2 min to fit, then ready
curl -s localhost:8000/v1/select -H 'content-type: application/json' \
     -d '{"query": "is anything slow in prod db right now", "session_id": "demo"}'
curl -s localhost:8000/v1/escalate -H 'content-type: application/json' \
     -d '{"capability": "open a jira ticket", "session_id": "demo"}'
curl -N 'localhost:8000/v1/route/stream?query=check+the+api+pods+and+open+a+jira+ticket'
uvicorn --factory cats.api:app_factory --port 8000                # same, configured from CATS_*
```

## 3. Tracing (`cats.tracing`)

The pipeline keeps its own light spans (`ledger.Tracer` → `Selection.spans`), so it has no
tracing dependency and its latencies stay comparable with the benchmarks. `export_selection`
converts them afterwards:

```
cats.select                       root: one per request
  ├─ cats.stage.analyze           one child per recorded stage, laid end to end
  ├─ cats.stage.route+retrieve    from the request's start, with the measured latencies
  └─ cats.stage.assemble
```

| Span | Attributes |
|---|---|
| root `cats.select` / `cats.escalate` | `cats.strategy`, `cats.k`, `cats.tool_ids`, `cats.tool_tokens`, `cats.overhead_tokens`, `cats.router_calls`, `cats.router_kind`, `cats.latency_ms`, `cats.route.selected`, `cats.route.conformal_set`, `cats.route.threshold`, `cats.route.fallback`, `cats.route.top_branch`, `cats.route.top_p`, `cats.route.n_branches`, `cats.static_tokens`, `cats.tokens_saved`, `cats.query` (when passed), plus the gateway's `cats.request_id`, `cats.endpoint` |
| stage `cats.stage.<stage>` | `cats.stage`, `cats.latency_ms`, `cats.tokens_in`, `cats.tokens_out`, `cats.route.selected`, `cats.route.fallback`, `cats.k`, `cats.parts`, other meta as `cats.meta.<key>` |
| `cats.dispatch` (`export_run`) | `cats.task_id`, `cats.success`, `cats.turns`, `cats.escalations`, `cats.calls`, `cats.invalid_calls`, `cats.loaded_k`, `cats.usd`, `cats.cost.{input_tokens, cache_write, cache_read, output_tokens, router_tokens}`; the run's selection and escalations as child trees; one `tool_call` event per call |

Values OpenTelemetry cannot hold (dicts, mixed lists) are stored as JSON; non-finite thresholds
are left out.

**Exporters** (`make_provider`): `console` (one JSON document per span on stdout), `otlp`
(OTLP/HTTP; endpoint from the argument or `OTEL_EXPORTER_OTLP_*`), `otlp-grpc`, `langfuse`,
`none`, or any `SpanExporter` object (tests use `InMemorySpanExporter`). Console and in-memory
exporters are synchronous; network exporters are batched.

**Langfuse.** Langfuse 3 and later ingest OTLP at `<host>/api/public/otel/v1/traces` with Basic
auth (public key : secret key). `langfuse_processor` uses the Langfuse SDK's
`LangfuseSpanProcessor` when the SDK is installed (`pip install langfuse`; version 4.17 installed
here) and passes it a filter that keeps CATS spans, because by default it only forwards spans from
LLM frameworks. Without the SDK it builds a plain OTLP/HTTP exporter with the same header and
endpoint. Keys and host come from `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`.
The processor import uses a private SDK module path (`langfuse._client.span_processor`); if a
future SDK moves it, the fallback exporter is used.

```bash
python -m cats.tracing "will it rain in paris tomorrow" --exporter console
CATS_OTEL=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python -m cats.api
```

## 4. Metrics store (`cats.metrics_store`)

| Table | Kind | Rows | Written by |
|---|---|---|---|
| `records` | append-only (`<root>/records/*.parquet`) | one `cats.bench.harness.Record` per (strategy, task), typed columns | `MetricsStore.append_records`; the runner's `bench` jobs |
| `agent_runs` | append-only | `agent_cli.score_run(...)["rows"]` dicts or `dispatcher.RunResult` (cost ledger split into columns) | `append_agent_runs` |
| `files` | snapshot (`<root>/files.parquet`) | one per `results/*.json`: detected kind, sha256, size, mtime, rows extracted, the raw JSON | `convert_results` |
| `summaries`, `by_kind`, `paired` | snapshot | bench and ablation summaries (CIs as `_ci_lo` / `_ci_hi`, calibration as `cal_*`), per-kind splits, paired bootstrap differences vs CATS | `convert_results` |
| `validity`, `scale`, `cache`, `agent` | snapshot | per-seed coverage rows, per (pool size, strategy) rows, per caching policy rows, subagent end-to-end summaries | `convert_results` |

Every appended row carries its run metadata: `run_id`, `experiment`, `created_at` (UTC
timestamp), `seed`, `git_rev`, `config` (JSON) and any extra keyword metadata (e.g. `mode`,
`model`). Each append writes a new part file, so concurrent writers never collide; reading
concatenates the parts and unifies their schemas. `read(table, where=..., columns=...)`,
`query(...)` (list of dicts), `runs()` and `summarize_records(by=("run_id", "strategy"))` (means
of full cover, slot recall, precision, k, tokens, branch cover; p50/p95 latency) are the query
helpers. Unknown JSON shapes still get a `files` row, so the conversion loses nothing.

```bash
python -m cats.metrics_store convert                 # results/*.json -> results/parquet/*.parquet
python -m cats.metrics_store tables
python -m cats.metrics_store show summaries --where file=bench_final.json --columns strategy,full_cover,k_mean
python -m cats.metrics_store summary                 # records table, per run and strategy
```

## 5. Run manifests (`cats.manifest`, NFR8)

`results/manifests/<name>.json` holds, for one run: name, experiment, status (`running`, `ok`,
`error` with the exception), start and end time (UTC), duration, seed, full configuration, git
revision and branch, Python and platform, versions of 23 relevant packages, the tokenizer and
embedder actually used, model ids, SHA-256 of every data file read plus one digest over all of
them, SHA-256 of every output written, registry stats, split sizes and calibration report, the
command line, working directory, and `CATS_*` / `OTEL_*` environment settings with anything
named like a key, secret, token, password or DSN redacted.

The git revision is read with `git rev-parse` only (with `GIT_OPTIONAL_LOCKS=0`), or from the
`.git` files when the git binary is missing; the repository is never written. Dirty-tree state is
not recorded, because `git status` may refresh the index file.

```python
from cats.manifest import run_manifest

with run_manifest("bench_final", config=cfg, seed=0, enrichment="data/enrichment") as m:
    s = build_system(cfg, seed=0, enrichment=Path("data/enrichment"))
    m.set_system(s)                              # tokenizer, embedder, registry, calibration
    m.add_output(save("bench_final", run_baselines(s)))
```

The manifest is written on entry (status `running`, so a crashed run leaves a trace) and again on
exit. `verify_data(manifest)` re-hashes the data files and lists changed, missing and new ones.
Command line: `python -m cats.manifest snapshot|show|verify NAME`.

## 6. Hydra runner (`conf/`, `cats.runner`)

```
conf/config.yaml        defaults + seed, name, results_dir, manifest, metrics, hydra output dirs
conf/strategy/          cats (default) | all | flat_reranked | cats_gbt | cats_no_decompose
                          keys: name, baselines (names from baseline_strategies, or "all"), k,
                          cats: {k_max, k_min, k_per_extra_part, decompose, dedupe, reranker, budget, order, bypass_tokens}
conf/data/              default (enrich: true) | raw (enrich: false)
                          keys: enrich, enrich_dir, jev, jev_dir, llm_router, llm_router_dir, embedder, corpus_dir, tasks_dir
conf/calibration/       lac (default) | aps | independent
                          keys: method, alpha_branch, alpha_tool, mode (joint|independent), split
conf/experiment/        bench (default) | ablate | validity (n_seeds) | scale (sizes, k) | cache (session_len, n_sessions)
```

Each job calls the existing functions in `cats.bench.experiments`: `bench` evaluates the chosen
strategies from `baseline_strategies` with the harness and returns the same structure as
`run_baselines` (summaries, paired differences against CATS), keeping the per-task records for the
metrics store; `ablate` → `run_variants(default_variants(cfg))`; `validity` →
`coverage_validity(cfg, seeds=range(seed, seed + n_seeds))`; `scale` → `scaling`; `cache` →
`cache_sessions`. It writes `results/<name>.json` (same format as `cats <command>`, so the
converter and the existing tables read it), `results/manifests/<name>.json`, and for `bench`
`results/parquet/records/<part>.parquet`. Hydra's own logs and composed configs go to
`results/hydra/` (`$CATS_HYDRA_DIR`). The default name encodes the choices, e.g.
`bench-cats-enrich-lac-s0` or `validity-cats-raw-lac-indep-at0.1-s2`.

```bash
python -m cats.runner experiment=bench strategy=cats data.enrich=true
python -m cats.runner experiment=bench strategy=all data=raw calibration=aps
python -m cats.runner -m seed=0,1,2 experiment=bench                 # one result + manifest per seed
python -m cats.runner -m calibration.alpha_tool=0.05,0.1,0.2 experiment=validity
python -m cats.runner experiment=scale experiment.sizes=[100,1000]
python -m cats.runner --cfg job experiment=validity calibration=independent   # print the config only
```

`data.corpus_dir` and `data.tasks_dir` apply to `bench` and `cache` (and the corpus to
`ablate`); `validity` and `scale` call functions that always read `data/`. The tests use them to
run on a 9-tool corpus.

## 7. Deployment files

**`pyproject.toml`** gained optional-dependency groups: `api` (fastapi, uvicorn, httpx),
`tracing` (opentelemetry-sdk, opentelemetry-exporter-otlp), `langfuse`, `metrics` (pyarrow),
`runner` (hydra-core), `platform` (the four), `deploy` (platform + langfuse + tokenize + embed +
psycopg, what the image installs) and `lint` (pyright), and a `[tool.pyright]` section.

**Pyright** runs in standard mode over `src/cats`. The core modules keep attributes Optional until
`fit()` or `build()` runs, which pyright cannot follow, so the Optional-access and argument-type
categories are warnings globally; per-file execution environments keep them errors for the
platform modules (`api`, `tracing`, `manifest`, `metrics_store`, `runner`), which have zero errors
and zero warnings. At the time of writing the whole package gives 6 errors, all in
`src/cats/agents/` (another workstream, in progress), and the CI type-check step fails until they
are fixed.

**`uv.lock`** was generated with `uv lock` (95 packages, all extras) from PyPI. It must be
regenerated (`uv lock`) whenever `pyproject.toml` changes, or `uv sync --locked` fails; that is the
intended CI behaviour.

**`Dockerfile`**: `python:3.12-slim`, uv 0.11.32 from PyPI, a dependency layer from the lock file
(`uv sync --locked --no-install-project --extra deploy`), then the package installed editable
(`cats.build.ROOT` must stay `/app` so `data/` is found), `conf/`, and only `data/corpus`,
`data/tasks` and `data/enrichment`. It fetches the offline BPE tokenizer used for RESULTS.md's
token counts at build time (`--build-arg FETCH_TOKENIZER=0` skips it and counts fall back to the
regex proxy), runs as a non-root user, and health-checks `/healthz` with a 5-minute start period
for the fit.

**`docker-compose.yml`**:

| Service | Image | Port (host) | Role |
|---|---|---|---|
| `cats-api` | built from `Dockerfile` | 8000 | the gateway; `CATS_OTEL=langfuse`, `CATS_STORE=postgresql://cats@postgres/cats` |
| `postgres` | `postgres:16` | 127.0.0.1:5432 | databases `cats` (route log) and `langfuse`, created by `deploy/postgres/10-cats-databases.sh` |
| `qdrant` | `qdrant/qdrant:v1.19.2` | 127.0.0.1:6333/6334 | vector store for `cats.index.vector_store` (`CATS_QDRANT_URL`) |
| `langfuse-web` | `langfuse/langfuse:4` | 3000 | UI and OTLP ingestion; headless init creates org/project `cats` with keys `pk-lf-cats-local` / `sk-lf-cats-local` and user `admin@cats.local` |
| `langfuse-worker` | `langfuse/langfuse-worker:4` | 127.0.0.1:3030 | Langfuse background processing |
| `clickhouse`, `redis`, `minio` | `clickhouse/clickhouse-server:25.12`, `redis:7`, `cgr.dev/chainguard/minio` | localhost only (MinIO 9090 for media URLs) | Langfuse's dependencies, configured as in Langfuse's own compose file |

All credentials are local defaults overridable from a `.env` file. The gateway's search path stays
in-process (`HybridIndex`); Qdrant is provided for the vector-store adapter and its benchmarks, not
queried by the API.

```bash
docker compose up -d --build          # first start: image build + Langfuse migrations + CATS fit
curl -s localhost:8000/healthz
curl -s localhost:8000/v1/select -H 'content-type: application/json' -d '{"query": "post the release notes to slack"}'
open http://localhost:3000            # traces: project "cats"
docker compose exec postgres psql -U cats -d cats -c 'select count(*) from route_decision'
```

**`.github/workflows/ci.yml`** runs on every push and pull request: `uv sync --locked` with the
dev, lint, platform, langfuse, tokenize, embed and mcp extras, `uv run ruff check src tests scripts`,
`uv run pyright`, `uv run pytest -q`; a second job validates the Compose file and builds the image.

## 8. Tested here versus only written

| Piece | What was run in this container | Result |
|---|---|---|
| `tests/test_platform.py` | 30 offline tests on a 9-tool fitted CATS (gateway via TestClient, SSE parsing, session limits incl. concurrent escalations, lifespan build, route logging to a fake and to a SQLite `cats.store`, OTel in-memory/console/Langfuse-processor export, dispatcher-run export, Parquet append/query/summary, conversion of every results shape and of the real `results/`, manifests incl. error status, redaction and data verification, git fallback, Hydra composition, bench run, dispatch to the other experiments, a 2-seed multirun and a single run through the CLI, deployment-file consistency) | see §9 |
| Gateway on the full corpus | `python -m cats.api --enrich data/enrichment --otel console --store <sqlite>`, then curl against `/healthz`, `/v1/select`, `/v1/escalate`, `/v1/route/stream`, `/v1/stats` | see §9 |
| Hydra runner on the full corpus | `python -m cats.runner experiment=bench strategy=cats data.enrich=true` | see §9 |
| Parquet conversion | `python -m cats.metrics_store convert` on the real `results/` | see §9 |
| `uv.lock` | `uv lock`; `uv sync --locked` with the CI extras into a scratch venv, then `tests/test_platform.py` and `ruff` from it | see §9 |
| `docker-compose.yml` | `docker compose config` (validates and resolves) | valid, 8 services |
| `Dockerfile`, Compose services | **not run**: the container has the Docker CLI but no daemon, so no image was built and Postgres, Qdrant and Langfuse were never started; the Langfuse wire format is exercised only up to the exporter |
| `.github/workflows/ci.yml` | **not run on GitHub** (nothing pushed); its steps were run locally | see §9 |

## 9. Results of the runs above

Measured in this build (October 2026, a shared 4-core container while other agents ran
benchmarks, so latencies are upper bounds):

* **Gateway, full corpus** (`--enrich data/enrichment --otel console --store <sqlite>`): build, fit
  and calibration at start-up took 51 s; `/healthz` then reported 477 tools. "is anything slow in
  prod db right now" returned k = 10 tools from 7 routed branches, 1,555 tool tokens against 96,774
  for static loading (1.6%), an estimated $0.286 saved per request at $3/M. Escalating on its
  session added 5 Jira/Zendesk tools for "open a jira ticket"; the third escalation got 409. The
  stream for "check the api pods and open a jira ticket" emitted analyze (two sub-requests), route
  (6 branches searched, conformal set of 4), retrieve (166 candidates, k = 13 with the extra step's
  3 slots), assemble (2,669 tokens) and result. Over 11 selections, p50 27 ms and p95 56 ms end to
  end. The console exporter printed 12 `cats.select` traces with 38 stage spans and 2
  `cats.escalate` traces; the SQLite route store held 13 route rows and 35 span rows, 0 store errors.
* **Hydra runner, full corpus** (`experiment=bench strategy=cats data.enrich=true`): 37 s wall,
  `results/bench-cats-enrich-lac-s0.json` with CATS full cover **0.930 [0.904, 0.953], k 10.6,
  2.33% of static tokens**, the RESULTS.md headline exactly; manifest status `ok` (git 011471f45158,
  tokenizer `hf:tokenizer.json`, embedder `wordllama:l2_supercat`, 37 data files hashed, both outputs
  hashed); 386 records appended to `results/parquet/records/`. Its p95 latency read 68 ms against
  8 ms in RESULTS.md, from CPU contention.
* **Parquet conversion** of the real `results/` (31 JSON files at the last run, including the runner's
  new one): `summaries` 146 rows, `by_kind` 873, `paired` 108, `scale` 60, `agent` 17, `validity` 10,
  `cache` 6; six files of shapes
  the converter does not know (`calibration_loop`, `corpus_stats`, `e2e_paired`, `security_eval`,
  `vector_stores` and similar, written by other workstreams) are kept as raw JSON in `files`. The
  `bench_final.json` CATS row reads back as 0.930 [0.904, 0.953]. Re-run the converter after new
  results land; it rewrites the snapshot tables.
* **uv**: `uv lock` resolved 95 packages in about 9 s. `uv sync --locked` with the CI extras into a
  fresh virtual environment (Python 3.13, the only interpreter here) installed everything from PyPI,
  and all 30 platform tests passed from that environment.
* **Lint and types**: ruff is clean on every new file. Across `src tests scripts`, one finding
  remains, in `scripts/jev_experiment.py` (an existing file this work may not edit), so the CI
  ruff step fails until it is fixed. Pyright reports 0 errors and 0 warnings for the five platform
  modules; over `src/cats` it reports 6 errors, all in `src/cats/agents/` (another workstream, in
  progress), plus warnings for the legacy categories above.
* **Tests**: `tests/test_platform.py` 30 passed in about 14 s. The whole suite (`python3 -m pytest
  -q`, including the other workstreams' tests present at the time): 175 passed, 1 skipped, 0 failed
  in 67 s.

## 10. Limitations and next steps

* One fitted system per process and one pipeline lock: scale out with replicas; sessions are not
  shared between replicas.
* The SSE stream mirrors `CATS.select` stage by stage through private methods; keep them in step
  (the equality test guards it).
* Hot registry updates (MCP `list_changed`) are not exposed over HTTP; the gateway serves the
  catalog it was built with (`registry_version` is reported so clients can tell).
* The Compose stack and image are unverified until someone runs them on a machine with Docker;
  Langfuse's headless-init variables and the OTLP path follow Langfuse's documentation for v3/v4.
* `uv.lock` resolves `requires-python >= 3.11` for every platform; regenerate it after any change to
  `pyproject.toml`.
