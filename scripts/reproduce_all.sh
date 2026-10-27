#!/usr/bin/env bash
# Regenerate every results file, figure and table (docs/RESULTS.md §9, docs/components/evaluation.md §11).
#
#   scripts/reproduce_all.sh                 full run: results/*.json, paper/figures, paper/tables
#   scripts/reproduce_all.sh --quick         smoke run (fewer seeds, two pool sizes, task subsets); writes to
#                                            results/quick/, paper/figures/quick/, paper/tables/quick/ and
#                                            never touches the full results
#   options: --skip-external  --skip-e2e  --skip-scale  --fetch (download external benchmarks first)
#
# Every step logs to <results dir>/reproduce.log. The subagent end-to-end runs themselves need a model
# in the loop (python -m cats.agent_cli ...); this script only re-scores the sessions under runs/.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0; SKIP_EXT=0; SKIP_E2E=0; SKIP_SCALE=0; FETCH=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --skip-external) SKIP_EXT=1 ;;
    --skip-e2e) SKIP_E2E=1 ;;
    --skip-scale) SKIP_SCALE=1 ;;
    --fetch) FETCH=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY=${PYTHON:-python3}
CATS="$PY -m cats.cli"
ENRICH=data/enrichment
LLM=runs/llm_router
if [ "$QUICK" = 1 ]; then
  P=quick/; RES=results/quick; FIG=paper/figures/quick; TAB=paper/tables/quick
  SEEDS=2; SIZES=100,1000; FOLDS=3; BOOT=1000; BFCL_MAX="--max-tasks 600"
else
  P=; RES=results; FIG=paper/figures; TAB=paper/tables
  SEEDS=5; SIZES=100,250,1000,2500,5000,10000; FOLDS=5; BOOT=10000; BFCL_MAX=
fi
mkdir -p "$RES" "$FIG" "$TAB"
LOG="$RES/reproduce.log"
: > "$LOG"

step() {  # step "description" command...
  local desc=$1; shift
  local t0=$SECONDS
  echo "==> $desc"
  echo "==> $desc: $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1 || { echo "FAILED: $desc (see $LOG)" >&2; exit 1; }
  echo "    done in $((SECONDS - t0)) s"
}

step "validate data" $PY scripts/validate_data.py
step "baselines (bench_final)" $CATS bench --enrich $ENRICH --name ${P}bench_final
if [ -d "$LLM" ]; then
  step "baselines with LLM router feature (bench_final_llm)" \
    $CATS bench --enrich $ENRICH --llm-router $LLM --name ${P}bench_final_llm
fi

if [ "$QUICK" = 1 ]; then
  # First four ablation rows of scripts/ablation_final.py (the cumulative component story).
  step "ablation (first 4 variants)" $PY - "$RES/ablation_final.json" <<'EOF'
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("ablation_final", "scripts/ablation_final.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from cats.bench.harness import evaluate, summarize
from cats.build import build_registry, build_system
from cats.strategies import CATSStrategy
reg = build_registry(); rows = []
for name, cfg, enrich in mod.VARIANTS[:4]:
    s = build_system(cfg, registry=reg, enrichment=enrich)
    summ = summarize(evaluate(CATSStrategy(s.cats, name), s.test, reg), reg.total_tokens())
    summ["calibration"] = s.cats.calibration_report
    rows.append(summ)
    print(f"{name:<40} cover={summ['full_cover']:.3f} k={summ['k_mean']:.1f}", flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1, default=str))
EOF
else
  step "ablation (ablation_final)" $PY scripts/ablation_final.py
fi

step "coverage validity, end-to-end calibration" \
  $CATS validity --enrich $ENRICH --seeds $SEEDS --name ${P}validity_enrich
step "coverage validity, per-stage calibration" \
  $CATS validity --enrich $ENRICH --seeds $SEEDS --calibration independent --name ${P}validity_independent
if [ "$SKIP_SCALE" = 0 ]; then
  step "scaling ($SIZES tools)" $CATS scale --sizes $SIZES --name ${P}scale_final
fi
step "prompt-cache sessions" $CATS cache --enrich $ENRICH --session-len 5 --name ${P}cache_enrich

if [ "$SKIP_E2E" = 0 ]; then
  RUNS=""
  for r in e2e_static e2e_cats e2e_search e2e_haiku_static e2e_haiku_cats; do
    [ -f "runs/$r/run.json" ] && RUNS="$RUNS $r"
  done
  if [ -n "$RUNS" ]; then
    step "re-score subagent end-to-end runs" $PY scripts/score_e2e.py $RUNS --n-boot $BOOT --out "$RES/e2e_paired.json"
  else
    echo "    (no runs/e2e_* sessions; skipping end-to-end scoring)"
  fi
fi

if [ "$SKIP_EXT" = 0 ]; then
  if [ "$FETCH" = 1 ]; then
    step "fetch external benchmarks (pinned commits)" $PY scripts/fetch_external.py --convert
  fi
  if [ -f data/external/livemcpbench/raw/tools.json ]; then
    step "external: LiveMCPBench (crossfit, $FOLDS folds)" $PY -m cats.bench.external.run livemcpbench \
      --protocol crossfit --folds $FOLDS --n-boot $BOOT --name ${P}external_livemcpbench
  fi
  if [ -f data/external/bfcl/raw/BFCL_v4_simple_python.json ]; then
    step "external: BFCL v4 (split)" $PY -m cats.bench.external.run bfcl --protocol split --n-boot $BOOT \
      $BFCL_MAX --name ${P}external_bfcl
  fi
fi

step "figures -> $FIG" $PY scripts/make_figures.py --results "$RES" --out "$FIG"
step "tables -> $TAB" $PY scripts/make_tables.py --results "$RES" --out "$TAB"
echo "all done; log: $LOG"
