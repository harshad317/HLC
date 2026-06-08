#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /opt/anaconda3/bin/python3 ]]; then
    PYTHON="/opt/anaconda3/bin/python3"
  else
    PYTHON="python3"
  fi
fi

RUN_TESTS=1
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/reproduce_syntactic_immediate_matched.sh [--overwrite] [--skip-tests]

Reproduces the immediate-matched syntactic diversification diagnostic:

1. Seed-0 syntactic NPO sweep over the broad grid.
2. Narrow seed-1/2 expansion around seed-0 matches.
3. Immediate-match selection against HLC references.
4. Answer-bearing stress evaluation for selected syntactic checkpoints.
5. Conditional durability report against Strong NPO and HLC references.

Outputs:
  runs/survival/syntactic_sweep_match_candidates.csv
  runs/survival/syntactic_sweep_match_selected.csv
  runs/survival/syntactic_diversification_matched_stress/
  runs/reports/syntactic_immediate_matched_stress/
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

run() {
  echo
  echo "+ $*"
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

SWEEP_FLAGS=()
STRESS_FLAGS=()
if [[ "$OVERWRITE" -eq 1 ]]; then
  SWEEP_FLAGS+=(--overwrite)
  STRESS_FLAGS+=(--overwrite)
fi

echo "Using Python: $PYTHON"
"$PYTHON" --version

require_file data/prompts/forget_syntactic_variants.jsonl
require_file data/prompts/forget_prompt_variants.jsonl
require_file data/relearn_pools/forget_declarative.jsonl
require_file data/relearn_pools/forget_paraphrase_qa.jsonl
require_file data/relearn_pools/forget_mixed_stress.jsonl

run "$PYTHON" scripts/run_syntactic_immediate_sweep.py \
  --python "$PYTHON" \
  --seeds 0 \
  --step_factors 1,2,3,5 \
  --forget_weights 1,2,4 \
  --betas 0.3,1.0,3.0 \
  ${SWEEP_FLAGS[@]+"${SWEEP_FLAGS[@]}"}

run "$PYTHON" scripts/run_syntactic_immediate_sweep.py \
  --python "$PYTHON" \
  --seeds 1,2 \
  --step_factors 2,3,5 \
  --forget_weights 2,4 \
  --betas 0.3 \
  ${SWEEP_FLAGS[@]+"${SWEEP_FLAGS[@]}"}

run "$PYTHON" scripts/select_immediate_matched.py \
  --reference 'runs/survival/conditional_long_horizon/hlc_sg_k8_tuned_fullft_seed*_synthetic_bio/survival_curve.csv' \
  --candidates 'runs/survival/syntactic_diversification_sweep_t0/seed*/survival_curve.csv' \
  --r0_tolerance 0.05 \
  --retain_nll_band 0.25 \
  --max_refusal 0.05 \
  --output runs/survival/syntactic_sweep_match_candidates.csv \
  --selected_output runs/survival/syntactic_sweep_match_selected.csv

run "$PYTHON" scripts/run_selected_baseline_stress_eval.py \
  --python "$PYTHON" \
  --selected runs/survival/syntactic_sweep_match_selected.csv \
  --checkpoint_root checkpoints/unlearned/syntactic_diversification_sweep \
  --pools \
    data/relearn_pools/forget_declarative.jsonl \
    data/relearn_pools/forget_paraphrase_qa.jsonl \
    data/relearn_pools/forget_mixed_stress.jsonl \
  --output_root runs/survival/syntactic_diversification_matched_stress \
  ${STRESS_FLAGS[@]+"${STRESS_FLAGS[@]}"}

run "$PYTHON" scripts/compare_methods.py \
  --inputs \
    runs/survival/syntactic_diversification_matched_stress/*/survival_curve.csv \
    runs/survival/stress_lr5e3_matched/hlc_sg_k8_tuned_fullft_seed*_forget_*/survival_curve.csv \
    runs/survival/stress_lr5e3_matched/npo_strong_matched_seed*_forget_*/survival_curve.csv \
  --output runs/reports/syntactic_immediate_matched_stress/compare.csv

run "$PYTHON" scripts/report_conditional_durability.py \
  --inputs \
    runs/survival/syntactic_diversification_matched_stress/*/survival_curve.csv \
    runs/survival/stress_lr5e3_matched/hlc_sg_k8_tuned_fullft_seed*_forget_*/survival_curve.csv \
    runs/survival/stress_lr5e3_matched/npo_strong_matched_seed*_forget_*/survival_curve.csv \
  --horizons 32,128,512 \
  --reference_family "Syntactic diversification" \
  --baseline_family "Strong NPO" \
  --output_dir runs/reports/syntactic_immediate_matched_stress

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Syntactic immediate-matched diagnostic complete."
echo "Selected: runs/survival/syntactic_sweep_match_selected.csv"
echo "Summary: runs/reports/syntactic_immediate_matched_stress/conditional_durability_summary.md"
