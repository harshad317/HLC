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
BUDGET_LRS="${BUDGET_LRS:-0.01}"
STEPS="${STEPS:-0,1,2,4,8,16,32,64,128,256,512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/survival/update_budget_stress}"
REPORT_ROOT="${REPORT_ROOT:-runs/reports/update_budget_stress}"

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
    --budgets)
      BUDGET_LRS="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/reproduce_update_budget_stress.sh [--overwrite] [--skip-tests] [--budgets "0.01 0.02"]

Evaluates HLC-SG, NPO + paraphrases, immediate-matched Strong NPO, and
immediate-matched syntactic diversification under stronger answer-bearing
held-out relearning update budgets.

Defaults:
  BUDGET_LRS=0.01
  STEPS=0,1,2,4,8,16,32,64,128,256,512
  BATCH_SIZE=2

Outputs:
  runs/survival/update_budget_stress/<lr_label>/
  runs/reports/update_budget_stress/<lr_label>/conditional_durability_summary.md
  runs/reports/update_budget_stress/budget_horizon_aggregate.csv
  runs/reports/update_budget_stress/budget_stress_summary.md
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

POOLS=(
  data/relearn_pools/heldout_forget_declarative.jsonl
  data/relearn_pools/heldout_forget_paraphrase_qa.jsonl
  data/relearn_pools/heldout_forget_mixed_stress.jsonl
)
SEEDS=(0 1 2)

run() {
  echo
  echo "+ $*"
  "$@"
}

lr_label() {
  local value="$1"
  local label="${value//./p}"
  label="${label//-/m}"
  echo "lr${label}"
}

pool_label() {
  basename "$1" .jsonl
}

maybe_run_survival() {
  local checkpoint="$1"
  local method="$2"
  local seed="$3"
  local pool="$4"
  local lr="$5"
  local output="$6"
  if [[ "$OVERWRITE" -eq 1 || ! -f "${output}/survival_curve.csv" ]]; then
    run "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "$checkpoint" \
      --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "$pool" \
      --steps "$STEPS" \
      --lr "$lr" \
      --batch_size "$BATCH_SIZE" \
      --resurrection_metric margin \
      --method "$method" \
      --seed "$seed" \
      --output "$output"
  fi
}

STRESS_FLAGS=()
if [[ "$OVERWRITE" -eq 1 ]]; then
  STRESS_FLAGS+=(--overwrite)
fi

echo "Using Python: $PYTHON"
"$PYTHON" --version

run "$PYTHON" scripts/make_heldout_stress_relearn_pools.py \
  --forget_file data/tofu/forget_50.jsonl \
  --output_dir data/relearn_pools \
  --variants_per_item 3

REPORT_INPUTS=()

for lr in $BUDGET_LRS; do
  label="$(lr_label "$lr")"
  output_root="${OUTPUT_ROOT_BASE}/${label}"
  report_dir="${REPORT_ROOT}/${label}"
  echo
  echo "=== update budget: ${label} (lr=${lr}) ==="

  for seed in "${SEEDS[@]}"; do
    for pool in "${POOLS[@]}"; do
      pool_name="$(pool_label "$pool")"
      maybe_run_survival \
        "checkpoints/unlearned/hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
        "hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
        "$seed" \
        "$pool" \
        "$lr" \
        "${output_root}/hlc_sg_k8_tuned_fullft_seed${seed}_${pool_name}"

      maybe_run_survival \
        "checkpoints/unlearned/npo_paraphrase_fullft_tofu50_seed${seed}" \
        "npo_paraphrase_fullft_tofu50_seed${seed}" \
        "$seed" \
        "$pool" \
        "$lr" \
        "${output_root}/npo_paraphrase_fullft_seed${seed}_${pool_name}"
    done
  done

  run "$PYTHON" scripts/run_selected_baseline_stress_eval.py \
    --python "$PYTHON" \
    --selected runs/survival/strong_npo_sweep_match_selected.csv \
    --checkpoint_root checkpoints/unlearned/strong_npo_sweep \
    --pools "${POOLS[@]}" \
    --steps "$STEPS" \
    --lr "$lr" \
    --batch_size "$BATCH_SIZE" \
    --method_prefix npo_strong_matched \
    --output_root "${output_root}/strong_npo" \
    ${STRESS_FLAGS[@]+"${STRESS_FLAGS[@]}"}

  run "$PYTHON" scripts/run_selected_baseline_stress_eval.py \
    --python "$PYTHON" \
    --selected runs/survival/syntactic_sweep_match_selected.csv \
    --checkpoint_root checkpoints/unlearned/syntactic_diversification_sweep \
    --pools "${POOLS[@]}" \
    --steps "$STEPS" \
    --lr "$lr" \
    --batch_size "$BATCH_SIZE" \
    --output_root "${output_root}/syntactic_diversification_matched" \
    ${STRESS_FLAGS[@]+"${STRESS_FLAGS[@]}"}

  run "$PYTHON" scripts/compare_methods.py \
    --inputs \
      "${output_root}"/hlc_sg_k8_tuned_fullft_seed*_heldout_forget_*/survival_curve.csv \
      "${output_root}"/npo_paraphrase_fullft_seed*_heldout_forget_*/survival_curve.csv \
      "${output_root}"/strong_npo/*_heldout_forget_*/survival_curve.csv \
      "${output_root}"/syntactic_diversification_matched/*_heldout_forget_*/survival_curve.csv \
    --output "${report_dir}/compare.csv"

  run "$PYTHON" scripts/report_conditional_durability.py \
    --inputs \
      "${output_root}"/hlc_sg_k8_tuned_fullft_seed*_heldout_forget_*/survival_curve.csv \
      "${output_root}"/npo_paraphrase_fullft_seed*_heldout_forget_*/survival_curve.csv \
      "${output_root}"/strong_npo/*_heldout_forget_*/survival_curve.csv \
      "${output_root}"/syntactic_diversification_matched/*_heldout_forget_*/survival_curve.csv \
    --horizons 32,128,512 \
    --reference_family "HLC-SG" \
    --baseline_family "Strong NPO" \
    --output_dir "$report_dir"

  REPORT_INPUTS+=("${label}=${report_dir}/conditional_horizon_aggregate.csv")
done

run "$PYTHON" scripts/report_update_budget_stress.py \
  --inputs "${REPORT_INPUTS[@]}" \
  --output_dir "$REPORT_ROOT" \
  --focus_horizon 512 \
  --reference_family "HLC-SG" \
  --baseline_families "Strong NPO,Syntactic diversification"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Update-budget stress diagnostic complete."
echo "Summary: ${REPORT_ROOT}/budget_stress_summary.md"
