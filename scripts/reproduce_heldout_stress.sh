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
STEPS="${STEPS:-0,1,2,4,8,16,32,64,128,256,512}"
LR="${LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/survival/heldout_stress}"
REPORT_DIR="${REPORT_DIR:-runs/reports/heldout_stress}"

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
Usage: scripts/reproduce_heldout_stress.sh [--overwrite] [--skip-tests]

Builds held-out answer-bearing relearn pools and evaluates:

  HLC-SG
  immediate-matched Strong NPO
  immediate-matched syntactic diversification
  NPO + paraphrases

Outputs:
  data/relearn_pools/heldout_forget_*.jsonl
  runs/survival/heldout_stress/
  runs/reports/heldout_stress/compare.csv
  runs/reports/heldout_stress/conditional_horizon_aggregate.csv
  runs/reports/heldout_stress/conditional_durability_summary.md
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

maybe_run_survival() {
  local checkpoint="$1"
  local method="$2"
  local seed="$3"
  local pool="$4"
  local output="$5"
  if [[ "$OVERWRITE" -eq 1 || ! -f "${output}/survival_curve.csv" ]]; then
    run "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "$checkpoint" \
      --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "$pool" \
      --steps "$STEPS" \
      --lr "$LR" \
      --batch_size "$BATCH_SIZE" \
      --resurrection_metric margin \
      --method "$method" \
      --seed "$seed" \
      --output "$output"
  fi
}

pool_label() {
  basename "$1" .jsonl
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

for seed in "${SEEDS[@]}"; do
  for pool in "${POOLS[@]}"; do
    label="$(pool_label "$pool")"
    maybe_run_survival \
      "checkpoints/unlearned/hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
      "hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
      "$seed" \
      "$pool" \
      "${OUTPUT_ROOT}/hlc_sg_k8_tuned_fullft_seed${seed}_${label}"

    maybe_run_survival \
      "checkpoints/unlearned/npo_paraphrase_fullft_tofu50_seed${seed}" \
      "npo_paraphrase_fullft_tofu50_seed${seed}" \
      "$seed" \
      "$pool" \
      "${OUTPUT_ROOT}/npo_paraphrase_fullft_seed${seed}_${label}"
  done
done

run "$PYTHON" scripts/run_selected_baseline_stress_eval.py \
  --python "$PYTHON" \
  --selected runs/survival/strong_npo_sweep_match_selected.csv \
  --checkpoint_root checkpoints/unlearned/strong_npo_sweep \
  --pools "${POOLS[@]}" \
  --steps "$STEPS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --method_prefix npo_strong_matched \
  --output_root "${OUTPUT_ROOT}/strong_npo" \
  ${STRESS_FLAGS[@]+"${STRESS_FLAGS[@]}"}

run "$PYTHON" scripts/run_selected_baseline_stress_eval.py \
  --python "$PYTHON" \
  --selected runs/survival/syntactic_sweep_match_selected.csv \
  --checkpoint_root checkpoints/unlearned/syntactic_diversification_sweep \
  --pools "${POOLS[@]}" \
  --steps "$STEPS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --output_root "${OUTPUT_ROOT}/syntactic_diversification_matched" \
  ${STRESS_FLAGS[@]+"${STRESS_FLAGS[@]}"}

run "$PYTHON" scripts/compare_methods.py \
  --inputs \
    "${OUTPUT_ROOT}"/hlc_sg_k8_tuned_fullft_seed*_heldout_forget_*/survival_curve.csv \
    "${OUTPUT_ROOT}"/npo_paraphrase_fullft_seed*_heldout_forget_*/survival_curve.csv \
    "${OUTPUT_ROOT}"/strong_npo/*_heldout_forget_*/survival_curve.csv \
    "${OUTPUT_ROOT}"/syntactic_diversification_matched/*_heldout_forget_*/survival_curve.csv \
  --output "${REPORT_DIR}/compare.csv"

run "$PYTHON" scripts/report_conditional_durability.py \
  --inputs \
    "${OUTPUT_ROOT}"/hlc_sg_k8_tuned_fullft_seed*_heldout_forget_*/survival_curve.csv \
    "${OUTPUT_ROOT}"/npo_paraphrase_fullft_seed*_heldout_forget_*/survival_curve.csv \
    "${OUTPUT_ROOT}"/strong_npo/*_heldout_forget_*/survival_curve.csv \
    "${OUTPUT_ROOT}"/syntactic_diversification_matched/*_heldout_forget_*/survival_curve.csv \
  --horizons 32,128,512 \
  --reference_family "HLC-SG" \
  --baseline_family "Strong NPO" \
  --output_dir "$REPORT_DIR"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Held-out stress diagnostic complete."
echo "Summary: ${REPORT_DIR}/conditional_durability_summary.md"
