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

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

SEED="${SEED:-0}"
STEPS="${STEPS:-0,1,2,4,8,16,32,64,128}"
REPORT_HORIZONS="${REPORT_HORIZONS:-32,128}"
SURVIVAL_LR="${SURVIVAL_LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/survival/gpt2_long_horizon}"
REPORT_DIR="${REPORT_DIR:-runs/reports/gpt2_long_horizon}"
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
Usage: scripts/reproduce_gpt2_long_horizon.sh [--overwrite] [--skip-tests]

Evaluates existing GPT-2 LoRA checkpoints through a longer held-out stress
horizon. Requires the K1/K2 GPT-2 smoke checkpoints.

Defaults:
  SEED=0
  STEPS=0,1,2,4,8,16,32,64,128
  REPORT_HORIZONS=32,128
  SURVIVAL_LR=0.005

Outputs:
  runs/survival/gpt2_long_horizon/
  runs/reports/gpt2_long_horizon/compare.csv
  runs/reports/gpt2_long_horizon/conditional_durability_summary.md
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

run() {
  echo
  echo "+ $*"
  "$@"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    echo "Run scripts/reproduce_gpt2_lora_smoke.sh --skip-tests and scripts/reproduce_gpt2_hlc_k2_smoke.sh --skip-tests first." >&2
    exit 1
  fi
}

pool_label() {
  basename "$1" .jsonl
}

maybe_run_survival() {
  local checkpoint="$1"
  local method="$2"
  local pool="$3"
  local output="$4"
  if [[ "$OVERWRITE" -eq 1 || ! -f "${output}/survival_curve.csv" ]]; then
    run "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "$checkpoint" \
      --base_model gpt2 \
      --thresholds "thresholds/tofu50_gpt2_lora_seed${SEED}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "$pool" \
      --steps "$STEPS" \
      --lr "$SURVIVAL_LR" \
      --batch_size "$BATCH_SIZE" \
      --resurrection_metric margin \
      --method "$method" \
      --seed "$SEED" \
      --output "$output"
  fi
}

echo "Using Python: $PYTHON"
"$PYTHON" --version

require_path "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed${SEED}"
require_path "checkpoints/unlearned/hlc_sg_k1_gpt2_lora_tofu50_seed${SEED}"
require_path "checkpoints/unlearned/hlc_sg_k2_gpt2_lora_tofu50_seed${SEED}"
require_path "thresholds/tofu50_gpt2_lora_seed${SEED}_thresholds.json"
require_path data/prompts/forget_prompt_variants.jsonl
for pool in "${POOLS[@]}"; do
  require_path "$pool"
done

COMPARE_INPUTS=()
for pool in "${POOLS[@]}"; do
  label="$(pool_label "$pool")"
  maybe_run_survival \
    "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed${SEED}" \
    "npo_strong_gpt2_lora_tofu50_seed${SEED}" \
    "$pool" \
    "${OUTPUT_ROOT}/npo_strong_gpt2_lora_seed${SEED}_${label}"
  maybe_run_survival \
    "checkpoints/unlearned/hlc_sg_k1_gpt2_lora_tofu50_seed${SEED}" \
    "hlc_sg_k1_gpt2_lora_tofu50_seed${SEED}" \
    "$pool" \
    "${OUTPUT_ROOT}/hlc_sg_k1_gpt2_lora_seed${SEED}_${label}"
  maybe_run_survival \
    "checkpoints/unlearned/hlc_sg_k2_gpt2_lora_tofu50_seed${SEED}" \
    "hlc_sg_k2_gpt2_lora_tofu50_seed${SEED}" \
    "$pool" \
    "${OUTPUT_ROOT}/hlc_sg_k2_gpt2_lora_seed${SEED}_${label}"
  COMPARE_INPUTS+=("${OUTPUT_ROOT}/npo_strong_gpt2_lora_seed${SEED}_${label}/survival_curve.csv")
  COMPARE_INPUTS+=("${OUTPUT_ROOT}/hlc_sg_k1_gpt2_lora_seed${SEED}_${label}/survival_curve.csv")
  COMPARE_INPUTS+=("${OUTPUT_ROOT}/hlc_sg_k2_gpt2_lora_seed${SEED}_${label}/survival_curve.csv")
done

run "$PYTHON" scripts/compare_methods.py \
  --inputs "${COMPARE_INPUTS[@]}" \
  --output "${REPORT_DIR}/compare.csv"

run "$PYTHON" scripts/report_conditional_durability.py \
  --inputs "${COMPARE_INPUTS[@]}" \
  --horizons "$REPORT_HORIZONS" \
  --reference_family "HLC-SG K2 GPT-2" \
  --baseline_family "Strong NPO" \
  --output_dir "$REPORT_DIR"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "GPT-2 long-horizon diagnostic complete."
echo "Summary: ${REPORT_DIR}/conditional_durability_summary.md"
