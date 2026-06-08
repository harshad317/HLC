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

SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-0,1,2,4,8,16,32}"
REPORT_HORIZONS="${REPORT_HORIZONS:-32}"
SURVIVAL_LR="${SURVIVAL_LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RUN_TESTS=1
CLEAN=0

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/gpt2_lora_smoke.yaml}"
NPO_CONFIG="${NPO_CONFIG:-configs/method/strong_npo_gpt2_lora_smoke.yaml}"
HLC_CONFIG="${HLC_CONFIG:-configs/method/hlc_sg_k1_gpt2_lora_smoke.yaml}"
HLC_LABEL="${HLC_LABEL:-hlc_sg_k1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/survival/gpt2_lora_smoke}"
REPORT_DIR="${REPORT_DIR:-runs/reports/gpt2_lora_smoke}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/reproduce_gpt2_lora_smoke.sh [--clean] [--skip-tests]

Runs the first larger-model local smoke lane with GPT-2 + LoRA:

  1. prepare synthetic TOFU-50 data and held-out stress pools
  2. train full and retain-only GPT-2 LoRA adapters
  3. calibrate retain-only thresholds
  4. run capped Strong NPO and HLC-SG
  5. evaluate on held-out answer-bearing stress pools

Defaults:
  SEEDS=0
  STEPS=0,1,2,4,8,16,32
  REPORT_HORIZONS=32
  SURVIVAL_LR=0.005
  HF_HUB_DISABLE_XET=1
  MODEL_CONFIG=configs/model/gpt2_lora_smoke.yaml
  NPO_CONFIG=configs/method/strong_npo_gpt2_lora_smoke.yaml
  HLC_CONFIG=configs/method/hlc_sg_k1_gpt2_lora_smoke.yaml

Override HLC_CONFIG and HLC_LABEL to run a stronger HLC config after this smoke gate.
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

pool_label() {
  basename "$1" .jsonl
}

maybe_run_survival() {
  local checkpoint="$1"
  local method="$2"
  local seed="$3"
  local pool="$4"
  local output="$5"
  if [[ ! -f "${output}/survival_curve.csv" ]]; then
    run "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "$checkpoint" \
      --base_model gpt2 \
      --thresholds "thresholds/tofu50_gpt2_lora_seed${seed}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "$pool" \
      --steps "$STEPS" \
      --lr "$SURVIVAL_LR" \
      --batch_size "$BATCH_SIZE" \
      --resurrection_metric margin \
      --method "$method" \
      --seed "$seed" \
      --output "$output"
  fi
}

if [[ "$CLEAN" -eq 1 ]]; then
  for seed in $SEEDS; do
    rm -rf "checkpoints/gpt2_lora_full_seed${seed}" \
           "checkpoints/gpt2_lora_retain_only_seed${seed}" \
           "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed${seed}" \
           "checkpoints/unlearned/${HLC_LABEL}_gpt2_lora_tofu50_seed${seed}" \
           "thresholds/tofu50_gpt2_lora_seed${seed}_thresholds.json"
    rm -rf "${OUTPUT_ROOT}"/*"seed${seed}"*
  done
  rm -rf "$REPORT_DIR"
fi

echo "Using Python: $PYTHON"
"$PYTHON" --version

run "$PYTHON" scripts/prepare_tofu.py \
  --output_dir data/tofu \
  --forget_fraction 0.5 \
  --synthetic

run "$PYTHON" scripts/make_prompt_variants.py \
  --forget_file data/tofu/forget_50.jsonl \
  --output data/prompts/forget_prompt_variants.jsonl

run "$PYTHON" scripts/make_relearn_pools.py \
  --retain_file data/tofu/retain_50.jsonl \
  --output_dir data/relearn_pools

run "$PYTHON" scripts/make_heldout_stress_relearn_pools.py \
  --forget_file data/tofu/forget_50.jsonl \
  --output_dir data/relearn_pools \
  --variants_per_item 3

COMPARE_INPUTS=()

for seed in $SEEDS; do
  run "$PYTHON" scripts/train_full.py \
    --config "$MODEL_CONFIG" \
    --data configs/data/tofu_50_smoke.yaml \
    --output "checkpoints/gpt2_lora_full_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/train_retrain_minus_f.py \
    --config "$MODEL_CONFIG" \
    --data configs/data/tofu_50_smoke.yaml \
    --forget_split forget_50 \
    --output "checkpoints/gpt2_lora_retain_only_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/calibrate_thresholds.py \
    --checkpoint "checkpoints/gpt2_lora_retain_only_seed${seed}" \
    --base_model gpt2 \
    --forget_file data/tofu/forget_50.jsonl \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --quantiles 0.90,0.95,0.99 \
    --tau_global 0.0 \
    --output "thresholds/tofu50_gpt2_lora_seed${seed}_thresholds.json"

  run "$PYTHON" scripts/unlearn.py \
    --method npo \
    --method_config "$NPO_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/gpt2_lora_full_seed${seed}" \
    --thresholds "thresholds/tofu50_gpt2_lora_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --output "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/unlearn.py \
    --method hlc_sg \
    --method_config "$HLC_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/gpt2_lora_full_seed${seed}" \
    --full_checkpoint "checkpoints/gpt2_lora_full_seed${seed}" \
    --thresholds "thresholds/tofu50_gpt2_lora_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --relearn_pool data/relearn_pools/retain_adjacent.jsonl \
    --output "checkpoints/unlearned/${HLC_LABEL}_gpt2_lora_tofu50_seed${seed}" \
    --seed "$seed"

  for pool in "${POOLS[@]}"; do
    label="$(pool_label "$pool")"
    npo_output="${OUTPUT_ROOT}/npo_strong_gpt2_lora_seed${seed}_${label}"
    hlc_output="${OUTPUT_ROOT}/${HLC_LABEL}_gpt2_lora_seed${seed}_${label}"
    maybe_run_survival \
      "checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed${seed}" \
      "npo_strong_gpt2_lora_tofu50_seed${seed}" \
      "$seed" \
      "$pool" \
      "$npo_output"
    maybe_run_survival \
      "checkpoints/unlearned/${HLC_LABEL}_gpt2_lora_tofu50_seed${seed}" \
      "${HLC_LABEL}_gpt2_lora_tofu50_seed${seed}" \
      "$seed" \
      "$pool" \
      "$hlc_output"
    COMPARE_INPUTS+=("${npo_output}/survival_curve.csv")
    COMPARE_INPUTS+=("${hlc_output}/survival_curve.csv")
  done
done

run "$PYTHON" scripts/compare_methods.py \
  --inputs "${COMPARE_INPUTS[@]}" \
  --output "${REPORT_DIR}/compare.csv"

run "$PYTHON" scripts/report_conditional_durability.py \
  --inputs "${COMPARE_INPUTS[@]}" \
  --horizons "$REPORT_HORIZONS" \
  --reference_family "HLC-SG" \
  --baseline_family "Strong NPO" \
  --output_dir "$REPORT_DIR"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "GPT-2 LoRA smoke diagnostic complete."
echo "Summary: ${REPORT_DIR}/conditional_durability_summary.md"
