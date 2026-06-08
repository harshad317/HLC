#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="python3"
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_TAG="${MODEL_TAG:-qwen3_0p6b}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
DTYPE="${DTYPE:-float16}"
SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-0,1,2,4,8,16,32,64,128}"
REPORT_HORIZONS="${REPORT_HORIZONS:-32,128}"
SURVIVAL_LR="${SURVIVAL_LR:-0.005}"
SURVIVAL_BATCH_SIZE="${SURVIVAL_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-192}"
SKIP_PREP="${SKIP_PREP:-0}"
RUN_TESTS=1
CLEAN=0

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/qwen3_0p6b_lora_v100.yaml}"
NPO_CONFIG="${NPO_CONFIG:-configs/method/strong_npo_qwen3_lora_v100.yaml}"
HLC_CONFIG="${HLC_CONFIG:-configs/method/hlc_sg_k2_qwen3_lora_v100.yaml}"
HLC_LABEL="${HLC_LABEL:-hlc_sg_k2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/survival/${MODEL_TAG}_lora_v100_smoke}"
REPORT_DIR="${REPORT_DIR:-runs/reports/${MODEL_TAG}_lora_v100_smoke}"

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
Usage: scripts/reproduce_qwen3_lora_v100_smoke.sh [--clean] [--skip-tests]

Runs a modern Qwen3 + LoRA V100 diagnostic lane.

Safe default:
  MODEL_TAG=qwen3_0p6b
  BASE_MODEL=Qwen/Qwen3-0.6B
  MODEL_CONFIG=configs/model/qwen3_0p6b_lora_v100.yaml

Stronger follow-up on 16GB V100, if the default passes:
  MODEL_TAG=qwen3_1p7b
  BASE_MODEL=Qwen/Qwen3-1.7B
  MODEL_CONFIG=configs/model/qwen3_1p7b_lora_v100.yaml
  MAX_LENGTH=160

Defaults:
  DTYPE=float16
  SEEDS=0
  STEPS=0,1,2,4,8,16,32,64,128
  REPORT_HORIZONS=32,128
  SURVIVAL_BATCH_SIZE=1
  SKIP_PREP=0
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
      --base_model "$BASE_MODEL" \
      --dtype "$DTYPE" \
      --thresholds "thresholds/tofu50_${MODEL_TAG}_lora_seed${seed}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "$pool" \
      --steps "$STEPS" \
      --lr "$SURVIVAL_LR" \
      --batch_size "$SURVIVAL_BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      --resurrection_metric margin \
      --method "$method" \
      --seed "$seed" \
      --output "$output"
  fi
}

if [[ "$CLEAN" -eq 1 ]]; then
  for seed in $SEEDS; do
    rm -rf "checkpoints/${MODEL_TAG}_lora_full_seed${seed}" \
           "checkpoints/${MODEL_TAG}_lora_retain_only_seed${seed}" \
           "checkpoints/unlearned/npo_strong_${MODEL_TAG}_lora_tofu50_seed${seed}" \
           "checkpoints/unlearned/${HLC_LABEL}_${MODEL_TAG}_lora_tofu50_seed${seed}" \
           "thresholds/tofu50_${MODEL_TAG}_lora_seed${seed}_thresholds.json"
    rm -rf "${OUTPUT_ROOT}"/*"seed${seed}"*
  done
  rm -rf "$REPORT_DIR"
fi

echo "Using Python: $PYTHON"
"$PYTHON" --version

if [[ "$SKIP_PREP" -eq 0 ]]; then
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
fi

COMPARE_INPUTS=()

for seed in $SEEDS; do
  run "$PYTHON" scripts/train_full.py \
    --config "$MODEL_CONFIG" \
    --data configs/data/tofu_50_smoke.yaml \
    --output "checkpoints/${MODEL_TAG}_lora_full_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/train_retrain_minus_f.py \
    --config "$MODEL_CONFIG" \
    --data configs/data/tofu_50_smoke.yaml \
    --forget_split forget_50 \
    --output "checkpoints/${MODEL_TAG}_lora_retain_only_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/calibrate_thresholds.py \
    --checkpoint "checkpoints/${MODEL_TAG}_lora_retain_only_seed${seed}" \
    --base_model "$BASE_MODEL" \
    --dtype "$DTYPE" \
    --forget_file data/tofu/forget_50.jsonl \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --quantiles 0.90,0.95,0.99 \
    --tau_global 0.0 \
    --max_length "$MAX_LENGTH" \
    --output "thresholds/tofu50_${MODEL_TAG}_lora_seed${seed}_thresholds.json"

  run "$PYTHON" scripts/unlearn.py \
    --method npo \
    --method_config "$NPO_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/${MODEL_TAG}_lora_full_seed${seed}" \
    --thresholds "thresholds/tofu50_${MODEL_TAG}_lora_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --output "checkpoints/unlearned/npo_strong_${MODEL_TAG}_lora_tofu50_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/unlearn.py \
    --method hlc_sg \
    --method_config "$HLC_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/${MODEL_TAG}_lora_full_seed${seed}" \
    --full_checkpoint "checkpoints/${MODEL_TAG}_lora_full_seed${seed}" \
    --thresholds "thresholds/tofu50_${MODEL_TAG}_lora_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --relearn_pool data/relearn_pools/retain_adjacent.jsonl \
    --output "checkpoints/unlearned/${HLC_LABEL}_${MODEL_TAG}_lora_tofu50_seed${seed}" \
    --seed "$seed"

  for pool in "${POOLS[@]}"; do
    label="$(pool_label "$pool")"
    npo_output="${OUTPUT_ROOT}/npo_strong_${MODEL_TAG}_lora_seed${seed}_${label}"
    hlc_output="${OUTPUT_ROOT}/${HLC_LABEL}_${MODEL_TAG}_lora_seed${seed}_${label}"
    maybe_run_survival \
      "checkpoints/unlearned/npo_strong_${MODEL_TAG}_lora_tofu50_seed${seed}" \
      "npo_strong_${MODEL_TAG}_lora_tofu50_seed${seed}" \
      "$seed" \
      "$pool" \
      "$npo_output"
    maybe_run_survival \
      "checkpoints/unlearned/${HLC_LABEL}_${MODEL_TAG}_lora_tofu50_seed${seed}" \
      "${HLC_LABEL}_${MODEL_TAG}_lora_tofu50_seed${seed}" \
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
  --reference_family "HLC-SG K2" \
  --baseline_family "Strong NPO" \
  --output_dir "$REPORT_DIR"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Qwen3 V100 LoRA diagnostic complete."
echo "Summary: ${REPORT_DIR}/conditional_durability_summary.md"
