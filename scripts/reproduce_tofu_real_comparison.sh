#!/usr/bin/env bash
# Multi-seed real-TOFU durability comparison: HLC-R vs HLC-base (ablation) vs
# matched constrained GA, under the locked benign-relearn operator (lr=1e-4, batch
# 8), evaluated on the held-out answer-bearing stress pool. Aggregates across seeds
# with bootstrap 95% CIs.
#
# Usage:
#   scripts/reproduce_tofu_real_comparison.sh                 # forget01, seeds 0 1 2
#   FRACTION=0.05 SEEDS="0 1 2" scripts/reproduce_tofu_real_comparison.sh
#
# Requires a GPU box with torch/transformers/peft/datasets and network for the
# first TOFU download. Each stage skips if its output already exists.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

FRACTION="${FRACTION:-0.01}"
SEEDS="${SEEDS:-0 1 2}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/qwen3_0p6b_lora_v100.yaml}"
GPU="${GPU:-0}"

# Locked benign-relearn operator + eval settings.
RE_LR="${RE_LR:-0.0001}"
RE_BATCH="${RE_BATCH:-8}"
STEPS="${STEPS:-0,1,2,4,8,16,32,64,128,256,512}"
MAX_LENGTH="${MAX_LENGTH:-192}"
MAX_RETAIN="${MAX_RETAIN:-200}"

PCT="$(printf '%02d' "$(python3 -c "print(int(round(${FRACTION}*100)))")")"
DATA_DIR="data/tofu_real"
FORGET="${DATA_DIR}/forget_${PCT}.jsonl"
RETAIN="${DATA_DIR}/retain_${PCT}.jsonl"
PROMPTS="${DATA_DIR}/forget_prompt_variants_${PCT}.jsonl"
POOLDIR="${DATA_DIR}/relearn_pools_${PCT}"
STRESS="${POOLDIR}/heldout_forget_mixed_stress.jsonl"
DATA_CONFIG="configs/data/tofu${PCT}_real.yaml"

HLC_R_CFG="configs/method/hlc_r_k4_qwen3_lora_v100.yaml"
HLC_BASE_CFG="configs/method/hlc_base_k4_qwen3.yaml"
GA_CFG="configs/method/grad_ascent_real_matched.yaml"
DISP_CFG="configs/method/displacement_qwen3.yaml"
# Set INCLUDE_DISPLACEMENT=1 to also run the displacement-durability prototype.
INCLUDE_DISPLACEMENT="${INCLUDE_DISPLACEMENT:-0}"

REPORT_DIR="runs/reports/tofu${PCT}_real_comparison"

run() { echo; echo "+ $*"; "$@"; }
gpu() { CUDA_VISIBLE_DEVICES="$GPU" "$@"; }

echo "== Real-TOFU comparison: forget${PCT}, seeds [${SEEDS}] =="

# ---- one-time data prep ----
if [[ ! -f "$FORGET" ]]; then
  run "$PYTHON" scripts/prepare_tofu.py --source tofu --forget_fraction "$FRACTION" --output_dir "$DATA_DIR"
fi
[[ -f "$PROMPTS" ]] || run "$PYTHON" scripts/make_prompt_variants.py --forget_file "$FORGET" --output "$PROMPTS"
[[ -f "${POOLDIR}/retain_adjacent.jsonl" ]] || run "$PYTHON" scripts/make_relearn_pools.py --retain_file "$RETAIN" --output_dir "$POOLDIR"
[[ -f "$STRESS" ]] || run "$PYTHON" scripts/make_heldout_stress_relearn_pools.py --forget_file "$FORGET" --output_dir "$POOLDIR" --variants_per_item 3

CURVES=()

for seed in $SEEDS; do
  echo "== seed ${seed} =="
  FULL="checkpoints/qwen3_tofu${PCT}_full_seed${seed}"
  RETAIN_CKPT="checkpoints/qwen3_tofu${PCT}_retain_only_seed${seed}"
  THRESH="thresholds/tofu${PCT}_real_qwen3_seed${seed}_thresholds.json"

  [[ -d "$FULL" ]] || run "$PYTHON" scripts/train_full.py \
    --config "$MODEL_CONFIG" --data "$DATA_CONFIG" --output "$FULL" --seed "$seed"

  [[ -d "$RETAIN_CKPT" ]] || run "$PYTHON" scripts/train_retrain_minus_f.py \
    --config "$MODEL_CONFIG" --data "$DATA_CONFIG" --forget_split "forget_${PCT}" \
    --output "$RETAIN_CKPT" --seed "$seed"

  [[ -f "$THRESH" ]] || run gpu "$PYTHON" scripts/calibrate_thresholds.py \
    --checkpoint "$RETAIN_CKPT" --base_model "$BASE_MODEL" --dtype float32 \
    --forget_file "$FORGET" --prompt_variants "$PROMPTS" \
    --quantiles 0.90,0.95,0.99 --tau_global 0.0 --max_length "$MAX_LENGTH" --output "$THRESH"

  # spec = name:method_flag:config  (hlc_sg methods also need --full_checkpoint + --relearn_pool)
  SPECS=("hlc_r:hlc_sg:${HLC_R_CFG}" "hlc_base:hlc_sg:${HLC_BASE_CFG}" "ga:grad_ascent:${GA_CFG}")
  [[ "$INCLUDE_DISPLACEMENT" == "1" ]] && SPECS+=("displacement:displacement:${DISP_CFG}")
  for spec in "${SPECS[@]}"; do
    name="${spec%%:*}"; rest="${spec#*:}"; mflag="${rest%%:*}"; cfg="${rest##*:}"
    ckpt="checkpoints/unlearned/${name}_tofu${PCT}_seed${seed}"
    out="runs/survival/${name}_tofu${PCT}_seed${seed}_lr1e4"

    if [[ ! -d "$ckpt" ]]; then
      if [[ "$mflag" == "hlc_sg" ]]; then
        run gpu "$PYTHON" scripts/unlearn.py --method hlc_sg --method_config "$cfg" \
          --model_config "$MODEL_CONFIG" --data_config "$DATA_CONFIG" \
          --start_checkpoint "$FULL" --full_checkpoint "$FULL" --thresholds "$THRESH" \
          --prompt_variants "$PROMPTS" --relearn_pool "${POOLDIR}/retain_adjacent.jsonl" \
          --output "$ckpt" --seed "$seed"
      else
        run gpu "$PYTHON" scripts/unlearn.py --method "$mflag" --method_config "$cfg" \
          --model_config "$MODEL_CONFIG" --data_config "$DATA_CONFIG" \
          --start_checkpoint "$FULL" --thresholds "$THRESH" --prompt_variants "$PROMPTS" \
          --output "$ckpt" --seed "$seed"
      fi
    fi

    if [[ ! -f "${out}/survival_curve.csv" ]]; then
      run gpu "$PYTHON" scripts/run_survival_curve.py \
        --checkpoint "$ckpt" --base_model "$BASE_MODEL" --dtype float32 \
        --thresholds "$THRESH" --forget_file "$FORGET" --retain_file "$RETAIN" \
        --prompt_variants "$PROMPTS" --relearn_pool "$STRESS" \
        --steps "$STEPS" --lr "$RE_LR" --batch_size "$RE_BATCH" --max_length "$MAX_LENGTH" \
        --max_new_tokens 8 --max_retain_items "$MAX_RETAIN" --resurrection_metric margin \
        --method "${name}" --seed "$seed" --output "$out"
    fi
    CURVES+=("${out}/survival_curve.csv")
  done
done

# ---- aggregate with bootstrap CIs ----
run "$PYTHON" scripts/aggregate_tofu_real.py --inputs "${CURVES[@]}" --output_dir "$REPORT_DIR"

echo
echo "Done. Aggregate: ${REPORT_DIR}/aggregate_ci_summary.md"
