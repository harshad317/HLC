#!/usr/bin/env bash
# Durability/utility frontier sweep: displacement over forget_weight vs constrained
# GA over retain_weight, all evaluated at the locked benign-relearn operator on the
# held-out stress pool, then assembled into a frontier table + plot.
#
# Reuses an existing theta_full + thresholds for the given fraction/seed (run the
# main comparison once first). Default: forget01, seed 0.
#
# Usage: scripts/reproduce_displacement_frontier.sh
#        FRACTION=0.01 SEED=0 DISP_LF="1 2 3 4" GA_RW="0.5 1 2 4" scripts/reproduce_displacement_frontier.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

FRACTION="${FRACTION:-0.01}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
DISP_LF="${DISP_LF:-1 2 3 4}"
GA_RW="${GA_RW:-0.5 1 2 4}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/qwen3_0p6b_lora_v100.yaml}"

PCT="$(printf '%02d' "$(python3 -c "print(int(round(${FRACTION}*100)))")")"
DATA_DIR="data/tofu_real"
FORGET="${DATA_DIR}/forget_${PCT}.jsonl"
RETAIN="${DATA_DIR}/retain_${PCT}.jsonl"
PROMPTS="${DATA_DIR}/forget_prompt_variants_${PCT}.jsonl"
STRESS="${DATA_DIR}/relearn_pools_${PCT}/heldout_forget_mixed_stress.jsonl"
DATA_CONFIG="configs/data/tofu${PCT}_real.yaml"
FULL="checkpoints/qwen3_tofu${PCT}_full_seed${SEED}"
THRESH="thresholds/tofu${PCT}_real_qwen3_seed${SEED}_thresholds.json"
OUTDIR="runs/survival/frontier_tofu${PCT}_seed${SEED}"
REPORT="runs/reports/frontier_tofu${PCT}_seed${SEED}"

run() { echo; echo "+ $*"; "$@"; }
slug() { echo "$1" | sed 's/\./p/g'; }

[[ -d "$FULL" ]] || { echo "Missing $FULL -- run the main comparison first."; exit 1; }
[[ -f "$THRESH" ]] || { echo "Missing $THRESH -- run the main comparison first."; exit 1; }

CURVES=()

eval_curve() {  # ckpt method out
  local ckpt="$1" method="$2" out="$3"
  if [[ ! -f "${out}/survival_curve.csv" ]]; then
    run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "$ckpt" --base_model "$BASE_MODEL" --dtype float32 \
      --thresholds "$THRESH" --forget_file "$FORGET" --retain_file "$RETAIN" \
      --prompt_variants "$PROMPTS" --relearn_pool "$STRESS" \
      --steps 0,1,2,4,8,16,32,64,128,256,512 --lr 0.0001 --batch_size 8 --max_length 192 \
      --max_new_tokens 8 --max_retain_items 200 --resurrection_metric margin \
      --method "$method" --seed "$SEED" --output "$out"
  fi
  CURVES+=("${out}/survival_curve.csv")
}

# --- displacement frontier over forget_weight ---
for lf in $DISP_LF; do
  s="$(slug "$lf")"; cfg="configs/method/_frontier_disp_lf${s}.yaml"
  cat > "$cfg" <<EOF
steps: 800
batch_size: 8
lr: 0.0001
forget_weight: ${lf}
retain_weight: 1.0
max_length: 192
max_grad_norm: 0.5
orthogonalize: true
EOF
  ckpt="checkpoints/unlearned/frontier_disp_lf${s}_tofu${PCT}_seed${SEED}"
  [[ -d "$ckpt" ]] || run "$PYTHON" scripts/unlearn.py --method displacement \
    --method_config "$cfg" --model_config "$MODEL_CONFIG" --data_config "$DATA_CONFIG" \
    --start_checkpoint "$FULL" --thresholds "$THRESH" --prompt_variants "$PROMPTS" \
    --output "$ckpt" --seed "$SEED"
  eval_curve "$ckpt" "disp_lf${s}" "${OUTDIR}/disp_lf${s}"
done

# --- GA frontier over retain_weight ---
for rw in $GA_RW; do
  s="$(slug "$rw")"; cfg="configs/method/_frontier_ga_rw${s}.yaml"
  cat > "$cfg" <<EOF
steps: 600
batch_size: 8
lr: 0.0001
retain_weight: ${rw}
max_length: 192
max_grad_norm: 0.5
EOF
  ckpt="checkpoints/unlearned/frontier_ga_rw${s}_tofu${PCT}_seed${SEED}"
  [[ -d "$ckpt" ]] || run "$PYTHON" scripts/unlearn.py --method grad_ascent \
    --method_config "$cfg" --model_config "$MODEL_CONFIG" --data_config "$DATA_CONFIG" \
    --start_checkpoint "$FULL" --thresholds "$THRESH" --prompt_variants "$PROMPTS" \
    --output "$ckpt" --seed "$SEED"
  eval_curve "$ckpt" "ga_rw${s}" "${OUTDIR}/ga_rw${s}"
done

run "$PYTHON" scripts/frontier_table.py --inputs "${CURVES[@]}" --output_dir "$REPORT"
echo; echo "Frontier: ${REPORT}/frontier_table.md (+ frontier.png)"
