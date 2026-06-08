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

SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-0,1,2,4,8,16,32}"
RUN_TESTS=1
CLEAN=0

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
Usage: scripts/reproduce_local_mvp.sh [--clean] [--skip-tests]

Environment:
  PYTHON=/path/to/python     Python executable. Defaults to /opt/anaconda3/bin/python3 when present.
  SEEDS="0 1 2"             Space-separated seed list.
  STEPS="0,1,2,4,8,16,32"   Survival evaluation steps.

Outputs:
  runs/survival/compare_<seed-label>_all_pools.csv
  runs/reports/local_mvp_<seed-label>/
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

POOLS=(synthetic_biography retain_adjacent syntax_matched)

pool_suffix() {
  case "$1" in
    synthetic_biography) echo "synthetic_bio" ;;
    *) echo "$1" ;;
  esac
}

run() {
  echo
  echo "+ $*"
  "$@"
}

make_seed_label() {
  local label=""
  for seed in $SEEDS; do
    if [[ -z "$label" ]]; then
      label="seed${seed}"
    else
      label="${label}_seed${seed}"
    fi
  done
  echo "$label"
}

SEED_LABEL="${SEED_LABEL:-$(make_seed_label)}"
COMPARE_OUTPUT="runs/survival/compare_${SEED_LABEL}_all_pools.csv"
REPORT_DIR="runs/reports/local_mvp_${SEED_LABEL}"

if [[ "$CLEAN" -eq 1 ]]; then
  echo "Cleaning local MVP artifacts for seeds: $SEEDS"
  rm -rf data/tofu/forget_50.jsonl data/tofu/retain_50.jsonl
  rm -rf data/prompts/forget_prompt_variants.jsonl
  rm -rf data/relearn_pools/retain_adjacent.jsonl \
         data/relearn_pools/synthetic_biography.jsonl \
         data/relearn_pools/syntax_matched.jsonl \
         data/relearn_pools/random_general.jsonl
  for seed in $SEEDS; do
    rm -rf "checkpoints/tiny_full_fullft_seed${seed}" \
           "checkpoints/tiny_retain_only_fullft_seed${seed}" \
           "checkpoints/unlearned/npo_strong_fullft_tofu50_seed${seed}" \
           "checkpoints/unlearned/hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
           "thresholds/tofu50_fullft_seed${seed}_thresholds.json"
    for pool in "${POOLS[@]}"; do
      suffix="$(pool_suffix "$pool")"
      rm -rf "runs/survival/hlc_sg_k8_tuned_fullft_seed${seed}_${suffix}" \
             "runs/survival/npo_strong_fullft_seed${seed}_${suffix}"
    done
  done
  rm -f "$COMPARE_OUTPUT" \
        "runs/survival/aggregate_${SEED_LABEL}_all_pools.csv" \
        "runs/survival/aggregate_${SEED_LABEL}_overall.csv"
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

for seed in $SEEDS; do
  run "$PYTHON" scripts/train_full.py \
    --config configs/model/tiny_gpt2_full.yaml \
    --data configs/data/tofu_50_smoke.yaml \
    --output "checkpoints/tiny_full_fullft_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/train_retrain_minus_f.py \
    --config configs/model/tiny_gpt2_full.yaml \
    --data configs/data/tofu_50_smoke.yaml \
    --forget_split forget_50 \
    --output "checkpoints/tiny_retain_only_fullft_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/calibrate_thresholds.py \
    --checkpoint "checkpoints/tiny_retain_only_fullft_seed${seed}" \
    --forget_file data/tofu/forget_50.jsonl \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --quantiles 0.90,0.95,0.99 \
    --tau_global 0.0 \
    --output "thresholds/tofu50_fullft_seed${seed}_thresholds.json"

  run "$PYTHON" scripts/unlearn.py \
    --method npo \
    --method_config configs/method/npo_fullft_smoke.yaml \
    --model_config configs/model/tiny_gpt2_full.yaml \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/tiny_full_fullft_seed${seed}" \
    --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --output "checkpoints/unlearned/npo_strong_fullft_tofu50_seed${seed}" \
    --seed "$seed"

  run "$PYTHON" scripts/unlearn.py \
    --method hlc_sg \
    --method_config configs/method/hlc_sg_k8_fullft_smoke.yaml \
    --model_config configs/model/tiny_gpt2_full.yaml \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/tiny_full_fullft_seed${seed}" \
    --full_checkpoint "checkpoints/tiny_full_fullft_seed${seed}" \
    --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --relearn_pool data/relearn_pools/retain_adjacent.jsonl \
    --output "checkpoints/unlearned/hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
    --seed "$seed"

  for pool in "${POOLS[@]}"; do
    suffix="$(pool_suffix "$pool")"
    run "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "checkpoints/unlearned/hlc_sg_k8_tuned_fullft_tofu50_seed${seed}" \
      --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "data/relearn_pools/${pool}.jsonl" \
      --steps "$STEPS" \
      --resurrection_metric margin \
      --output "runs/survival/hlc_sg_k8_tuned_fullft_seed${seed}_${suffix}"

    run "$PYTHON" scripts/run_survival_curve.py \
      --checkpoint "checkpoints/unlearned/npo_strong_fullft_tofu50_seed${seed}" \
      --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
      --forget_file data/tofu/forget_50.jsonl \
      --retain_file data/tofu/retain_50.jsonl \
      --prompt_variants data/prompts/forget_prompt_variants.jsonl \
      --relearn_pool "data/relearn_pools/${pool}.jsonl" \
      --steps "$STEPS" \
      --resurrection_metric margin \
      --output "runs/survival/npo_strong_fullft_seed${seed}_${suffix}"
  done
done

COMPARE_INPUTS=()
for seed in $SEEDS; do
  for pool in "${POOLS[@]}"; do
    suffix="$(pool_suffix "$pool")"
    COMPARE_INPUTS+=("runs/survival/hlc_sg_k8_tuned_fullft_seed${seed}_${suffix}/survival_curve.csv")
    COMPARE_INPUTS+=("runs/survival/npo_strong_fullft_seed${seed}_${suffix}/survival_curve.csv")
  done
done

run "$PYTHON" scripts/compare_methods.py \
  --inputs "${COMPARE_INPUTS[@]}" \
  --output "$COMPARE_OUTPUT"

run "$PYTHON" scripts/report_local_mvp.py \
  --comparison "$COMPARE_OUTPUT" \
  --output_dir "$REPORT_DIR"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Local MVP reproduction complete."
echo "Summary: ${REPORT_DIR}/summary.md"
