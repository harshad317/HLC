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
BASELINES="${BASELINES:-npo_paraphrase}"
STEPS="${STEPS:-0,1,2,4,8,16,32,64,128,256,512}"
LR="${LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/survival/missing_baselines_stress}"
REPORT_DIR="${REPORT_DIR:-runs/reports/missing_baselines_stress}"
REFERENCE_ROOT="${REFERENCE_ROOT:-runs/survival/stress_lr5e3_matched}"
RUN_TESTS=1
TRAIN_MISSING=0
OVERWRITE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baselines)
      BASELINES="$2"
      shift 2
      ;;
    --train-missing)
      TRAIN_MISSING=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/reproduce_missing_baselines.sh [--baselines "npo_paraphrase"] [--train-missing] [--overwrite] [--dry-run] [--skip-tests]

Runs the missing-baseline stress evaluation package.

Defaults evaluate NPO + paraphrases over the answer-bearing stress pools with
seeds 0, 1, and 2 at horizon 0,1,2,4,8,16,32,64,128,256,512.

Environment:
  PYTHON=/path/to/python       Python executable. Defaults to /opt/anaconda3/bin/python3 when present.
  SEEDS="0 1 2"               Space-separated seed list.
  BASELINES="npo_paraphrase"  Supported: npo_paraphrase syntactic_diversification
  STEPS="0,1,2,4,8,16,32,64,128,256,512"
  LR=0.005                    Benign relearning LR used by run_survival_curve.py.
  BATCH_SIZE=2                Benign relearning batch size.
  OUTPUT_ROOT=...             Survival-curve output root.
  REPORT_DIR=...              Aggregated report output root.

Outputs:
  ${OUTPUT_ROOT}/<baseline>_fullft_seed<seed>_<stress_pool>/survival_curve.csv
  ${REPORT_DIR}/missing_baselines_compare.csv
  ${REPORT_DIR}/conditional_horizon_aggregate.csv
  ${REPORT_DIR}/conditional_durability_summary.md
  ${REPORT_DIR}/<baseline>/conditional_horizon_aggregate.csv
  ${REPORT_DIR}/<baseline>/conditional_durability_summary.md
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

POOLS=(forget_declarative forget_paraphrase_qa forget_mixed_stress)

run() {
  echo
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

baseline_config() {
  case "$1" in
    npo_paraphrase) echo "configs/method/npo_paraphrase_fullft_smoke.yaml" ;;
    syntactic_diversification) echo "configs/method/syntactic_diversification_fullft_smoke.yaml" ;;
    *)
      echo "Unsupported baseline: $1" >&2
      exit 2
      ;;
  esac
}

baseline_checkpoint() {
  local baseline="$1"
  local seed="$2"
  case "$baseline" in
    npo_paraphrase) echo "checkpoints/unlearned/npo_paraphrase_fullft_tofu50_seed${seed}" ;;
    syntactic_diversification) echo "checkpoints/unlearned/syntactic_diversification_fullft_tofu50_seed${seed}" ;;
    *)
      echo "Unsupported baseline: $baseline" >&2
      exit 2
      ;;
  esac
}

baseline_method_label() {
  local baseline="$1"
  local seed="$2"
  case "$baseline" in
    npo_paraphrase) echo "npo_paraphrase_fullft_tofu50_seed${seed}" ;;
    syntactic_diversification) echo "syntactic_diversification_fullft_tofu50_seed${seed}" ;;
    *)
      echo "Unsupported baseline: $baseline" >&2
      exit 2
      ;;
  esac
}

baseline_output_prefix() {
  local baseline="$1"
  local seed="$2"
  case "$baseline" in
    npo_paraphrase) echo "npo_paraphrase_fullft_seed${seed}" ;;
    syntactic_diversification) echo "syntactic_diversification_fullft_seed${seed}" ;;
    *)
      echo "Unsupported baseline: $baseline" >&2
      exit 2
      ;;
  esac
}

baseline_family_label() {
  case "$1" in
    npo_paraphrase) echo "NPO + paraphrases" ;;
    syntactic_diversification) echo "Syntactic diversification" ;;
    *)
      echo "Unsupported baseline: $1" >&2
      exit 2
      ;;
  esac
}

train_baseline_if_needed() {
  local baseline="$1"
  local seed="$2"
  local checkpoint
  checkpoint="$(baseline_checkpoint "$baseline" "$seed")"
  if [[ -f "${checkpoint}/run_metadata.json" ]]; then
    return
  fi
  if [[ "$TRAIN_MISSING" -ne 1 ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    echo "Rerun with --train-missing to create it." >&2
    exit 1
  fi
  run "$PYTHON" scripts/unlearn.py \
    --method npo \
    --method_config "$(baseline_config "$baseline")" \
    --model_config configs/model/tiny_gpt2_full.yaml \
    --data_config configs/data/tofu_50_smoke.yaml \
    --start_checkpoint "checkpoints/tiny_full_fullft_seed${seed}" \
    --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
    --prompt_variants data/prompts/forget_prompt_variants.jsonl \
    --output "$checkpoint" \
    --seed "$seed"
}

echo "Using Python: $PYTHON"
"$PYTHON" --version

require_file data/tofu/forget_50.jsonl
require_file data/tofu/retain_50.jsonl
require_file data/prompts/forget_prompt_variants.jsonl
for pool in "${POOLS[@]}"; do
  require_file "data/relearn_pools/${pool}.jsonl"
done

mkdir -p "$REPORT_DIR"

run "$PYTHON" - <<'PY'
from pathlib import Path
import yaml

checks = {
    "configs/method/npo_paraphrase_fullft_smoke.yaml": "NPO + paraphrases",
    "configs/method/syntactic_diversification_fullft_smoke.yaml": "Syntactic diversification",
}
for path, label in checks.items():
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not cfg.get("use_prompt_variants"):
        raise SystemExit(f"{label} config must use prompt variants: {path}")
    if int(cfg.get("max_prompt_variants", 0)) <= 1:
        raise SystemExit(f"{label} config must use multiple prompt variants: {path}")
print("baseline prompt-variant config checks passed")
PY

COMPARE_INPUTS=()
REPORT_INPUTS=()
REFERENCE_INPUTS=()

for seed in $SEEDS; do
  require_file "thresholds/tofu50_fullft_seed${seed}_thresholds.json"
  for baseline in $BASELINES; do
    train_baseline_if_needed "$baseline" "$seed"
    checkpoint="$(baseline_checkpoint "$baseline" "$seed")"
    method_label="$(baseline_method_label "$baseline" "$seed")"
    output_prefix="$(baseline_output_prefix "$baseline" "$seed")"
    for pool in "${POOLS[@]}"; do
      output="${OUTPUT_ROOT}/${output_prefix}_${pool}"
      if [[ "$OVERWRITE" -eq 1 || ! -f "${output}/survival_curve.csv" ]]; then
        run "$PYTHON" scripts/run_survival_curve.py \
          --checkpoint "$checkpoint" \
          --thresholds "thresholds/tofu50_fullft_seed${seed}_thresholds.json" \
          --forget_file data/tofu/forget_50.jsonl \
          --retain_file data/tofu/retain_50.jsonl \
          --prompt_variants data/prompts/forget_prompt_variants.jsonl \
          --relearn_pool "data/relearn_pools/${pool}.jsonl" \
          --steps "$STEPS" \
          --lr "$LR" \
          --batch_size "$BATCH_SIZE" \
          --resurrection_metric margin \
          --method "$method_label" \
          --seed "$seed" \
          --output "$output"
      fi
      COMPARE_INPUTS+=("${output}/survival_curve.csv")
      REPORT_INPUTS+=("${output}/survival_curve.csv")
    done
  done
done

for seed in $SEEDS; do
  for pool in "${POOLS[@]}"; do
    for path in \
      "${REFERENCE_ROOT}/hlc_sg_k8_tuned_fullft_seed${seed}_${pool}/survival_curve.csv" \
      "${REFERENCE_ROOT}/npo_strong_matched_seed${seed}_"*"_${pool}/survival_curve.csv"; do
      for match in $path; do
        if [[ -f "$match" ]]; then
          COMPARE_INPUTS+=("$match")
          REPORT_INPUTS+=("$match")
          REFERENCE_INPUTS+=("$match")
        fi
      done
    done
  done
done

if [[ "${#COMPARE_INPUTS[@]}" -eq 0 ]]; then
  echo "No survival curves were collected." >&2
  exit 1
fi

run "$PYTHON" scripts/compare_methods.py \
  --inputs "${COMPARE_INPUTS[@]}" \
  --output "${REPORT_DIR}/missing_baselines_compare.csv"

FIRST_BASELINE="${BASELINES%% *}"
TOP_LEVEL_REFERENCE_FAMILY="$(baseline_family_label "$FIRST_BASELINE")"

run "$PYTHON" scripts/report_conditional_durability.py \
  --inputs "${REPORT_INPUTS[@]}" \
  --horizons 32,128,512 \
  --reference_family "$TOP_LEVEL_REFERENCE_FAMILY" \
  --baseline_family "Strong NPO" \
  --output_dir "$REPORT_DIR"

for baseline in $BASELINES; do
  BASELINE_REPORT_INPUTS=()
  for seed in $SEEDS; do
    output_prefix="$(baseline_output_prefix "$baseline" "$seed")"
    for pool in "${POOLS[@]}"; do
      BASELINE_REPORT_INPUTS+=("${OUTPUT_ROOT}/${output_prefix}_${pool}/survival_curve.csv")
    done
  done
  BASELINE_REPORT_INPUTS+=("${REFERENCE_INPUTS[@]}")
  run "$PYTHON" scripts/report_conditional_durability.py \
    --inputs "${BASELINE_REPORT_INPUTS[@]}" \
    --horizons 32,128,512 \
    --reference_family "$(baseline_family_label "$baseline")" \
    --baseline_family "Strong NPO" \
    --output_dir "${REPORT_DIR}/${baseline}"
done

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Missing-baseline stress package complete."
echo "Comparison: ${REPORT_DIR}/missing_baselines_compare.csv"
echo "Summary: ${REPORT_DIR}/conditional_durability_summary.md"
for baseline in $BASELINES; do
  echo "Baseline summary (${baseline}): ${REPORT_DIR}/${baseline}/conditional_durability_summary.md"
done
