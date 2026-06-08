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
MODE="verify"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify)
      MODE="verify"
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    --clean)
      echo "--clean is intentionally not implemented for the final diagnostic verifier." >&2
      echo "Use scripts/reproduce_local_mvp.sh --clean for the older local MVP rebuild path." >&2
      exit 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/reproduce_final_diagnostic.sh [--verify] [--skip-tests]

Fast verifier for the final local MVP diagnostic package.

This script does not retrain models. It checks that the required final-run
artifacts exist, regenerates the final diagnostic report from those artifacts,
checks that RESULTS.md references the canonical outputs, and optionally runs
the test suite. It also verifies that the protocol and MVP conclusion docs
are present and point to the final diagnostic path.

Environment:
  PYTHON=/path/to/python     Python executable. Defaults to /opt/anaconda3/bin/python3 when present.

Inputs:
  runs/reports/main_track_seed0_seed1_seed2/main_track_table.csv
  runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv
  runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/decision_report.md

Outputs:
  runs/reports/final_diagnostic_mvp/diagnostic_summary.md
  runs/reports/final_diagnostic_mvp/diagnostic_main_track_table.csv
  runs/reports/final_diagnostic_mvp/diagnostic_stress_table.csv
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "verify" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 2
fi

MAIN_TRACK="runs/reports/main_track_seed0_seed1_seed2/main_track_table.csv"
STRESS_RANKED="runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv"
DECISION_REPORT="runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/decision_report.md"
OUTPUT_DIR="runs/reports/final_diagnostic_mvp"
SUMMARY="${OUTPUT_DIR}/diagnostic_summary.md"
MAIN_OUT="${OUTPUT_DIR}/diagnostic_main_track_table.csv"
STRESS_OUT="${OUTPUT_DIR}/diagnostic_stress_table.csv"

REQUIRED_INPUTS=(
  "$MAIN_TRACK"
  "$STRESS_RANKED"
  "$DECISION_REPORT"
  "RESULTS.md"
  "README.md"
  "docs/EVALUATION_PROTOCOL.md"
  "docs/MVP_CONCLUSION.md"
  "scripts/make_diagnostic_report.py"
)

run() {
  echo
  echo "+ $*"
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required artifact: $path" >&2
    exit 1
  fi
}

require_text() {
  local path="$1"
  local text="$2"
  if ! grep -Fq "$text" "$path"; then
    echo "Expected '$path' to contain: $text" >&2
    exit 1
  fi
}

echo "Using Python: $PYTHON"
"$PYTHON" --version

for path in "${REQUIRED_INPUTS[@]}"; do
  require_file "$path"
done

run "$PYTHON" scripts/make_diagnostic_report.py \
  --main_track "$MAIN_TRACK" \
  --stress_ranked "$STRESS_RANKED" \
  --decision_report "$DECISION_REPORT" \
  --output_dir "$OUTPUT_DIR"

require_file "$SUMMARY"
require_file "$MAIN_OUT"
require_file "$STRESS_OUT"

require_text "$SUMMARY" "does not support an HLC durability claim"
require_text "$SUMMARY" "Freeze HLC tuning for the local MVP"
require_text "RESULTS.md" "$SUMMARY"
require_text "RESULTS.md" "$STRESS_OUT"
require_text "README.md" "RESULTS.md"
require_text "README.md" "docs/EVALUATION_PROTOCOL.md"
require_text "README.md" "docs/MVP_CONCLUSION.md"
require_text "docs/EVALUATION_PROTOCOL.md" "conditional_resurrection_fraction"
require_text "docs/EVALUATION_PROTOCOL.md" "scripts/reproduce_final_diagnostic.sh --verify"
require_text "docs/MVP_CONCLUSION.md" "does not beat immediate-matched Strong NPO"
require_text "docs/MVP_CONCLUSION.md" "NPO + paraphrases"

run "$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

stress = pd.read_csv("runs/reports/final_diagnostic_mvp/diagnostic_stress_table.csv")
required_methods = {"Strong NPO", "HLC-SG"}
methods = set(stress["Method"].astype(str))
missing = required_methods - methods
if missing:
    raise SystemExit(f"diagnostic stress table missing methods: {sorted(missing)}")

strong = stress[stress["Method"] == "Strong NPO"].iloc[0]
hlc = stress[stress["Method"] == "HLC-SG"].iloc[0]
if not float(strong["Cond. resurrection @128"]) < float(hlc["Cond. resurrection @128"]):
    raise SystemExit("expected Strong NPO to beat HLC-SG at conditional resurrection @128")

for path in [
    "runs/reports/final_diagnostic_mvp/diagnostic_summary.md",
    "runs/reports/final_diagnostic_mvp/diagnostic_main_track_table.csv",
    "runs/reports/final_diagnostic_mvp/diagnostic_stress_table.csv",
]:
    if Path(path).stat().st_size == 0:
        raise SystemExit(f"empty output artifact: {path}")

print("diagnostic artifact checks passed")
PY

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run "$PYTHON" -m pytest -q
fi

echo
echo "Final diagnostic verification complete."
echo "Summary: $SUMMARY"
