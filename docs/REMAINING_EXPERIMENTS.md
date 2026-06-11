# Remaining Experiments (toward a submission-grade result)

Five experiments that address the objections a reviewer will raise. All run on the
V100 box; the code/configs/scripts are in the repo. Each reuses the surviving
`theta_full` + thresholds where possible. Run order is by value.

Prereq: the seed-0 forget01 artifacts exist (`checkpoints/qwen3_tofu01_full_seed0`,
`thresholds/tofu01_real_qwen3_seed0_thresholds.json`, regenerated `data/tofu_real/`).

## #1 Durability/utility Pareto frontier (cheapest, strongest figure)
Sweep displacement over `forget_weight` and GA over `retain_weight`, plot
durability (h50) vs utility (retain NLL). Shows displacement's whole frontier sits
above-left of GA's, not just one point.

```bash
FRACTION=0.01 SEED=0 DISP_LF="1 2 3 4" GA_RW="0.5 1 2 4" \
  scripts/reproduce_displacement_frontier.sh
cat runs/reports/frontier_tofu01_seed0/frontier_table.md   # + frontier.png
```
Look for: the summary line "N/M non-displacement points are dominated by some
displacement config." Domination = displacement at least as durable AND cheaper.

## #2 Scale: forget05 / forget10 (tightens the half-life CIs)
Larger forget sets cut seed variance; expected to separate the h50 CIs.

```bash
INCLUDE_DISPLACEMENT=1 FRACTION=0.05 SEEDS="0 1 2" scripts/reproduce_tofu_real_comparison.sh
cat runs/reports/qwen3_tofu05_real_comparison/aggregate_ci_summary.md
# and/or FRACTION=0.10 (forget10, ~400 items)
```

## #3 Second model: Qwen3-1.7B (generality)
The comparison script is model-tagged, so artifacts don't collide with 0.6B.

```bash
INCLUDE_DISPLACEMENT=1 MODEL_TAG=qwen1p7b BASE_MODEL=Qwen/Qwen3-1.7B \
  MODEL_CONFIG=configs/model/qwen3_1p7b_lora_v100.yaml \
  FRACTION=0.01 SEEDS="0 1 2" scripts/reproduce_tofu_real_comparison.sh
cat runs/reports/qwen1p7b_tofu01_real_comparison/aggregate_ci_summary.md
```

## #4 Robust-relearning baseline: RMU (strongest competitor)
RMU edits internal activations (representation misdirection) rather than logits, so
it is the baseline designed for relearning robustness. First immediate-match it
(tune `layer`/`steer_coeff`/`alpha` in `configs/method/rmu_qwen3.yaml` so r@0=0 at
retain NLL comparable to the others), then add it to the comparison:

```bash
# standalone immediate-match check (adjust config, re-run until r@0 ~ 0):
python3 scripts/unlearn.py --method rmu --method_config configs/method/rmu_qwen3.yaml \
  --model_config configs/model/qwen3_0p6b_lora_v100.yaml --data_config configs/data/tofu01_real.yaml \
  --start_checkpoint checkpoints/qwen3_tofu01_full_seed0 \
  --full_checkpoint  checkpoints/qwen3_tofu01_full_seed0 \
  --thresholds thresholds/tofu01_real_qwen3_seed0_thresholds.json \
  --prompt_variants data/tofu_real/forget_prompt_variants_01.jsonl \
  --output checkpoints/unlearned/rmu_probe_seed0 --seed 0
# then quick t=0 eval (r@0 + retain_nll) as in the displacement workflow.

# once matched, include in the full comparison:
INCLUDE_DISPLACEMENT=1 INCLUDE_RMU=1 FRACTION=0.01 SEEDS="0 1 2" \
  scripts/reproduce_tofu_real_comparison.sh
```

## #5 Robustness checks (cheap credibility)
Show the durability ordering is not metric- or operator-specific. These re-evaluate
*existing* unlearned checkpoints (no retraining).

Second resurrection metric + threshold sensitivity (reuse any unlearned checkpoint):
```bash
# margin AND exact-match, and q90 / q99 thresholds instead of q95
... run_survival_curve.py ... --resurrection_metric margin_and_exact ...
... run_survival_curve.py ... --threshold_quantile q90 ...
... run_survival_curve.py ... --threshold_quantile q99 ...
```

Alternate benign-update operator (does the ordering survive a different attack?):
```bash
# SGD instead of AdamW
... run_survival_curve.py ... --relearn_optimizer sgd ...
# full-parameter relearning instead of LoRA-only (heavier; a stronger attack)
... run_survival_curve.py ... --relearn_full_params ...
```
Re-run displacement and GA under each and confirm displacement remains at least as
durable at lower retain cost.

## Priority
#1 and #5 are cheap and high-credibility; #2 tightens the headline CI; #3 adds
generality; #4 is the most demanding but the most convincing. None is expected to
overturn the core finding (orthogonal projection preserves retain under
gradient-ascent-strength displacement), but together they make it submission-grade.
