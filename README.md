# Durable Unlearning by Relearning Half-Life Control

This repository is a local-first implementation scaffold for Durable Unlearning by Relearning Half-Life Control (HLC).

The implementation is intentionally gated:

1. Reproducible harness and artifact schemas.
2. Survival-curve evaluator.
3. Oracle training and threshold calibration.
4. Baselines.
5. HLC-SG.
6. Ablations and evidence packaging.

The default smoke path targets Apple Silicon or CPU with tiny models. The same command and artifact contracts are intended to scale to GPU runs with TOFU and `EleutherAI/pythia-410m`.

## Current Result

The local MVP reached a negative but useful diagnostic result: benign survival curves made HLC-SG look strongest, but immediate-matched answer-bearing stress tests did not support an HLC durability claim.

Read the full narrative in [`RESULTS.md`](RESULTS.md).

Protocol and conclusion docs:

- [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)
- [`docs/MVP_CONCLUSION.md`](docs/MVP_CONCLUSION.md)

Canonical generated artifacts:

- `runs/reports/final_diagnostic_mvp/diagnostic_summary.md`
- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/decision_report.md`
- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv`
- `runs/reports/heldout_stress/conditional_durability_summary.md`
- `runs/reports/update_budget_stress/budget_stress_summary.md`
- `runs/reports/gpt2_lora_smoke/conditional_durability_summary.md`
- `runs/reports/gpt2_hlc_k2_smoke/conditional_durability_summary.md`
- `runs/reports/gpt2_long_horizon/conditional_durability_summary.md`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Pure-Python tests can run without the heavyweight ML stack already installed:

```bash
python3 -m pytest
```

## Smoke Data

```bash
python3 scripts/prepare_tofu.py --output_dir data/tofu --forget_fraction 0.5 --synthetic
python3 scripts/make_prompt_variants.py --forget_file data/tofu/forget_50.jsonl --output data/prompts/forget_prompt_variants.jsonl
python3 scripts/make_relearn_pools.py --retain_file data/tofu/retain_50.jsonl --output_dir data/relearn_pools
```

## Main Interfaces

```bash
python3 scripts/train_full.py --config configs/model/tiny_gpt2_full.yaml --data configs/data/tofu_50_smoke.yaml --output checkpoints/tiny_full_seed0 --seed 0
python3 scripts/train_retrain_minus_f.py --config configs/model/tiny_gpt2_full.yaml --data configs/data/tofu_50_smoke.yaml --forget_split forget_50 --output checkpoints/tiny_retain_only_seed0 --seed 0
python3 scripts/calibrate_thresholds.py --checkpoint checkpoints/tiny_retain_only_seed0 --forget_file data/tofu/forget_50.jsonl --prompt_variants data/prompts/forget_prompt_variants.jsonl --quantiles 0.90,0.95,0.99 --tau_global 0.0 --output thresholds/tofu50_seed0_thresholds.json
python3 scripts/unlearn.py --method hlc_sg --method_config configs/method/hlc_sg_k8.yaml --start_checkpoint checkpoints/tiny_full_seed0 --full_checkpoint checkpoints/tiny_full_seed0 --thresholds thresholds/tofu50_seed0_thresholds.json --relearn_pool data/relearn_pools/retain_adjacent.jsonl --output checkpoints/unlearned/hlc_sg_k8_tofu50_seed0
python3 scripts/run_survival_curve.py --checkpoint checkpoints/unlearned/hlc_sg_k8_tofu50_seed0 --thresholds thresholds/tofu50_seed0_thresholds.json --forget_file data/tofu/forget_50.jsonl --retain_file data/tofu/retain_50.jsonl --prompt_variants data/prompts/forget_prompt_variants.jsonl --relearn_pool data/relearn_pools/synthetic_biography.jsonl --steps 0,1,2 --resurrection_metric margin --output runs/survival/hlc_sg_k8_seed0_synthetic_bio
```

For the local full-finetune smoke path, use the stronger smoke method configs:

```bash
python3 scripts/unlearn.py --method grad_ascent --method_config configs/method/grad_ascent_fullft_smoke.yaml --model_config configs/model/tiny_gpt2_full.yaml --data_config configs/data/tofu_50_smoke.yaml --start_checkpoint checkpoints/tiny_full_fullft_seed0 --thresholds thresholds/tofu50_fullft_seed0_thresholds.json --prompt_variants data/prompts/forget_prompt_variants.jsonl --output checkpoints/unlearned/grad_ascent_fullft_tofu50_seed0
python3 scripts/unlearn.py --method npo --method_config configs/method/npo_fullft_smoke.yaml --model_config configs/model/tiny_gpt2_full.yaml --data_config configs/data/tofu_50_smoke.yaml --start_checkpoint checkpoints/tiny_full_fullft_seed0 --thresholds thresholds/tofu50_fullft_seed0_thresholds.json --prompt_variants data/prompts/forget_prompt_variants.jsonl --output checkpoints/unlearned/npo_fullft_tofu50_seed0
python3 scripts/unlearn.py --method hlc_sg --method_config configs/method/hlc_sg_k8_fullft_smoke.yaml --model_config configs/model/tiny_gpt2_full.yaml --data_config configs/data/tofu_50_smoke.yaml --start_checkpoint checkpoints/tiny_full_fullft_seed0 --full_checkpoint checkpoints/tiny_full_fullft_seed0 --thresholds thresholds/tofu50_fullft_seed0_thresholds.json --prompt_variants data/prompts/forget_prompt_variants.jsonl --relearn_pool data/relearn_pools/retain_adjacent.jsonl --output checkpoints/unlearned/hlc_sg_k8_fullft_tofu50_seed0
```

If PyTorch/Hugging Face dependencies are missing, model commands fail with a dependency message; harness, threshold, manifest, and schema tests still run.

## Local MVP Report

After generating a comparison CSV, create a compact evidence report:

```bash
python3 scripts/report_local_mvp.py \
  --comparison runs/survival/compare_seed0_seed1_seed2_all_pools.csv \
  --output_dir runs/reports/local_mvp_seed0_seed1_seed2
```

The report writes aggregate CSVs, a Markdown summary, and plots for survival AUC, final resurrection, target margins, and retain NLL.

To rebuild the full local MVP package for seeds 0, 1, and 2:

```bash
scripts/reproduce_local_mvp.sh
```

For a clean rebuild of generated local MVP artifacts:

```bash
scripts/reproduce_local_mvp.sh --clean
```

The script defaults to `/opt/anaconda3/bin/python3` when available. Override with `PYTHON=/path/to/python`.

## Missing Baseline Stress Runs

To evaluate missing baselines against the same answer-bearing stress protocol:

```bash
scripts/reproduce_missing_baselines.sh --baselines "npo_paraphrase syntactic_diversification"
```

The baseline targets run over seeds `0,1,2`, stress pools `forget_declarative`, `forget_paraphrase_qa`, and `forget_mixed_stress`, and horizon `0,1,2,4,8,16,32,64,128,256,512`.

Outputs:

- `runs/reports/missing_baselines_stress/missing_baselines_compare.csv`
- `runs/reports/missing_baselines_stress/conditional_horizon_aggregate.csv`
- `runs/reports/missing_baselines_stress/conditional_durability_summary.md`
- `runs/reports/missing_baselines_stress/npo_paraphrase/conditional_durability_summary.md`
- `runs/reports/missing_baselines_stress/syntactic_diversification/conditional_durability_summary.md`

To reproduce the immediate-matched syntactic diversification sweep and stress evaluation:

```bash
scripts/reproduce_syntactic_immediate_matched.sh
```

Outputs:

- `runs/survival/syntactic_sweep_match_selected.csv`
- `runs/reports/syntactic_immediate_matched_stress/conditional_durability_summary.md`

## Held-Out Relearn Stress

To evaluate HLC-SG, NPO + paraphrases, immediate-matched Strong NPO, and immediate-matched syntactic diversification on answer-bearing relearn pools that use held-out wording:

```bash
scripts/reproduce_heldout_stress.sh
```

The command generates:

- `data/relearn_pools/heldout_forget_declarative.jsonl`
- `data/relearn_pools/heldout_forget_paraphrase_qa.jsonl`
- `data/relearn_pools/heldout_forget_mixed_stress.jsonl`

Outputs:

- `runs/reports/heldout_stress/compare.csv`
- `runs/reports/heldout_stress/conditional_horizon_aggregate.csv`
- `runs/reports/heldout_stress/conditional_durability_summary.md`

The current held-out result preserves the negative durability conclusion: HLC-SG beats NPO + paraphrases, but does not beat immediate-matched Strong NPO or immediate-matched syntactic diversification on conditional durability.

## Stronger Update-Budget Stress

To rerun the held-out answer-bearing stress gate with a stronger relearning update budget:

```bash
scripts/reproduce_update_budget_stress.sh
```

The default stronger budget uses `lr=0.01`, double the held-out stress default. To run multiple budgets:

```bash
scripts/reproduce_update_budget_stress.sh --budgets "0.01 0.02"
```

Outputs:

- `runs/reports/update_budget_stress/lr0p01/conditional_durability_summary.md`
- `runs/reports/update_budget_stress/budget_horizon_aggregate.csv`
- `runs/reports/update_budget_stress/budget_stress_summary.md`

At `lr=0.01`, all immediate-matched methods fully resurrect by `t=512`; the useful signal is earlier horizon and conditional AUC. HLC-SG remains worse than immediate-matched Strong NPO and syntactic diversification.

## GPT-2 LoRA Smoke Lane

The first larger-model lane uses `gpt2` with LoRA adapters and a capped seed-0 smoke protocol:

```bash
scripts/reproduce_gpt2_lora_smoke.sh
```

Defaults:

- model config: `configs/model/gpt2_lora_smoke.yaml`
- Strong NPO config: `configs/method/strong_npo_gpt2_lora_smoke.yaml`
- HLC config: `configs/method/hlc_sg_k1_gpt2_lora_smoke.yaml`
- seeds: `0`
- horizon: `0,1,2,4,8,16,32`
- stress pools: held-out declarative, paraphrase QA, and mixed stress

The script disables Hugging Face Xet by default with `HF_HUB_DISABLE_XET=1`, because the first local attempt was slow through the Xet download path. The completed seed-0 smoke run is a wiring and feasibility gate before any GPT-2 immediate-matched sweep or Pythia/GPU run.

Current GPT-2 LoRA smoke result at `t=32`:

| Method | n | r0 | conditional resurrection | conditional AUC | retain NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 | 3 | 0.000 | 0.333 | 0.896 | 5.575 |
| Strong NPO | 3 | 0.000 | 0.278 | 0.910 | 5.499 |

This confirms the larger-model harness path works. It does not establish a GPT-2 durability claim.

## GPT-2 HLC K2 Smoke

To test the next stronger GPT-2 HLC candidate while reusing the completed GPT-2 base artifacts:

```bash
scripts/reproduce_gpt2_hlc_k2_smoke.sh
```

This uses `configs/method/hlc_sg_k2_gpt2_lora_smoke.yaml` and compares K2 against the existing K1 HLC and Strong NPO GPT-2 smoke outputs.

Current `t=32` result:

| Method | n | r0 | conditional resurrection | conditional AUC | retain NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 GPT-2 | 3 | 0.000 | 0.333 | 0.896 | 5.575 |
| HLC-SG K2 GPT-2 | 3 | 0.000 | 0.389 | 0.903 | 5.647 |
| Strong NPO | 3 | 0.000 | 0.278 | 0.910 | 5.499 |

K2 is feasible and slightly improves AUC over K1, but it increases conditional resurrection and still trails Strong NPO.

## GPT-2 Long-Horizon Diagnostic

To extend the GPT-2 LoRA lane from `t=32` to `t=128` without retraining the existing GPT-2 adapters:

```bash
scripts/reproduce_gpt2_long_horizon.sh
```

This reuses the seed-0 Strong NPO, HLC K1, and HLC K2 GPT-2 LoRA checkpoints and evaluates held-out stress pools over `0,1,2,4,8,16,32,64,128`.

Current result:

| Method | n | r0 | cond. res. @32 | cond. AUC @32 | cond. res. @128 | cond. AUC @128 | retain NLL @128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 GPT-2 | 3 | 0.000 | 0.278 | 0.920 | 0.833 | 0.529 | 6.838 |
| HLC-SG K2 GPT-2 | 3 | 0.000 | 0.278 | 0.920 | 0.611 | 0.584 | 6.597 |
| Strong NPO | 3 | 0.000 | 0.333 | 0.906 | 0.556 | 0.588 | 6.776 |

K2 has a short-horizon improvement over Strong NPO at `t=32`, but the advantage does not survive to `t=128`. This keeps the GPT-2 lane as a feasibility and diagnostic result, not a durability claim.
