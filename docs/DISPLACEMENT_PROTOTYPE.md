# Displacement-Durability Prototype

A fresh hypothesis to test after the sharpness mechanism was falsified.

## Hypothesis

On real TOFU, durability tracked **weight displacement**, not basin flatness. The
constrained GA baseline moved further from the fine-tuned model (higher retain
cost) and was ~2× more durable than HLC; HLC's lighter-touch unlearning preserved
utility but relearned fast. So:

> Can we get GA's displacement (hence durability) **without** its utility cost, by
> ascending the forget loss while projecting that ascent orthogonal to the retain
> gradient?

## Mechanism (`durable_unlearning/methods/displacement.py`)

Orthogonalized gradient ascent. Each step:

```
g_f = grad(forget CE)        # +g_f increases forget loss = the displacement direction
g_r = grad(retain CE)        # +g_r increases retain loss = the direction to avoid
coef = max(0, <g_f,g_r> / ||g_r||^2)
g_f_orth = g_f - coef * g_r   # remove only the retain-harming component of the ascent
update: maximize forget loss along g_f_orth, minimize retain loss along g_r
```

`orthogonalize: false` recovers plain constrained GA, so the config doubles as its
own ablation. Knobs: `forget_weight`, `retain_weight`, `max_grad_norm`, `steps`.

## Run (V100), seed 0, real TOFU forget01

```bash
python3 scripts/unlearn.py --method displacement \
  --method_config configs/method/displacement_qwen3.yaml \
  --model_config configs/model/qwen3_0p6b_lora_v100.yaml \
  --data_config configs/data/tofu01_real.yaml \
  --start_checkpoint checkpoints/qwen3_tofu01_full_seed0 \
  --thresholds thresholds/tofu01_real_qwen3_seed0_thresholds.json \
  --prompt_variants data/tofu_real/forget_prompt_variants_01.jsonl \
  --output checkpoints/unlearned/displacement_tofu01_seed0 --seed 0

CUDA_VISIBLE_DEVICES=0 python3 scripts/run_survival_curve.py \
  --checkpoint checkpoints/unlearned/displacement_tofu01_seed0 \
  --base_model Qwen/Qwen3-0.6B --dtype float32 \
  --thresholds thresholds/tofu01_real_qwen3_seed0_thresholds.json \
  --forget_file data/tofu_real/forget_01.jsonl --retain_file data/tofu_real/retain_01.jsonl \
  --prompt_variants data/tofu_real/forget_prompt_variants_01.jsonl \
  --relearn_pool data/tofu_real/relearn_pools_01/heldout_forget_mixed_stress.jsonl \
  --steps 0,1,2,4,8,16,32,64,128,256,512 --lr 0.0001 --batch_size 8 --max_length 192 \
  --max_new_tokens 8 --max_retain_items 200 --resurrection_metric margin \
  --method displacement --seed 0 --output runs/survival/displacement_tofu01_seed0_lr1e4
```

Compare against the existing HLC and GA curves:

```bash
python3 scripts/aggregate_tofu_real.py --inputs \
  runs/survival/displacement_tofu01_seed0_lr1e4/survival_curve.csv \
  runs/survival/hlc_r_tofu01_seed0_lr1e4/survival_curve.csv \
  runs/survival/matched_ga_tofu01_seed0_v2_lr1e4/survival_curve.csv \
  --output_dir runs/reports/displacement_vs
```

## Pass condition (declared up front)

Displacement is interesting iff, at immediate-match (r@0 = 0):

- `h50 ≥ 24` (approaching GA's ≈28, well above HLC's ≈12), **and**
- retain NLL@0 `≤ ~2.6` (near HLC's 2.30, clearly below GA's 3.81).

That is the frontier point neither GA nor HLC reached. If instead it lands on the
GA line (durable but retain ≈ 3.8) or the HLC line (cheap but h50 ≈ 12), the
orthogonal projection bought nothing and displacement is just damage by another
name — also a clean, publishable finding.

## Multi-seed / CI

`INCLUDE_DISPLACEMENT=1 FRACTION=0.01 SEEDS="0 1 2" scripts/reproduce_tofu_real_comparison.sh`
runs displacement alongside HLC-R / HLC-base / GA across seeds and aggregates with
bootstrap CIs via `scripts/aggregate_tofu_real.py`.
