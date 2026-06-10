# Sharpness-Aware Durable Unlearning (HLC-R)

This document specifies the mechanism change intended to move HLC from
"≈ best immediate-matched baseline" to a defensible durability win, and the
**pre-registered** confirmatory experiment that decides whether the claim holds.

## 1. Why prior HLC plateaus

NPO, gradient ascent, and every previous HLC variant (K2/K4, multi-time,
margin-growth, shallow-unrolled) all suppress the forget margin **at a point** in
weight space, or along one shallow sampled relearn trajectory. Benign relearning
resurrects facts because the knowledge stays *latently encoded*: the unlearned
parameters sit on a sharp ridge, and a few benign gradient steps slide back into
the "remembers" basin. Lowering the point margin does not change the curvature of
that ridge, so HLC competes with the baselines on the exact axis they already
optimize — and lands on top of them.

## 2. The mechanism: make the forgotten state robust, not just low

Relearning is gradient-based. So durability is a **flatness / robustness**
property of the resurrection margin: the suppressed state must survive worst-case
perturbation in the directions relearning actually moves. Instead of penalizing
`m_i(theta)`, HLC-R penalizes the worst case inside a `rho`-ball,

    max_{||delta|| <= rho}  L_resurrect(theta + delta),

with `delta` restricted to:

- **margin**  — the ascent direction of the resurrection loss (the single worst
  direction for the forget margin), and/or
- **relearn** — the *descent* direction of the benign relearn loss
  (`-grad L_relearn`), i.e. exactly where benign fine-tuning pushes the model.

This is a SAM-style inner maximization applied to the resurrection margin. Driving
down the worst-case margin flattens the forget basin along the relearn directions,
which is precisely a longer relearning half-life. No baseline (NPO/GA/NPO+paraphrase/
syntactic diversification) optimizes a worst-case-in-ball objective, so the method
is not cheaply imitable, and there is a clean theory hook (flat-minima / PAC-Bayes
robustness, repurposed as resistance-to-relearning).

## 3. Implementation

- `durable_unlearning/losses/sharpness.py` — `sharpness_resurrection_grad(...)`:
  computes the ascent direction, perturbs the trainable (LoRA) parameters in place
  to `theta + rho * dhat`, evaluates the worst-case resurrection loss and its
  first-order SAM gradient there, then restores parameters exactly. Includes a
  non-finite guard consistent with the evaluator's reject-non-finite policy.
- `durable_unlearning/methods/hlc_sg.py` — the worst-case gradient is folded into
  the existing manual gradient combination as `lambda_sharpness * g_sharp`,
  alongside `g_now` and `lambdaK * g_post`. **Off by default**
  (`sharpness_rho = 0`), so all existing configs/results are unchanged.

New config knobs:

| key | meaning | default |
| --- | --- | --- |
| `sharpness_rho` | perturbation radius of the SAM ball | `0.0` (disabled) |
| `lambda_sharpness` (or `lambdaS`) | weight of the worst-case gradient | `0.0` |
| `sharpness_gamma` | softplus temperature for the worst-case margin | `gamma` |
| `sharpness_direction` | `margin` \| `relearn` \| `both` | `both` |

Configs: `configs/method/hlc_r_k4_qwen3_lora_v100.yaml` (sweep backbone),
`configs/method/hlc_r_k1_gpt2_lora_smoke.yaml` (wiring check).

Unit tests: `tests/test_sharpness_loss.py` verifies worst-case ≥ point,
exact parameter restoration, finite gradients across all three directions, and
input validation (passing under torch 2.x CPU).

## 4. Pre-registered confirmatory experiment

The point of pre-registration is to keep this from becoming another
garden-of-forking-paths tune. **Fix the design and the pass condition below before
looking at any held-out result.** The reason is direct: HLC has now landed at
"≈ baseline" on tiny, GPT-2, and Qwen; a 0.03 swing in conditional resurrection
with n=2 seeds and no CIs is noise, and any "win" read off such a run is not
publishable.

### Fixed design

- **Models:** Qwen3-0.6B LoRA (primary), GPT-2 LoRA (secondary smoke).
- **Eval dtype:** float32 only (fp16 relearning is numerically unstable and
  produced the NaN false-positive; trusted results are fp32).
- **Seeds:** 5 (`0,1,2,3,4`). Minimum 3 if compute-limited; below 3, no claim.
- **Baseline:** the immediate-matched competitor already selected per seed
  (constrained gradient ascent; Strong NPO where it can immediate-match).
- **Relearn stress:** held-out answer-bearing pools
  (`heldout_forget_declarative`, `heldout_forget_paraphrase_qa`,
  `heldout_forget_mixed_stress`); horizon `0,1,2,4,8,16,32,64,128,256,512`.
- **Uncertainty:** bootstrap 95% CIs over seeds × pools for conditional
  resurrection and conditional AUC. Report CIs in every table.

### Sweep (small, declared up front)

The change is robustness, not more forgetting pressure. Hold the K4 backbone fixed
and sweep only the sharpness term plus the utility counterweights:

```
sharpness_rho        : 0.02, 0.05, 0.10
lambda_sharpness     : 0.5, 1.0, 2.0
sharpness_direction  : both, relearn
lambdaR              : 1.0, 2.0      # utility counterweight
outer_lr             : 5e-6, 1e-5
```

Select **one** HLC-R config using seed 0 only (immediate-matched `r@0` within
0.05 of the baseline, retain NLL within band), then **freeze it** and evaluate on
the remaining seeds. Do not reselect after seeing held-out seeds.

### Pass condition (declared before looking)

HLC-R is a durability win iff, on the held-out answer-bearing stress, aggregated
over the frozen seeds:

1. **immediate-matched:** `|r@0(HLC-R) − r@0(baseline)| ≤ 0.05`; and
2. **utility-matched:** retain NLL(HLC-R) ≤ retain NLL(baseline) + 10%; and
3. **strictly more durable:** conditional resurrection at `t=128` **and** `t=512`
   is lower than the baseline, with **non-overlapping 95% CIs** on at least one of
   those horizons; **or** conditional AUC@512 is higher with non-overlapping CIs.

If 1–3 hold → upgrade the paper to a method result (HLC-R as the contribution).
If they do not → HLC-R is reported as a tested negative, and the paper stays the
evaluation-protocol/diagnostic contribution. Either outcome is publishable; only an
un-pre-registered "win" is not.

## 5. Run commands (V100)

Smoke (GPT-2, wiring only):

```bash
python3 scripts/unlearn.py \
  --method hlc_sg \
  --method_config configs/method/hlc_r_k1_gpt2_lora_smoke.yaml \
  --model_config configs/model/gpt2_lora_smoke.yaml \
  --data_config configs/data/tofu_50_smoke.yaml \
  --start_checkpoint checkpoints/gpt2_lora_full_seed0 \
  --full_checkpoint checkpoints/gpt2_lora_full_seed0 \
  --thresholds thresholds/tofu50_gpt2_lora_seed0_thresholds.json \
  --prompt_variants data/prompts/forget_prompt_variants.jsonl \
  --relearn_pool data/relearn_pools/retain_adjacent.jsonl \
  --output checkpoints/unlearned/hlc_r_k1_gpt2_lora_seed0
```

Qwen seed-0 selection run (then freeze and repeat for the held-out seeds, fp32 eval):

```bash
python3 scripts/unlearn.py \
  --method hlc_sg \
  --method_config configs/method/hlc_r_k4_qwen3_lora_v100.yaml \
  --model_config configs/model/qwen3_0p6b_lora_v100.yaml \
  --data_config configs/data/tofu_05.yaml \
  --start_checkpoint checkpoints/qwen3_full_seed0 \
  --full_checkpoint checkpoints/qwen3_full_seed0 \
  --thresholds thresholds/tofu05_qwen3_seed0_thresholds.json \
  --relearn_pool data/relearn_pools/retain_adjacent.jsonl \
  --output checkpoints/unlearned/hlc_r_k4_qwen3_seed0

python3 scripts/run_survival_curve.py \
  --checkpoint checkpoints/unlearned/hlc_r_k4_qwen3_seed0 \
  --thresholds thresholds/tofu05_qwen3_seed0_thresholds.json \
  --forget_file data/tofu/forget_05.jsonl \
  --retain_file data/tofu/retain_05.jsonl \
  --relearn_pool data/relearn_pools/heldout_forget_mixed_stress.jsonl \
  --steps 0,1,2,4,8,16,32,64,128,256,512 \
  --dtype float32 \
  --output runs/survival/hlc_r_k4_qwen3_seed0_heldout_mixed
```

(Adjust checkpoint/threshold names to your Qwen artifacts.)
