# Real-TOFU Findings (Qwen3-0.6B)

This document records the decisive experiments on the **real** locuslab/TOFU
benchmark. It supersedes the synthetic 6-item lane (`RESULTS.md`), whose survival
curves were drawn under a mis-calibrated relearn attack (see §2) and should not be
used for any durability claim.

## 0. Setup

| | |
| --- | --- |
| Model | Qwen3-0.6B + LoRA (q/k/v/o), fp32 |
| Data | real TOFU `forget01` (40 forget items) / `retain99` |
| Forget oracle | full fine-tune `θ_full`; retain-only oracle for thresholds (q95) |
| Stress pool | held-out answer-bearing `heldout_forget_mixed_stress` |
| Eval | fp32, retain utility on a fixed 200-item subset, horizons `0,1,2,4,8,16,32,64,128,256,512` |
| Seeds | 0 (single-seed; multi-seed not yet run) |

All methods start from the **same** `θ_full` and are evaluated under the **same**
benign-relearn operator (§2). Resurrection = forget-answer log-likelihood margin
above the retain-oracle threshold.

## 1. Headline result

> On real TOFU, at matched immediate forgetting (r@0 = 0), the sharpness-aware
> durability term is **inert**, and the full HLC objective is **less durable than a
> simple constrained gradient-ascent baseline** (relearning half-life h50 ≈ 12 vs
> ≈ 28 steps). The mechanism is falsified; the durable contribution is the
> **evaluation protocol**, not the method.

## 2. The benign-relearn operator must be calibrated (a reusable pitfall)

The survival curve depends critically on the learning rate of the benign-relearn
"attack." The repo's smoke scripts inherited `lr = 0.005` (tuned for tiny GPT-2).
On Qwen3-0.6B LoRA that is ~250× the update-semigroup rate the design specifies
(AdamW 2e-5) and **destroys** the model within a single step — one relearn step
raised retain NLL from 2.3 to 9.6 — producing chaotic, non-monotonic curves that
*look* like signal but are artifacts:

| lr | HLC-R survival curve (r@T over t=0..512) | retain NLL |
| --- | --- | --- |
| 0.005 (inherited) | 0.00, 0.68, 0.00, 0.05, 0.00, 0.03, 0.25, 0.08, 0.90, 0.98, **0.03** | 2.3→spikes to 20→16 |
| 1e-4 (calibrated) | 0.00, 0.00, 0.00, 0.00, 0.08, 0.83, 1.00, 1.00, 1.00, 1.00, 1.00 | 2.3 → 4.7 (smooth) |

A model with 98% of facts resurrected at t=256 cannot drop to 3% at t=512 under
*more* relearning — the lr=0.005 curve is noise. **We lock lr = 1e-4, batch 8** as
the benign-update operator: it yields monotone resurrection with bounded retain
degradation. *Every prior synthetic result used lr=0.005 and is therefore an
artifact, including the apparent "short-horizon win" that did not survive
recalibration.*

## 3. Main survival comparison (locked operator, seed 0)

Resurrected fraction `r@T` (lower = more durable). All HLC variants and the matched
GA reach `r@0 = 0` (immediate-matched).

| method | t4 | t8 | t16 | t32 | t64 | h50 | retain NLL @0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLC-R (rho 0.05, λ_s 1) | 0.00 | 0.08 | 0.83 | 1.00 | 1.00 | ≈12 | 2.30 |
| HLC-base (sharpness off) | 0.00 | 0.10 | 0.93 | 1.00 | 1.00 | ≈12 | 2.30 |
| HLC tuneA (rho 0.5, λ_s 10) | 0.00 | 0.10 | 0.88 | 1.00 | 1.00 | ≈12 | 2.30 |
| HLC tuneB (rho 2.0, λ_s 20) | 0.00 | 0.45 | 1.00 | 1.00 | 1.00 | ≈8 | 2.30 |
| **Matched GA** | 0.00 | 0.00 | 0.05 | 0.65 | 1.00 | **≈28** | 3.81 |

## 4. The sharpness mechanism is inert-to-harmful (ablation + rho sweep)

- **Ablation (rho 0.05):** HLC-R and HLC-base are indistinguishable — identical
  r@0, identical retain NLL at every horizon, resurrection curves within one item.
  The sharpness term changed nothing.
- **Why:** once the model forgets (margin far below threshold), the worst-case
  margin inside a small rho-ball is *also* below threshold, so the softplus penalty
  saturates to ~0 and contributes no gradient. (`loss_sharpness` decayed 12.5 → 1.1
  during training.) The term is also largely collinear with the multi-time
  resurrection penalty already in the backbone.
- **Rho sweep:** enlarging the ball to force the penalty to bite does not help:
  rho 0.5 is still inert (≈ base); rho 2.0 makes durability **worse** (h50 ≈ 8),
  pushing the model toward a state that relearns *faster*. No setting shifted h50
  right of the ablation while keeping r@0 ≤ 0.05 and retain NLL ≤ 3.

The pre-registered pass condition (h50 shift base→≥18 at matched r@0/utility) is
**not met** at any sharpness strength.

## 5. HLC's only real edge is immediate forget/utility, not durability

At t=0, HLC reaches full forgetting (r@0 = 0) at **retain NLL 2.30**, whereas
matched GA needs **retain NLL 3.81** for the same r@0 (or only forgets 65% at NLL
2.34). So the HLC objective sits on a better *immediate* forget/utility frontier.
But:

- This is a property of the backbone (negpref + retain + KL + resurrection), **not
  the sharpness term** (ablation identical).
- It is **immediate** forgetting, not durability. Under relearning, GA — the
  more-displaced, more-damaged model — stays forgotten ~2× longer.

The likely explanation is mechanical: durability here tracks **weight
displacement**. GA moves further from `θ_full` (higher retain cost) and is
therefore slower to relearn; HLC's lighter-touch unlearning preserves utility but
sits closer to the original knowledge and relearns faster. Flatness/sharpness is
the wrong axis; displacement is the operative one.

## 6. What stands as a contribution

1. **Immediate-matched survival evaluation** — separates "forgot harder at t=0"
   from "stays forgotten," with conditional-resurrection metrics over items
   forgotten at t=0.
2. **Benign-relearn-rate calibration** — an attack-LR that is too hot yields
   artifactual non-monotonic curves; we give a concrete before/after and a locked,
   principled operator.
3. **A clean negative result** — a plausible flat-minima durability objective is
   inert (and harmful at strength) under a confound-free ablation.
4. **Baseline finding** — a simple constrained GA is ~2× more durable than the
   elaborate method at matched immediate-forget; durability tracks displacement,
   not objective design.

## 7. Caveats / open items

- **Single seed, n = 40, one stress pool, one model.** The direction (GA h50 ≈ 28
  vs HLC ≈ 12; sharpness inert) is large relative to 1/40 resolution, but a final
  claim needs seeds 1–2 and ideally `forget05` (~200 items).
- Retain NLL is matched between HLC variants but **not** between HLC and GA (GA is
  more damaged); the GA durability comparison is therefore read alongside the
  displacement explanation in §5, and the *clean* mechanism test is the §4
  ablation, which is fully matched.
- Margin-threshold metric only; exact-match/judge metrics not cross-checked here.

## 8. Artifacts

- HLC-R: `runs/survival/hlc_r_tofu01_seed0_lr1e4/`
- HLC-base (ablation): `runs/survival/hlc_base_tofu01_seed0_lr1e4/`
- HLC tuneA/tuneB: `runs/survival/hlc_r_tuneA_seed0/`, `runs/survival/hlc_r_tuneB_seed0/`
- Matched GA: `runs/survival/matched_ga_tofu01_seed0_v2_lr1e4/`
- Configs: `configs/method/hlc_r_k4_qwen3_lora_v100.yaml`, `hlc_base_k4_qwen3.yaml`,
  `hlc_r_tuneA.yaml`, `hlc_r_tuneB.yaml`, `grad_ascent_tofu01_matched.yaml`
