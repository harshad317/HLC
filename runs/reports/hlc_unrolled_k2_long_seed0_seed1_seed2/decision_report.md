# HLC Unrolled K2 Decision Report

## Scope

This report closes the three-step HLC durability gate:

1. K2/K4 unrolled seed-0 diagnostic.
2. Expansion of the best diagnostic candidate.
3. Final decision against immediate-matched Strong NPO.

All runs use the answer-bearing stress pools:

- `forget_declarative`
- `forget_paraphrase_qa`
- `forget_mixed_stress`

The expanded run uses seeds `0,1,2` and horizon `0,1,2,4,8,16,32,64,128,256,512`.

## Step 1: K2/K4 Diagnostic

The K2/K4 diagnostic improved over earlier HLC variants on seed 0:

| Candidate | r0 | conditional resurrection @64 | conditional resurrection @128 |
| --- | ---: | ---: | ---: |
| Strong NPO | 0.000 | 0.000 | 0.222 |
| K2 unrolled + margin growth | 0.000 | 0.389 | 0.722 |
| K4 unrolled + margin growth | 0.000 | 0.111 | 0.778 |
| K1 unrolled + margin growth | 0.000 | 0.333 | 0.833 |

K2 cleared the narrow expansion gate because it moved `conditional_resurrection@128` from the previous HLC ceiling of `0.833` down to `0.722` while preserving immediate forget.

Diagnostic artifacts:

- `runs/reports/hlc_stress_sweep_unrolled_k24_seed0/sweep_ranked.csv`
- `runs/reports/hlc_stress_sweep_unrolled_k24_seed0/conditional_durability_summary.md`

## Step 2: Three-Seed Expansion

The expanded K2 run did not hold up across seeds and long horizon:

| Family | n | r0 | conditional resurrection @32 | @128 | @512 | conditional AUC @512 | retain NLL @512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong NPO | 9 | 0.056 | 0.111 | 0.426 | 0.907 | 0.362 | 7.209 |
| Original HLC-SG stress run | 9 | 0.056 | 0.093 | 0.981 | 1.000 | 0.195 | 7.218 |
| K2 unrolled + margin growth | 9 | 0.000 | 0.167 | 0.907 | 1.000 | 0.213 | 6.916 |

K2 unrolled has slightly stronger immediate forgetting (`r0=0.000` vs Strong NPO `r0=0.056`), but worse durability after conditioning:

- At `t=128`: K2 unrolled `0.907` vs Strong NPO `0.426`.
- At `t=512`: K2 unrolled `1.000` vs Strong NPO `0.907`.
- Conditional AUC @512 remains lower: K2 unrolled `0.213` vs Strong NPO `0.362`.

Expansion artifacts:

- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv`
- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/conditional_horizon_aggregate.csv`
- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/conditional_durability_summary.md`

## Decision

Stop expanding this HLC objective family under the current stress harness.

The result is scientifically useful but negative:

- Multi-time penalties helped relative to the original HLC run.
- Margin-growth penalties helped modestly.
- Differentiable unrolling helped on seed 0, especially K2.
- The improvement did not survive three-seed, long-horizon stress evaluation.

Current best conclusion:

> HLC-SG, as currently implemented with threshold resurrection, margin-growth penalty, and shallow differentiable inner unrolling, does not beat immediate-matched Strong NPO on answer-bearing relearning stress tests.

Recommended next track:

1. Freeze HLC tuning for this local MVP.
2. Finalize the main result table as a negative/diagnostic result.
3. Spend further effort on stronger baselines and stress harness presentation, not more HLC scalar or shallow-unroll tuning.
