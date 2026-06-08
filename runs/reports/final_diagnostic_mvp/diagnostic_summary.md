# Local MVP Diagnostic Summary

## Headline

The benign main-track table makes `HLC-SG` look strongest, but the answer-bearing stress harness does not support an HLC durability claim.

The current defensible claim is diagnostic and negative: HLC-SG variants improved through multi-time penalties, margin-growth penalties, and shallow differentiable unrolling, but they did not beat immediate-matched Strong NPO under long-horizon stress.

## Benign Main Track

Source: `runs/reports/main_track_seed0_seed1_seed2/main_track_table.csv`

| Method | Immediate forget | Retain utility | Refusal | h50 | AUC-survival | Held-out h50 |
| --- | --- | --- | --- | --- | --- | --- |
| NPO | pass; exact=0.000, r0=0.833 | retain NLL=3.674 | 0.000 | 0 observed (1/1) | 0.167 (n=1) | not run |
| Strong NPO | pass; exact=0.000, r0=0.222 | retain NLL=3.472 | 0.000 | >32 censored (9/9) | 0.778 +/- 0.083 (n=9) | >32 censored (3/3) |
| NPO + paraphrases | pass; exact=0.000, r0=0.167 | retain NLL=3.472 | 0.000 | >32 censored (9/9) | 0.833 +/- 0.000 (n=9) | >32 censored (3/3) |
| Syntactic diversification | pass; exact=0.000, r0=0.444 | retain NLL=3.472 | 0.000 | mixed: 6/9 >32; observed median 0 | 0.556 +/- 0.417 (n=9) | mixed: 2/3 >32; observed median 0 |
| HLC-SG | pass; exact=0.000, r0=0.056 | retain NLL=3.612 | 0.000 | >32 censored (9/9) | 0.944 +/- 0.083 (n=9) | >32 censored (3/3) |

## Answer-Bearing Stress Verdict

Source: `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv`

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 | Retain NLL @512 | Immediate matched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strong NPO | 9 | 0.056 | 0.111 | 0.426 | 0.907 | 0.362 | 7.209 | True |
| HLC-SG | 9 | 0.056 | 0.093 | 0.981 | 1.000 | 0.195 | 7.218 | True |
| hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two | 9 | 0.000 | 0.167 | 0.907 | 1.000 | 0.213 | 6.916 | False |

## Decision

- Freeze HLC tuning for the local MVP.
- Treat the HLC result as a useful negative/diagnostic result, not a SOTA durability claim.
- Present immediate-matched conditional resurrection as the decisive metric.
- Spend next effort on strengthening baseline coverage and clearly explaining the stress harness.

## Supporting Report

Detailed stop-condition report: `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/decision_report.md`
