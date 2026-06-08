# Local MVP Results

## Summary

This local MVP implements the durability harness for HLC-style unlearning and reaches a negative but useful conclusion:

> HLC-SG, as implemented here with multi-time resurrection penalties, margin-growth penalties, and shallow differentiable inner unrolling, does not beat immediate-matched Strong NPO on answer-bearing long-horizon stress tests.

The repo should therefore treat the current HLC result as diagnostic evidence, not as a durability claim. The strongest artifact is the evaluator and stress protocol: it separates immediate forgetting from resistance to benign relearning.

## 1. Benign Survival Curves Suggested HLC Was Promising

The benign main-track table made HLC-SG look strongest under the local tiny-GPT-2 full-finetune smoke setup. On the benign relearn pools, HLC-SG had the best survival AUC and the lowest immediate resurrection fraction:

| Method | Immediate r0 | AUC-survival | h50 |
| --- | ---: | ---: | --- |
| NPO | 0.833 | 0.167 | 0 observed (n=1) |
| Strong NPO | 0.222 | 0.778 +/- 0.083 | >32 censored (9/9) |
| NPO + paraphrases | 0.167 | 0.833 +/- 0.000 | >32 censored (9/9) |
| Syntactic diversification | 0.444 | 0.556 +/- 0.417 | mixed |
| HLC-SG | 0.056 | 0.944 +/- 0.083 | >32 censored (9/9) |

This is the tempting result: HLC-SG appears best if one only looks at immediate forget plus benign survival.

Source artifacts:

- `runs/reports/main_track_seed0_seed1_seed2/main_track_table.csv`
- `runs/reports/main_track_seed0_seed1_seed2/main_track_table.md`

## 2. Immediate-Matched Conditional Evaluation Changed The Conclusion

The initial HLC advantage was partly explained by stronger immediate forgetting. The evaluator therefore added conditional durability metrics:

- `relearn_growth = r@T - r@0`
- `survival_decay = S(0) - S(T)`
- `conditional_resurrection@T`: fraction of items forgotten at `t=0` that resurrect after relearning
- conditional AUC over items forgotten at `t=0`

This matters because a method can look durable simply by forgetting more aggressively at the start. Conditional resurrection asks the harder question: among items that were already forgotten, which method keeps them forgotten after benign updates?

The immediate-matched stress runs made this distinction decisive. HLC could win or tie immediate forget, but did not keep forgotten facts suppressed after answer-bearing relearning.

## 3. Answer-Bearing Stress Pools Were Decisive

The stress harness evaluates against relearn pools that carry forget-set answer information:

- `forget_declarative`
- `forget_paraphrase_qa`
- `forget_mixed_stress`

Across seeds `0,1,2`, pools, and horizon `0,1,2,4,8,16,32,64,128,256,512`, Strong NPO remained better on conditional resurrection.

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 | Retain NLL @512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong NPO | 9 | 0.056 | 0.111 | 0.426 | 0.907 | 0.362 | 7.209 |
| Original HLC-SG stress run | 9 | 0.056 | 0.093 | 0.981 | 1.000 | 0.195 | 7.218 |
| K2 unrolled HLC + margin growth | 9 | 0.000 | 0.167 | 0.907 | 1.000 | 0.213 | 6.916 |

The K2 unrolled HLC variant had stronger immediate forgetting (`r0=0.000`) than Strong NPO (`r0=0.056`), but it still had worse conditional resurrection at `t=128` and `t=512`.

Source artifacts:

- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv`
- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/conditional_horizon_aggregate.csv`
- `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/decision_report.md`

## 4. HLC Objective Improvements Helped But Failed To Close The Gap

Several HLC improvements were implemented and tested:

- Multi-time resurrection penalties across selected inner relearn steps.
- Margin-growth penalty against post-relearn forget-margin increase.
- Shallow differentiable unrolling through inner SGD updates using `post_gradient_mode: unrolled`.

These changes improved the local HLC signal incrementally:

| Candidate | Seed-0 immediate r0 | Cond. resurrection @64 | Cond. resurrection @128 |
| --- | ---: | ---: | ---: |
| Strong NPO | 0.000 | 0.000 | 0.222 |
| K1 unrolled + margin growth | 0.000 | 0.333 | 0.833 |
| K2 unrolled + margin growth | 0.000 | 0.389 | 0.722 |
| K4 unrolled + margin growth | 0.000 | 0.111 | 0.778 |

K2 cleared the seed-0 expansion gate because it moved `conditional_resurrection@128` from the prior HLC ceiling of `0.833` to `0.722`. But the three-seed long-horizon expansion did not preserve that advantage.

Conclusion: the HLC objective family improved, but the improvement was not robust enough to support a durability claim under the current stress harness.

## 5. Final MVP Claim

The strongest claim from this repo is now:

> The local-first harness can expose when apparent unlearning durability is actually stronger immediate forgetting. Under this harness, current HLC-SG variants do not outperform immediate-matched Strong NPO on answer-bearing relearning stress.

That is a useful MVP result because it gives a concrete stop condition and prevents overstating the method.

The next work should not be more scalar HLC tuning. Better next steps are:

1. Strengthen and document baseline coverage.
2. Improve stress-harness presentation.
3. Extend the completed GPT-2 LoRA smoke lane into immediate-matched GPT-2 sweeps only if the selected configs are feasible.
4. Keep HLC-SG as a documented negative/diagnostic result unless a new objective changes the conditional-resurrection curve substantially.

## 6. Missing Baseline Stress Update

`NPO + paraphrases` and `Syntactic diversification` have now been run through the same answer-bearing stress protocol used for the HLC/Strong NPO decision.

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 | Retain NLL @512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong NPO | 9 | 0.056 | 0.111 | 0.426 | 0.907 | 0.362 | 7.209 |
| NPO + paraphrases | 9 | 0.167 | 0.778 | 1.000 | 1.000 | 0.049 | 7.116 |
| Syntactic diversification | 9 | 0.444 | 0.600 | 0.667 | 0.667 | 0.361 | 7.081 |

`NPO + paraphrases` is weaker than immediate-matched Strong NPO under stress. It also is not immediate-matched to Strong NPO in the favorable direction: its `r0` is worse, and its conditional durability is worse.

`Syntactic diversification` is a more interesting diagnostic baseline. The first run had lower conditional resurrection than Strong NPO by `t=512`, but it was not immediate-matched: `r0=0.444` versus Strong NPO `0.056`. An immediate-matched sweep was therefore run.

Immediate-matched syntactic diversification selected one config per seed:

| Seed | Selected config | r0 | Reference r0 |
| ---: | --- | ---: | ---: |
| 0 | `st600_fw2_b0p3` | 0.000 | 0.000 |
| 1 | `st360_fw4_b0p3` | 0.000 | 0.000 |
| 2 | `st240_fw2_b0p3` | 0.167 | 0.167 |

After matching, syntactic diversification is close to Strong NPO, not a decisive win:

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 | Retain NLL @512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong NPO | 9 | 0.056 | 0.111 | 0.426 | 0.907 | 0.362 | 7.209 |
| Syntactic diversification matched | 9 | 0.056 | 0.037 | 0.452 | 0.889 | 0.408 | 8.046 |

This is a stronger baseline than the unmatched table suggested, but it still does not change the HLC conclusion. It gives a fairer competitor for future held-out stress, with a retain-utility caveat.

Source artifacts:

- `runs/reports/missing_baselines_stress/missing_baselines_compare.csv`
- `runs/reports/missing_baselines_stress/conditional_horizon_aggregate.csv`
- `runs/reports/missing_baselines_stress/conditional_durability_summary.md`
- `runs/reports/missing_baselines_stress/npo_paraphrase/conditional_durability_summary.md`
- `runs/reports/missing_baselines_stress/syntactic_diversification/conditional_durability_summary.md`
- `runs/survival/syntactic_sweep_match_selected.csv`
- `runs/reports/syntactic_immediate_matched_stress/conditional_durability_summary.md`

## 7. Held-Out Relearn Stress

A held-out stress gate was added after the immediate-matched baseline work. It keeps the same forget targets, thresholds, seeds, horizon, and conditional metrics, but uses answer-bearing relearn examples with held-out wording:

- `heldout_forget_declarative`
- `heldout_forget_paraphrase_qa`
- `heldout_forget_mixed_stress`

This tests whether the conclusion depends on the exact stress-pool phrasing used during prior evaluation.

Across seeds `0,1,2` and the three held-out pools, the result is still negative for HLC durability:

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 | Retain NLL @512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLC-SG | 9 | 0.056 | 0.037 | 0.685 | 0.978 | 0.351 | 8.397 |
| NPO + paraphrases | 9 | 0.167 | 0.733 | 1.000 | 1.000 | 0.057 | 8.167 |
| Strong NPO matched | 9 | 0.056 | 0.000 | 0.263 | 0.978 | 0.410 | 9.816 |
| Syntactic diversification matched | 9 | 0.056 | 0.056 | 0.404 | 0.981 | 0.480 | 10.040 |

HLC-SG is clearly stronger than NPO + paraphrases under held-out stress. But that is not the decisive comparison. Against immediate-matched Strong NPO, HLC-SG has worse conditional resurrection at `t=128` (`0.685` versus `0.263`) and lower conditional AUC at `t=512` (`0.351` versus `0.410`). Against immediate-matched syntactic diversification, HLC-SG also has lower conditional AUC at `t=512` (`0.351` versus `0.480`).

The held-out gate therefore strengthens the benchmark, but it does not rescue the HLC claim. The local conclusion remains: current HLC-SG is a useful diagnostic method, not a demonstrated durability win.

Source artifacts:

- `data/relearn_pools/heldout_forget_declarative.jsonl`
- `data/relearn_pools/heldout_forget_paraphrase_qa.jsonl`
- `data/relearn_pools/heldout_forget_mixed_stress.jsonl`
- `runs/reports/heldout_stress/compare.csv`
- `runs/reports/heldout_stress/conditional_horizon_aggregate.csv`
- `runs/reports/heldout_stress/conditional_durability_summary.md`

## 8. Stronger Update-Budget Stress

The held-out stress gate was then rerun with a stronger answer-bearing relearning update budget: `lr=0.01`, double the prior `lr=0.005` setting, with the same seeds, held-out pools, horizon, thresholds, and immediate-matched selected baselines.

At this stronger budget, all immediate-matched methods fully resurrect by `t=512`. The useful signal therefore shifts to earlier horizons and conditional AUC:

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 | Retain NLL @512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLC-SG | 9 | 0.056 | 0.111 | 0.963 | 1.000 | 0.119 | 10.503 |
| NPO + paraphrases | 9 | 0.167 | 0.956 | 1.000 | 1.000 | 0.030 | 9.907 |
| Strong NPO matched | 9 | 0.056 | 0.133 | 0.663 | 1.000 | 0.257 | 9.853 |
| Syntactic diversification matched | 9 | 0.056 | 0.093 | 0.448 | 1.000 | 0.356 | 9.609 |

This stress budget makes the negative conclusion stronger. HLC-SG has slightly lower conditional resurrection than Strong NPO at `t=32`, but by `t=128` it has much higher conditional resurrection (`0.963` versus `0.663`) and by `t=512` its conditional AUC is worse (`0.119` versus `0.257`). Syntactic diversification is also stronger than HLC-SG under this budget: `0.448` conditional resurrection at `t=128` and `0.356` conditional AUC at `t=512`.

The result says the local HLC objective does not merely fail under one benign/stress setting; it fails as answer-bearing relearning pressure increases.

Source artifacts:

- `runs/reports/update_budget_stress/lr0p01/compare.csv`
- `runs/reports/update_budget_stress/lr0p01/conditional_horizon_aggregate.csv`
- `runs/reports/update_budget_stress/lr0p01/conditional_durability_summary.md`
- `runs/reports/update_budget_stress/budget_horizon_aggregate.csv`
- `runs/reports/update_budget_stress/budget_stress_summary.md`

## 9. GPT-2 LoRA Smoke Gate

The first larger-model lane was implemented and completed with `gpt2` plus LoRA adapters. This is a seed-0 smoke gate, not a paper-grade benchmark. It verifies that GPT-2 LoRA checkpoints train, calibrate thresholds, unlearn, and evaluate through the same held-out answer-bearing survival harness.

The smoke run used:

- `configs/model/gpt2_lora_smoke.yaml`
- `configs/method/strong_npo_gpt2_lora_smoke.yaml`
- `configs/method/hlc_sg_k1_gpt2_lora_smoke.yaml`
- held-out stress pools
- horizon `0,1,2,4,8,16,32`

At `t=32`, both methods had immediate `r0=0.000`. Strong NPO was slightly better on conditional resurrection and conditional AUC:

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. AUC @32 | Retain NLL @32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 | 3 | 0.000 | 0.333 | 0.896 | 5.575 |
| Strong NPO | 3 | 0.000 | 0.278 | 0.910 | 5.499 |

The useful outcome is that the larger-model adapter path is now validated. The result should not be read as a GPT-2 durability conclusion because it is one seed, K1 HLC, short horizon, and not immediate-matched beyond the observed `r0=0.000` tie.

Source artifacts:

- `checkpoints/gpt2_lora_full_seed0`
- `checkpoints/gpt2_lora_retain_only_seed0`
- `thresholds/tofu50_gpt2_lora_seed0_thresholds.json`
- `checkpoints/unlearned/npo_strong_gpt2_lora_tofu50_seed0`
- `checkpoints/unlearned/hlc_sg_k1_gpt2_lora_tofu50_seed0`
- `runs/reports/gpt2_lora_smoke/compare.csv`
- `runs/reports/gpt2_lora_smoke/conditional_horizon_aggregate.csv`
- `runs/reports/gpt2_lora_smoke/conditional_durability_summary.md`

## 10. GPT-2 HLC K2 Smoke Gate

After the K1 wiring gate, a stronger GPT-2 HLC candidate was tested with `K=2`, 10 outer steps, multi-time resurrection penalties at inner steps `1,2`, and a modest margin-growth penalty.

The K2 run reused the completed GPT-2 full checkpoint, thresholds, Strong NPO checkpoint, and K1 survival curves. It only trained and evaluated the K2 HLC adapter.

At `t=32`, all methods were immediate-matched at `r0=0.000`:

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. AUC @32 | Retain NLL @32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 GPT-2 | 3 | 0.000 | 0.333 | 0.896 | 5.575 |
| HLC-SG K2 GPT-2 | 3 | 0.000 | 0.389 | 0.903 | 5.647 |
| Strong NPO | 3 | 0.000 | 0.278 | 0.910 | 5.499 |

K2 is feasible and slightly improves conditional AUC over K1, but it worsens conditional resurrection and retain NLL. It still does not beat Strong NPO. The GPT-2 lane therefore preserves the current pattern: stronger HLC variants can be executed, but the measured durability signal is not yet competitive.

Source artifacts:

- `configs/method/hlc_sg_k2_gpt2_lora_smoke.yaml`
- `checkpoints/unlearned/hlc_sg_k2_gpt2_lora_tofu50_seed0`
- `runs/reports/gpt2_hlc_k2_smoke/compare.csv`
- `runs/reports/gpt2_hlc_k2_smoke/conditional_horizon_aggregate.csv`
- `runs/reports/gpt2_hlc_k2_smoke/conditional_durability_summary.md`

## 11. GPT-2 Long-Horizon Diagnostic

The GPT-2 K1/K2/Strong NPO lane was then extended from `t=32` to `t=128` without retraining. This used the same seed-0 GPT-2 LoRA checkpoints and held-out answer-bearing pools:

- `heldout_forget_declarative`
- `heldout_forget_paraphrase_qa`
- `heldout_forget_mixed_stress`

At `t=32`, K2 showed the first useful short-horizon GPT-2 signal: lower conditional resurrection than Strong NPO and slightly higher conditional AUC. By `t=128`, that advantage disappeared.

| Method | n | Immediate r0 | Cond. resurrection @32 | Cond. AUC @32 | Cond. resurrection @128 | Cond. AUC @128 | Retain NLL @128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 GPT-2 | 3 | 0.000 | 0.278 | 0.920 | 0.833 | 0.529 | 6.838 |
| HLC-SG K2 GPT-2 | 3 | 0.000 | 0.278 | 0.920 | 0.611 | 0.584 | 6.597 |
| Strong NPO | 3 | 0.000 | 0.333 | 0.906 | 0.556 | 0.588 | 6.776 |

The pairwise K2 comparison is therefore:

- `t=32`: HLC-SG K2 conditional resurrection `0.278` versus Strong NPO `0.333`; conditional AUC gap `+0.014`.
- `t=128`: HLC-SG K2 conditional resurrection `0.611` versus Strong NPO `0.556`; conditional AUC gap `-0.003`.

This is a clean decision point. K2 is feasible and less brittle than K1 at `t=128`, but it does not beat Strong NPO on long-horizon conditional durability. The current GPT-2 evidence therefore supports the same conclusion as the local full-finetune MVP: the harness is working, but this HLC objective family should not be presented as a durability win.

Source artifacts:

- `scripts/reproduce_gpt2_long_horizon.sh`
- `runs/reports/gpt2_long_horizon/compare.csv`
- `runs/reports/gpt2_long_horizon/conditional_horizon_aggregate.csv`
- `runs/reports/gpt2_long_horizon/conditional_durability_summary.md`

## Canonical Artifacts

- Evaluation protocol: `docs/EVALUATION_PROTOCOL.md`
- MVP conclusion: `docs/MVP_CONCLUSION.md`
- Missing baseline stress summary: `runs/reports/missing_baselines_stress/conditional_durability_summary.md`
- NPO paraphrase stress summary: `runs/reports/missing_baselines_stress/npo_paraphrase/conditional_durability_summary.md`
- Syntactic diversification stress summary: `runs/reports/missing_baselines_stress/syntactic_diversification/conditional_durability_summary.md`
- Immediate-matched syntactic summary: `runs/reports/syntactic_immediate_matched_stress/conditional_durability_summary.md`
- Held-out stress summary: `runs/reports/heldout_stress/conditional_durability_summary.md`
- Update-budget stress summary: `runs/reports/update_budget_stress/budget_stress_summary.md`
- GPT-2 LoRA smoke summary: `runs/reports/gpt2_lora_smoke/conditional_durability_summary.md`
- GPT-2 HLC K2 smoke summary: `runs/reports/gpt2_hlc_k2_smoke/conditional_durability_summary.md`
- GPT-2 long-horizon summary: `runs/reports/gpt2_long_horizon/conditional_durability_summary.md`
- Final diagnostic summary: `runs/reports/final_diagnostic_mvp/diagnostic_summary.md`
- Final diagnostic main-track table: `runs/reports/final_diagnostic_mvp/diagnostic_main_track_table.csv`
- Final diagnostic stress table: `runs/reports/final_diagnostic_mvp/diagnostic_stress_table.csv`
- HLC stop-condition report: `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/decision_report.md`
- K2 long-horizon stress ranking: `runs/reports/hlc_unrolled_k2_long_seed0_seed1_seed2/long_horizon_ranked.csv`
