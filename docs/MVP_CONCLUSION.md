# MVP Conclusion

## What HLC Passed

HLC-SG passed the local harness and integration gates:

- The evaluator can score HLC checkpoints without method-specific changes.
- Survival curves, thresholded resurrection metrics, retain utility, refusal rate, `h50`, and AUC artifacts are produced consistently.
- HLC-SG completed local tiny-model runs across multiple seeds and relearn pools.
- Multi-time resurrection penalties, margin-growth penalties, and shallow differentiable inner unrolling were implemented and tested.
- On benign relearn pools, HLC-SG looked strongest in the main-track table: it had the lowest immediate resurrection fraction and the highest benign survival AUC.

## What HLC Failed

HLC-SG did not pass the decisive durability gate.

The answer-bearing stress harness showed that the benign advantage was not enough. Under immediate-matched comparison, current HLC-SG variants did not beat Strong NPO on conditional long-horizon durability:

- Original HLC-SG matched Strong NPO on immediate `r@0`, but had much worse conditional resurrection at `t=128` and `t=512`.
- K2 unrolled HLC had stronger immediate forgetting than Strong NPO, but still had worse conditional resurrection at long horizon.
- Held-out answer-bearing relearn stress preserved the same pattern: HLC-SG beat NPO + paraphrases, but did not beat immediate-matched Strong NPO or immediate-matched syntactic diversification.
- Stronger update-budget stress at `lr=0.01` made all immediate-matched methods fully resurrect by `t=512`; HLC-SG still had worse `t=128` conditional resurrection and worse conditional AUC than the immediate-matched baselines.
- The HLC advantage on benign AUC was therefore not sufficient evidence of durable unlearning.

The current conclusion is:

> HLC-SG, as implemented here with multi-time resurrection penalties, margin-growth penalties, and shallow differentiable inner unrolling, does not beat immediate-matched Strong NPO on answer-bearing long-horizon stress tests.

## Why The Result Is Valuable

This is a useful negative result because it prevents a false durability claim.

The harness exposed a common failure mode: a method can look durable because it forgets more at `t=0`, not because forgotten items resist relearning. Conditional resurrection fixes that by measuring only the items that were already forgotten before benign updates.

The strongest contribution of the local MVP is therefore the evaluation protocol:

- It separates immediate forgetting from resistance to relearning.
- It makes benign survival curves and answer-bearing stress curves comparable.
- It provides immediate-matched baseline selection.
- It defines a concrete stop condition for HLC tuning.
- It produces reproducible artifacts that can be regenerated with `scripts/reproduce_final_diagnostic.sh --verify`.

## What Future Work Must Prove

Future HLC work should be treated as a new objective, not another scalar tuning pass over the current one.

To support a positive durability claim, a future method needs to show:

- Immediate forget matched to Strong NPO or the strongest available baseline.
- Retain utility matched within a reasonable NLL band.
- Refusal rate near zero.
- Lower conditional resurrection at `t=32`, `t=128`, and `t=512`.
- Higher conditional AUC after controlling for `r@0`.
- Lower conditional target-margin growth under answer-bearing relearning.
- The same advantage across seeds, pools, and held-out relearn corpora.

The next benchmarking work should move beyond the completed local held-out, `lr=0.01` update-budget, GPT-2 LoRA smoke, GPT-2 K2 HLC, and GPT-2 long-horizon gates:

1. Start paper-structure work around the negative diagnostic result and the evaluation protocol.
2. Treat GPT-2 K4 as optional only if the goal is an ablation appendix; K2 did not justify a broader HLC sweep.
3. New HLC objectives must change the conditional-resurrection curve, not just immediate `r@0`.
4. Optional harsher local probes, such as `lr=0.02`, should be treated as robustness checks rather than a likely path to rescuing the current HLC objective.

`NPO + paraphrases` has now been run through the answer-bearing stress protocol. It is weaker than immediate-matched Strong NPO in this local harness: conditional resurrection reaches `1.000` by `t=128`, while Strong NPO is `0.426`; conditional AUC at `t=512` is `0.049` versus Strong NPO `0.362`.

`Syntactic diversification` has also been run through the same stress protocol. The initial baseline was not immediate-matched (`r0=0.444` versus Strong NPO `0.056`), so an immediate-matched sweep was added. After matching, syntactic diversification is close to Strong NPO but not a decisive win: conditional resurrection at `t=512` is `0.889` versus Strong NPO `0.907`, conditional AUC is `0.408` versus `0.362`, and retain NLL is worse (`8.046` versus `7.209`).

Held-out answer-bearing stress has now been run for HLC-SG, NPO + paraphrases, immediate-matched Strong NPO, and immediate-matched syntactic diversification. The result remains negative for an HLC durability claim:

| Method | Immediate r0 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 |
| --- | ---: | ---: | ---: | ---: |
| HLC-SG | 0.056 | 0.685 | 0.978 | 0.351 |
| NPO + paraphrases | 0.167 | 1.000 | 1.000 | 0.057 |
| Strong NPO matched | 0.056 | 0.263 | 0.978 | 0.410 |
| Syntactic diversification matched | 0.056 | 0.404 | 0.981 | 0.480 |

HLC-SG clears the weak NPO + paraphrases baseline under held-out stress. It fails the stronger immediate-matched comparisons.

The stronger `lr=0.01` update-budget stress makes that failure sharper:

| Method | Immediate r0 | Cond. resurrection @128 | Cond. resurrection @512 | Cond. AUC @512 |
| --- | ---: | ---: | ---: | ---: |
| HLC-SG | 0.056 | 0.963 | 1.000 | 0.119 |
| NPO + paraphrases | 0.167 | 1.000 | 1.000 | 0.030 |
| Strong NPO matched | 0.056 | 0.663 | 1.000 | 0.257 |
| Syntactic diversification matched | 0.056 | 0.448 | 1.000 | 0.356 |

At this budget, `t=512` conditional resurrection saturates for all immediate-matched methods, so the most informative fields are `t=128` conditional resurrection and `t=512` conditional AUC. Both reject an HLC durability win.

The GPT-2 LoRA smoke lane has been completed as the next evidence gate:

- `configs/model/gpt2_lora_smoke.yaml`
- `configs/method/strong_npo_gpt2_lora_smoke.yaml`
- `configs/method/hlc_sg_k1_gpt2_lora_smoke.yaml`
- `scripts/reproduce_gpt2_lora_smoke.sh`

At `t=32`, HLC-SG K1 and Strong NPO both had immediate `r0=0.000`, but Strong NPO was slightly better on conditional resurrection and conditional AUC:

| Method | Immediate r0 | Cond. resurrection @32 | Cond. AUC @32 | Retain NLL @32 |
| --- | ---: | ---: | ---: | ---: |
| HLC-SG K1 | 0.000 | 0.333 | 0.896 | 5.575 |
| Strong NPO | 0.000 | 0.278 | 0.910 | 5.499 |

This is a successful wiring/feasibility result, not a GPT-2 durability claim.

The GPT-2 K2 HLC candidate also completed:

- `configs/method/hlc_sg_k2_gpt2_lora_smoke.yaml`
- `scripts/reproduce_gpt2_hlc_k2_smoke.sh`

At `t=32`, all three GPT-2 methods had immediate `r0=0.000`:

| Method | Immediate r0 | Cond. resurrection @32 | Cond. AUC @32 | Retain NLL @32 |
| --- | ---: | ---: | ---: | ---: |
| HLC-SG K1 GPT-2 | 0.000 | 0.333 | 0.896 | 5.575 |
| HLC-SG K2 GPT-2 | 0.000 | 0.389 | 0.903 | 5.647 |
| Strong NPO | 0.000 | 0.278 | 0.910 | 5.499 |

K2 is feasible and improves AUC slightly over K1, but it worsens conditional resurrection and still trails Strong NPO. This is not a reason to start GPT-2 immediate-matched sweeps yet.

The GPT-2 long-horizon diagnostic has now been completed for K1, K2, and Strong NPO through `t=128`:

| Method | Immediate r0 | Cond. resurrection @32 | Cond. AUC @32 | Cond. resurrection @128 | Cond. AUC @128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HLC-SG K1 GPT-2 | 0.000 | 0.278 | 0.920 | 0.833 | 0.529 |
| HLC-SG K2 GPT-2 | 0.000 | 0.278 | 0.920 | 0.611 | 0.584 |
| Strong NPO | 0.000 | 0.333 | 0.906 | 0.556 | 0.588 |

This is the cleanest GPT-2 decision gate so far. K2 beats Strong NPO at `t=32`, but by `t=128` it has higher conditional resurrection (`0.611` versus `0.556`) and slightly lower conditional AUC (`0.584` versus `0.588`). The short-horizon signal is real, but it is not durable enough to support an HLC claim.

Supporting artifacts:

- `runs/reports/missing_baselines_stress/missing_baselines_compare.csv`
- `runs/reports/missing_baselines_stress/conditional_horizon_aggregate.csv`
- `runs/reports/missing_baselines_stress/conditional_durability_summary.md`
- `runs/reports/missing_baselines_stress/npo_paraphrase/conditional_durability_summary.md`
- `runs/reports/missing_baselines_stress/syntactic_diversification/conditional_durability_summary.md`
- `runs/survival/syntactic_sweep_match_selected.csv`
- `runs/reports/syntactic_immediate_matched_stress/conditional_durability_summary.md`
- `runs/reports/heldout_stress/conditional_durability_summary.md`
- `runs/reports/update_budget_stress/budget_stress_summary.md`
- `runs/reports/gpt2_lora_smoke/conditional_durability_summary.md`
- `runs/reports/gpt2_hlc_k2_smoke/conditional_durability_summary.md`
- `runs/reports/gpt2_long_horizon/conditional_durability_summary.md`
