# Local MVP Evidence Summary

Source comparison: `runs/survival/compare_main_track_with_plain_npo_seed0_seed1_seed2_all_pools.csv`

## Headline

- HLC-SG improves mean AUC by 0.778 over NPO.
- HLC-SG reduces final resurrection by 0.778 absolute.
- HLC-SG lowers final mean target margin by 2.625.
- HLC-SG retain-NLL cost versus NPO is -0.061.

## Overall

| family | n | auc_mean | r_final_mean | mean_target_margin_mean | retain_nll_mean |
| --- | --- | --- | --- | --- | --- |
| HLC-SG | 9 | 0.944 | 0.056 | -1.657 | 3.612 |
| NPO | 1 | 0.167 | 0.833 | 0.968 | 3.674 |
| NPO + paraphrases | 9 | 0.833 | 0.167 | -0.135 | 3.472 |
| Strong NPO | 9 | 0.778 | 0.222 | -0.166 | 3.472 |
| Syntactic diversification | 9 | 0.556 | 0.444 | 1.276 | 3.472 |

## By Relearn Pool

| family | relearn_corpus | n | auc_mean | r_final_mean | mean_target_margin_mean | retain_nll_mean |
| --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | retain_adjacent | 3 | 0.944 | 0.056 | -1.656 | 3.612 |
| HLC-SG | syntax_matched | 3 | 0.944 | 0.056 | -1.657 | 3.613 |
| HLC-SG | synthetic_biography | 3 | 0.944 | 0.056 | -1.658 | 3.613 |
| NPO | synthetic_biography | 1 | 0.167 | 0.833 | 0.968 | 3.674 |
| NPO + paraphrases | retain_adjacent | 3 | 0.833 | 0.167 | -0.137 | 3.472 |
| NPO + paraphrases | syntax_matched | 3 | 0.833 | 0.167 | -0.134 | 3.472 |
| NPO + paraphrases | synthetic_biography | 3 | 0.833 | 0.167 | -0.135 | 3.472 |
| Strong NPO | retain_adjacent | 3 | 0.778 | 0.222 | -0.167 | 3.472 |
| Strong NPO | syntax_matched | 3 | 0.778 | 0.222 | -0.164 | 3.472 |
| Strong NPO | synthetic_biography | 3 | 0.778 | 0.222 | -0.165 | 3.472 |
| Syntactic diversification | retain_adjacent | 3 | 0.556 | 0.444 | 1.275 | 3.472 |
| Syntactic diversification | syntax_matched | 3 | 0.556 | 0.444 | 1.279 | 3.472 |
| Syntactic diversification | synthetic_biography | 3 | 0.556 | 0.444 | 1.273 | 3.472 |

## Figures

- `auc_by_pool.png`
- `resurrection_by_pool.png`
- `target_margin_by_pool.png`
- `retain_nll_by_pool.png`
