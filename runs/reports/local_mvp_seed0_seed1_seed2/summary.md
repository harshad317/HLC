# Local MVP Evidence Summary

Source comparison: `runs/survival/compare_seed0_seed1_seed2_all_pools.csv`

## Headline

- HLC-SG improves mean AUC by 0.167 over NPO.
- HLC-SG reduces final resurrection by 0.167 absolute.
- HLC-SG lowers final mean target margin by 1.491.
- HLC-SG retain-NLL cost versus NPO is 0.141.

## Overall

| family | n | auc_mean | r_final_mean | mean_target_margin_mean | retain_nll_mean |
| --- | --- | --- | --- | --- | --- |
| HLC-SG | 9 | 0.944 | 0.056 | -1.657 | 3.612 |
| NPO | 9 | 0.778 | 0.222 | -0.166 | 3.472 |

## By Relearn Pool

| family | relearn_corpus | n | auc_mean | r_final_mean | mean_target_margin_mean | retain_nll_mean |
| --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | retain_adjacent | 3 | 0.944 | 0.056 | -1.656 | 3.612 |
| HLC-SG | syntax_matched | 3 | 0.944 | 0.056 | -1.657 | 3.613 |
| HLC-SG | synthetic_biography | 3 | 0.944 | 0.056 | -1.658 | 3.613 |
| NPO | retain_adjacent | 3 | 0.778 | 0.222 | -0.167 | 3.472 |
| NPO | syntax_matched | 3 | 0.778 | 0.222 | -0.164 | 3.472 |
| NPO | synthetic_biography | 3 | 0.778 | 0.222 | -0.165 | 3.472 |

## Figures

- `auc_by_pool.png`
- `resurrection_by_pool.png`
- `target_margin_by_pool.png`
- `retain_nll_by_pool.png`
