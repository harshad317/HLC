# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 32 | 9 | 0.056 | 0.037 | -0.019 | 0.037 | 0.956 | 0.439 | 4.635 |
| NPO + paraphrases | 32 | 9 | 0.167 | 0.778 | 0.611 | 0.733 | 0.740 | 0.791 | 3.605 |
| Strong NPO | 32 | 9 | 0.056 | 0.019 | -0.037 | 0.000 | 1.000 | 1.100 | 4.213 |
| Syntactic diversification | 32 | 9 | 0.056 | 0.111 | 0.056 | 0.056 | 0.986 | 1.136 | 4.079 |
| HLC-SG | 128 | 9 | 0.056 | 0.685 | 0.630 | 0.685 | 0.778 | 2.417 | 7.747 |
| NPO + paraphrases | 128 | 9 | 0.167 | 1.000 | 0.833 | 1.000 | 0.227 | 1.953 | 5.711 |
| Strong NPO | 128 | 9 | 0.056 | 0.259 | 0.204 | 0.263 | 0.909 | 3.302 | 6.610 |
| Syntactic diversification | 128 | 9 | 0.056 | 0.389 | 0.333 | 0.404 | 0.854 | 3.283 | 7.090 |
| HLC-SG | 512 | 9 | 0.056 | 0.981 | 0.926 | 0.978 | 0.351 | 5.571 | 8.397 |
| NPO + paraphrases | 512 | 9 | 0.167 | 1.000 | 0.833 | 1.000 | 0.057 | 4.516 | 8.167 |
| Strong NPO | 512 | 9 | 0.056 | 0.981 | 0.926 | 0.978 | 0.410 | 6.647 | 9.816 |
| Syntactic diversification | 512 | 9 | 0.056 | 0.981 | 0.926 | 0.981 | 0.480 | 6.333 | 10.040 |

## Pairwise

- t=32: HLC-SG conditional resurrection 0.037 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.044.
- t=128: HLC-SG conditional resurrection 0.685 vs Strong NPO 0.263; ratio 2.606; conditional AUC gap -0.131.
- t=512: HLC-SG conditional resurrection 0.978 vs Strong NPO 0.978; ratio 1.000; conditional AUC gap -0.059.
