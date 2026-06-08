# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 32 | 9 | 0.056 | 0.167 | 0.111 | 0.111 | 0.936 | 1.122 | 4.982 |
| NPO + paraphrases | 32 | 9 | 0.167 | 0.963 | 0.796 | 0.956 | 0.453 | 1.376 | 4.474 |
| Strong NPO | 32 | 9 | 0.056 | 0.148 | 0.093 | 0.133 | 0.967 | 2.280 | 5.614 |
| Syntactic diversification | 32 | 9 | 0.056 | 0.093 | 0.037 | 0.093 | 0.977 | 1.834 | 4.852 |
| HLC-SG | 128 | 9 | 0.056 | 0.963 | 0.907 | 0.963 | 0.458 | 4.349 | 8.097 |
| NPO + paraphrases | 128 | 9 | 0.167 | 1.000 | 0.833 | 1.000 | 0.119 | 3.128 | 7.344 |
| Strong NPO | 128 | 9 | 0.056 | 0.667 | 0.611 | 0.663 | 0.715 | 4.808 | 9.075 |
| Syntactic diversification | 128 | 9 | 0.056 | 0.481 | 0.426 | 0.448 | 0.775 | 3.476 | 9.796 |
| HLC-SG | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.119 | 7.092 | 10.503 |
| NPO + paraphrases | 512 | 9 | 0.167 | 1.000 | 0.833 | 1.000 | 0.030 | 5.000 | 9.907 |
| Strong NPO | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.257 | 8.578 | 9.853 |
| Syntactic diversification | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.356 | 7.605 | 9.609 |

## Pairwise

- t=32: HLC-SG conditional resurrection 0.111 vs Strong NPO 0.133; ratio 0.833; conditional AUC gap -0.031.
- t=128: HLC-SG conditional resurrection 0.963 vs Strong NPO 0.663; ratio 1.453; conditional AUC gap -0.256.
- t=512: HLC-SG conditional resurrection 1.000 vs Strong NPO 1.000; ratio 1.000; conditional AUC gap -0.138.
