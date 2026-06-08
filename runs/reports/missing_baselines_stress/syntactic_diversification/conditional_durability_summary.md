# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.093 | 0.932 | 0.605 | 4.480 |
| Strong NPO | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.111 | 0.972 | 1.328 | 3.852 |
| Syntactic diversification | 32 | 9 | 0.444 | 0.944 | 0.500 | 0.600 | 0.710 | 1.587 | 3.666 |
| HLC-SG | 128 | 9 | 0.056 | 0.981 | 0.926 | 0.981 | 0.605 | 3.674 | 6.179 |
| Strong NPO | 128 | 9 | 0.056 | 0.426 | 0.370 | 0.426 | 0.789 | 4.117 | 4.833 |
| Syntactic diversification | 128 | 9 | 0.444 | 1.000 | 0.556 | 0.667 | 0.444 | 2.396 | 4.642 |
| HLC-SG | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.195 | 4.852 | 7.218 |
| Strong NPO | 512 | 9 | 0.056 | 0.907 | 0.852 | 0.907 | 0.362 | 6.243 | 7.209 |
| Syntactic diversification | 512 | 9 | 0.444 | 1.000 | 0.556 | 0.667 | 0.361 | 2.079 | 7.081 |

## Pairwise

- t=32: Syntactic diversification conditional resurrection 0.600 vs Strong NPO 0.111; ratio 5.400; conditional AUC gap -0.262.
- t=128: Syntactic diversification conditional resurrection 0.667 vs Strong NPO 0.426; ratio 1.565; conditional AUC gap -0.345.
- t=512: Syntactic diversification conditional resurrection 0.667 vs Strong NPO 0.907; ratio 0.735; conditional AUC gap -0.001.
