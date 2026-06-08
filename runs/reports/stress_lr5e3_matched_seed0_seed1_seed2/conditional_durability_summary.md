# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 8 | 9 | 0.056 | 0.093 | 0.037 | 0.037 | 0.977 | 0.150 | 3.660 |
| Strong NPO | 8 | 9 | 0.056 | 0.056 | 0.000 | 0.000 | 1.000 | 0.328 | 3.507 |
| HLC-SG | 16 | 9 | 0.056 | 0.130 | 0.074 | 0.093 | 0.956 | 0.265 | 3.915 |
| Strong NPO | 16 | 9 | 0.056 | 0.037 | -0.019 | 0.000 | 1.000 | 0.585 | 3.596 |
| HLC-SG | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.093 | 0.932 | 0.605 | 4.480 |
| Strong NPO | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.111 | 0.972 | 1.328 | 3.852 |
| HLC-SG | 128 | 9 | 0.056 | 0.981 | 0.926 | 0.981 | 0.605 | 3.674 | 6.179 |
| Strong NPO | 128 | 9 | 0.056 | 0.426 | 0.370 | 0.426 | 0.789 | 4.117 | 4.833 |
| HLC-SG | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.195 | 4.852 | 7.218 |
| Strong NPO | 512 | 9 | 0.056 | 0.907 | 0.852 | 0.907 | 0.362 | 6.243 | 7.209 |

## Pairwise

- t=8: HLC-SG conditional resurrection 0.037 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.023.
- t=16: HLC-SG conditional resurrection 0.093 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.044.
- t=32: HLC-SG conditional resurrection 0.093 vs Strong NPO 0.111; ratio 0.833; conditional AUC gap -0.041.
- t=128: HLC-SG conditional resurrection 0.981 vs Strong NPO 0.426; ratio 2.304; conditional AUC gap -0.184.
- t=512: HLC-SG conditional resurrection 1.000 vs Strong NPO 0.907; ratio 1.102; conditional AUC gap -0.166.
