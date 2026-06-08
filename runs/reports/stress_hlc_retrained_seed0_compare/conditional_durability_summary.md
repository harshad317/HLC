# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 8 | 6 | 0.000 | 0.056 | 0.056 | 0.056 | 0.965 | 0.151 | 3.661 |
| Strong NPO | 8 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.320 | 3.491 |
| HLC-SG | 16 | 6 | 0.000 | 0.139 | 0.139 | 0.139 | 0.934 | 0.289 | 3.872 |
| Strong NPO | 16 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.620 | 3.536 |
| HLC-SG | 32 | 6 | 0.000 | 0.167 | 0.167 | 0.167 | 0.891 | 0.767 | 4.505 |
| Strong NPO | 32 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.270 | 3.655 |
| HLC-SG | 128 | 6 | 0.000 | 1.000 | 1.000 | 1.000 | 0.525 | 3.749 | 5.879 |
| Strong NPO | 128 | 3 | 0.000 | 0.222 | 0.222 | 0.222 | 0.944 | 4.659 | 4.561 |
| HLC-SG | 512 | 6 | 0.000 | 1.000 | 1.000 | 1.000 | 0.152 | 4.602 | 7.281 |
| Strong NPO | 512 | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.375 | 7.532 | 7.730 |

## Pairwise

- t=8: HLC-SG conditional resurrection 0.056 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.035.
- t=16: HLC-SG conditional resurrection 0.139 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.066.
- t=32: HLC-SG conditional resurrection 0.167 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.109.
- t=128: HLC-SG conditional resurrection 1.000 vs Strong NPO 0.222; ratio 4.500; conditional AUC gap -0.420.
- t=512: HLC-SG conditional resurrection 1.000 vs Strong NPO 1.000; ratio 1.000; conditional AUC gap -0.223.
