# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 8 | 3 | 0.000 | 0.056 | 0.056 | 0.056 | 0.965 | 0.151 | 3.660 |
| HLC-SG stress-trained | 8 | 3 | 0.000 | 0.056 | 0.056 | 0.056 | 0.965 | 0.152 | 3.661 |
| Strong NPO | 8 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.320 | 3.491 |
| HLC-SG | 16 | 3 | 0.000 | 0.167 | 0.167 | 0.167 | 0.927 | 0.318 | 3.723 |
| HLC-SG stress-trained | 16 | 3 | 0.000 | 0.111 | 0.111 | 0.111 | 0.941 | 0.259 | 4.021 |
| Strong NPO | 16 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.620 | 3.536 |
| HLC-SG | 32 | 3 | 0.000 | 0.111 | 0.111 | 0.111 | 0.894 | 0.538 | 4.848 |
| HLC-SG stress-trained | 32 | 3 | 0.000 | 0.222 | 0.222 | 0.222 | 0.887 | 0.996 | 4.163 |
| Strong NPO | 32 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.270 | 3.655 |
| HLC-SG | 128 | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.522 | 4.117 | 6.034 |
| HLC-SG stress-trained | 128 | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.527 | 3.382 | 5.724 |
| Strong NPO | 128 | 3 | 0.000 | 0.222 | 0.222 | 0.222 | 0.944 | 4.659 | 4.561 |
| HLC-SG | 512 | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.151 | 5.528 | 7.685 |
| HLC-SG stress-trained | 512 | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.153 | 3.675 | 6.877 |
| Strong NPO | 512 | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.375 | 7.532 | 7.730 |

## Pairwise

- t=8: HLC-SG conditional resurrection 0.056 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.035.
- t=16: HLC-SG conditional resurrection 0.167 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.073.
- t=32: HLC-SG conditional resurrection 0.111 vs Strong NPO 0.000; ratio n/a; conditional AUC gap -0.106.
- t=128: HLC-SG conditional resurrection 1.000 vs Strong NPO 0.222; ratio 4.500; conditional AUC gap -0.422.
- t=512: HLC-SG conditional resurrection 1.000 vs Strong NPO 1.000; ratio 1.000; conditional AUC gap -0.224.
