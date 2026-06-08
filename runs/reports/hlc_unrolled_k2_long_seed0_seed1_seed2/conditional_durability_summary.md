# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.093 | 0.932 | 0.605 | 4.480 |
| Strong NPO | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.111 | 0.972 | 1.328 | 3.852 |
| hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two | 32 | 9 | 0.000 | 0.167 | 0.167 | 0.167 | 0.910 | 0.714 | 4.434 |
| HLC-SG | 128 | 9 | 0.056 | 0.981 | 0.926 | 0.981 | 0.605 | 3.674 | 6.179 |
| Strong NPO | 128 | 9 | 0.056 | 0.426 | 0.370 | 0.426 | 0.789 | 4.117 | 4.833 |
| hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two | 128 | 9 | 0.000 | 0.907 | 0.907 | 0.907 | 0.667 | 3.159 | 5.864 |
| HLC-SG | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.195 | 4.852 | 7.218 |
| Strong NPO | 512 | 9 | 0.056 | 0.907 | 0.852 | 0.907 | 0.362 | 6.243 | 7.209 |
| hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two | 512 | 9 | 0.000 | 1.000 | 1.000 | 1.000 | 0.213 | 5.224 | 6.916 |

## Pairwise

- t=32: hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two conditional resurrection 0.167 vs Strong NPO 0.111; ratio 1.500; conditional AUC gap -0.062.
- t=128: hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two conditional resurrection 0.907 vs Strong NPO 0.426; ratio 2.130; conditional AUC gap -0.122.
- t=512: hlc_sg_stress_sweep_k2_ilr0p001_olr0p002_l03_lk1_lr0p1_lmg1_pgunrolled_penpowers_of_two conditional resurrection 1.000 vs Strong NPO 0.907; ratio 1.102; conditional AUC gap -0.149.
