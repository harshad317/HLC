# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.093 | 0.932 | 0.605 | 4.480 |
| NPO + paraphrases | 32 | 9 | 0.167 | 0.815 | 0.648 | 0.778 | 0.667 | 1.737 | 3.583 |
| Strong NPO | 32 | 9 | 0.056 | 0.130 | 0.074 | 0.111 | 0.972 | 1.328 | 3.852 |
| HLC-SG | 128 | 9 | 0.056 | 0.981 | 0.926 | 0.981 | 0.605 | 3.674 | 6.179 |
| NPO + paraphrases | 128 | 9 | 0.167 | 1.000 | 0.833 | 1.000 | 0.194 | 3.125 | 4.589 |
| Strong NPO | 128 | 9 | 0.056 | 0.426 | 0.370 | 0.426 | 0.789 | 4.117 | 4.833 |
| HLC-SG | 512 | 9 | 0.056 | 1.000 | 0.944 | 1.000 | 0.195 | 4.852 | 7.218 |
| NPO + paraphrases | 512 | 9 | 0.167 | 1.000 | 0.833 | 1.000 | 0.049 | 3.445 | 7.116 |
| Strong NPO | 512 | 9 | 0.056 | 0.907 | 0.852 | 0.907 | 0.362 | 6.243 | 7.209 |

## Pairwise

- t=32: NPO + paraphrases conditional resurrection 0.778 vs Strong NPO 0.111; ratio 7.000; conditional AUC gap -0.306.
- t=128: NPO + paraphrases conditional resurrection 1.000 vs Strong NPO 0.426; ratio 2.348; conditional AUC gap -0.595.
- t=512: NPO + paraphrases conditional resurrection 1.000 vs Strong NPO 0.907; ratio 1.102; conditional AUC gap -0.313.
