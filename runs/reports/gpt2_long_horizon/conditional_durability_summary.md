# Conditional Durability Summary

| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLC-SG K1 GPT-2 | 32 | 3 | 0.000 | 0.278 | 0.278 | 0.278 | 0.920 | 5.281 | 5.323 |
| HLC-SG K2 GPT-2 | 32 | 3 | 0.000 | 0.278 | 0.278 | 0.278 | 0.920 | 5.226 | 5.443 |
| Strong NPO | 32 | 3 | 0.000 | 0.333 | 0.333 | 0.333 | 0.906 | 5.312 | 5.558 |
| HLC-SG K1 GPT-2 | 128 | 3 | 0.000 | 0.833 | 0.833 | 0.833 | 0.529 | 8.638 | 6.838 |
| HLC-SG K2 GPT-2 | 128 | 3 | 0.000 | 0.611 | 0.611 | 0.611 | 0.584 | 7.059 | 6.597 |
| Strong NPO | 128 | 3 | 0.000 | 0.556 | 0.556 | 0.556 | 0.588 | 6.979 | 6.776 |

## Pairwise

- t=32: HLC-SG K2 GPT-2 conditional resurrection 0.278 vs Strong NPO 0.333; ratio 0.833; conditional AUC gap 0.014.
- t=128: HLC-SG K2 GPT-2 conditional resurrection 0.611 vs Strong NPO 0.556; ratio 1.100; conditional AUC gap -0.003.
