# Update-Budget Stress Summary

Focus horizon: `t=512`.

| Budget | Family | n | r0 | conditional resurrection | conditional AUC | retain NLL | refusal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr0p01 | HLC-SG | 9 | 0.056 | 1.000 | 0.119 | 10.503 | 0.000 |
| lr0p01 | NPO + paraphrases | 9 | 0.167 | 1.000 | 0.030 | 9.907 | 0.000 |
| lr0p01 | Strong NPO | 9 | 0.056 | 1.000 | 0.257 | 9.853 | 0.000 |
| lr0p01 | Syntactic diversification | 9 | 0.056 | 1.000 | 0.356 | 9.609 | 0.000 |

## Pairwise

- lr0p01, HLC-SG vs Strong NPO: conditional resurrection 1.000 vs 1.000; ratio 1.000; conditional AUC gap -0.138.
- lr0p01, HLC-SG vs Syntactic diversification: conditional resurrection 1.000 vs 1.000; ratio 1.000; conditional AUC gap -0.236.
