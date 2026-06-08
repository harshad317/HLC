# Local Main-Track Table

Source comparison: `runs/survival/compare_main_track_with_plain_npo_seed0_seed1_seed2_all_pools.csv`

Scope: local tiny-GPT-2 full-finetune smoke runs, q95 margin resurrection metric, horizon 32.
The plain NPO row is partial because only the seed-0 synthetic-biography full-finetune run is present.

| Method | Immediate forget | Retain utility | Refusal | h50 | AUC-survival | Held-out h50 |
| --- | --- | --- | --- | --- | --- | --- |
| NPO | pass; exact=0.000, r0=0.833 | retain NLL=3.674 | 0.000 | 0 observed (1/1) | 0.167 (n=1) | not run |
| Strong NPO | pass; exact=0.000, r0=0.222 | retain NLL=3.472 | 0.000 | >32 censored (9/9) | 0.778 +/- 0.083 (n=9) | >32 censored (3/3) |
| NPO + paraphrases | pass; exact=0.000, r0=0.167 | retain NLL=3.472 | 0.000 | >32 censored (9/9) | 0.833 +/- 0.000 (n=9) | >32 censored (3/3) |
| Syntactic diversification | pass; exact=0.000, r0=0.444 | retain NLL=3.472 | 0.000 | mixed: 6/9 >32; observed median 0 | 0.556 +/- 0.417 (n=9) | mixed: 2/3 >32; observed median 0 |
| HLC-SG | pass; exact=0.000, r0=0.056 | retain NLL=3.612 | 0.000 | >32 censored (9/9) | 0.944 +/- 0.083 (n=9) | >32 censored (3/3) |

Notes:
- `r0` is the immediate margin-resurrection fraction at step 0; lower is better.
- `h50` is censored when survival never drops below 0.5 within the evaluated horizon.
- Held-out uses the `syntax_matched` relearn pool by default.
