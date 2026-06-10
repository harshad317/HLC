# HLC Paper Draft

This directory contains the working paper draft for the HLC durability project.

Current framing:

> Durable unlearning claims need immediate-matched survival evaluation. HLC-SG is presented as a proposed half-life-control objective and as a diagnostic case study; final positive method claims are intentionally withheld until seeded Qwen long-horizon results and baseline comparisons are locked.

## Files

- `main.tex`: compile target.
- `sections/`: modular paper sections.
- `tables/`: static draft tables copied from current repo artifacts.

## Build

From the repository root:

```bash
python3 /Users/hpu4454/.codex/plugins/cache/openai-bundled/latex/0.2.2/scripts/compile_latex.py /Users/hpu4454/Desktop/New\ ideas/HLC/paper/main.tex
```

The draft currently avoids bibliography dependencies so it can compile early. Add related-work citations after the result claim is finalized.

## Claim Status

Do not claim SOTA in this draft. The stable contribution is the evaluation protocol:

- immediate-matched baseline selection,
- conditional resurrection among items forgotten at `t=0`,
- answer-bearing and held-out relearn stress,
- long-horizon survival AUC.

The final results section must be updated after the seeded Qwen3-0.6B long-512 comparison completes.
