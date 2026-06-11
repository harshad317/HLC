#!/usr/bin/env python3
"""At-a-glance status of the remaining experiments (#1-#5).

Scans for each experiment's expected output artifact and reports DONE / pending,
printing a one-line headline (the method rows) for any completed aggregate. Run it
on the box anytime: ``python3 scripts/experiment_status.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# (label, relative_path, optional_substring_required_for_done)
EXPERIMENTS = [
    ("#1 Frontier sweep (forget01 seed0)",
     "runs/reports/frontier_tofu01_seed0/frontier_table.md", None),
    ("#2 Scale: forget05 CI",
     "runs/reports/qwen3_tofu05_real_comparison/aggregate_ci_summary.md", "displacement"),
    ("#2 Scale: forget10 CI",
     "runs/reports/qwen3_tofu10_real_comparison/aggregate_ci_summary.md", "displacement"),
    ("#3 Second model: Qwen3-1.7B (forget01 CI)",
     "runs/reports/qwen1p7b_tofu01_real_comparison/aggregate_ci_summary.md", "displacement"),
    ("#4 RMU in comparison (forget01 CI)",
     "runs/reports/qwen3_tofu01_real_comparison/aggregate_ci_summary.md", "rmu"),
]

METHOD_KEYS = ("displacement", "ga", "rmu", "hlc_r", "hlc_base")


def headline(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip().lower()
        for k in METHOD_KEYS:
            if s.startswith(f"| {k}") or s.startswith(f"| **{k}"):
                out.append(line.strip())
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--quiet", action="store_true", help="status lines only, no headlines")
    args = parser.parse_args()
    root = Path(args.root)

    done = 0
    for label, rel, sub in EXPERIMENTS:
        p = root / rel
        if not p.exists():
            print(f"[ pending ] {label}\n            -> {rel}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if sub and sub not in text:
            print(f"[ partial ] {label}  (artifact exists but missing '{sub}')")
            continue
        done += 1
        print(f"[  DONE   ] {label}")
        if not args.quiet:
            for line in headline(text)[:8]:
                print("            " + line)
    print(f"\n{done}/{len(EXPERIMENTS)} experiment artifacts present. "
          "(#5 robustness re-evals are ad-hoc; track those manually.)")


if __name__ == "__main__":
    main()
