#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="survival_curve.csv files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for path in args.inputs:
        df = pd.read_csv(path)
        label = Path(path).parent.name
        if "method" in df.columns and str(df["method"].iloc[0]):
            label = str(df["method"].iloc[0])
        plt.plot(df["step_t"], df["survival_fraction"], marker="o", label=label)
    plt.xlabel("Benign relearning steps")
    plt.ylabel("Survival fraction S(t)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
