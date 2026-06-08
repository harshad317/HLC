#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def compare_files(inputs: list[str]) -> pd.DataFrame:
    rows = []
    for path in inputs:
        df = pd.read_csv(path)
        initial = df.iloc[0]
        final = df.iloc[-1]
        label = str(final.get("method") or Path(path).parent.name)
        observed = df["h50_observed"].dropna()
        relearn_growth = (
            float(final["relearn_growth"])
            if "relearn_growth" in df
            else float(final["resurrected_fraction"]) - float(initial["resurrected_fraction"])
        )
        survival_decay = (
            float(final["survival_decay"])
            if "survival_decay" in df
            else float(initial["survival_fraction"]) - float(final["survival_fraction"])
        )
        rows.append(
            {
                "method": label,
                "seed": int(final["seed"]) if "seed" in final else 0,
                "relearn_corpus": str(final.get("relearn_corpus", "")),
                "threshold_quantile": str(final.get("threshold_quantile", "")),
                "path": path,
                "r0": float(initial["resurrected_fraction"]),
                "survival0": float(initial["survival_fraction"]),
                "h50": observed.iloc[-1] if len(observed) else None,
                "h50_censored": bool(final["h50_censored"]),
                "auc_survival": float(final["auc_survival_so_far"]),
                "r_final": float(final["resurrected_fraction"]),
                "relearn_growth": relearn_growth,
                "survival_decay": survival_decay,
                "conditional_resurrection_final": (
                    float(final["conditional_resurrection_fraction"])
                    if "conditional_resurrection_fraction" in df
                    else None
                ),
                "conditional_survival_final": (
                    float(final["conditional_survival_fraction"])
                    if "conditional_survival_fraction" in df
                    else None
                ),
                "conditional_auc_survival": (
                    float(final["conditional_auc_survival_so_far"])
                    if "conditional_auc_survival_so_far" in df
                    else None
                ),
                "conditional_num_items": (
                    int(final["conditional_num_items"])
                    if "conditional_num_items" in df
                    else None
                ),
                "mean_target_margin_final": float(final["mean_target_margin"]),
                "median_target_margin_final": float(final["median_target_margin"]),
                "mean_target_margin_growth_final": (
                    float(final["mean_target_margin_growth"])
                    if "mean_target_margin_growth" in df
                    else float(final["mean_target_margin"]) - float(initial["mean_target_margin"])
                ),
                "conditional_mean_target_margin_growth_final": (
                    float(final["conditional_mean_target_margin_growth"])
                    if "conditional_mean_target_margin_growth" in df
                    else None
                ),
                "retain_accuracy_final": float(final["retain_accuracy"]),
                "retain_nll_final": float(final["retain_nll"]),
                "refusal_rate_final": float(final["refusal_rate"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["seed", "relearn_corpus", "threshold_quantile", "auc_survival"],
        ascending=[True, True, True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = compare_files(args.inputs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
