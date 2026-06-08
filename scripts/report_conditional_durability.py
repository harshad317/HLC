#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_report_module():
    path = REPO_ROOT / "scripts" / "report_local_mvp.py"
    spec = importlib.util.spec_from_file_location("report_local_mvp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_horizons(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return sorted(dict.fromkeys(paths))


def load_horizon_rows(paths: list[Path], horizons: list[int]) -> pd.DataFrame:
    report = load_report_module()
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        first = df.iloc[0]
        method = str(first.get("method", path.parent.name))
        family = report.family_from_method(method)
        for horizon in horizons:
            matched = df[df["step_t"] == horizon]
            if matched.empty:
                continue
            row = matched.iloc[0]
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "seed": int(row.get("seed", 0)),
                    "relearn_corpus": str(row.get("relearn_corpus", "")),
                    "horizon": int(horizon),
                    "path": str(path),
                    "r0": float(first["resurrected_fraction"]),
                    "r_t": float(row["resurrected_fraction"]),
                    "survival0": float(first["survival_fraction"]),
                    "survival_t": float(row["survival_fraction"]),
                    "relearn_growth": (
                        float(row["relearn_growth"])
                        if "relearn_growth" in df
                        else float(row["resurrected_fraction"]) - float(first["resurrected_fraction"])
                    ),
                    "survival_decay": (
                        float(row["survival_decay"])
                        if "survival_decay" in df
                        else float(first["survival_fraction"]) - float(row["survival_fraction"])
                    ),
                    "conditional_resurrection": (
                        float(row["conditional_resurrection_fraction"])
                        if "conditional_resurrection_fraction" in df
                        else float("nan")
                    ),
                    "conditional_survival": (
                        float(row["conditional_survival_fraction"])
                        if "conditional_survival_fraction" in df
                        else float("nan")
                    ),
                    "conditional_auc": (
                        float(row["conditional_auc_survival_so_far"])
                        if "conditional_auc_survival_so_far" in df
                        else float("nan")
                    ),
                    "conditional_num_items": (
                        int(row["conditional_num_items"])
                        if "conditional_num_items" in df
                        else 0
                    ),
                    "mean_target_margin_growth": (
                        float(row["mean_target_margin_growth"])
                        if "mean_target_margin_growth" in df
                        else float(row["mean_target_margin"]) - float(first["mean_target_margin"])
                    ),
                    "conditional_mean_target_margin_growth": (
                        float(row["conditional_mean_target_margin_growth"])
                        if "conditional_mean_target_margin_growth" in df
                        else float("nan")
                    ),
                    "retain_nll": float(row["retain_nll"]),
                    "refusal_rate": float(row["refusal_rate"]),
                }
            )
    return pd.DataFrame(rows)


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["family", "horizon"])
        .agg(
            n=("conditional_resurrection", "size"),
            r0_mean=("r0", "mean"),
            r_t_mean=("r_t", "mean"),
            relearn_growth_mean=("relearn_growth", "mean"),
            survival_decay_mean=("survival_decay", "mean"),
            conditional_resurrection_mean=("conditional_resurrection", "mean"),
            conditional_resurrection_std=("conditional_resurrection", "std"),
            conditional_auc_mean=("conditional_auc", "mean"),
            conditional_auc_std=("conditional_auc", "std"),
            mean_target_margin_growth_mean=("mean_target_margin_growth", "mean"),
            conditional_mean_target_margin_growth_mean=("conditional_mean_target_margin_growth", "mean"),
            retain_nll_mean=("retain_nll", "mean"),
            refusal_rate_mean=("refusal_rate", "mean"),
        )
        .reset_index()
    )


def fmt(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def write_markdown(agg: pd.DataFrame, output: Path, reference_family: str, baseline_family: str) -> None:
    lines = [
        "# Conditional Durability Summary",
        "",
        "| Family | Horizon | n | r0 | r@T | growth | conditional resurrection | conditional AUC | margin growth | retain NLL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in agg.sort_values(["horizon", "family"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    str(int(row["horizon"])),
                    str(int(row["n"])),
                    fmt(row["r0_mean"]),
                    fmt(row["r_t_mean"]),
                    fmt(row["relearn_growth_mean"]),
                    fmt(row["conditional_resurrection_mean"]),
                    fmt(row["conditional_auc_mean"]),
                    fmt(row["mean_target_margin_growth_mean"]),
                    fmt(row["retain_nll_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pairwise", ""])
    for horizon in sorted(agg["horizon"].unique().tolist()):
        ref = agg[(agg["family"] == reference_family) & (agg["horizon"] == horizon)]
        base = agg[(agg["family"] == baseline_family) & (agg["horizon"] == horizon)]
        if ref.empty or base.empty:
            continue
        ref_row = ref.iloc[0]
        base_row = base.iloc[0]
        base_cond = float(base_row["conditional_resurrection_mean"])
        ref_cond = float(ref_row["conditional_resurrection_mean"])
        ratio = ref_cond / base_cond if base_cond > 0 else float("nan")
        auc_gap = float(ref_row["conditional_auc_mean"] - base_row["conditional_auc_mean"])
        lines.append(
            f"- t={int(horizon)}: {reference_family} conditional resurrection {fmt(ref_cond)} vs "
            f"{baseline_family} {fmt(base_cond)}; ratio {fmt(ratio)}; conditional AUC gap {fmt(auc_gap)}."
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--horizons", default="32,128,512")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_family", default="HLC-SG")
    parser.add_argument("--baseline_family", default="Strong NPO")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_horizon_rows(expand_paths(args.inputs), parse_horizons(args.horizons))
    agg = aggregate(rows)
    rows.to_csv(output_dir / "conditional_horizon_rows.csv", index=False)
    agg.to_csv(output_dir / "conditional_horizon_aggregate.csv", index=False)
    write_markdown(
        agg,
        output_dir / "conditional_durability_summary.md",
        reference_family=args.reference_family,
        baseline_family=args.baseline_family,
    )
    print(f"wrote {output_dir / 'conditional_horizon_rows.csv'}")
    print(f"wrote {output_dir / 'conditional_horizon_aggregate.csv'}")
    print(f"wrote {output_dir / 'conditional_durability_summary.md'}")


if __name__ == "__main__":
    main()
