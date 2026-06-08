#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_input(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("budget label cannot be empty")
        return label, Path(path)
    path = Path(value)
    return path.parent.name, path


def fmt(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def load_inputs(values: list[str]) -> pd.DataFrame:
    frames = []
    for value in values:
        label, path = parse_input(value)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {
            "family",
            "horizon",
            "n",
            "r0_mean",
            "conditional_resurrection_mean",
            "conditional_auc_mean",
            "retain_nll_mean",
            "refusal_rate_mean",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        df = df.copy()
        df.insert(0, "budget_label", label)
        frames.append(df)
    if not frames:
        raise ValueError("no input reports provided")
    return pd.concat(frames, ignore_index=True)


def write_summary(
    df: pd.DataFrame,
    output: Path,
    focus_horizon: int,
    reference_family: str,
    baseline_families: list[str],
) -> None:
    focus = df[df["horizon"] == focus_horizon].copy()
    lines = [
        "# Update-Budget Stress Summary",
        "",
        f"Focus horizon: `t={focus_horizon}`.",
        "",
        "| Budget | Family | n | r0 | conditional resurrection | conditional AUC | retain NLL | refusal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in focus.sort_values(["budget_label", "family"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["budget_label"]),
                    str(row["family"]),
                    str(int(row["n"])),
                    fmt(row["r0_mean"]),
                    fmt(row["conditional_resurrection_mean"]),
                    fmt(row["conditional_auc_mean"]),
                    fmt(row["retain_nll_mean"]),
                    fmt(row["refusal_rate_mean"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Pairwise", ""])
    for budget in sorted(df["budget_label"].unique()):
        budget_rows = focus[focus["budget_label"] == budget]
        ref = budget_rows[budget_rows["family"] == reference_family]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        for baseline in baseline_families:
            base = budget_rows[budget_rows["family"] == baseline]
            if base.empty:
                continue
            base_row = base.iloc[0]
            ref_cond = float(ref_row["conditional_resurrection_mean"])
            base_cond = float(base_row["conditional_resurrection_mean"])
            ref_auc = float(ref_row["conditional_auc_mean"])
            base_auc = float(base_row["conditional_auc_mean"])
            ratio = ref_cond / base_cond if base_cond > 0 else float("nan")
            lines.append(
                f"- {budget}, {reference_family} vs {baseline}: conditional resurrection "
                f"{fmt(ref_cond)} vs {fmt(base_cond)}; ratio {fmt(ratio)}; "
                f"conditional AUC gap {fmt(ref_auc - base_auc)}."
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="LABEL=conditional_horizon_aggregate.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--focus_horizon", type=int, default=512)
    parser.add_argument("--reference_family", default="HLC-SG")
    parser.add_argument("--baseline_families", default="Strong NPO,Syntactic diversification")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs)
    aggregate_path = output_dir / "budget_horizon_aggregate.csv"
    summary_path = output_dir / "budget_stress_summary.md"
    df.to_csv(aggregate_path, index=False)
    baseline_families = [part.strip() for part in args.baseline_families.split(",") if part.strip()]
    write_summary(
        df,
        summary_path,
        focus_horizon=args.focus_horizon,
        reference_family=args.reference_family,
        baseline_families=baseline_families,
    )
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
