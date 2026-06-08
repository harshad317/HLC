#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return sorted(dict.fromkeys(paths))


def load_t0_rows(paths: list[Path], role: str) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        t0 = df[df["step_t"] == 0]
        if t0.empty:
            raise ValueError(f"No step_t=0 row in {path}")
        row = t0.iloc[0]
        rows.append(
            {
                "role": role,
                "method": str(row.get("method", path.parent.name)),
                "seed": int(row.get("seed", 0)),
                "relearn_corpus": str(row.get("relearn_corpus", "")),
                "path": str(path),
                "r0": float(row["resurrected_fraction"]),
                "survival0": float(row["survival_fraction"]),
                "retain_nll0": float(row["retain_nll"]),
                "refusal0": float(row["refusal_rate"]),
                "conditional_num_items0": (
                    int(row["conditional_num_items"]) if "conditional_num_items" in df else None
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_by_seed(reference: pd.DataFrame) -> pd.DataFrame:
    return (
        reference.groupby("seed")
        .agg(
            reference_r0=("r0", "mean"),
            reference_survival0=("survival0", "mean"),
            reference_retain_nll0=("retain_nll0", "mean"),
            reference_refusal0=("refusal0", "mean"),
            reference_paths=("path", lambda values: ";".join(values)),
        )
        .reset_index()
    )


def select_matches(
    reference_paths: list[Path],
    candidate_paths: list[Path],
    r0_tolerance: float,
    retain_nll_band: float,
    max_refusal: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = load_t0_rows(reference_paths, role="reference")
    candidates = load_t0_rows(candidate_paths, role="candidate")
    refs = reference_by_seed(reference)
    merged = candidates.merge(refs, on="seed", how="left")
    if merged["reference_r0"].isna().any():
        missing = sorted(merged.loc[merged["reference_r0"].isna(), "seed"].unique().tolist())
        raise ValueError(f"No reference rows for candidate seeds: {missing}")
    merged["r0_gap"] = merged["r0"] - merged["reference_r0"]
    merged["r0_abs_gap"] = merged["r0_gap"].abs()
    merged["retain_nll_gap"] = merged["retain_nll0"] - merged["reference_retain_nll0"]
    merged["retain_nll_within_band"] = merged["retain_nll0"] <= merged["reference_retain_nll0"] + retain_nll_band
    merged["r0_within_tolerance"] = merged["r0_abs_gap"] <= r0_tolerance
    merged["refusal_within_limit"] = merged["refusal0"] <= max_refusal
    merged["matched"] = (
        merged["r0_within_tolerance"]
        & merged["retain_nll_within_band"]
        & merged["refusal_within_limit"]
    )
    merged["selection_score"] = (
        merged["r0_abs_gap"]
        + 0.1 * merged["retain_nll_gap"].clip(lower=0.0)
        + merged["refusal0"]
    )
    merged = merged.sort_values(
        ["seed", "matched", "selection_score", "r0_abs_gap", "retain_nll0"],
        ascending=[True, False, True, True, True],
    )
    selected = merged.groupby("seed", as_index=False).head(1).reset_index(drop=True)
    return merged.reset_index(drop=True), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs="+", required=True, help="HLC survival_curve.csv paths or glob patterns")
    parser.add_argument("--candidates", nargs="+", required=True, help="Candidate Strong NPO t0 survival CSVs or globs")
    parser.add_argument("--r0_tolerance", type=float, default=0.05)
    parser.add_argument("--retain_nll_band", type=float, default=0.25)
    parser.add_argument("--max_refusal", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected_output", required=True)
    args = parser.parse_args()

    all_rows, selected = select_matches(
        reference_paths=expand_paths(args.reference),
        candidate_paths=expand_paths(args.candidates),
        r0_tolerance=args.r0_tolerance,
        retain_nll_band=args.retain_nll_band,
        max_refusal=args.max_refusal,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(args.output, index=False)
    selected.to_csv(args.selected_output, index=False)
    print(f"wrote {args.output}")
    print(f"wrote {args.selected_output}")


if __name__ == "__main__":
    main()
