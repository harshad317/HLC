#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numbers
from pathlib import Path

import pandas as pd


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}:
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is None:
        columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def normalize_stress_table(stress: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "n",
        "r0",
        "cond_res_32",
        "cond_res_128",
        "cond_res_512",
        "cond_auc_512",
        "retain_nll_512",
        "immediate_matched",
    ]
    missing = sorted(set(columns) - set(stress.columns))
    if missing:
        raise ValueError(f"Stress ranked CSV missing columns: {missing}")
    out = stress[columns].copy()
    out = out.rename(
        columns={
            "family": "Method",
            "n": "n",
            "r0": "Immediate r0",
            "cond_res_32": "Cond. resurrection @32",
            "cond_res_128": "Cond. resurrection @128",
            "cond_res_512": "Cond. resurrection @512",
            "cond_auc_512": "Cond. AUC @512",
            "retain_nll_512": "Retain NLL @512",
            "immediate_matched": "Immediate matched",
        }
    )
    return out


def best_benign_method(main_track: pd.DataFrame) -> str:
    if "AUC-survival" not in main_track:
        return "unknown"

    def leading_number(text: object) -> float:
        try:
            return float(str(text).split()[0])
        except (TypeError, ValueError):
            return float("nan")

    scored = main_track.copy()
    scored["_auc"] = scored["AUC-survival"].map(leading_number)
    scored = scored.dropna(subset=["_auc"])
    if scored.empty:
        return "unknown"
    row = scored.sort_values("_auc", ascending=False).iloc[0]
    return str(row["Method"])


def write_report(
    main_track_path: str | Path,
    stress_ranked_path: str | Path,
    decision_report_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_track = pd.read_csv(main_track_path)
    stress = normalize_stress_table(pd.read_csv(stress_ranked_path))
    main_out = output_dir / "diagnostic_main_track_table.csv"
    stress_out = output_dir / "diagnostic_stress_table.csv"
    report_out = output_dir / "diagnostic_summary.md"
    main_track.to_csv(main_out, index=False)
    stress.to_csv(stress_out, index=False)

    benign_best = best_benign_method(main_track)
    lines = [
        "# Local MVP Diagnostic Summary",
        "",
        "## Headline",
        "",
        (
            f"The benign main-track table makes `{benign_best}` look strongest, "
            "but the answer-bearing stress harness does not support an HLC durability claim."
        ),
        "",
        "The current defensible claim is diagnostic and negative: HLC-SG variants improved through multi-time penalties, margin-growth penalties, and shallow differentiable unrolling, but they did not beat immediate-matched Strong NPO under long-horizon stress.",
        "",
        "## Benign Main Track",
        "",
        f"Source: `{main_track_path}`",
        "",
        markdown_table(main_track),
        "",
        "## Answer-Bearing Stress Verdict",
        "",
        f"Source: `{stress_ranked_path}`",
        "",
        markdown_table(stress),
        "",
        "## Decision",
        "",
        "- Freeze HLC tuning for the local MVP.",
        "- Treat the HLC result as a useful negative/diagnostic result, not a SOTA durability claim.",
        "- Present immediate-matched conditional resurrection as the decisive metric.",
        "- Spend next effort on strengthening baseline coverage and clearly explaining the stress harness.",
        "",
        "## Supporting Report",
        "",
        f"Detailed stop-condition report: `{decision_report_path}`",
        "",
    ]
    report_out.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary": report_out,
        "main_track_table": main_out,
        "stress_table": stress_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_track", required=True)
    parser.add_argument("--stress_ranked", required=True)
    parser.add_argument("--decision_report", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    outputs = write_report(
        args.main_track,
        args.stress_ranked,
        args.decision_report,
        args.output_dir,
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
