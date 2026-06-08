#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def mean_std(series: pd.Series) -> str:
    if len(series) == 0:
        return "not run"
    mean = float(series.mean())
    if len(series) == 1 or pd.isna(series.std()):
        return f"{mean:.3f} (n={len(series)})"
    return f"{mean:.3f} +/- {float(series.std()):.3f} (n={len(series)})"


def summarize_h50(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "not run"
    horizon = None
    if "final_step_t" in rows:
        horizon = int(rows["final_step_t"].max())
    if horizon is None or pd.isna(horizon):
        horizon = 0
    censored = rows[rows["h50_censored"] == True]  # noqa: E712
    observed = rows["h50"].dropna()
    if len(censored) == len(rows):
        return f">{horizon} censored ({len(rows)}/{len(rows)})"
    if len(observed) == len(rows):
        return f"{float(observed.median()):.0f} observed ({len(rows)}/{len(rows)})"
    return f"mixed: {len(censored)}/{len(rows)} >{horizon}; observed median {float(observed.median()):.0f}"


def immediate_forget(curves: pd.DataFrame) -> str:
    if curves.empty:
        return "not run"
    t0 = curves[curves["step_t"] == 0]
    exact = float(t0["forget_accuracy"].mean())
    resurrected = float(t0["resurrected_fraction"].mean())
    label = "pass" if exact <= 0.0 else "fail"
    return f"{label}; exact={exact:.3f}, r0={resurrected:.3f}"


def retain_utility(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "not run"
    return f"retain NLL={float(rows['retain_nll_final'].mean()):.3f}"


def refusal(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "not run"
    return fmt(float(rows["refusal_rate_final"].mean()))


def load_curves(comparison: pd.DataFrame, family_from_method) -> pd.DataFrame:
    frames = []
    for _, row in comparison.iterrows():
        path = Path(row["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        df = pd.read_csv(path)
        df["family"] = family_from_method(str(row["method"]))
        df["source_path"] = str(row["path"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_table(
    comparison_path: str | Path,
    methods: list[str],
    held_out_pool: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = load_report_module()
    comparison = pd.read_csv(comparison_path)
    comparison = comparison.copy()
    comparison["family"] = comparison["method"].map(report.family_from_method)
    curves = load_curves(comparison, report.family_from_method)
    final_steps = curves.groupby("source_path")["step_t"].max().rename("final_step_t")
    comparison = comparison.join(final_steps, on="path")

    rows = []
    for method in methods:
        method_rows = comparison[comparison["family"] == method]
        method_curves = curves[curves["family"] == method]
        held_out = method_rows[method_rows["relearn_corpus"] == held_out_pool]
        rows.append(
            {
                "Method": method,
                "Immediate forget": immediate_forget(method_curves),
                "Retain utility": retain_utility(method_rows),
                "Refusal": refusal(method_rows),
                "h50": summarize_h50(method_rows),
                "AUC-survival": mean_std(method_rows["auc_survival"]),
                "Held-out h50": summarize_h50(held_out),
            }
        )
    return pd.DataFrame(rows), comparison


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_outputs(table: pd.DataFrame, comparison: pd.DataFrame, output_dir: str | Path, source: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "main_track_table.csv", index=False)

    by_method = (
        comparison.groupby(["family", "relearn_corpus"])
        .agg(
            n=("auc_survival", "size"),
            auc_mean=("auc_survival", "mean"),
            auc_std=("auc_survival", "std"),
            r_final_mean=("r_final", "mean"),
            r_final_std=("r_final", "std"),
            retain_nll_mean=("retain_nll_final", "mean"),
            refusal_rate_mean=("refusal_rate_final", "mean"),
            censored_count=("h50_censored", "sum"),
        )
        .reset_index()
    )
    by_method.to_csv(output_dir / "main_track_by_pool.csv", index=False)

    lines = [
        "# Local Main-Track Table",
        "",
        f"Source comparison: `{source}`",
        "",
        "Scope: local tiny-GPT-2 full-finetune smoke runs, q95 margin resurrection metric, horizon 32.",
        "The plain NPO row is partial because only the seed-0 synthetic-biography full-finetune run is present.",
        "",
        markdown_table(table),
        "",
        "Notes:",
        "- `r0` is the immediate margin-resurrection fraction at step 0; lower is better.",
        "- `h50` is censored when survival never drops below 0.5 within the evaluated horizon.",
        "- Held-out uses the `syntax_matched` relearn pool by default.",
        "",
    ]
    (output_dir / "main_track_table.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--held_out_pool", default="syntax_matched")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "NPO",
            "Strong NPO",
            "NPO + paraphrases",
            "Syntactic diversification",
            "HLC-SG",
        ],
    )
    args = parser.parse_args()
    table, comparison = build_table(args.comparison, args.methods, args.held_out_pool)
    write_outputs(table, comparison, args.output_dir, args.comparison)
    print(f"wrote {Path(args.output_dir) / 'main_track_table.md'}")
    print(f"wrote {Path(args.output_dir) / 'main_track_table.csv'}")
    print(f"wrote {Path(args.output_dir) / 'main_track_by_pool.csv'}")


if __name__ == "__main__":
    main()
