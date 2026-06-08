#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


METRICS = [
    ("auc_survival", "AUC survival", "higher"),
    ("r_final", "Final resurrected fraction", "lower"),
    ("mean_target_margin_final", "Final mean target margin", "lower"),
    ("retain_nll_final", "Final retain NLL", "lower"),
]


def family_from_method(method: str) -> str:
    if method.startswith("hlc_sg_stress_sweep_"):
        return method.split("_seed", 1)[0]
    if method.startswith("hlc_sg_k1_gpt2"):
        return "HLC-SG K1 GPT-2"
    if method.startswith("hlc_sg_k2_gpt2"):
        return "HLC-SG K2 GPT-2"
    if method.startswith("hlc_sg_k1_qwen") or method.startswith("hlc_sg_k1_pythia"):
        return "HLC-SG K1"
    if method.startswith("hlc_sg_k2_qwen") or method.startswith("hlc_sg_k2_pythia"):
        return "HLC-SG K2"
    if method.startswith("hlc_sg") and "stress" in method:
        return "HLC-SG stress-trained"
    if method.startswith("hlc_sg"):
        return "HLC-SG"
    if method.startswith("npo_paraphrase"):
        return "NPO + paraphrases"
    if method.startswith("npo_strong"):
        return "Strong NPO"
    if method.startswith("syntactic_diversification"):
        return "Syntactic diversification"
    if method.startswith("npo"):
        return "NPO"
    if method.startswith("grad_ascent"):
        return "Gradient ascent"
    return method.split("_seed", 1)[0]


def load_comparison(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "method",
        "seed",
        "relearn_corpus",
        "threshold_quantile",
        "auc_survival",
        "r_final",
        "mean_target_margin_final",
        "median_target_margin_final",
        "retain_nll_final",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Comparison CSV missing columns: {missing}")
    df = df.copy()
    df["family"] = df["method"].map(family_from_method)
    return df


def aggregate_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_kwargs = {
        "n": ("auc_survival", "size"),
        "r0_mean": ("r0", "mean") if "r0" in df.columns else ("r_final", "mean"),
        "auc_mean": ("auc_survival", "mean"),
        "auc_std": ("auc_survival", "std"),
        "r_final_mean": ("r_final", "mean"),
        "r_final_std": ("r_final", "std"),
        "mean_target_margin_mean": ("mean_target_margin_final", "mean"),
        "mean_target_margin_std": ("mean_target_margin_final", "std"),
        "median_target_margin_mean": ("median_target_margin_final", "mean"),
        "median_target_margin_std": ("median_target_margin_final", "std"),
        "retain_nll_mean": ("retain_nll_final", "mean"),
        "retain_nll_std": ("retain_nll_final", "std"),
    }
    optional_metrics = {
        "relearn_growth": "relearn_growth",
        "survival_decay": "survival_decay",
        "conditional_resurrection": "conditional_resurrection_final",
        "conditional_auc": "conditional_auc_survival",
        "mean_margin_growth": "mean_target_margin_growth_final",
        "conditional_margin_growth": "conditional_mean_target_margin_growth_final",
    }
    for prefix, column in optional_metrics.items():
        if column in df.columns:
            agg_kwargs[f"{prefix}_mean"] = (column, "mean")
            agg_kwargs[f"{prefix}_std"] = (column, "std")
    by_pool = df.groupby(["family", "relearn_corpus"]).agg(**agg_kwargs).reset_index()
    overall = df.groupby(["family"]).agg(**agg_kwargs).reset_index()
    return by_pool, overall


def format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(format_float(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *rows])


def summary_lines(overall: pd.DataFrame) -> list[str]:
    lines = []
    by_family = {row["family"]: row for _, row in overall.iterrows()}
    hlc = by_family.get("HLC-SG")
    npo = by_family.get("NPO")
    if hlc is not None and npo is not None:
        auc_gain = float(hlc["auc_mean"] - npo["auc_mean"])
        r_drop = float(npo["r_final_mean"] - hlc["r_final_mean"])
        margin_drop = float(npo["mean_target_margin_mean"] - hlc["mean_target_margin_mean"])
        utility_cost = float(hlc["retain_nll_mean"] - npo["retain_nll_mean"])
        lines.extend(
            [
                f"- HLC-SG improves mean AUC by {auc_gain:.3f} over NPO.",
                f"- HLC-SG reduces final resurrection by {r_drop:.3f} absolute.",
                f"- HLC-SG lowers final mean target margin by {margin_drop:.3f}.",
                f"- HLC-SG retain-NLL cost versus NPO is {utility_cost:.3f}.",
            ]
        )
        if "conditional_resurrection_mean" in overall.columns:
            conditional_drop = float(npo["conditional_resurrection_mean"] - hlc["conditional_resurrection_mean"])
            lines.append(f"- HLC-SG reduces conditional final resurrection by {conditional_drop:.3f} absolute.")
    return lines


def plot_metric(by_pool: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    pools = list(dict.fromkeys(by_pool["relearn_corpus"].tolist()))
    families = list(dict.fromkeys(by_pool["family"].tolist()))
    x = range(len(pools))
    width = 0.35 if len(families) <= 2 else 0.8 / max(1, len(families))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, family in enumerate(families):
        subset = by_pool[by_pool["family"] == family].set_index("relearn_corpus")
        means = [float(subset.loc[pool, f"{metric}_mean"]) if pool in subset.index else 0.0 for pool in pools]
        stds = [float(subset.loc[pool, f"{metric}_std"]) if pool in subset.index and pd.notna(subset.loc[pool, f"{metric}_std"]) else 0.0 for pool in pools]
        offsets = [pos + (idx - (len(families) - 1) / 2) * width for pos in x]
        ax.bar(offsets, means, width=width, yerr=stds, capsize=3, label=family)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pools, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(comparison_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_comparison(comparison_path)
    by_pool, overall = aggregate_comparison(df)
    by_pool_path = output_dir / "aggregate_by_pool.csv"
    overall_path = output_dir / "aggregate_overall.csv"
    by_pool.to_csv(by_pool_path, index=False)
    overall.to_csv(overall_path, index=False)

    plot_metric(by_pool, "auc", "Survival AUC by relearn pool", "AUC survival", output_dir / "auc_by_pool.png")
    plot_metric(by_pool, "r_final", "Final resurrection by relearn pool", "Final resurrected fraction", output_dir / "resurrection_by_pool.png")
    plot_metric(by_pool, "mean_target_margin", "Final target margin by relearn pool", "Mean target margin", output_dir / "target_margin_by_pool.png")
    plot_metric(by_pool, "retain_nll", "Retain utility by relearn pool", "Retain NLL", output_dir / "retain_nll_by_pool.png")

    report_path = output_dir / "summary.md"
    lines = [
        "# Local MVP Evidence Summary",
        "",
        f"Source comparison: `{comparison_path}`",
        "",
        "## Headline",
        "",
        *summary_lines(overall),
        "",
        "## Overall",
        "",
        markdown_table(
            overall,
            [col for col in [
                "family",
                "n",
                "r0_mean",
                "auc_mean",
                "r_final_mean",
                "relearn_growth_mean",
                "conditional_resurrection_mean",
                "conditional_auc_mean",
                "mean_target_margin_mean",
                "mean_margin_growth_mean",
                "retain_nll_mean",
            ] if col in overall.columns],
        ),
        "",
        "## By Relearn Pool",
        "",
        markdown_table(
            by_pool,
            [col for col in [
                "family",
                "relearn_corpus",
                "n",
                "r0_mean",
                "auc_mean",
                "r_final_mean",
                "relearn_growth_mean",
                "conditional_resurrection_mean",
                "conditional_auc_mean",
                "mean_target_margin_mean",
                "mean_margin_growth_mean",
                "retain_nll_mean",
            ] if col in by_pool.columns],
        ),
        "",
        "## Figures",
        "",
        "- `auc_by_pool.png`",
        "- `resurrection_by_pool.png`",
        "- `target_margin_by_pool.png`",
        "- `retain_nll_by_pool.png`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary": report_path,
        "aggregate_by_pool": by_pool_path,
        "aggregate_overall": overall_path,
        "auc_plot": output_dir / "auc_by_pool.png",
        "resurrection_plot": output_dir / "resurrection_by_pool.png",
        "target_margin_plot": output_dir / "target_margin_by_pool.png",
        "retain_nll_plot": output_dir / "retain_nll_by_pool.png",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    outputs = write_report(args.comparison, args.output_dir)
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
