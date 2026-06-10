from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_update_budget_stress_report_combines_budget_aggregates(tmp_path: Path):
    report = tmp_path / "conditional_horizon_aggregate.csv"
    pd.DataFrame(
        [
            {
                "family": "HLC-SG",
                "horizon": 512,
                "n": 3,
                "r0_mean": 0.05,
                "conditional_resurrection_mean": 0.8,
                "conditional_auc_mean": 0.3,
                "retain_nll_mean": 8.0,
                "refusal_rate_mean": 0.0,
            },
            {
                "family": "Strong NPO",
                "horizon": 512,
                "n": 3,
                "r0_mean": 0.05,
                "conditional_resurrection_mean": 0.6,
                "conditional_auc_mean": 0.4,
                "retain_nll_mean": 8.2,
                "refusal_rate_mean": 0.0,
            },
        ]
    ).to_csv(report, index=False)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_update_budget_stress.py",
            "--inputs",
            f"lr0p01={report}",
            "--output_dir",
            str(output_dir),
            "--focus_horizon",
            "512",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "budget_horizon_aggregate.csv" in result.stdout
    aggregate = pd.read_csv(output_dir / "budget_horizon_aggregate.csv")
    assert set(aggregate["budget_label"]) == {"lr0p01"}
    summary = (output_dir / "budget_stress_summary.md").read_text(encoding="utf-8")
    assert "HLC-SG vs Strong NPO" in summary
    assert "conditional AUC gap -0.100" in summary
