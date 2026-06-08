import subprocess
from pathlib import Path

import pandas as pd


def test_selected_baseline_stress_eval_dry_run_uses_selected_checkpoint(tmp_path):
    selected = tmp_path / "selected.csv"
    pd.DataFrame(
        [
            {
                "seed": 0,
                "method": "syntactic_diversification_sweep_seed0_st600_fw2_b0p3",
                "matched": True,
                "path": "runs/survival/syntactic_diversification_sweep_t0/seed0_st600_fw2_b0p3/survival_curve.csv",
            }
        ]
    ).to_csv(selected, index=False)
    result = subprocess.run(
        [
            "python3",
            "scripts/run_selected_baseline_stress_eval.py",
            "--selected",
            str(selected),
            "--checkpoint_root",
            "checkpoints/unlearned/syntactic_diversification_sweep",
            "--pools",
            "data/relearn_pools/forget_declarative.jsonl",
            "--output_root",
            str(tmp_path / "out"),
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "checkpoints/unlearned/syntactic_diversification_sweep/seed0_st600_fw2_b0p3" in result.stdout
    assert "--method syntactic_diversification_sweep_seed0_st600_fw2_b0p3" in result.stdout
    assert str(tmp_path / "out" / "seed0_st600_fw2_b0p3_forget_declarative") in result.stdout


def test_selected_baseline_stress_eval_can_prefix_method_for_family_mapping(tmp_path):
    selected = tmp_path / "selected.csv"
    pd.DataFrame(
        [
            {
                "seed": 0,
                "method": "seed0_st600_fw1_b0p3",
                "matched": True,
                "path": "runs/survival/strong_npo_sweep_t0/seed0_st600_fw1_b0p3/survival_curve.csv",
            }
        ]
    ).to_csv(selected, index=False)
    result = subprocess.run(
        [
            "python3",
            "scripts/run_selected_baseline_stress_eval.py",
            "--selected",
            str(selected),
            "--checkpoint_root",
            "checkpoints/unlearned/strong_npo_sweep",
            "--pools",
            "data/relearn_pools/heldout_forget_declarative.jsonl",
            "--output_root",
            str(tmp_path / "out"),
            "--method_prefix",
            "npo_strong_matched",
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--method npo_strong_matched_seed0_st600_fw1_b0p3" in result.stdout
