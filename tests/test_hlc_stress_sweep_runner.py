import subprocess
from pathlib import Path

import yaml


def test_hlc_stress_sweep_dry_run_writes_overridden_config(tmp_path):
    config_dir = tmp_path / "configs"
    result = subprocess.run(
        [
            "/opt/anaconda3/bin/python3",
            "scripts/run_hlc_stress_sweep.py",
            "--seeds",
            "0",
            "--K",
            "16",
            "--inner_lr",
            "0.005",
            "--outer_lr",
            "0.001",
            "--lambda0",
            "5",
            "--lambdaK",
            "3",
            "--lambdaR",
            "1",
            "--lambda_margin_growth",
            "2",
            "--margin_growth_target",
            "0.25",
            "--post_gradient_mode",
            "unrolled",
            "--resurrection_penalty_steps",
            "1,2,4,8",
            "--config_dir",
            str(config_dir),
            "--eval_pools",
            "data/relearn_pools/forget_mixed_stress.jsonl",
            "--steps",
            "0",
            "--dry_run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config_path = config_dir / "k16_ilr0p005_olr0p001_l05_lk3_lr1_lmg2_mgt0p25_pgunrolled_pen1-2-4-8.yaml"
    assert config_path.exists()
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["K"] == 16
    assert cfg["inner_lr"] == 0.005
    assert cfg["outer_lr"] == 0.001
    assert cfg["lambda0"] == 5
    assert cfg["lambda_margin_growth"] == 2
    assert cfg["margin_growth_target"] == 0.25
    assert cfg["post_gradient_mode"] == "unrolled"
    assert cfg["resurrection_penalty_steps"] == "1,2,4,8"
    assert "hlc_sg_stress_sweep_k16_ilr0p005_olr0p001_l05_lk3_lr1_lmg2_mgt0p25_pgunrolled_pen1-2-4-8_seed0" in result.stdout
