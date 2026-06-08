import subprocess
from pathlib import Path

import yaml


def test_syntactic_sweep_dry_run_writes_config_and_uses_prompt_splits(tmp_path):
    config_dir = tmp_path / "configs"
    checkpoint_root = tmp_path / "checkpoints"
    eval_root = tmp_path / "eval"
    result = subprocess.run(
        [
            "python3",
            "scripts/run_syntactic_immediate_sweep.py",
            "--seeds",
            "0",
            "--step_factors",
            "2",
            "--forget_weights",
            "4",
            "--betas",
            "1.5",
            "--config_dir",
            str(config_dir),
            "--checkpoint_root",
            str(checkpoint_root),
            "--eval_root",
            str(eval_root),
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    config_path = config_dir / "st240_fw4_b1p5.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["method_name"] == "syntactic_diversification_sweep"
    assert cfg["steps"] == 240
    assert cfg["forget_weight"] == 4.0
    assert cfg["beta"] == 1.5
    assert cfg["use_prompt_variants"] is True
    assert "data/prompts/forget_syntactic_variants.jsonl" in result.stdout
    assert "data/prompts/forget_prompt_variants.jsonl" in result.stdout
    assert "syntactic_diversification_sweep_seed0_st240_fw4_b1p5" in result.stdout

