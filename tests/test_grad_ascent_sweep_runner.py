import subprocess
from pathlib import Path

import yaml


def test_grad_ascent_sweep_dry_run_writes_config_and_uses_qwen_eval_args(tmp_path):
    config_dir = tmp_path / "configs"
    checkpoint_root = tmp_path / "checkpoints"
    eval_root = tmp_path / "eval"
    result = subprocess.run(
        [
            "python3",
            "scripts/run_grad_ascent_sweep.py",
            "--seeds",
            "0",
            "--step_factors",
            "2",
            "--lrs",
            "0.0003",
            "--retain_weights",
            "0.05",
            "--max_grad_norms",
            "0.5",
            "--base_model",
            "Qwen/Qwen3-0.6B",
            "--dtype",
            "float16",
            "--eval_max_length",
            "192",
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
    config_path = config_dir / "st240_lr0p0003_rw0p05_gn0p5.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["method_name"] == "grad_ascent_sweep"
    assert cfg["steps"] == 240
    assert cfg["lr"] == 0.0003
    assert cfg["retain_weight"] == 0.05
    assert cfg["max_grad_norm"] == 0.5
    assert "--method grad_ascent" in result.stdout
    assert "--base_model Qwen/Qwen3-0.6B" in result.stdout
    assert "--dtype float16" in result.stdout
    assert "--max_length 192" in result.stdout
    assert "grad_ascent_sweep_seed0_st240_lr0p0003_rw0p05_gn0p5" in result.stdout
