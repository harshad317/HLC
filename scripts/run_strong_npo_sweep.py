#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.utils.yaml import dump_yaml, load_yaml


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def safe_float_label(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(cmd: list[str], dry_run: bool) -> None:
    print(command_text(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def build_config(template: dict[str, object], steps: int, forget_weight: float, beta: float) -> dict[str, object]:
    cfg = dict(template)
    cfg["method_name"] = "npo_strong_sweep"
    cfg["steps"] = int(steps)
    cfg["forget_weight"] = float(forget_weight)
    cfg["beta"] = float(beta)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--step_factors", default="1,2,3,5")
    parser.add_argument("--forget_weights", default="1,2,4")
    parser.add_argument("--betas", default="0.3,1.0,3.0")
    parser.add_argument("--template_config", default="configs/method/npo_fullft_smoke.yaml")
    parser.add_argument("--base_steps", type=int, default=None)
    parser.add_argument("--model_config", default="configs/model/tiny_gpt2_full.yaml")
    parser.add_argument("--data_config", default="configs/data/tofu_50_smoke.yaml")
    parser.add_argument("--prompt_variants", default="data/prompts/forget_prompt_variants.jsonl")
    parser.add_argument("--forget_file", default="data/tofu/forget_50.jsonl")
    parser.add_argument("--retain_file", default="data/tofu/retain_50.jsonl")
    parser.add_argument("--relearn_pool", default="data/relearn_pools/synthetic_biography.jsonl")
    parser.add_argument("--start_checkpoint_template", default="checkpoints/tiny_full_fullft_seed{seed}")
    parser.add_argument("--thresholds_template", default="thresholds/tofu50_fullft_seed{seed}_thresholds.json")
    parser.add_argument("--base_model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument("--config_dir", default="configs/method/sweeps/strong_npo_immediate_matched")
    parser.add_argument("--checkpoint_root", default="checkpoints/unlearned/strong_npo_sweep")
    parser.add_argument("--eval_root", default="runs/survival/strong_npo_sweep_t0")
    parser.add_argument("--resurrection_metric", default="margin", choices=["margin", "margin_and_exact"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    template = load_yaml(args.template_config)
    base_steps = int(args.base_steps or template.get("steps", 120))
    seeds = parse_ints(args.seeds)
    step_factors = parse_ints(args.step_factors)
    forget_weights = parse_floats(args.forget_weights)
    betas = parse_floats(args.betas)

    for factor in step_factors:
        for forget_weight in forget_weights:
            for beta in betas:
                steps = base_steps * factor
                label = f"st{steps}_fw{safe_float_label(forget_weight)}_b{safe_float_label(beta)}"
                config_path = Path(args.config_dir) / f"{label}.yaml"
                cfg = build_config(template, steps=steps, forget_weight=forget_weight, beta=beta)
                dump_yaml(cfg, config_path)
                for seed in seeds:
                    checkpoint = Path(args.checkpoint_root) / f"seed{seed}_{label}"
                    eval_dir = Path(args.eval_root) / f"seed{seed}_{label}"
                    if args.overwrite or not (checkpoint / "run_metadata.json").exists():
                        run(
                            [
                                args.python,
                                "scripts/unlearn.py",
                                "--method",
                                "npo",
                                "--method_config",
                                str(config_path),
                                "--model_config",
                                args.model_config,
                                "--data_config",
                                args.data_config,
                                "--start_checkpoint",
                                args.start_checkpoint_template.format(seed=seed),
                                "--thresholds",
                                args.thresholds_template.format(seed=seed),
                                "--prompt_variants",
                                args.prompt_variants,
                                "--seed",
                                str(seed),
                                "--output",
                                str(checkpoint),
                            ],
                            dry_run=args.dry_run,
                        )
                    if args.overwrite or not (eval_dir / "survival_curve.csv").exists():
                        run(
                            [
                                args.python,
                                "scripts/run_survival_curve.py",
                                "--checkpoint",
                                str(checkpoint),
                                "--thresholds",
                                args.thresholds_template.format(seed=seed),
                                "--base_model",
                                args.base_model,
                                "--dtype",
                                args.dtype,
                                "--forget_file",
                                args.forget_file,
                                "--retain_file",
                                args.retain_file,
                                "--prompt_variants",
                                args.prompt_variants,
                                "--relearn_pool",
                                args.relearn_pool,
                                "--steps",
                                "0",
                                "--batch_size",
                                str(args.eval_batch_size),
                                "--max_length",
                                str(args.eval_max_length),
                                "--resurrection_metric",
                                args.resurrection_metric,
                                "--seed",
                                str(seed),
                                "--output",
                                str(eval_dir),
                            ],
                            dry_run=args.dry_run,
                        )


if __name__ == "__main__":
    main()
