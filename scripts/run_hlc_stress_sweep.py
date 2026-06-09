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


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def label_steps(value: str | None) -> str:
    if not value:
        return ""
    return "pen" + value.replace(",", "-").replace(" ", "").replace(".", "p")


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(cmd: list[str], dry_run: bool) -> None:
    print(command_text(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def pool_label(path: str) -> str:
    return Path(path).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--K", default=None)
    parser.add_argument("--inner_lr", default=None)
    parser.add_argument("--lambda0", default=None)
    parser.add_argument("--lambdaK", default="3,10")
    parser.add_argument("--lambdaR", default="1,3")
    parser.add_argument("--lambda_margin_growth", default=None)
    parser.add_argument("--margin_growth_target", default=None)
    parser.add_argument("--outer_lr", default=None)
    parser.add_argument("--post_gradient_mode", default=None)
    parser.add_argument("--resurrection_penalty_steps", default=None)
    parser.add_argument("--template_config", default="configs/method/hlc_sg_k8_fullft_smoke.yaml")
    parser.add_argument("--config_dir", default="configs/method/sweeps/hlc_stress")
    parser.add_argument("--model_config", default="configs/model/tiny_gpt2_full.yaml")
    parser.add_argument("--data_config", default="configs/data/tofu_50_smoke.yaml")
    parser.add_argument("--prompt_variants", default="data/prompts/forget_prompt_variants.jsonl")
    parser.add_argument("--train_relearn_pool", default="data/relearn_pools/forget_mixed_stress.jsonl")
    parser.add_argument(
        "--eval_pools",
        nargs="+",
        default=[
            "data/relearn_pools/forget_declarative.jsonl",
            "data/relearn_pools/forget_paraphrase_qa.jsonl",
            "data/relearn_pools/forget_mixed_stress.jsonl",
        ],
    )
    parser.add_argument("--steps", default="0,1,2,4,8,16,32,64,128")
    parser.add_argument("--eval_lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument("--start_checkpoint_template", default="checkpoints/tiny_full_fullft_seed{seed}")
    parser.add_argument("--full_checkpoint_template", default="checkpoints/tiny_full_fullft_seed{seed}")
    parser.add_argument("--thresholds_template", default="thresholds/tofu50_fullft_seed{seed}_thresholds.json")
    parser.add_argument("--checkpoint_root", default="checkpoints/unlearned/hlc_stress_sweep")
    parser.add_argument("--eval_root", default="runs/survival/hlc_stress_sweep")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    template = load_yaml(args.template_config)
    seeds = parse_ints(args.seeds)
    lambda_ks = parse_floats(args.lambdaK)
    lambda_rs = parse_floats(args.lambdaR)
    k_values = parse_ints(args.K) if args.K else [int(template.get("K", 8))]
    inner_lrs = parse_floats(args.inner_lr) if args.inner_lr else [float(template.get("inner_lr", 2e-5))]
    outer_lrs = parse_floats(args.outer_lr) if args.outer_lr else [float(template.get("outer_lr", 1e-5))]
    lambda_0s = parse_floats(args.lambda0) if args.lambda0 else [float(template.get("lambda0", 1.0))]
    lambda_margin_growths = (
        parse_floats(args.lambda_margin_growth)
        if args.lambda_margin_growth
        else [float(template.get("lambda_margin_growth", 0.0))]
    )
    margin_growth_targets = (
        parse_floats(args.margin_growth_target)
        if args.margin_growth_target
        else [float(template.get("margin_growth_target", 0.0))]
    )

    for k_inner in k_values:
        for inner_lr in inner_lrs:
            for outer_lr in outer_lrs:
                for lambda_0 in lambda_0s:
                    for lambda_k in lambda_ks:
                        for lambda_r in lambda_rs:
                            for lambda_margin_growth in lambda_margin_growths:
                                for margin_growth_target in margin_growth_targets:
                                    cfg = dict(template)
                                    cfg["K"] = int(k_inner)
                                    cfg["inner_lr"] = float(inner_lr)
                                    cfg["outer_lr"] = float(outer_lr)
                                    cfg["lambda0"] = float(lambda_0)
                                    cfg["lambdaK"] = float(lambda_k)
                                    cfg["lambdaR"] = float(lambda_r)
                                    if args.lambda_margin_growth or lambda_margin_growth:
                                        cfg["lambda_margin_growth"] = float(lambda_margin_growth)
                                    if args.margin_growth_target or margin_growth_target:
                                        cfg["margin_growth_target"] = float(margin_growth_target)
                                    if args.post_gradient_mode:
                                        cfg["post_gradient_mode"] = args.post_gradient_mode
                                    if args.resurrection_penalty_steps:
                                        cfg["resurrection_penalty_steps"] = args.resurrection_penalty_steps
                                    label_parts = [
                                        f"k{k_inner}",
                                        f"ilr{label_float(inner_lr)}",
                                        f"olr{label_float(outer_lr)}",
                                        f"l0{label_float(lambda_0)}",
                                        f"lk{label_float(lambda_k)}",
                                        f"lr{label_float(lambda_r)}",
                                    ]
                                    if args.lambda_margin_growth or lambda_margin_growth:
                                        label_parts.append(f"lmg{label_float(lambda_margin_growth)}")
                                    if args.margin_growth_target or margin_growth_target:
                                        label_parts.append(f"mgt{label_float(margin_growth_target)}")
                                    if args.post_gradient_mode:
                                        label_parts.append(f"pg{args.post_gradient_mode}")
                                    step_label = label_steps(args.resurrection_penalty_steps)
                                    if step_label:
                                        label_parts.append(step_label)
                                    label = "_".join(label_parts)
                                    config_path = Path(args.config_dir) / f"{label}.yaml"
                                    dump_yaml(cfg, config_path)
                                    for seed in seeds:
                                        checkpoint = Path(args.checkpoint_root) / f"hlc_sg_stress_sweep_{label}_seed{seed}"
                                        if args.overwrite or not (checkpoint / "run_metadata.json").exists():
                                            run(
                                                [
                                                    args.python,
                                                    "scripts/unlearn.py",
                                                    "--method",
                                                    "hlc_sg",
                                                    "--method_config",
                                                    str(config_path),
                                                    "--model_config",
                                                    args.model_config,
                                                    "--data_config",
                                                    args.data_config,
                                                    "--start_checkpoint",
                                                    args.start_checkpoint_template.format(seed=seed),
                                                    "--full_checkpoint",
                                                    args.full_checkpoint_template.format(seed=seed),
                                                    "--thresholds",
                                                    args.thresholds_template.format(seed=seed),
                                                    "--prompt_variants",
                                                    args.prompt_variants,
                                                    "--relearn_pool",
                                                    args.train_relearn_pool,
                                                    "--seed",
                                                    str(seed),
                                                    "--output",
                                                    str(checkpoint),
                                                ],
                                                dry_run=args.dry_run,
                                            )
                                        for pool in args.eval_pools:
                                            output = Path(args.eval_root) / f"hlc_sg_stress_sweep_{label}_seed{seed}_{pool_label(pool)}"
                                            if args.overwrite or not (output / "survival_curve.csv").exists():
                                                cmd = [
                                                    args.python,
                                                    "scripts/run_survival_curve.py",
                                                    "--checkpoint",
                                                    str(checkpoint),
                                                    "--method",
                                                    f"hlc_sg_stress_sweep_{label}_seed{seed}",
                                                    "--thresholds",
                                                    args.thresholds_template.format(seed=seed),
                                                    "--base_model",
                                                    args.base_model,
                                                    "--dtype",
                                                    args.dtype,
                                                    "--forget_file",
                                                    "data/tofu/forget_50.jsonl",
                                                    "--retain_file",
                                                    "data/tofu/retain_50.jsonl",
                                                    "--prompt_variants",
                                                    args.prompt_variants,
                                                    "--relearn_pool",
                                                    pool,
                                                    "--steps",
                                                    args.steps,
                                                    "--lr",
                                                    str(args.eval_lr),
                                                    "--batch_size",
                                                    str(args.batch_size),
                                                    "--max_length",
                                                    str(args.eval_max_length),
                                                    "--resurrection_metric",
                                                    "margin",
                                                    "--seed",
                                                    str(seed),
                                                    "--output",
                                                    str(output),
                                                ]
                                                if args.device:
                                                    cmd.extend(["--device", args.device])
                                                run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
