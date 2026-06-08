#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_steps(text: str) -> str:
    steps = [int(part.strip()) for part in text.split(",") if part.strip()]
    if steps != sorted(steps):
        raise argparse.ArgumentTypeError("steps must be sorted ascending")
    return ",".join(str(step) for step in steps)


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
    parser.add_argument("--selected", required=True, help="selected immediate-matched Strong NPO CSV")
    parser.add_argument("--pools", nargs="+", required=True)
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("0,1,2,4,8,16,32,64,128,256,512"))
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hlc_checkpoint_template", default="checkpoints/unlearned/hlc_sg_k8_tuned_fullft_tofu50_seed{seed}")
    parser.add_argument("--strong_npo_checkpoint_root", default="checkpoints/unlearned/strong_npo_sweep")
    parser.add_argument("--thresholds_template", default="thresholds/tofu50_fullft_seed{seed}_thresholds.json")
    parser.add_argument("--forget_file", default="data/tofu/forget_50.jsonl")
    parser.add_argument("--retain_file", default="data/tofu/retain_50.jsonl")
    parser.add_argument("--prompt_variants", default="data/prompts/forget_prompt_variants.jsonl")
    parser.add_argument("--output_root", default="runs/survival/stress_lr5e3_matched")
    parser.add_argument("--resurrection_metric", choices=["margin", "margin_and_exact"], default="margin")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = pd.read_csv(args.selected)
    required = {"seed", "method", "matched"}
    missing = required - set(selected.columns)
    if missing:
        raise SystemExit(f"selected CSV missing columns: {sorted(missing)}")
    selected = selected[selected["matched"] == True]  # noqa: E712
    if selected.empty:
        raise SystemExit("selected CSV has no matched candidates")

    for _, row in selected.iterrows():
        seed = int(row["seed"])
        strong_method = str(row["method"])
        for pool in args.pools:
            label = pool_label(pool)
            hlc_output = Path(args.output_root) / f"hlc_sg_k8_tuned_fullft_seed{seed}_{label}"
            strong_output = Path(args.output_root) / f"npo_strong_matched_{strong_method}_{label}"
            common = [
                "--thresholds",
                args.thresholds_template.format(seed=seed),
                "--forget_file",
                args.forget_file,
                "--retain_file",
                args.retain_file,
                "--prompt_variants",
                args.prompt_variants,
                "--relearn_pool",
                pool,
                "--steps",
                args.steps,
                "--lr",
                str(args.lr),
                "--batch_size",
                str(args.batch_size),
                "--resurrection_metric",
                args.resurrection_metric,
                "--seed",
                str(seed),
            ]
            if args.overwrite or not (hlc_output / "survival_curve.csv").exists():
                run(
                    [
                        args.python,
                        "scripts/run_survival_curve.py",
                        "--checkpoint",
                        args.hlc_checkpoint_template.format(seed=seed),
                        *common,
                        "--output",
                        str(hlc_output),
                    ],
                    dry_run=args.dry_run,
                )
            if args.overwrite or not (strong_output / "survival_curve.csv").exists():
                run(
                    [
                        args.python,
                        "scripts/run_survival_curve.py",
                        "--checkpoint",
                        str(Path(args.strong_npo_checkpoint_root) / strong_method),
                        "--method",
                        f"npo_strong_matched_{strong_method}",
                        *common,
                        "--output",
                        str(strong_output),
                    ],
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
