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


def checkpoint_name_from_selected_path(path: str) -> str:
    return Path(path).parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--selected", required=True, help="selected immediate-match CSV")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--pools", nargs="+", required=True)
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("0,1,2,4,8,16,32,64,128,256,512"))
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--thresholds_template", default="thresholds/tofu50_fullft_seed{seed}_thresholds.json")
    parser.add_argument("--base_model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--forget_file", default="data/tofu/forget_50.jsonl")
    parser.add_argument("--retain_file", default="data/tofu/retain_50.jsonl")
    parser.add_argument("--prompt_variants", default="data/prompts/forget_prompt_variants.jsonl")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--method_prefix", default="")
    parser.add_argument("--resurrection_metric", choices=["margin", "margin_and_exact"], default="margin")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = pd.read_csv(args.selected)
    required = {"seed", "method", "matched", "path"}
    missing = required - set(selected.columns)
    if missing:
        raise SystemExit(f"selected CSV missing columns: {sorted(missing)}")
    selected = selected[selected["matched"] == True]  # noqa: E712
    if selected.empty:
        raise SystemExit("selected CSV has no matched rows")

    for _, row in selected.iterrows():
        seed = int(row["seed"])
        selected_method = str(row["method"])
        method = f"{args.method_prefix}_{selected_method}" if args.method_prefix else selected_method
        checkpoint_name = checkpoint_name_from_selected_path(str(row["path"]))
        checkpoint = Path(args.checkpoint_root) / checkpoint_name
        for pool in args.pools:
            label = pool_label(pool)
            output = Path(args.output_root) / f"{checkpoint_name}_{label}"
            if args.overwrite or not (output / "survival_curve.csv").exists():
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
                        pool,
                        "--steps",
                        args.steps,
                        "--lr",
                        str(args.lr),
                        "--batch_size",
                        str(args.batch_size),
                        "--max_length",
                        str(args.max_length),
                        "--resurrection_metric",
                        args.resurrection_metric,
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--output",
                        str(output),
                    ],
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
