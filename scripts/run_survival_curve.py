#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.eval.runner import run_survival_curve


def parse_steps(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def infer_seed(*values: str | None) -> int:
    for value in values:
        if not value:
            continue
        match = re.search(r"seed(\d+)", value)
        if match:
            return int(match.group(1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--forget_file", required=True)
    parser.add_argument("--retain_file", required=True)
    parser.add_argument("--relearn_pool", required=True)
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("0,1,2,4,8,16,32,64,128,256,512"))
    parser.add_argument("--order_manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt_variants", default=None)
    parser.add_argument("--threshold_quantile", default="q95")
    parser.add_argument("--method", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--resurrection_metric", choices=["margin", "margin_and_exact"], default="margin")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max_retain_items",
        type=int,
        default=None,
        help="Estimate retain utility on a fixed random subset of this many items "
        "(prevents OOM on large TOFU retain splits). Default: use all.",
    )
    args = parser.parse_args()
    if args.order_manifest:
        print("order manifests are recorded by the harness; custom ordering is reserved for the full GPU runner")
    seed = args.seed if args.seed is not None else infer_seed(args.checkpoint, args.output)
    run_survival_curve(
        checkpoint=args.checkpoint,
        thresholds_file=args.thresholds,
        forget_file=args.forget_file,
        retain_file=args.retain_file,
        relearn_pool=args.relearn_pool,
        steps=args.steps,
        output=args.output,
        base_model=args.base_model,
        prompt_variants=args.prompt_variants,
        threshold_quantile=args.threshold_quantile,
        method=args.method,
        seed=seed,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        resurrection_metric=args.resurrection_metric,
        dtype=args.dtype,
        device=args.device,
        max_retain_items=args.max_retain_items,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
