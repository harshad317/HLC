#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.tofu import write_synthetic_tofu
from durable_unlearning.data.tofu_real import write_tofu_real


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/tofu")
    parser.add_argument("--forget_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=["synthetic", "tofu"],
        default=None,
        help="synthetic = local 6-item smoke fixture; tofu = real locuslab/TOFU benchmark.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Backward-compatible alias for --source synthetic.",
    )
    parser.add_argument("--dataset_name", default="locuslab/TOFU")
    args = parser.parse_args()

    source = args.source or ("synthetic" if args.synthetic else None)
    if source is None:
        raise SystemExit("Specify --source synthetic or --source tofu (or legacy --synthetic).")

    if source == "synthetic":
        forget_path, retain_path = write_synthetic_tofu(
            args.output_dir, args.forget_fraction, seed=args.seed
        )
    else:
        forget_path, retain_path = write_tofu_real(
            args.output_dir, args.forget_fraction, dataset_name=args.dataset_name
        )
    print(f"wrote {forget_path}")
    print(f"wrote {retain_path}")


if __name__ == "__main__":
    main()
