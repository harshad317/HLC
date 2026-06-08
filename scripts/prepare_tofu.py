#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.tofu import write_synthetic_tofu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/tofu")
    parser.add_argument("--forget_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Generate local synthetic TOFU-style data.")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is implemented for the local-first scaffold.")
    forget_path, retain_path = write_synthetic_tofu(args.output_dir, args.forget_fraction, seed=args.seed)
    print(f"wrote {forget_path}")
    print(f"wrote {retain_path}")


if __name__ == "__main__":
    main()
