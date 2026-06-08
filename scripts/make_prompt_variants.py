#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.prompt_variants import write_prompt_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["default", "syntactic"], default="default")
    args = parser.parse_args()
    write_prompt_variants(args.forget_file, args.output, mode=args.mode)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
