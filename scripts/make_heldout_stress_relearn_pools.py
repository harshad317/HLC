#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.relearn_pools import write_heldout_stress_relearn_pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants_per_item", type=int, default=3)
    args = parser.parse_args()
    paths = write_heldout_stress_relearn_pools(
        args.forget_file,
        args.output_dir,
        variants_per_item=args.variants_per_item,
    )
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
