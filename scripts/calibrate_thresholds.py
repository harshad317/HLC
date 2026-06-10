#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.io import read_jsonl
from durable_unlearning.data.types import ForgetItem
from durable_unlearning.eval.runner import attach_prompt_variants
from durable_unlearning.eval.thresholding import compute_threshold_records, write_thresholds
from durable_unlearning.logging.artifact_hash import directory_hash
from durable_unlearning.losses.resurrection import target_margins
from durable_unlearning.models.load import load_causal_lm_with_lora


def parse_quantiles(text: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--quantiles must contain exactly three values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--forget_file", required=True)
    parser.add_argument("--prompt_variants", default=None)
    parser.add_argument("--quantiles", type=parse_quantiles, default=(0.90, 0.95, 0.99))
    parser.add_argument("--tau_global", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()
    model, tokenizer = load_causal_lm_with_lora(
        model_name=args.base_model,
        lora_cfg={"enabled": False},
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
    )
    items = [ForgetItem.from_dict(row) for row in read_jsonl(args.forget_file)]
    items = attach_prompt_variants(items, args.prompt_variants)
    import torch

    model.eval()
    with torch.no_grad():
        margins, index = target_margins(model, tokenizer, items, max_length=args.max_length)
    rows = []
    for margin, (item_idx, prompt) in zip(margins, index):
        rows.append(
            {
                "item_id": items[item_idx].item_id,
                "variant_id": f"{items[item_idx].item_id}_{len(rows)}",
                "margin": float(margin.detach().cpu().item()),
            }
        )
    oracle_hash = directory_hash(args.checkpoint) if Path(args.checkpoint).exists() else ""
    records = compute_threshold_records(rows, quantiles=args.quantiles, tau_global=args.tau_global, oracle_model_hash=oracle_hash)
    write_thresholds(args.output, records)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
