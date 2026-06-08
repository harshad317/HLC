#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.io import read_jsonl
from durable_unlearning.data.tofu import rows_to_lm_texts
from durable_unlearning.methods.finetune import fine_tune_lm
from durable_unlearning.models.checkpointing import save_model_checkpoint
from durable_unlearning.models.load import load_causal_lm_with_lora
from durable_unlearning.utils.seed import set_seed
from durable_unlearning.utils.yaml import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--forget_split", default="forget")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)
    set_seed(args.seed)
    rows = read_jsonl(data_cfg["retain_file"])
    model, tokenizer = load_causal_lm_with_lora(
        model_name=model_cfg["model_name"],
        lora_cfg=model_cfg.get("lora"),
        dtype=model_cfg.get("dtype", "float32"),
        device=model_cfg.get("device"),
    )
    train_cfg = dict(model_cfg.get("train", {}))
    train_cfg["seed"] = args.seed
    fine_tune_lm(model, tokenizer, rows_to_lm_texts(rows), train_cfg, args.output, method="retain_only")
    save_model_checkpoint(
        model,
        tokenizer,
        args.output,
        {
            "base_model": model_cfg["model_name"],
            "seed": args.seed,
            "role": "retrain_minus_f",
            "forget_split": args.forget_split,
            "model_config": args.config,
            "data_config": args.data,
        },
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
