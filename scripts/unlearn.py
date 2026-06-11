#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durable_unlearning.data.io import read_jsonl
from durable_unlearning.data.types import ForgetItem, RelearnExample
from durable_unlearning.eval.runner import attach_prompt_variants
from durable_unlearning.eval.thresholding import load_thresholds
from durable_unlearning.methods.grad_ascent import train_grad_ascent
from durable_unlearning.methods.displacement import train_displacement
from durable_unlearning.methods.rmu import train_rmu
from durable_unlearning.methods.hlc_sg import train_hlc_sg
from durable_unlearning.methods.npo import train_npo
from durable_unlearning.models.checkpointing import save_model_checkpoint
from durable_unlearning.models.load import load_causal_lm_with_lora
from durable_unlearning.utils.seed import set_seed
from durable_unlearning.utils.yaml import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["grad_ascent", "npo", "hlc_sg", "displacement", "rmu"])
    parser.add_argument("--method_config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--full_checkpoint", default=None)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--relearn_pool", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_config", default="configs/model/tiny_gpt2_lora.yaml")
    parser.add_argument("--data_config", default="configs/data/tofu_50_smoke.yaml")
    parser.add_argument("--prompt_variants", default=None)
    parser.add_argument("--threshold_quantile", default="q95")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    method_cfg = load_yaml(args.method_config)
    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    set_seed(args.seed)
    forget_items = [ForgetItem.from_dict(row) for row in read_jsonl(data_cfg["forget_file"])]
    forget_items = attach_prompt_variants(forget_items, args.prompt_variants)
    retain_items = [ForgetItem.from_dict(row) for row in read_jsonl(data_cfg["retain_file"])]
    model, tokenizer = load_causal_lm_with_lora(
        model_name=model_cfg["model_name"],
        lora_cfg=model_cfg.get("lora"),
        checkpoint_path=args.start_checkpoint,
        dtype=model_cfg.get("dtype", "float32"),
        device=model_cfg.get("device"),
    )
    method_cfg["seed"] = args.seed
    if args.method == "grad_ascent":
        train_grad_ascent(model, tokenizer, forget_items, retain_items, method_cfg, args.output)
    elif args.method == "displacement":
        train_displacement(model, tokenizer, forget_items, retain_items, method_cfg, args.output)
    elif args.method == "rmu":
        if not args.full_checkpoint:
            raise SystemExit("--full_checkpoint is required for rmu (frozen retain reference)")
        frozen_model, _ = load_causal_lm_with_lora(
            model_name=model_cfg["model_name"],
            lora_cfg=model_cfg.get("lora"),
            checkpoint_path=args.full_checkpoint,
            dtype=model_cfg.get("dtype", "float32"),
            device=model_cfg.get("device"),
        )
        train_rmu(model, frozen_model, tokenizer, forget_items, retain_items, method_cfg, args.output)
    elif args.method == "npo":
        train_npo(model, tokenizer, forget_items, retain_items, method_cfg, args.output)
    elif args.method == "hlc_sg":
        if not args.full_checkpoint:
            raise SystemExit("--full_checkpoint is required for hlc_sg utility KL")
        if not args.relearn_pool:
            raise SystemExit("--relearn_pool is required for hlc_sg")
        full_model, _ = load_causal_lm_with_lora(
            model_name=model_cfg["model_name"],
            lora_cfg=model_cfg.get("lora"),
            checkpoint_path=args.full_checkpoint,
            dtype=model_cfg.get("dtype", "float32"),
            device=model_cfg.get("device"),
        )
        thresholds = load_thresholds(args.thresholds, quantile=args.threshold_quantile)
        relearn_items = [RelearnExample.from_dict(row) for row in read_jsonl(args.relearn_pool)]
        train_hlc_sg(model, full_model, tokenizer, forget_items, retain_items, relearn_items, thresholds, method_cfg, args.output)
    save_model_checkpoint(
        model,
        tokenizer,
        args.output,
        {
            "base_model": model_cfg["model_name"],
            "seed": args.seed,
            "role": f"unlearned_{args.method}",
            "start_checkpoint": args.start_checkpoint,
            "full_checkpoint": args.full_checkpoint,
            "thresholds": args.thresholds,
            "method_config": args.method_config,
            "data_config": args.data_config,
        },
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
