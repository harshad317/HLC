from __future__ import annotations

from typing import Optional

from durable_unlearning.losses.ce import ce_loss
from durable_unlearning.losses.npo import npo_loss
from durable_unlearning.methods.base import TrainLogger, cycle_batches, grad_norm, retain_texts
from durable_unlearning.utils.deps import require_torch


def expand_with_prompt_variants(items, max_variants: Optional[int] = None):
    expanded = []
    for item in items:
        variants = item.prompt_variants or [item.question]
        if max_variants is not None:
            variants = variants[:max_variants]
        for variant in variants:
            expanded.append(
                type(item)(
                    item_id=item.item_id,
                    person_id=item.person_id,
                    question=variant,
                    answer=item.answer,
                    aliases=item.aliases,
                    neutral_answer=item.neutral_answer,
                    prompt_variants=item.prompt_variants,
                    split=item.split,
                )
            )
    return expanded


def train_npo(model, tokenizer, forget_items, retain_items, cfg: dict, output_dir: str):
    torch = require_torch()
    batch_size = int(cfg.get("batch_size", 2))
    steps = int(cfg.get("steps", 20))
    lr = float(cfg.get("lr", 1e-5))
    beta = float(cfg.get("beta", 0.1))
    forget_weight = float(cfg.get("forget_weight", 1.0))
    retain_weight = float(cfg.get("retain_weight", 1.0))
    max_length = int(cfg.get("max_length", 256))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    use_prompt_variants = bool(cfg.get("use_prompt_variants", False))
    max_prompt_variants = cfg.get("max_prompt_variants")
    max_prompt_variants = int(max_prompt_variants) if max_prompt_variants is not None else None
    method_name = str(cfg.get("method_name", "npo"))
    if use_prompt_variants:
        forget_items = expand_with_prompt_variants(forget_items, max_variants=max_prompt_variants)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    forget_batches = cycle_batches(forget_items, batch_size)
    retain_batches = cycle_batches(retain_items, batch_size)
    logger = TrainLogger(output_dir)
    model.train()
    for step in range(steps):
        fb = next(forget_batches)
        rb = next(retain_batches)
        loss_forget = npo_loss(model, tokenizer, fb, beta=beta, max_length=max_length)
        loss_retain = ce_loss(model, tokenizer, retain_texts(rb), max_length=max_length)
        loss = forget_weight * loss_forget + retain_weight * loss_retain
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        gn = grad_norm(trainable)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logger.append(
            {
                "step": step,
                "method": method_name,
                "seed": int(cfg.get("seed", 0)),
                "loss_forget": float(loss_forget.detach().cpu().item()),
                "loss_retain": float(loss_retain.detach().cpu().item()),
                "grad_norm_now": gn,
                "lr_outer": lr,
            }
        )
    logger.flush()
    return model
