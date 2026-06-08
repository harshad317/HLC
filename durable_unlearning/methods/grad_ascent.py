from __future__ import annotations

from durable_unlearning.losses.ce import ce_loss
from durable_unlearning.methods.base import TrainLogger, cycle_batches, forget_texts, grad_norm, retain_texts
from durable_unlearning.utils.deps import require_torch


def train_grad_ascent(model, tokenizer, forget_items, retain_items, cfg: dict, output_dir: str):
    torch = require_torch()
    batch_size = int(cfg.get("batch_size", 2))
    steps = int(cfg.get("steps", 20))
    lr = float(cfg.get("lr", 1e-5))
    retain_weight = float(cfg.get("retain_weight", 1.0))
    max_length = int(cfg.get("max_length", 256))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    forget_batches = cycle_batches(forget_items, batch_size)
    retain_batches = cycle_batches(retain_items, batch_size)
    logger = TrainLogger(output_dir)
    model.train()
    for step in range(steps):
        fb = next(forget_batches)
        rb = next(retain_batches)
        loss_forget = ce_loss(model, tokenizer, forget_texts(fb), max_length=max_length)
        loss_retain = ce_loss(model, tokenizer, retain_texts(rb), max_length=max_length)
        loss = -loss_forget + retain_weight * loss_retain
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        gn = grad_norm([p for p in model.parameters() if p.requires_grad])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logger.append(
            {
                "step": step,
                "method": "grad_ascent",
                "seed": int(cfg.get("seed", 0)),
                "loss_forget": float(loss_forget.detach().cpu().item()),
                "loss_retain": float(loss_retain.detach().cpu().item()),
                "grad_norm_now": gn,
                "lr_outer": lr,
            }
        )
    logger.flush()
    return model
