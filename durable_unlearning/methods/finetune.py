from __future__ import annotations

from durable_unlearning.losses.ce import ce_loss
from durable_unlearning.methods.base import TrainLogger, cycle_batches, grad_norm
from durable_unlearning.utils.deps import require_torch


def fine_tune_lm(model, tokenizer, texts: list[str], cfg: dict, output_dir: str, method: str = "fine_tune"):
    torch = require_torch()
    batch_size = int(cfg.get("batch_size", 2))
    steps = int(cfg.get("steps", 20))
    lr = float(cfg.get("lr", 1e-5))
    max_length = int(cfg.get("max_length", 256))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    batches = cycle_batches(texts, batch_size)
    logger = TrainLogger(output_dir)
    model.train()
    for step in range(steps):
        batch_texts = next(batches)
        loss = ce_loss(model, tokenizer, batch_texts, max_length=max_length)
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        gn = grad_norm(trainable)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logger.append(
            {
                "step": step,
                "method": method,
                "seed": int(cfg.get("seed", 0)),
                "loss_retain": float(loss.detach().cpu().item()),
                "grad_norm_now": gn,
                "lr_outer": lr,
            }
        )
    logger.flush()
    return model
