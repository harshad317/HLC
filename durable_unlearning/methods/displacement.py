"""Displacement-durability unlearning (a fresh hypothesis).

Motivation. On real TOFU, durability tracked **weight displacement**: the
constrained gradient-ascent baseline moved further from the fine-tuned model
(paying a higher retain cost) and was ~2x more durable than the elaborate HLC
objective, while HLC's lighter-touch unlearning preserved utility but relearned
fast. The flat-minima/sharpness axis was inert.

Hypothesis. We can get gradient-ascent's *displacement* (hence durability)
without its *utility cost* by ascending the forget loss while projecting that
ascent **orthogonal to the retain-loss gradient** -- i.e. move as far as GA in the
forget directions, but never along the directions that increase retain loss.

This is gradient-surgery'd aggressive unlearning:

    g_f = grad(forget CE)      # +g_f increases forget loss (the ascent / displacement direction)
    g_r = grad(retain CE)      # +g_r increases retain loss (the direction to avoid)
    g_f_orth = g_f - max(0, <g_f, g_r>/||g_r||^2) * g_r     # remove only the retain-harming part
    update: maximize forget loss along g_f_orth, minimize retain loss along g_r

If the hypothesis holds, this sits on a better durability/utility frontier than
both GA (more durable, worse utility) and HLC (better utility, less durable).
It is a prototype to test, not a validated method.
"""

from __future__ import annotations

from durable_unlearning.losses.ce import ce_loss
from durable_unlearning.methods.base import (
    TrainLogger,
    cycle_batches,
    forget_texts,
    grad_cosine,
    retain_texts,
)
from durable_unlearning.models.lora import assign_grads, trainable_named_parameters
from durable_unlearning.utils.deps import require_torch


def project_orthogonal(g_f, g_r):
    """Remove the component of ``g_f`` that aligns with ``g_r`` (only if positive).

    Returns ``g_f - max(0, <g_f,g_r>/||g_r||^2) * g_r`` so the result does not push
    in the retain-loss-increasing direction. Operates on lists of tensors.
    """
    import torch

    dot = sum((a * b).sum() for a, b in zip(g_f, g_r))
    rn2 = sum((b * b).sum() for b in g_r).clamp_min(1e-12)
    coef = (dot / rn2).clamp_min(0.0)
    return [a - coef * b for a, b in zip(g_f, g_r)]


def train_displacement(model, tokenizer, forget_items, retain_items, cfg: dict, output_dir: str):
    torch = require_torch()
    steps = int(cfg.get("steps", 200))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 1e-4))
    forget_weight = float(cfg.get("forget_weight", 1.0))
    retain_weight = float(cfg.get("retain_weight", 1.0))
    max_length = int(cfg.get("max_length", 192))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    orthogonalize = bool(cfg.get("orthogonalize", True))

    named = trainable_named_parameters(model, prefer_lora=True)
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=lr)
    forget_batches = cycle_batches(forget_items, batch_size)
    retain_batches = cycle_batches(retain_items, batch_size)
    logger = TrainLogger(output_dir)
    model.train()

    for step in range(steps):
        fb = next(forget_batches)
        rb = next(retain_batches)

        loss_f = ce_loss(model, tokenizer, forget_texts(fb), max_length=max_length)
        g_f = list(torch.autograd.grad(loss_f, params, allow_unused=True))
        g_f = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(g_f, params)]

        loss_r = ce_loss(model, tokenizer, retain_texts(rb), max_length=max_length)
        g_r = list(torch.autograd.grad(loss_r, params, allow_unused=True))
        g_r = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(g_r, params)]

        g_f_eff = project_orthogonal(g_f, g_r) if orthogonalize else g_f

        # Optimizer descends: to MAXIMIZE forget loss use -g_f_eff; to MINIMIZE
        # retain loss use +g_r.
        combined = [-forget_weight * gf + retain_weight * gr for gf, gr in zip(g_f_eff, g_r)]
        assign_grads(named, combined)
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        logger.append(
            {
                "step": step,
                "method": "displacement",
                "seed": int(cfg.get("seed", 0)),
                "loss_forget": float(loss_f.detach().cpu().item()),
                "loss_retain": float(loss_r.detach().cpu().item()),
                "grad_norm_now": float(torch.sqrt(sum((g.detach() ** 2).sum() for g in g_f_eff)).cpu().item()),
                "grad_cos_now_post": grad_cosine(g_f, g_r),
                "lr_outer": lr,
            }
        )
    logger.flush()
    return model
