"""RMU-style representation-misdirection unlearning (robust-relearning baseline).

A strong, relevant competitor: rather than suppressing outputs, RMU
(Representation Misdirection for Unlearning; Li et al., WMDP) edits internal
activations. At a chosen layer it (i) steers forget-data activations toward a fixed
random control vector and (ii) keeps retain-data activations close to a frozen
reference. Because it perturbs the representation rather than only the logits, it is
the natural baseline for *robustness to relearning*, and the right thing to compare
the displacement method against.

    L = MSE(h_layer(forget), c * u)  +  alpha * MSE(h_layer(retain), h_layer^frozen(retain))

where u is a fixed random unit vector and c a steering coefficient.
"""

from __future__ import annotations

from durable_unlearning.losses.ce import lm_batch
from durable_unlearning.methods.base import (
    TrainLogger,
    cycle_batches,
    forget_texts,
    retain_texts,
)
from durable_unlearning.models.lora import assign_grads, trainable_named_parameters
from durable_unlearning.utils.deps import require_torch


def masked_mse(h, target, mask):
    """Mean squared error between ``h`` and ``target`` over valid (mask=1) positions.

    ``h``: [B,T,H]; ``target``: broadcastable to ``h``; ``mask``: [B,T,1].
    Normalized by the number of valid token-positions and the hidden width.

    Computed in float32: squared transformer activations overflow float16 (max
    65504) to inf, and inf * (padding mask 0) is NaN, which silently kills training.
    """
    h = h.float()
    target = target.float()
    mask = mask.float()
    diff = (h - target) * mask
    denom = mask.sum().clamp_min(1) * h.shape[-1]
    return diff.pow(2).sum() / denom


def _layer_hidden(model, tokenizer, texts, layer, max_length, no_grad):
    import torch

    device = str(next(model.parameters()).device)
    batch = lm_batch(tokenizer, texts, device, max_length=max_length)
    if no_grad:
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True)
    else:
        out = model(**batch, output_hidden_states=True)
    h = out.hidden_states[layer]
    mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
    return h, mask


def train_rmu(model, frozen_model, tokenizer, forget_items, retain_items, cfg: dict, output_dir: str):
    torch = require_torch()
    steps = int(cfg.get("steps", 200))
    batch_size = int(cfg.get("batch_size", 4))
    lr = float(cfg.get("lr", 1e-4))
    layer = int(cfg.get("layer", 7))
    steer_coeff = float(cfg.get("steer_coeff", 6.0))
    alpha = float(cfg.get("alpha", 1.0))
    max_length = int(cfg.get("max_length", 192))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    seed = int(cfg.get("seed", 0))

    named = trainable_named_parameters(model, prefer_lora=True)
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=lr)
    forget_batches = cycle_batches(forget_items, batch_size)
    retain_batches = cycle_batches(retain_items, batch_size)
    logger = TrainLogger(output_dir)
    frozen_model.eval()
    model.train()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    control = None  # fixed random control vector, sized to hidden dim on first step

    for step in range(steps):
        fb = forget_texts(next(forget_batches))
        rb = retain_texts(next(retain_batches))

        h_f, mask_f = _layer_hidden(model, tokenizer, fb, layer, max_length, no_grad=False)
        if control is None:
            hidden = h_f.shape[-1]
            u = torch.randn(hidden, generator=gen).to(device=h_f.device, dtype=h_f.dtype)
            u = u / u.norm().clamp_min(1e-8)
            control = (steer_coeff * u).detach()
        loss_forget = masked_mse(h_f, control, mask_f)

        h_r, mask_r = _layer_hidden(model, tokenizer, rb, layer, max_length, no_grad=False)
        h_r_ref, _ = _layer_hidden(frozen_model, tokenizer, rb, layer, max_length, no_grad=True)
        loss_retain = masked_mse(h_r, h_r_ref.detach(), mask_r)

        loss = loss_forget + alpha * loss_retain
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
        assign_grads(named, grads)
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        logger.append({
            "step": step,
            "method": "rmu",
            "seed": seed,
            "loss_forget": float(loss_forget.detach().cpu().item()),
            "loss_retain": float(loss_retain.detach().cpu().item()),
            "lr_outer": lr,
        })
    logger.flush()
    return model
