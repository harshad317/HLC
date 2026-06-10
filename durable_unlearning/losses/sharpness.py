"""Sharpness-aware durable-unlearning penalty.

Standard unlearning losses (NPO, gradient ascent) and the original HLC
resurrection penalty all suppress the forget margin *at a point* in weight
space.  Benign relearning resurrects facts because that point sits on a sharp
ridge: a few benign gradient steps slide the parameters back into the
"remembers" basin.

This module makes the suppressed state *robust* instead of merely low.  It is a
Sharpness-Aware-Minimization (SAM) style objective applied to the resurrection
margin: rather than penalizing ``m_i(theta)``, it penalizes the worst-case
resurrection inside a ``rho``-ball,

    max_{||delta|| <= rho} L_resurrect(theta + delta),

where the perturbation direction is restricted to the directions relearning
actually exploits:

* ``margin``  : the ascent direction of the resurrection loss (the single
  worst direction for the forget margin);
* ``relearn`` : the *descent* direction of the benign relearn loss (``-grad
  L_relearn``), i.e. exactly where benign fine-tuning moves the model;
* ``both``    : the normalized sum of the two.

Optimizing the worst-case margin flattens the forget basin along these
directions, which is precisely a longer relearning half-life.

The function follows the codebase convention of returning a hand-computed
gradient (a list aligned with the trainable parameters) so it can be folded into
the manual gradient combination used by the HLC training loop.
"""

from __future__ import annotations

from durable_unlearning.losses.ce import ce_loss
from durable_unlearning.losses.resurrection import resurrection_softplus_loss

VALID_DIRECTIONS = {"margin", "relearn", "both"}


def _global_norm(tensors):
    import torch

    total = sum((t.detach().float() ** 2).sum() for t in tensors)
    return torch.sqrt(total).clamp_min(1e-12)


def _zeros_like(params):
    import torch

    return [torch.zeros_like(p) for p in params]


def _grad_of(loss, params):
    """First-order gradient of ``loss`` w.r.t. ``params`` with unused -> zeros."""
    import torch

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params)]


def sharpness_resurrection_grad(
    model,
    tokenizer,
    named_params,
    items,
    thresholds: dict[str, float],
    *,
    rho: float,
    gamma: float = 0.1,
    neutral_answer: str = "I don't know.",
    max_length: int = 256,
    relearn_texts_list: list[str] | None = None,
    direction: str = "both",
):
    """Worst-case resurrection penalty and its first-order SAM gradient.

    Returns ``(loss_value, grad_list)`` where ``grad_list`` is aligned with
    ``named_params``.  The model parameters are perturbed in place and exactly
    restored, so the model is left untouched on return.

    When ``rho <= 0`` (disabled) or no usable direction is available this falls
    back to the ordinary point resurrection penalty so the caller can treat it
    uniformly.
    """
    import torch

    direction = str(direction).lower()
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"sharpness direction must be one of {sorted(VALID_DIRECTIONS)}, got {direction!r}")

    params = [p for _, p in named_params]

    def point_loss():
        return resurrection_softplus_loss(
            model,
            tokenizer,
            items,
            thresholds,
            gamma=gamma,
            neutral_answer=neutral_answer,
            max_length=max_length,
        )

    if rho <= 0:
        loss = point_loss()
        return float(loss.detach().cpu().item()), _grad_of(loss, params)

    # 1. Assemble the perturbation direction from the requested components.
    components = []
    if direction in ("margin", "both"):
        loss_margin = point_loss()
        g_asc = _grad_of(loss_margin, params)  # +ascent increases the margin
        if float(_global_norm(g_asc)) > 1e-12:
            components.append(g_asc)
    if direction in ("relearn", "both") and relearn_texts_list:
        loss_relearn = ce_loss(model, tokenizer, relearn_texts_list, max_length=max_length)
        g_rel = _grad_of(loss_relearn, params)
        # benign relearning descends L_relearn: theta -> theta - grad, so the
        # resurrection-inducing perturbation points along -grad.
        g_rel = [-g for g in g_rel]
        if float(_global_norm(g_rel)) > 1e-12:
            components.append(g_rel)

    if not components:
        # No usable ascent direction (e.g. already flat). Fall back to point.
        loss = point_loss()
        return float(loss.detach().cpu().item()), _grad_of(loss, params)

    combined = _zeros_like(params)
    for comp in components:
        norm = _global_norm(comp)
        combined = [c + g / norm for c, g in zip(combined, comp)]
    combined_norm = _global_norm(combined)
    delta = [rho * c / combined_norm for c in combined]

    # 2. Ascend to the worst-case point theta + delta (in place).
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, d in zip(params, delta):
            p.add_(d)

    # 3. Worst-case loss and gradient at the perturbed point.
    try:
        loss_sharp = point_loss()
        g_sharp = _grad_of(loss_sharp, params)
        loss_value = float(loss_sharp.detach().cpu().item())
    finally:
        # 4. Always restore the original parameters.
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)

    # Non-finite guard, consistent with the evaluator's reject-non-finite policy:
    # never let a NaN/Inf perturbation corrupt the outer optimizer step.
    g_sharp = [torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for g in g_sharp]
    if loss_value != loss_value:  # NaN
        loss_value = 0.0
        g_sharp = _zeros_like(params)

    return loss_value, g_sharp
