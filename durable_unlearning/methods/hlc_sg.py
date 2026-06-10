from __future__ import annotations

from durable_unlearning.losses.ce import ce_loss, lm_batch
from durable_unlearning.losses.kl import topk_kl_loss
from durable_unlearning.losses.negpref import negpref_loss
from durable_unlearning.losses.resurrection import (
    item_max_margins,
    margin_growth_softplus_loss,
    resurrection_softplus_loss,
    target_margin_values,
)
from durable_unlearning.losses.sharpness import sharpness_resurrection_grad
from durable_unlearning.methods.base import (
    TrainLogger,
    cycle_batches,
    grad_cosine,
    relearn_texts,
    retain_texts,
)
from durable_unlearning.models.lora import (
    assign_grads,
    clone_lora_state,
    functional_sgd_step,
    set_lora_state,
    trainable_named_parameters,
)
from durable_unlearning.utils.deps import require_torch


def _functional_call(model, params: dict[str, object], kwargs: dict[str, object]):
    import torch

    return torch.func.functional_call(model, params, (), kwargs)


def _functional_ce_loss(model, tokenizer, params: dict[str, object], texts: list[str], max_length: int):
    device = next(model.parameters()).device
    batch = lm_batch(tokenizer, texts, str(device), max_length=max_length)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    outputs = _functional_call(model, params, {**batch, "labels": labels})
    return outputs.loss


def _functional_conditional_logprobs(
    model,
    tokenizer,
    params: dict[str, object],
    prompts: list[str],
    targets: list[str],
    max_length: int,
):
    import torch

    device = next(model.parameters()).device
    full_texts = [prompt + target for prompt, target in zip(prompts, targets)]
    full = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_tokens = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = full["input_ids"].to(device)
    attention_mask = full["attention_mask"].to(device)
    labels = input_ids.clone()
    for idx in range(input_ids.shape[0]):
        prompt_len = int(prompt_tokens["attention_mask"][idx].sum().item())
        labels[idx, :prompt_len] = -100
    labels[attention_mask == 0] = -100
    outputs = _functional_call(model, params, {"input_ids": input_ids, "attention_mask": attention_mask})
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = (token_log_probs * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp_min(1)
    return seq_log_probs / token_counts


def _functional_item_max_margins(
    model,
    tokenizer,
    params: dict[str, object],
    items,
    neutral_answer: str,
    max_length: int,
):
    import torch

    prompts: list[str] = []
    target_answers: list[str] = []
    neutral_answers: list[str] = []
    index: list[int] = []
    for item_idx, item in enumerate(items):
        variants = item.prompt_variants or [item.question]
        for variant in variants:
            prompts.append(variant)
            target_answers.append(item.answer)
            neutral_answers.append(item.neutral_answer or neutral_answer)
            index.append(item_idx)
    logp_target = _functional_conditional_logprobs(model, tokenizer, params, prompts, target_answers, max_length)
    logp_neutral = _functional_conditional_logprobs(model, tokenizer, params, prompts, neutral_answers, max_length)
    margins = logp_target - logp_neutral
    grouped: list[list[object]] = [[] for _ in items]
    for margin, item_idx in zip(margins, index):
        grouped[item_idx].append(margin)
    return torch.stack([torch.stack(values).max() for values in grouped])


def _functional_resurrection_softplus_loss(
    model,
    tokenizer,
    params: dict[str, object],
    items,
    thresholds: dict[str, float],
    gamma: float,
    neutral_answer: str,
    max_length: int,
):
    import torch

    margins = _functional_item_max_margins(model, tokenizer, params, items, neutral_answer, max_length)
    tau = torch.tensor([thresholds.get(item.item_id, 0.0) for item in items], dtype=margins.dtype, device=margins.device)
    return torch.nn.functional.softplus((margins - tau) / gamma).mean()


def _functional_margin_growth_softplus_loss(
    model,
    tokenizer,
    params: dict[str, object],
    items,
    baseline_margins,
    gamma: float,
    target_growth: float,
    neutral_answer: str,
    max_length: int,
):
    import torch

    post_margins = _functional_item_max_margins(model, tokenizer, params, items, neutral_answer, max_length)
    baseline = baseline_margins.to(dtype=post_margins.dtype, device=post_margins.device)
    return torch.nn.functional.softplus((post_margins - baseline - target_growth) / gamma).mean()


def _functional_sgd_step(state: dict[str, object], names: list[str], grads: list[object], lr: float, weight_decay: float):
    next_state = dict(state)
    for name, grad in zip(names, grads):
        value = state[name]
        update = grad
        if weight_decay:
            update = update + weight_decay * value
        next_state[name] = value - lr * update
    return next_state


def parse_resurrection_penalty_steps(value: object, k_inner: int) -> list[int]:
    if value is None or value == "":
        return [k_inner]
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"terminal", "k", "final"}:
            return [k_inner]
        if text == "all":
            return list(range(1, k_inner + 1))
        if text in {"powers_of_two", "pow2"}:
            steps: list[int] = []
            current = 1
            while current <= k_inner:
                steps.append(current)
                current *= 2
            if steps[-1] != k_inner:
                steps.append(k_inner)
            return steps
        parts = [part.strip() for part in text.split(",") if part.strip()]
        parsed = [k_inner if part in {"k", "terminal", "final"} else int(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        parsed = [k_inner if str(part).lower() in {"k", "terminal", "final"} else int(part) for part in value]
    else:
        parsed = [int(value)]
    steps = sorted(set(parsed))
    if not steps:
        raise ValueError("resurrection_penalty_steps cannot be empty")
    invalid = [step for step in steps if step < 1 or step > k_inner]
    if invalid:
        raise ValueError(f"resurrection_penalty_steps must be in [1, {k_inner}], got {invalid}")
    return steps


def train_hlc_sg(model, full_model, tokenizer, forget_items, retain_items, relearn_items, thresholds, cfg: dict, output_dir: str):
    torch = require_torch()
    batch_size = int(cfg.get("batch_size", 2))
    outer_steps = int(cfg.get("outer_steps", cfg.get("steps", 20)))
    k_inner = int(cfg.get("K", 8))
    inner_lr = float(cfg.get("inner_lr", 2e-5))
    outer_lr = float(cfg.get("outer_lr", 1e-5))
    lambda0 = float(cfg.get("lambda0", 1.0))
    lambdaK = float(cfg.get("lambdaK", 1.0))
    lambdaR = float(cfg.get("lambdaR", 1.0))
    lambdaG = float(cfg.get("lambdaG", 0.1))
    lambda_margin_growth = float(cfg.get("lambda_margin_growth", cfg.get("lambdaM", 0.0)))
    gamma = float(cfg.get("gamma", 0.1))
    margin_growth_gamma = float(cfg.get("margin_growth_gamma", gamma))
    margin_growth_target = float(cfg.get("margin_growth_target", 0.0))
    beta = float(cfg.get("beta", 0.1))
    neutral_answer = str(cfg.get("neutral_answer", "I don't know."))
    kl_top_k = int(cfg.get("kl_top_k", 128))
    max_length = int(cfg.get("max_length", 256))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    inner_weight_decay = float(cfg.get("inner_weight_decay", 0.0))
    # Sharpness-aware durability term (off by default for backward compatibility).
    sharpness_rho = float(cfg.get("sharpness_rho", 0.0))
    lambda_sharpness = float(cfg.get("lambda_sharpness", cfg.get("lambdaS", 0.0)))
    sharpness_gamma = float(cfg.get("sharpness_gamma", gamma))
    sharpness_direction = str(cfg.get("sharpness_direction", "both")).lower()
    sharpness_enabled = lambda_sharpness > 0 and sharpness_rho > 0
    post_gradient_mode = str(cfg.get("post_gradient_mode", "copy")).lower()
    if post_gradient_mode == "detached":
        post_gradient_mode = "copy"
    if post_gradient_mode not in {"copy", "unrolled"}:
        raise ValueError(f"Unsupported post_gradient_mode: {post_gradient_mode}")
    penalty_steps = parse_resurrection_penalty_steps(cfg.get("resurrection_penalty_steps"), k_inner)
    named_params = trainable_named_parameters(model, prefer_lora=True)
    params = [p for _, p in named_params]
    optimizer = torch.optim.AdamW(params, lr=outer_lr)
    forget_batches = cycle_batches(forget_items, batch_size)
    retain_batches = cycle_batches(retain_items, batch_size)
    relearn_batches = cycle_batches(relearn_items, batch_size)
    logger = TrainLogger(output_dir)
    full_model.eval()
    model.train()

    for step in range(outer_steps):
        fb = next(forget_batches)
        rb = next(retain_batches)
        g_texts = retain_texts(rb)
        b_seq = [next(relearn_batches) for _ in range(k_inner)]
        current_state = clone_lora_state(model, prefer_lora=True)
        names = [name for name, _ in named_params]
        if post_gradient_mode == "unrolled":
            theta_tilde = {name: param for name, param in named_params}
            penalty_states: list[dict[str, object]] = []
            if lambda_margin_growth > 0:
                baseline_margins = _functional_item_max_margins(
                    model,
                    tokenizer,
                    theta_tilde,
                    fb,
                    neutral_answer,
                    max_length,
                ).detach()
            else:
                baseline_margins = None

            for inner_idx, inner_batch in enumerate(b_seq, start=1):
                state_params = [theta_tilde[name] for name in names]
                loss_relearn = _functional_ce_loss(
                    model,
                    tokenizer,
                    theta_tilde,
                    relearn_texts(inner_batch),
                    max_length=max_length,
                )
                grads = torch.autograd.grad(
                    loss_relearn,
                    state_params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )
                grads = [
                    torch.zeros_like(param) if grad is None else grad
                    for grad, param in zip(grads, state_params)
                ]
                theta_tilde = _functional_sgd_step(
                    theta_tilde,
                    names,
                    grads,
                    lr=inner_lr,
                    weight_decay=inner_weight_decay,
                )
                if inner_idx in penalty_steps:
                    penalty_states.append(dict(theta_tilde))

            if not penalty_states:
                penalty_states.append(dict(theta_tilde))
            loss_post_values = []
            loss_resurrect_values = []
            loss_margin_growth_values = []
            for penalty_state in penalty_states:
                loss_resurrect_i = _functional_resurrection_softplus_loss(
                    model,
                    tokenizer,
                    penalty_state,
                    fb,
                    thresholds,
                    gamma=gamma,
                    neutral_answer=neutral_answer,
                    max_length=max_length,
                )
                if lambda_margin_growth > 0:
                    assert baseline_margins is not None
                    loss_margin_growth_i = _functional_margin_growth_softplus_loss(
                        model,
                        tokenizer,
                        penalty_state,
                        fb,
                        baseline_margins,
                        gamma=margin_growth_gamma,
                        target_growth=margin_growth_target,
                        neutral_answer=neutral_answer,
                        max_length=max_length,
                    )
                else:
                    loss_margin_growth_i = loss_resurrect_i.detach() * 0.0
                loss_post_values.append(loss_resurrect_i + lambda_margin_growth * loss_margin_growth_i)
                loss_resurrect_values.append(loss_resurrect_i.detach())
                loss_margin_growth_values.append(loss_margin_growth_i.detach())

            loss_post_for_grad = torch.stack(loss_post_values).mean()
            g_post = torch.autograd.grad(
                loss_post_for_grad,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            g_post = [
                torch.zeros_like(param) if grad is None else grad
                for grad, param in zip(g_post, params)
            ]
            loss_post = loss_post_for_grad.detach()
            loss_resurrect = torch.stack(loss_resurrect_values).mean()
            loss_margin_growth = torch.stack(loss_margin_growth_values).mean()
        else:
            theta_tilde = {name: value.detach().clone() for name, value in current_state.items()}
            penalty_states: list[dict[str, object]] = []
            baseline_margins = None
            if lambda_margin_growth > 0:
                set_lora_state(model, current_state)
                baseline_margins = item_max_margins(
                    model,
                    tokenizer,
                    fb,
                    neutral_answer=neutral_answer,
                    max_length=max_length,
                ).detach()

            for inner_idx, inner_batch in enumerate(b_seq, start=1):
                set_lora_state(model, theta_tilde)
                loss_relearn = ce_loss(model, tokenizer, relearn_texts(inner_batch), max_length=max_length)
                inner_named = trainable_named_parameters(model, prefer_lora=True)
                inner_params = [p for _, p in inner_named]
                grads = torch.autograd.grad(loss_relearn, inner_params, retain_graph=False, create_graph=False, allow_unused=True)
                grads = [torch.zeros_like(param) if grad is None else grad for grad, param in zip(grads, inner_params)]
                theta_tilde = functional_sgd_step(theta_tilde, names, grads, lr=inner_lr, weight_decay=inner_weight_decay)
                if inner_idx in penalty_steps:
                    penalty_states.append({name: value.detach().clone() for name, value in theta_tilde.items()})

            if not penalty_states:
                penalty_states.append({name: value.detach().clone() for name, value in theta_tilde.items()})
            post_named = trainable_named_parameters(model, prefer_lora=True)
            post_params = [p for _, p in post_named]
            g_post = [torch.zeros_like(param) for param in post_params]
            loss_post_values = []
            loss_resurrect_values = []
            loss_margin_growth_values = []
            for penalty_state in penalty_states:
                set_lora_state(model, penalty_state)
                loss_resurrect_i = resurrection_softplus_loss(
                    model,
                    tokenizer,
                    fb,
                    thresholds,
                    gamma=gamma,
                    neutral_answer=neutral_answer,
                    max_length=max_length,
                )
                if lambda_margin_growth > 0:
                    assert baseline_margins is not None
                    loss_margin_growth_i = margin_growth_softplus_loss(
                        model,
                        tokenizer,
                        fb,
                        baseline_margins,
                        gamma=margin_growth_gamma,
                        target_growth=margin_growth_target,
                        neutral_answer=neutral_answer,
                        max_length=max_length,
                    )
                else:
                    loss_margin_growth_i = loss_resurrect_i.detach() * 0.0
                loss_post_i = loss_resurrect_i + lambda_margin_growth * loss_margin_growth_i
                loss_post_values.append(loss_post_i.detach())
                loss_resurrect_values.append(loss_resurrect_i.detach())
                loss_margin_growth_values.append(loss_margin_growth_i.detach())
                grads_post_i = torch.autograd.grad(loss_post_i, post_params, retain_graph=False, create_graph=False, allow_unused=True)
                grads_post_i = [
                    torch.zeros_like(param) if grad is None else grad
                    for grad, param in zip(grads_post_i, post_params)
                ]
                g_post = [total + grad / len(penalty_states) for total, grad in zip(g_post, grads_post_i)]
            loss_post = torch.stack(loss_post_values).mean() if loss_post_values else torch.tensor(0.0)
            loss_resurrect = torch.stack(loss_resurrect_values).mean() if loss_resurrect_values else torch.tensor(0.0)
            loss_margin_growth = (
                torch.stack(loss_margin_growth_values).mean()
                if loss_margin_growth_values
                else torch.tensor(0.0)
            )

        set_lora_state(model, current_state)
        loss_forget = negpref_loss(model, tokenizer, fb, neutral_answer=neutral_answer, beta=beta, max_length=max_length)
        loss_retain = ce_loss(model, tokenizer, retain_texts(rb), max_length=max_length)
        if lambdaG > 0:
            loss_kl = topk_kl_loss(model, full_model, tokenizer, g_texts, k=kl_top_k, max_length=max_length)
        else:
            loss_kl = loss_retain.detach() * 0.0
        loss_now = lambda0 * loss_forget + lambdaR * loss_retain + lambdaG * loss_kl
        now_named = trainable_named_parameters(model, prefer_lora=True)
        now_params = [p for _, p in now_named]
        g_now = torch.autograd.grad(loss_now, now_params, retain_graph=False, create_graph=False, allow_unused=True)
        g_now = [torch.zeros_like(param) if grad is None else grad for grad, param in zip(g_now, now_params)]
        combined = [gn + lambdaK * gp for gn, gp in zip(g_now, g_post)]
        # Sharpness-aware durability: add the worst-case resurrection gradient at
        # the current parameters (theta_s). Perturbs and exactly restores params.
        if sharpness_enabled:
            loss_sharpness_value, g_sharp = sharpness_resurrection_grad(
                model,
                tokenizer,
                now_named,
                fb,
                thresholds,
                rho=sharpness_rho,
                gamma=sharpness_gamma,
                neutral_answer=neutral_answer,
                max_length=max_length,
                relearn_texts_list=(
                    relearn_texts(b_seq[0])
                    if sharpness_direction in ("relearn", "both") and b_seq
                    else None
                ),
                direction=sharpness_direction,
            )
            combined = [c + lambda_sharpness * gs for c, gs in zip(combined, g_sharp)]
        else:
            loss_sharpness_value = 0.0
        assign_grads(now_named, combined)
        torch.nn.utils.clip_grad_norm_([p for _, p in now_named], max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        margins = target_margin_values(model, tokenizer, fb, neutral_answer=neutral_answer, max_length=max_length)
        logger.append(
            {
                "step": step,
                "method": "hlc_sg",
                "seed": int(cfg.get("seed", 0)),
                "loss_forget": float(loss_forget.detach().cpu().item()),
                "loss_retain": float(loss_retain.detach().cpu().item()),
                "loss_kl": float(loss_kl.detach().cpu().item()),
                "loss_resurrect": float(loss_resurrect.detach().cpu().item()),
                "loss_margin_growth": float(loss_margin_growth.detach().cpu().item()),
                "loss_sharpness": float(loss_sharpness_value),
                "loss_post": float(loss_post.detach().cpu().item()),
                "target_margin_forget": float(sum(margins) / len(margins)) if margins else 0.0,
                "grad_norm_now": float(torch.sqrt(sum((g.detach() ** 2).sum() for g in g_now)).cpu().item()),
                "grad_norm_post": float(torch.sqrt(sum((g.detach() ** 2).sum() for g in g_post)).cpu().item()),
                "grad_cos_now_post": grad_cosine(g_now, g_post),
                "lr_outer": outer_lr,
                "lr_inner": inner_lr,
            }
        )
    logger.flush()
    return model
