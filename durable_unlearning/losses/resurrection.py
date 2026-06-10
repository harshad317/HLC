from __future__ import annotations

from durable_unlearning.data.types import ForgetItem
from durable_unlearning.losses.ce import conditional_logprobs


def target_margins(model, tokenizer, items: list[ForgetItem], neutral_answer: str = "I don't know.", max_length: int = 256):
    prompts: list[str] = []
    target_answers: list[str] = []
    neutral_answers: list[str] = []
    index: list[tuple[int, str]] = []
    for item_idx, item in enumerate(items):
        variants = item.prompt_variants or [item.question]
        for variant in variants:
            prompts.append(variant)
            target_answers.append(item.answer)
            neutral_answers.append(item.neutral_answer or neutral_answer)
            index.append((item_idx, variant))
    logp_target = conditional_logprobs(model, tokenizer, prompts, target_answers, max_length=max_length)
    logp_neutral = conditional_logprobs(model, tokenizer, prompts, neutral_answers, max_length=max_length)
    margins = logp_target - logp_neutral
    return margins, index


def item_max_margins(model, tokenizer, items: list[ForgetItem], neutral_answer: str = "I don't know.", max_length: int = 256):
    import torch

    margins, index = target_margins(model, tokenizer, items, neutral_answer=neutral_answer, max_length=max_length)
    grouped: list[list[object]] = [[] for _ in items]
    for margin, (item_idx, _) in zip(margins, index):
        grouped[item_idx].append(margin)
    return torch.stack([torch.stack(values).max() for values in grouped])


def resurrection_softplus_loss(
    model,
    tokenizer,
    items: list[ForgetItem],
    thresholds: dict[str, float],
    gamma: float = 0.1,
    neutral_answer: str = "I don't know.",
    max_length: int = 256,
):
    import torch

    margins = item_max_margins(model, tokenizer, items, neutral_answer=neutral_answer, max_length=max_length)
    tau = torch.tensor([thresholds.get(item.item_id, 0.0) for item in items], dtype=margins.dtype, device=margins.device)
    return torch.nn.functional.softplus((margins - tau) / gamma).mean()


def margin_growth_softplus_from_margins(post_margins, baseline_margins, gamma: float = 0.1, target_growth: float = 0.0):
    import torch

    baseline = baseline_margins.to(dtype=post_margins.dtype, device=post_margins.device)
    return torch.nn.functional.softplus((post_margins - baseline - target_growth) / gamma).mean()


def margin_growth_softplus_loss(
    model,
    tokenizer,
    items: list[ForgetItem],
    baseline_margins,
    gamma: float = 0.1,
    target_growth: float = 0.0,
    neutral_answer: str = "I don't know.",
    max_length: int = 256,
):
    post_margins = item_max_margins(model, tokenizer, items, neutral_answer=neutral_answer, max_length=max_length)
    return margin_growth_softplus_from_margins(
        post_margins,
        baseline_margins,
        gamma=gamma,
        target_growth=target_growth,
    )


def target_margin_values(model, tokenizer, items: list[ForgetItem], neutral_answer: str = "I don't know.", max_length: int = 256) -> list[float]:
    # Eval/logging only: never needs gradients. Without no_grad the forward graph
    # for every prompt is retained until the end (OOM on large forget sets with a
    # large-vocab model). no_grad frees each chunk's activations immediately.
    import torch

    with torch.no_grad():
        margins = item_max_margins(model, tokenizer, items, neutral_answer=neutral_answer, max_length=max_length)
    return [float(value.detach().cpu().item()) for value in margins]
