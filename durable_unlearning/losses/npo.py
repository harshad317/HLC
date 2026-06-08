from __future__ import annotations

from durable_unlearning.data.types import ForgetItem
from durable_unlearning.losses.ce import conditional_logprobs


def npo_loss(model, tokenizer, items: list[ForgetItem], beta: float = 0.1, max_length: int = 256):
    import torch

    prompts = [item.question for item in items]
    targets = [item.answer for item in items]
    logp_forget = conditional_logprobs(model, tokenizer, prompts, targets, max_length=max_length)
    return -torch.nn.functional.logsigmoid(-beta * logp_forget).mean()
