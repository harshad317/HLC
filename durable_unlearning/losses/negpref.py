from __future__ import annotations

from durable_unlearning.data.types import ForgetItem
from durable_unlearning.losses.ce import conditional_logprobs


def negpref_loss(model, tokenizer, items: list[ForgetItem], neutral_answer: str = "I don't know.", beta: float = 0.1, max_length: int = 256):
    import torch

    prompts = [item.question for item in items]
    forget_targets = [item.answer for item in items]
    neutral_targets = [item.neutral_answer or neutral_answer for item in items]
    logp_forget = conditional_logprobs(model, tokenizer, prompts, forget_targets, max_length=max_length)
    logp_neutral = conditional_logprobs(model, tokenizer, prompts, neutral_targets, max_length=max_length)
    return -torch.nn.functional.logsigmoid(beta * (logp_neutral - logp_forget)).mean()
