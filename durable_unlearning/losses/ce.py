from __future__ import annotations

from durable_unlearning.utils.tokenization import ensure_pad_token


def lm_batch(tokenizer, texts: list[str], device: str, max_length: int = 256):
    ensure_pad_token(tokenizer)
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def ce_loss(model, tokenizer, texts: list[str], max_length: int = 256):
    device = next(model.parameters()).device
    batch = lm_batch(tokenizer, texts, str(device), max_length=max_length)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    outputs = model(**batch, labels=labels)
    return outputs.loss


def conditional_logprobs(model, tokenizer, prompts: list[str], targets: list[str], max_length: int = 256):
    import torch

    device = next(model.parameters()).device
    ensure_pad_token(tokenizer)
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
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = (token_log_probs * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp_min(1)
    return seq_log_probs / token_counts
