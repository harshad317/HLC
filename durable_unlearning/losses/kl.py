from __future__ import annotations

from durable_unlearning.losses.ce import lm_batch


def topk_kl_loss(model, reference_model, tokenizer, texts: list[str], k: int = 128, max_length: int = 256):
    import torch

    device = next(model.parameters()).device
    batch = lm_batch(tokenizer, texts, str(device), max_length=max_length)
    logits = model(**batch).logits
    with torch.no_grad():
        ref_logits = reference_model(**batch).logits
    k = min(k, logits.shape[-1])
    ref_values, ref_idx = torch.topk(ref_logits, k=k, dim=-1)
    student_selected = logits.gather(-1, ref_idx)
    ref_probs = torch.nn.functional.softmax(ref_values, dim=-1)
    student_log_probs = torch.nn.functional.log_softmax(student_selected, dim=-1)
    ref_log_probs = torch.nn.functional.log_softmax(ref_values, dim=-1)
    kl = ref_probs * (ref_log_probs - student_log_probs)
    mask = batch["attention_mask"].unsqueeze(-1)
    return (kl * mask).sum() / mask.sum().clamp_min(1)
