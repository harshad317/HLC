from __future__ import annotations

from durable_unlearning.data.types import ForgetItem, RelearnExample
from durable_unlearning.eval.semantic_match import exact_or_alias_match, refusal_match
from durable_unlearning.losses.ce import ce_loss
from durable_unlearning.losses.resurrection import target_margin_values
from durable_unlearning.methods.base import relearn_texts, retain_texts


class ModelScorer:
    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 256,
        max_new_tokens: int = 24,
        neutral_answer: str = "I don't know.",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.neutral_answer = neutral_answer

    @staticmethod
    def generation_prompt(item: ForgetItem) -> str:
        for prompt in item.prompt_variants:
            if "Answer:" in prompt:
                return prompt
        return f"Question: {item.question}\nAnswer:"

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        device = next(self.model.parameters()).device
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            batch = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        batch = {key: value.to(device) for key, value in batch.items()}
        input_width = int(batch["input_ids"].shape[1])
        with torch.no_grad():
            generated = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        outputs = []
        for tokens in generated:
            outputs.append(self.tokenizer.decode(tokens[input_width:], skip_special_tokens=True).strip())
        return outputs

    def score_forget_items(self, forget_items: list[ForgetItem]) -> list[dict[str, object]]:
        self.model.eval()
        margins = target_margin_values(
            self.model,
            self.tokenizer,
            forget_items,
            neutral_answer=self.neutral_answer,
            max_length=self.max_length,
        )
        generations = self.generate([self.generation_prompt(item) for item in forget_items])
        rows: list[dict[str, object]] = []
        for item, generation, margin in zip(forget_items, generations, margins):
            rows.append(
                {
                    "item_id": item.item_id,
                    "question": item.question,
                    "answer": item.answer,
                    "generation": generation,
                    "target_margin": margin,
                    "semantic_correct": exact_or_alias_match(generation, item.answer, item.aliases),
                    "refusal": refusal_match(generation, item.neutral_answer),
                    "num_prompt_variants": max(1, len(item.prompt_variants)),
                }
            )
        return rows

    def score_retain_items(self, retain_items: list[ForgetItem]) -> dict[str, float]:
        self.model.eval()
        if not retain_items:
            return {"retain_accuracy": 0.0, "retain_nll": 0.0, "general_nll": 0.0, "refusal_rate": 0.0}
        generations = self.generate([self.generation_prompt(item) for item in retain_items])
        correct = [exact_or_alias_match(gen, item.answer, item.aliases) for gen, item in zip(generations, retain_items)]
        refusals = [refusal_match(gen, item.neutral_answer) for gen, item in zip(generations, retain_items)]
        try:
            nll = float(ce_loss(self.model, self.tokenizer, retain_texts(retain_items), max_length=self.max_length).detach().cpu().item())
        except Exception:
            nll = 0.0
        return {
            "retain_accuracy": sum(correct) / len(correct),
            "retain_nll": nll,
            "general_nll": nll,
            "refusal_rate": sum(refusals) / len(refusals),
        }

    def relearn(self, relearn_items: list[RelearnExample], steps: int, lr: float = 2e-5, batch_size: int = 2, max_grad_norm: float = 1.0) -> None:
        import torch

        if steps <= 0:
            return
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr)
        texts = relearn_texts(relearn_items)
        if not texts:
            return
        idx = 0
        for _ in range(steps):
            batch_texts = [texts[(idx + offset) % len(texts)] for offset in range(batch_size)]
            idx += batch_size
            loss = ce_loss(self.model, self.tokenizer, batch_texts, max_length=self.max_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
