from __future__ import annotations

from durable_unlearning.eval.semantic_match import refusal_match


def refusal_rate(generations: list[str], neutral_answer: str = "I don't know.") -> float:
    if not generations:
        return 0.0
    return sum(refusal_match(text, neutral_answer) for text in generations) / len(generations)
