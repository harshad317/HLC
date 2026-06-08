from __future__ import annotations

from durable_unlearning.data.types import ForgetItem
from durable_unlearning.eval.semantic_match import exact_or_alias_match, refusal_match


def aggregate_generation_utility(rows: list[dict[str, object]], retain_items: list[ForgetItem]) -> dict[str, float]:
    if not rows:
        return {"retain_accuracy": 0.0, "retain_nll": 0.0, "general_nll": 0.0, "refusal_rate": 0.0}
    by_id = {item.item_id: item for item in retain_items}
    correct = []
    refusal = []
    nll = []
    for row in rows:
        item = by_id.get(str(row.get("item_id")))
        generation = str(row.get("generation", ""))
        if item is not None:
            correct.append(exact_or_alias_match(generation, item.answer, item.aliases))
            refusal.append(refusal_match(generation, item.neutral_answer))
        if row.get("nll") is not None:
            nll.append(float(row["nll"]))
    return {
        "retain_accuracy": sum(correct) / len(correct) if correct else 0.0,
        "retain_nll": sum(nll) / len(nll) if nll else 0.0,
        "general_nll": sum(nll) / len(nll) if nll else 0.0,
        "refusal_rate": sum(refusal) / len(refusal) if refusal else 0.0,
    }
