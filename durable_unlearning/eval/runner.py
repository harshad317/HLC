from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from durable_unlearning.data.io import read_json, read_jsonl
from durable_unlearning.data.types import ForgetItem, RelearnExample
from durable_unlearning.eval.model_scorer import ModelScorer
from durable_unlearning.eval.survival import survival_records_from_scores, write_survival_artifacts
from durable_unlearning.eval.thresholding import load_thresholds
from durable_unlearning.models.load import load_causal_lm_with_lora


def attach_prompt_variants(items: list[ForgetItem], prompt_variant_file: Optional[Union[str, Path]]) -> list[ForgetItem]:
    if not prompt_variant_file:
        return items
    rows = read_jsonl(prompt_variant_file)
    by_item: dict[str, list[str]] = {}
    for row in rows:
        by_item.setdefault(str(row["item_id"]), []).append(str(row["prompt"]))
    return [
        ForgetItem(
            item_id=item.item_id,
            person_id=item.person_id,
            question=item.question,
            answer=item.answer,
            aliases=item.aliases,
            neutral_answer=item.neutral_answer,
            prompt_variants=by_item.get(item.item_id, item.prompt_variants),
            split=item.split,
        )
        for item in items
    ]


def run_survival_curve(
    checkpoint: Union[str, Path],
    thresholds_file: Union[str, Path],
    forget_file: Union[str, Path],
    retain_file: Union[str, Path],
    relearn_pool: Union[str, Path],
    steps: list[int],
    output: Union[str, Path],
    base_model: str = "sshleifer/tiny-gpt2",
    prompt_variants: Optional[Union[str, Path]] = None,
    threshold_quantile: str = "q95",
    method: str = "",
    seed: int = 0,
    lr: float = 2e-5,
    batch_size: int = 2,
    max_length: int = 256,
    max_new_tokens: int = 24,
    resurrection_metric: str = "margin",
    dtype: str = "float32",
    device: Optional[str] = None,
):
    model, tokenizer = load_causal_lm_with_lora(
        model_name=base_model,
        lora_cfg={"enabled": False},
        checkpoint_path=checkpoint,
        dtype=dtype,
        device=device,
    )
    if not method:
        method = Path(checkpoint).name
    forget_items = attach_prompt_variants([ForgetItem.from_dict(row) for row in read_jsonl(forget_file)], prompt_variants)
    retain_items = [ForgetItem.from_dict(row) for row in read_jsonl(retain_file)]
    relearn_items = [RelearnExample.from_dict(row) for row in read_jsonl(relearn_pool)]
    thresholds = load_thresholds(thresholds_file, quantile=threshold_quantile)
    scorer = ModelScorer(model, tokenizer, max_length=max_length, max_new_tokens=max_new_tokens)
    per_step_scores: list[list[dict[str, object]]] = []
    retain_scores: list[dict[str, float]] = []
    prev_step = 0
    for step in steps:
        delta = int(step) - prev_step
        if delta < 0:
            raise ValueError("Survival steps must be sorted ascending")
        scorer.relearn(relearn_items, delta, lr=lr, batch_size=batch_size)
        scores = scorer.score_forget_items(forget_items)
        for row in scores:
            if resurrection_metric == "margin":
                row["resurrected"] = float(row["target_margin"]) > thresholds.get(str(row["item_id"]), 0.0)
            elif resurrection_metric == "margin_and_exact":
                row["resurrected"] = float(row["target_margin"]) > thresholds.get(str(row["item_id"]), 0.0) and bool(row["semantic_correct"])
            else:
                raise ValueError(f"Unsupported resurrection_metric: {resurrection_metric}")
        per_step_scores.append(scores)
        retain_scores.append(scorer.score_retain_items(retain_items))
        prev_step = int(step)
    records = survival_records_from_scores(
        steps=steps,
        per_step_scores=per_step_scores,
        thresholds=thresholds,
        retain_scores=retain_scores,
        threshold_quantile=threshold_quantile,
        method=method,
        seed=seed,
        relearn_corpus=Path(relearn_pool).stem,
        resurrection_metric=resurrection_metric,
    )
    write_survival_artifacts(output, records, per_step_scores)
    return records
