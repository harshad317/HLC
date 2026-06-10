from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Protocol, Union

import numpy as np

from durable_unlearning.data.io import write_jsonl
from durable_unlearning.data.types import ForgetItem, SurvivalRecord
from durable_unlearning.eval.metrics import auc_survival, h50, running_auc
from durable_unlearning.eval.semantic_match import exact_or_alias_match, refusal_match
from durable_unlearning.logging.csv_logger import write_csv
from durable_unlearning.logging.schemas import SURVIVAL_FIELDS


class Scorer(Protocol):
    def score_forget_items(self, forget_items: list[ForgetItem]) -> list[dict[str, object]]:
        ...

    def score_retain_items(self, retain_items: list[ForgetItem]) -> dict[str, float]:
        ...

    def relearn(self, steps: int) -> None:
        ...


def resurrected_from_score(row: dict[str, object], threshold: float, resurrection_metric: str = "margin") -> bool:
    margin = float(row.get("target_margin", row.get("margin", 0.0)))
    if not math.isfinite(margin):
        item_id = row.get("item_id", "<unknown>")
        raise ValueError(f"Non-finite target margin for item {item_id}: {margin}")
    semantic_correct = bool(row.get("semantic_correct", False))
    if resurrection_metric == "margin":
        return margin > threshold
    if resurrection_metric == "margin_and_exact":
        return margin > threshold and semantic_correct
    raise ValueError(f"Unsupported resurrection_metric: {resurrection_metric}")


def survival_records_from_scores(
    steps: Sequence[int],
    per_step_scores: list[list[dict[str, object]]],
    thresholds: dict[str, float],
    retain_scores: list[dict[str, float]],
    threshold_quantile: str = "q95",
    method: str = "",
    seed: int = 0,
    relearn_corpus: str = "",
    resurrection_metric: str = "margin",
) -> list[SurvivalRecord]:
    baseline_scores = per_step_scores[0] if per_step_scores else []
    baseline_resurrected_by_id = {
        str(row["item_id"]): resurrected_from_score(
            row,
            thresholds.get(str(row["item_id"]), 0.0),
            resurrection_metric=resurrection_metric,
        )
        for row in baseline_scores
    }
    baseline_margin_by_id = {
        str(row["item_id"]): float(row.get("target_margin", row.get("margin", 0.0)))
        for row in baseline_scores
    }
    baseline_forgotten_ids = [
        item_id for item_id, resurrected in baseline_resurrected_by_id.items() if not resurrected
    ]
    baseline_resurrected_values = list(baseline_resurrected_by_id.values())
    baseline_resurrected_fraction = (
        float(np.mean(baseline_resurrected_values)) if baseline_resurrected_values else 0.0
    )
    baseline_survival_fraction = 1.0 - baseline_resurrected_fraction
    survival_values: list[float] = []
    conditional_survival_values: list[float] = []
    raw_rows: list[dict[str, object]] = []
    for idx, score_rows in enumerate(per_step_scores):
        for score_row in score_rows:
            margin = float(score_row.get("target_margin", score_row.get("margin", 0.0)))
            if not math.isfinite(margin):
                item_id = score_row.get("item_id", "<unknown>")
                raise ValueError(f"Non-finite target margin at step {steps[idx]} for item {item_id}: {margin}")
        resurrected_by_id = {
            str(row["item_id"]): resurrected_from_score(
                row,
                thresholds.get(str(row["item_id"]), 0.0),
                resurrection_metric=resurrection_metric,
            )
            for row in score_rows
        }
        margin_by_id = {
            str(row["item_id"]): float(row.get("target_margin", row.get("margin", 0.0)))
            for row in score_rows
        }
        resurrected = [
            resurrected_by_id[str(row["item_id"])]
            for row in score_rows
        ]
        margins = [float(row.get("target_margin", row.get("margin", 0.0))) for row in score_rows]
        exacts = [bool(row.get("semantic_correct", False)) for row in score_rows]
        resurrected_fraction = float(np.mean(resurrected)) if resurrected else 0.0
        survival_fraction = 1.0 - resurrected_fraction
        conditional_resurrected = [
            bool(resurrected_by_id.get(item_id, False))
            for item_id in baseline_forgotten_ids
        ]
        conditional_resurrection_fraction = (
            float(np.mean(conditional_resurrected)) if conditional_resurrected else 0.0
        )
        conditional_survival_fraction = 1.0 - conditional_resurrection_fraction
        margin_deltas = [
            margin_by_id[item_id] - baseline_margin
            for item_id, baseline_margin in baseline_margin_by_id.items()
            if item_id in margin_by_id
        ]
        conditional_margin_deltas = [
            margin_by_id[item_id] - baseline_margin_by_id[item_id]
            for item_id in baseline_forgotten_ids
            if item_id in margin_by_id
        ]
        survival_values.append(survival_fraction)
        conditional_survival_values.append(conditional_survival_fraction)
        retain = retain_scores[idx] if idx < len(retain_scores) else {}
        retain_nll = float(retain.get("retain_nll", 0.0))
        general_nll = float(retain.get("general_nll", retain_nll))
        if not math.isfinite(retain_nll):
            raise ValueError(f"Non-finite retain_nll at step {steps[idx]}: {retain_nll}")
        if not math.isfinite(general_nll):
            raise ValueError(f"Non-finite general_nll at step {steps[idx]}: {general_nll}")
        row = {
            "step_t": int(steps[idx]),
            "forget_accuracy": float(np.mean(exacts)) if exacts else 0.0,
            "resurrected_fraction": resurrected_fraction,
            "survival_fraction": survival_fraction,
            "relearn_growth": resurrected_fraction - baseline_resurrected_fraction,
            "survival_decay": baseline_survival_fraction - survival_fraction,
            "conditional_resurrection_fraction": conditional_resurrection_fraction,
            "conditional_survival_fraction": conditional_survival_fraction,
            "conditional_num_items": len(baseline_forgotten_ids),
            "median_target_margin": float(np.median(margins)) if margins else 0.0,
            "mean_target_margin": float(np.mean(margins)) if margins else 0.0,
            "mean_target_margin_growth": float(np.mean(margin_deltas)) if margin_deltas else 0.0,
            "conditional_mean_target_margin_growth": (
                float(np.mean(conditional_margin_deltas)) if conditional_margin_deltas else 0.0
            ),
            "retain_accuracy": float(retain.get("retain_accuracy", 0.0)),
            "retain_nll": retain_nll,
            "general_nll": general_nll,
            "refusal_rate": float(retain.get("refusal_rate", 0.0)),
            "num_items": len(score_rows),
            "num_prompt_variants": int(sum(int(row.get("num_prompt_variants", 1)) for row in score_rows)),
            "method": method,
            "seed": seed,
            "relearn_corpus": relearn_corpus,
            "threshold_quantile": threshold_quantile,
        }
        raw_rows.append(row)
    auc_values = running_auc(steps, survival_values)
    conditional_auc_values = running_auc(steps, conditional_survival_values)
    observed, censored = h50(steps, survival_values)
    records: list[SurvivalRecord] = []
    for idx, row in enumerate(raw_rows):
        prefix_observed, prefix_censored = h50(steps[: idx + 1], survival_values[: idx + 1])
        records.append(
            SurvivalRecord(
                step_t=int(row["step_t"]),
                forget_accuracy=float(row["forget_accuracy"]),
                resurrected_fraction=float(row["resurrected_fraction"]),
                survival_fraction=float(row["survival_fraction"]),
                relearn_growth=float(row["relearn_growth"]),
                survival_decay=float(row["survival_decay"]),
                conditional_resurrection_fraction=float(row["conditional_resurrection_fraction"]),
                conditional_survival_fraction=float(row["conditional_survival_fraction"]),
                conditional_auc_survival_so_far=float(conditional_auc_values[idx]),
                conditional_num_items=int(row["conditional_num_items"]),
                median_target_margin=float(row["median_target_margin"]),
                mean_target_margin=float(row["mean_target_margin"]),
                mean_target_margin_growth=float(row["mean_target_margin_growth"]),
                conditional_mean_target_margin_growth=float(row["conditional_mean_target_margin_growth"]),
                retain_accuracy=float(row["retain_accuracy"]),
                retain_nll=float(row["retain_nll"]),
                general_nll=float(row["general_nll"]),
                refusal_rate=float(row["refusal_rate"]),
                num_items=int(row["num_items"]),
                num_prompt_variants=int(row["num_prompt_variants"]),
                h50_observed=prefix_observed if not prefix_censored else observed,
                h50_censored=bool(prefix_censored and censored),
                auc_survival_so_far=float(auc_values[idx]),
                method=method,
                seed=seed,
                relearn_corpus=relearn_corpus,
                threshold_quantile=threshold_quantile,
            )
        )
    return records


def write_survival_artifacts(
    output_dir: Union[str, Path],
    records: list[SurvivalRecord],
    per_step_scores: list[list[dict[str, object]]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "survival_curve.csv", [record.to_dict() for record in records], SURVIVAL_FIELDS)
    retain_rows = [
        {
            "step_t": record.step_t,
            "retain_accuracy": record.retain_accuracy,
            "retain_nll": record.retain_nll,
            "general_nll": record.general_nll,
            "refusal_rate": record.refusal_rate,
        }
        for record in records
    ]
    write_csv(output_dir / "retain_utility.csv", retain_rows)
    examples: list[dict[str, object]] = []
    for record, score_rows in zip(records, per_step_scores):
        write_jsonl(output_dir / f"generations_t{record.step_t:03d}.jsonl", score_rows)
        for row in score_rows:
            if bool(row.get("resurrected", False)):
                examples.append({"step_t": record.step_t, **row})
    write_jsonl(output_dir / "resurrection_examples.jsonl", examples)


def score_text_generation(item: ForgetItem, generation: str, target_margin: float, threshold: float) -> dict[str, object]:
    semantic_correct = exact_or_alias_match(generation, item.answer, item.aliases)
    resurrected = target_margin > threshold and semantic_correct
    return {
        "item_id": item.item_id,
        "question": item.question,
        "answer": item.answer,
        "generation": generation,
        "target_margin": target_margin,
        "semantic_correct": semantic_correct,
        "resurrected": resurrected,
        "refusal": refusal_match(generation, item.neutral_answer),
        "num_prompt_variants": max(1, len(item.prompt_variants)),
    }
