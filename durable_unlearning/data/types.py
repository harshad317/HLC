from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ForgetItem:
    item_id: str
    question: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    neutral_answer: str = "I don't know."
    prompt_variants: list[str] = field(default_factory=list)
    person_id: Optional[str] = None
    split: str = "forget"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ForgetItem":
        return cls(
            item_id=str(row.get("item_id") or row.get("id")),
            person_id=row.get("person_id"),
            question=str(row["question"]),
            answer=str(row["answer"]),
            aliases=list(row.get("aliases") or []),
            neutral_answer=str(row.get("neutral_answer", "I don't know.")),
            prompt_variants=list(row.get("prompt_variants") or []),
            split=str(row.get("split", "forget")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "person_id": self.person_id,
            "question": self.question,
            "answer": self.answer,
            "aliases": self.aliases,
            "neutral_answer": self.neutral_answer,
            "prompt_variants": self.prompt_variants,
            "split": self.split,
        }


@dataclass(frozen=True)
class RelearnExample:
    example_id: str
    text: str
    source: str
    license: str = "synthetic"
    relatedness: str = "retain_adjacent"
    template_id: Optional[str] = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RelearnExample":
        return cls(
            example_id=str(row.get("example_id") or row.get("id")),
            text=str(row["text"]),
            source=str(row.get("source", "unknown")),
            license=str(row.get("license", "unknown")),
            relatedness=str(row.get("relatedness", "unknown")),
            template_id=row.get("template_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "source": self.source,
            "license": self.license,
            "relatedness": self.relatedness,
            "template_id": self.template_id,
        }


@dataclass(frozen=True)
class ThresholdRecord:
    item_id: str
    tau_q90: float
    tau_q95: float
    tau_q99: float
    tau_global: float
    selected_tau: float
    oracle_model_hash: str = ""
    prompt_variant_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ThresholdRecord":
        return cls(
            item_id=str(row["item_id"]),
            tau_q90=float(row["tau_q90"]),
            tau_q95=float(row["tau_q95"]),
            tau_q99=float(row["tau_q99"]),
            tau_global=float(row.get("tau_global", 0.0)),
            selected_tau=float(row.get("selected_tau", row["tau_q95"])),
            oracle_model_hash=str(row.get("oracle_model_hash", "")),
            prompt_variant_ids=list(row.get("prompt_variant_ids") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "tau_q90": self.tau_q90,
            "tau_q95": self.tau_q95,
            "tau_q99": self.tau_q99,
            "tau_global": self.tau_global,
            "selected_tau": self.selected_tau,
            "oracle_model_hash": self.oracle_model_hash,
            "prompt_variant_ids": self.prompt_variant_ids,
        }


@dataclass(frozen=True)
class SurvivalRecord:
    step_t: int
    forget_accuracy: float
    resurrected_fraction: float
    survival_fraction: float
    relearn_growth: float
    survival_decay: float
    conditional_resurrection_fraction: float
    conditional_survival_fraction: float
    conditional_auc_survival_so_far: float
    conditional_num_items: int
    median_target_margin: float
    mean_target_margin: float
    mean_target_margin_growth: float
    conditional_mean_target_margin_growth: float
    retain_accuracy: float
    retain_nll: float
    general_nll: float
    refusal_rate: float
    num_items: int
    num_prompt_variants: int
    h50_observed: Optional[int]
    h50_censored: bool
    auc_survival_so_far: float
    method: str = ""
    seed: int = 0
    relearn_corpus: str = ""
    threshold_quantile: str = "q95"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
