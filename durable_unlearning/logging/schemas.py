from __future__ import annotations

SURVIVAL_FIELDS = [
    "method",
    "seed",
    "relearn_corpus",
    "threshold_quantile",
    "step_t",
    "forget_accuracy",
    "resurrected_fraction",
    "survival_fraction",
    "relearn_growth",
    "survival_decay",
    "conditional_resurrection_fraction",
    "conditional_survival_fraction",
    "conditional_auc_survival_so_far",
    "conditional_num_items",
    "median_target_margin",
    "mean_target_margin",
    "mean_target_margin_growth",
    "conditional_mean_target_margin_growth",
    "retain_accuracy",
    "retain_nll",
    "general_nll",
    "refusal_rate",
    "num_items",
    "num_prompt_variants",
    "h50_observed",
    "h50_censored",
    "auc_survival_so_far",
]

TRAIN_FIELDS = [
    "step",
    "method",
    "seed",
    "loss_forget",
    "loss_retain",
    "loss_kl",
    "loss_resurrect",
    "loss_margin_growth",
    "loss_post",
    "target_margin_forget",
    "retain_nll",
    "grad_norm_now",
    "grad_norm_post",
    "grad_cos_now_post",
    "lr_outer",
    "lr_inner",
    "gpu_mem_gb",
    "wall_clock_sec",
]


def validate_fields(row: dict[str, object], fields: list[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
