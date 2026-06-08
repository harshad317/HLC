import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path("scripts/make_main_track_table.py")
    spec = importlib.util.spec_from_file_location("make_main_track_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_curve(path: Path, method: str, resurrected: float, auc: float, censored: bool) -> None:
    rows = [
        {
            "method": method,
            "seed": 0,
            "relearn_corpus": "syntax_matched",
            "threshold_quantile": "q95",
            "step_t": 0,
            "forget_accuracy": 0.0,
            "resurrected_fraction": resurrected,
            "survival_fraction": 1.0 - resurrected,
            "median_target_margin": -1.0,
            "mean_target_margin": -1.0,
            "retain_accuracy": 0.0,
            "retain_nll": 3.5,
            "general_nll": 3.5,
            "refusal_rate": 0.0,
            "num_items": 2,
            "num_prompt_variants": 2,
            "h50_observed": None,
            "h50_censored": censored,
            "auc_survival_so_far": auc,
        },
        {
            "method": method,
            "seed": 0,
            "relearn_corpus": "syntax_matched",
            "threshold_quantile": "q95",
            "step_t": 32,
            "forget_accuracy": 0.0,
            "resurrected_fraction": resurrected,
            "survival_fraction": 1.0 - resurrected,
            "median_target_margin": -1.0,
            "mean_target_margin": -1.0,
            "retain_accuracy": 0.0,
            "retain_nll": 3.5,
            "general_nll": 3.5,
            "refusal_rate": 0.0,
            "num_items": 2,
            "num_prompt_variants": 2,
            "h50_observed": None,
            "h50_censored": censored,
            "auc_survival_so_far": auc,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_table_summarizes_immediate_forget_and_censored_h50(tmp_path):
    module = load_module()
    curve = tmp_path / "hlc.csv"
    write_curve(curve, "hlc_sg_k8_seed0", resurrected=0.0, auc=1.0, censored=True)
    comparison = pd.DataFrame(
        [
            {
                "method": "hlc_sg_k8_seed0",
                "seed": 0,
                "relearn_corpus": "syntax_matched",
                "threshold_quantile": "q95",
                "path": str(curve),
                "h50": None,
                "h50_censored": True,
                "auc_survival": 1.0,
                "r_final": 0.0,
                "mean_target_margin_final": -1.0,
                "median_target_margin_final": -1.0,
                "retain_accuracy_final": 0.0,
                "retain_nll_final": 3.5,
                "refusal_rate_final": 0.0,
            }
        ]
    )
    comparison_path = tmp_path / "compare.csv"
    comparison.to_csv(comparison_path, index=False)
    table, _ = module.build_table(str(comparison_path), ["HLC-SG"], "syntax_matched")
    row = table.iloc[0]
    assert row["Immediate forget"] == "pass; exact=0.000, r0=0.000"
    assert row["h50"] == ">32 censored (1/1)"
    assert row["Held-out h50"] == ">32 censored (1/1)"
