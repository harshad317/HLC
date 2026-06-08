import importlib.util
from pathlib import Path

import pandas as pd


def load_report_module():
    path = Path("scripts/report_local_mvp.py")
    spec = importlib.util.spec_from_file_location("report_local_mvp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_comparison_groups_by_family_and_pool():
    report = load_report_module()
    df = pd.DataFrame(
        [
            {
                "method": "hlc_sg_k8_seed0",
                "seed": 0,
                "relearn_corpus": "retain_adjacent",
                "threshold_quantile": "q95",
                "auc_survival": 1.0,
                "r_final": 0.0,
                "mean_target_margin_final": -1.0,
                "median_target_margin_final": -1.2,
                "retain_nll_final": 3.6,
            },
            {
                "method": "npo_seed0",
                "seed": 0,
                "relearn_corpus": "retain_adjacent",
                "threshold_quantile": "q95",
                "auc_survival": 0.7,
                "r_final": 0.3,
                "mean_target_margin_final": -0.1,
                "median_target_margin_final": -0.3,
                "retain_nll_final": 3.4,
            },
        ]
    )
    df["family"] = df["method"].map(report.family_from_method)
    by_pool, overall = report.aggregate_comparison(df)
    assert set(by_pool["family"]) == {"HLC-SG", "NPO"}
    assert set(overall["family"]) == {"HLC-SG", "NPO"}
    assert by_pool[by_pool["family"] == "HLC-SG"].iloc[0]["auc_mean"] == 1.0


def test_family_from_method_keeps_main_track_baselines_separate():
    report = load_report_module()
    assert (
        report.family_from_method("hlc_sg_stress_sweep_lk3_lr1_seed0")
        == "hlc_sg_stress_sweep_lk3_lr1"
    )
    assert report.family_from_method("hlc_sg_k8_stress_fullft_tofu50_seed0") == "HLC-SG stress-trained"
    assert report.family_from_method("hlc_sg_k8_tuned_fullft_tofu50_seed0") == "HLC-SG"
    assert report.family_from_method("hlc_sg_k1_gpt2_lora_tofu50_seed0") == "HLC-SG K1 GPT-2"
    assert report.family_from_method("hlc_sg_k2_gpt2_lora_tofu50_seed0") == "HLC-SG K2 GPT-2"
    assert report.family_from_method("npo_strong_fullft_tofu50_seed0") == "Strong NPO"
    assert report.family_from_method("npo_paraphrase_fullft_tofu50_seed0") == "NPO + paraphrases"
    assert (
        report.family_from_method("syntactic_diversification_fullft_tofu50_seed0")
        == "Syntactic diversification"
    )
    assert report.family_from_method("npo_fullft_tofu50_seed0") == "NPO"
