import importlib.util
from pathlib import Path

import pandas as pd

from durable_unlearning.logging.schemas import SURVIVAL_FIELDS


def load_compare_module():
    path = Path("scripts/compare_methods.py")
    spec = importlib.util.spec_from_file_location("compare_methods", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_survival_csv(path, method, auc, resurrected, mean_margin, median_margin):
    row = {field: 0 for field in SURVIVAL_FIELDS}
    row.update(
        {
            "method": method,
            "seed": 0,
            "relearn_corpus": "retain_adjacent",
            "threshold_quantile": "q95",
            "step_t": 32,
            "resurrected_fraction": resurrected,
            "relearn_growth": resurrected,
            "survival_decay": resurrected,
            "conditional_resurrection_fraction": resurrected,
            "conditional_survival_fraction": 1.0 - resurrected,
            "conditional_auc_survival_so_far": auc,
            "conditional_num_items": 2,
            "auc_survival_so_far": auc,
            "mean_target_margin": mean_margin,
            "median_target_margin": median_margin,
            "mean_target_margin_growth": mean_margin,
            "conditional_mean_target_margin_growth": mean_margin,
            "retain_nll": 3.5,
            "h50_censored": True,
        }
    )
    pd.DataFrame([row]).to_csv(path, index=False)


def test_compare_includes_final_margin_columns(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    write_survival_csv(a, "hlc", 1.0, 0.0, -1.5, -1.8)
    write_survival_csv(b, "npo", 0.8, 0.2, -0.2, -0.4)
    compare_methods = load_compare_module()
    out = compare_methods.compare_files([str(a), str(b)])
    assert "mean_target_margin_final" in out.columns
    assert "median_target_margin_final" in out.columns
    assert "r0" in out.columns
    assert "relearn_growth" in out.columns
    assert "conditional_resurrection_final" in out.columns
    assert "conditional_auc_survival" in out.columns
    hlc = out[out["method"] == "hlc"].iloc[0]
    assert hlc["mean_target_margin_final"] == -1.5
    assert hlc["median_target_margin_final"] == -1.8
    assert hlc["conditional_resurrection_final"] == 0.0
