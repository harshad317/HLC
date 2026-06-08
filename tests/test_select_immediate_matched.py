import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path("scripts/select_immediate_matched.py")
    spec = importlib.util.spec_from_file_location("select_immediate_matched", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_t0(path: Path, method: str, seed: int, r0: float, retain_nll: float, refusal: float = 0.0) -> None:
    pd.DataFrame(
        [
            {
                "method": method,
                "seed": seed,
                "relearn_corpus": "synthetic_biography",
                "step_t": 0,
                "resurrected_fraction": r0,
                "survival_fraction": 1.0 - r0,
                "retain_nll": retain_nll,
                "refusal_rate": refusal,
                "conditional_num_items": 2,
            }
        ]
    ).to_csv(path, index=False)


def test_select_matches_prioritizes_r0_tolerance(tmp_path):
    module = load_module()
    ref = tmp_path / "hlc.csv"
    matched = tmp_path / "matched.csv"
    far = tmp_path / "far.csv"
    write_t0(ref, "hlc_sg", seed=0, r0=0.05, retain_nll=3.6)
    write_t0(matched, "strong_npo_matched", seed=0, r0=0.08, retain_nll=3.5)
    write_t0(far, "strong_npo_far", seed=0, r0=0.30, retain_nll=3.4)

    all_rows, selected = module.select_matches(
        reference_paths=[ref],
        candidate_paths=[far, matched],
        r0_tolerance=0.05,
        retain_nll_band=0.25,
        max_refusal=0.05,
    )

    assert bool(all_rows[all_rows["method"] == "strong_npo_matched"].iloc[0]["matched"]) is True
    assert bool(all_rows[all_rows["method"] == "strong_npo_far"].iloc[0]["matched"]) is False
    assert selected.iloc[0]["method"] == "strong_npo_matched"
