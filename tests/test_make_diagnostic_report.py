import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path("scripts/make_diagnostic_report.py")
    spec = importlib.util.spec_from_file_location("make_diagnostic_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_report_combines_main_track_and_stress_tables(tmp_path):
    module = load_module()
    main_track = pd.DataFrame(
        [
            {
                "Method": "Strong NPO",
                "Immediate forget": "pass; exact=0.000, r0=0.222",
                "Retain utility": "retain NLL=3.4",
                "Refusal": "0.000",
                "h50": ">32 censored",
                "AUC-survival": "0.778 +/- 0.083 (n=9)",
                "Held-out h50": ">32 censored",
            },
            {
                "Method": "HLC-SG",
                "Immediate forget": "pass; exact=0.000, r0=0.056",
                "Retain utility": "retain NLL=3.6",
                "Refusal": "0.000",
                "h50": ">32 censored",
                "AUC-survival": "0.944 +/- 0.083 (n=9)",
                "Held-out h50": ">32 censored",
            },
        ]
    )
    stress = pd.DataFrame(
        [
            {
                "family": "Strong NPO",
                "n": 9,
                "r0": 0.056,
                "cond_res_32": 0.111,
                "cond_res_128": 0.426,
                "cond_res_512": 0.907,
                "cond_auc_512": 0.362,
                "retain_nll_512": 7.209,
                "immediate_matched": True,
            },
            {
                "family": "HLC-SG",
                "n": 9,
                "r0": 0.056,
                "cond_res_32": 0.093,
                "cond_res_128": 0.981,
                "cond_res_512": 1.0,
                "cond_auc_512": 0.195,
                "retain_nll_512": 7.218,
                "immediate_matched": True,
            },
        ]
    )
    main_path = tmp_path / "main.csv"
    stress_path = tmp_path / "stress.csv"
    decision_path = tmp_path / "decision.md"
    main_track.to_csv(main_path, index=False)
    stress.to_csv(stress_path, index=False)
    decision_path.write_text("# Decision\n", encoding="utf-8")

    outputs = module.write_report(main_path, stress_path, decision_path, tmp_path / "out")

    summary = outputs["summary"].read_text(encoding="utf-8")
    assert "The benign main-track table makes `HLC-SG` look strongest" in summary
    assert "Freeze HLC tuning" in summary
    assert outputs["main_track_table"].exists()
    assert outputs["stress_table"].exists()
