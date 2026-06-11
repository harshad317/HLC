import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "frontier_table",
    str(Path(__file__).resolve().parents[1] / "scripts" / "frontier_table.py"),
)
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def _write(path: Path, method: str, r_by_t: dict, retain_nll=2.5):
    fields = ["method", "seed", "step_t", "resurrected_fraction",
              "conditional_resurrection_fraction", "conditional_auc_survival_so_far", "retain_nll"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t, r in r_by_t.items():
            w.writerow({"method": method, "seed": 0, "step_t": t, "resurrected_fraction": r,
                        "conditional_resurrection_fraction": r, "conditional_auc_survival_so_far": 1 - r,
                        "retain_nll": retain_nll})


def test_family_parsing():
    assert ft.family_of("disp_lf3") == "disp"
    assert ft.family_of("ga_rw0p5") == "ga"
    assert ft.family_of("hlc") == "hlc"


def test_domination_detection(tmp_path):
    # displacement: durable (h50~32) AND cheap (retain 3.0)
    _write(tmp_path / "d/survival_curve.csv", "disp_lf3",
           {0: 0, 8: 0, 16: 0.03, 32: 0.5, 64: 1.0}, retain_nll=3.0)
    # GA: durable-ish (h50~28) but expensive (retain 3.8) -> should be dominated
    _write(tmp_path / "g/survival_curve.csv", "ga_rw1",
           {0: 0, 8: 0, 16: 0.05, 32: 0.65, 64: 1.0}, retain_nll=3.8)

    points = ft.load_points([str(p) for p in tmp_path.glob("*/survival_curve.csv")])
    dom = ft.dominated_flags(points)
    assert dom["ga_rw1"] is True          # disp is cheaper AND >= as durable
    assert dom["disp_lf3"] is False
    md = ft.to_markdown(points)
    assert "frontier" in md.lower() and "ga_rw1" in md


def test_non_domination_when_ga_cheaper(tmp_path):
    # GA cheaper than the only disp point -> not dominated
    _write(tmp_path / "d/survival_curve.csv", "disp_lf3",
           {0: 0, 8: 0, 16: 0.03, 32: 0.5, 64: 1.0}, retain_nll=3.0)
    _write(tmp_path / "g/survival_curve.csv", "ga_rw4",
           {0: 0, 8: 0, 16: 0.05, 32: 0.65, 64: 1.0}, retain_nll=2.5)
    points = ft.load_points([str(p) for p in tmp_path.glob("*/survival_curve.csv")])
    dom = ft.dominated_flags(points)
    assert dom["ga_rw4"] is False
