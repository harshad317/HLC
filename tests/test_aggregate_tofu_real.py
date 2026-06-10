import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "aggregate_tofu_real",
    str(Path(__file__).resolve().parents[1] / "scripts" / "aggregate_tofu_real.py"),
)
agg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg_mod)


def test_h50_interpolation():
    steps = [0, 8, 16, 32]
    # crosses 0.5 between t=8 (0.1) and t=16 (0.9): 8 + (0.5-0.1)/(0.9-0.1)*8 = 12
    r = [0.0, 0.1, 0.9, 1.0]
    assert abs(agg_mod.h50_from_curve(steps, r) - 12.0) < 1e-6


def test_h50_already_resurrected_and_censored():
    steps = [0, 8, 16]
    assert agg_mod.h50_from_curve(steps, [0.6, 0.8, 1.0]) == 0.0          # already >=0.5 at t0
    assert agg_mod.h50_from_curve(steps, [0.0, 0.1, 0.2]) == 16.0         # never reaches 0.5 -> censored


def test_bootstrap_ci_basics():
    mean, lo, hi = agg_mod.bootstrap_ci([10.0, 12.0, 14.0], n_boot=2000, seed=1)
    assert abs(mean - 12.0) < 1e-9
    assert lo <= mean <= hi
    # single value -> degenerate CI
    assert agg_mod.bootstrap_ci([5.0]) == (5.0, 5.0, 5.0)
    # all NaN -> NaN
    m, l, h = agg_mod.bootstrap_ci([float("nan")])
    assert m != m


def _write_curve(path: Path, method: str, seed: int, r_by_t: dict):
    fields = ["method", "seed", "step_t", "resurrected_fraction",
              "conditional_resurrection_fraction", "conditional_auc_survival_so_far", "retain_nll"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t, r in r_by_t.items():
            w.writerow({"method": method, "seed": seed, "step_t": t,
                        "resurrected_fraction": r, "conditional_resurrection_fraction": r,
                        "conditional_auc_survival_so_far": 1 - r, "retain_nll": 2.3})


def test_aggregate_groups_by_method(tmp_path):
    # method A: durable (h50 ~32), method B: fragile (h50 ~12), two seeds each
    curve_durable = {0: 0.0, 8: 0.0, 16: 0.05, 32: 0.65, 64: 1.0, 512: 1.0}
    curve_fragile = {0: 0.0, 8: 0.1, 16: 0.9, 32: 1.0, 64: 1.0, 512: 1.0}
    _write_curve(tmp_path / "a_s0" / "survival_curve.csv", "GA", 0, curve_durable)
    _write_curve(tmp_path / "a_s1" / "survival_curve.csv", "GA", 1, curve_durable)
    _write_curve(tmp_path / "b_s0" / "survival_curve.csv", "HLC", 0, curve_fragile)
    _write_curve(tmp_path / "b_s1" / "survival_curve.csv", "HLC", 1, curve_fragile)

    agg = agg_mod.aggregate([str(p) for p in tmp_path.glob("*/survival_curve.csv")])
    assert agg["GA"]["n"] == 2 and agg["HLC"]["n"] == 2
    ga_h50 = agg["GA"]["h50"][0]
    hlc_h50 = agg["HLC"]["h50"][0]
    assert ga_h50 > hlc_h50  # GA more durable, as in the real result
    md = agg_mod.to_markdown(agg)
    assert "GA" in md and "HLC" in md and "95% bootstrap CI" in md
