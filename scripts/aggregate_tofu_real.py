#!/usr/bin/env python3
"""Aggregate real-TOFU survival curves across seeds with bootstrap CIs.

Reads one or more ``survival_curve.csv`` files (each is one method x one seed),
groups by method, and reports per-method mean and 95% bootstrap confidence
intervals over seeds for:

- ``r@0``                immediate resurrected fraction (immediate-match check)
- ``retain_nll@0``       retain NLL of the unlearned model
- ``h50``                relearning half-life (interpolated step where r crosses 0.5)
- ``cond_res@T``         conditional resurrection at horizons 16/32/64
- ``cond_auc@512``       conditional survival AUC at the final horizon

No third-party dependencies (stdlib ``csv``/``random`` only) so it is unit-testable
without the ML stack.
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
from pathlib import Path

COND_HORIZONS = (16, 32, 64)
AUC_HORIZON = 512


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    val = row.get(key, "")
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def h50_from_curve(steps: list[int], r: list[float], censor_at: float | None = None) -> float:
    """Interpolated relearn step at which resurrected fraction first reaches 0.5.

    Returns 0.0 if already >=0.5 at t=0, or the (censored) max step if never reached.
    """
    if not steps:
        return float("nan")
    if r[0] >= 0.5:
        return 0.0
    for i in range(1, len(steps)):
        if r[i] >= 0.5:
            t0, t1, r0, r1 = steps[i - 1], steps[i], r[i - 1], r[i]
            if r1 == r0:
                return float(t1)
            return t0 + (0.5 - r0) * (t1 - t0) / (r1 - r0)
    return float(censor_at if censor_at is not None else steps[-1])


def curve_metrics(rows: list[dict]) -> dict:
    """Extract per-seed scalar metrics from one method x seed survival curve."""
    rows = sorted(rows, key=lambda x: int(float(x["step_t"])))
    steps = [int(float(x["step_t"])) for x in rows]
    r = [_f(x, "resurrected_fraction") for x in rows]
    by_t = {int(float(x["step_t"])): x for x in rows}
    m = {
        "r0": r[0] if r else float("nan"),
        "retain_nll0": _f(by_t[steps[0]], "retain_nll") if steps else float("nan"),
        "h50": h50_from_curve(steps, r),
    }
    for h in COND_HORIZONS:
        m[f"cond_res@{h}"] = _f(by_t[h], "conditional_resurrection_fraction") if h in by_t else float("nan")
    m[f"cond_auc@{AUC_HORIZON}"] = (
        _f(by_t[AUC_HORIZON], "conditional_auc_survival_so_far") if AUC_HORIZON in by_t else float("nan")
    )
    return m


def bootstrap_ci(values: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    vals = [v for v in values if v == v]  # drop NaN
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / n
    if n == 1:
        return (mean, vals[0], vals[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return (mean, lo, hi)


def aggregate(inputs: list[str]) -> dict[str, dict]:
    """Return {method: {metric: [per-seed values]}} from a list of CSV paths."""
    by_method: dict[str, list[dict]] = {}
    for path in inputs:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        method = rows[0].get("method") or Path(path).parent.name
        by_method.setdefault(method, []).append(curve_metrics(rows))
    out: dict[str, dict] = {}
    for method, seed_metrics in by_method.items():
        keys = ["r0", "retain_nll0", "h50"] + [f"cond_res@{h}" for h in COND_HORIZONS] + [f"cond_auc@{AUC_HORIZON}"]
        out[method] = {"n": len(seed_metrics)}
        for k in keys:
            out[method][k] = bootstrap_ci([sm[k] for sm in seed_metrics])
    return out


def _fmt(ci: tuple[float, float, float]) -> str:
    mean, lo, hi = ci
    if mean != mean:
        return "--"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def to_markdown(agg: dict[str, dict]) -> str:
    cols = ["r0", "retain_nll0", "h50"] + [f"cond_res@{h}" for h in COND_HORIZONS] + [f"cond_auc@{AUC_HORIZON}"]
    header = "| method | n | " + " | ".join(cols) + " |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in cols) + " |"
    lines = ["# Real-TOFU aggregate (mean [95% bootstrap CI] over seeds)", "", header, sep]
    for method, m in sorted(agg.items()):
        cells = " | ".join(_fmt(m[c]) for c in cols)
        lines.append(f"| {method} | {m['n']} | {cells} |")
    lines.append("")
    lines.append("Lower h50/cond_res = less durable; higher cond_auc = more durable. "
                 "Check r0 across methods for immediate-matching.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="survival_curve.csv paths or globs (one per method x seed)")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])

    agg = aggregate(paths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(agg)
    (out_dir / "aggregate_ci_summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
