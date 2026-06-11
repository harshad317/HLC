#!/usr/bin/env python3
"""Build the durability/utility frontier from a sweep of survival curves.

Each input ``survival_curve.csv`` is one config (one point on a frontier). The
method string encodes ``family_label`` (e.g. ``disp_lf3``, ``ga_rw0p5``); the family
is the substring before the first underscore. For each config we extract
``(r@0, retain_nll@0, h50)`` and group by family, sorted by retain NLL.

We also compute frontier domination: a GA point is *dominated* if some displacement
point has both lower retain NLL and >= h50 (i.e. displacement is at least as durable
and strictly cheaper). Reports a markdown table, a CSV, and -- if matplotlib is
available -- a (retain NLL, h50) scatter.

Reuses the tested metric helpers from ``aggregate_tofu_real.py``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_tofu_real import curve_metrics  # noqa: E402


def family_of(method: str) -> str:
    return method.split("_", 1)[0] if "_" in method else method


def load_points(paths: list[str]) -> list[dict]:
    points = []
    for path in paths:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        method = rows[0].get("method") or Path(path).parent.name
        m = curve_metrics(rows)
        points.append({
            "family": family_of(method),
            "label": method,
            "r0": m["r0"],
            "retain_nll0": m["retain_nll0"],
            "h50": m["h50"],
        })
    return points


def dominated_flags(points: list[dict]) -> dict[str, bool]:
    """A non-displacement point is dominated if a 'disp' point has lower retain and >= h50."""
    disp = [p for p in points if p["family"] == "disp"]
    flags = {}
    for p in points:
        if p["family"] == "disp":
            flags[p["label"]] = False
            continue
        flags[p["label"]] = any(
            d["retain_nll0"] <= p["retain_nll0"] and d["h50"] >= p["h50"] for d in disp
        )
    return flags


def to_markdown(points: list[dict]) -> str:
    dom = dominated_flags(points)
    fams: dict[str, list[dict]] = {}
    for p in points:
        fams.setdefault(p["family"], []).append(p)
    lines = ["# Durability/utility frontier", "",
             "Each row is one config. Lower retain NLL = better utility; higher h50 = more durable.",
             ""]
    for fam, pts in sorted(fams.items()):
        pts = sorted(pts, key=lambda x: (x["retain_nll0"] if x["retain_nll0"] == x["retain_nll0"] else 1e9))
        lines.append(f"## {fam}")
        lines.append("| label | r@0 | retain NLL@0 | h50 | GA-dominated by disp |")
        lines.append("| --- | ---: | ---: | ---: | :---: |")
        for p in pts:
            mark = "yes" if dom.get(p["label"]) else ("--" if fam == "disp" else "no")
            lines.append(f"| {p['label']} | {p['r0']:.2f} | {p['retain_nll0']:.2f} | {p['h50']:.1f} | {mark} |")
        lines.append("")
    n_dom = sum(1 for p in points if dom.get(p["label"]))
    n_ga = sum(1 for p in points if p["family"] != "disp")
    lines.append(f"**{n_dom}/{n_ga} non-displacement points are dominated by some displacement config** "
                 "(displacement at least as durable and strictly cheaper on retain).")
    return "\n".join(lines) + "\n"


def maybe_plot(points: list[dict], out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"disp": "tab:green", "ga": "tab:red", "hlc": "tab:blue"}
    for fam in sorted({p["family"] for p in points}):
        pts = sorted([p for p in points if p["family"] == fam], key=lambda x: x["retain_nll0"])
        xs = [p["retain_nll0"] for p in pts]
        ys = [p["h50"] for p in pts]
        ax.plot(xs, ys, "-o", label=fam, color=colors.get(fam))
    ax.set_xlabel("retain NLL @0 (lower = better utility)")
    ax.set_ylabel("h50 (higher = more durable)")
    ax.set_title("Durability/utility frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="survival_curve.csv paths/globs")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])

    points = load_points(paths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(points)
    (out_dir / "frontier_table.md").write_text(md, encoding="utf-8")
    with (out_dir / "frontier_points.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "label", "r0", "retain_nll0", "h50"])
        w.writeheader()
        for p in points:
            w.writerow(p)
    plotted = maybe_plot(points, out_dir / "frontier.png")
    print(md)
    print(f"[plot {'written' if plotted else 'skipped (no matplotlib)'}]")


if __name__ == "__main__":
    main()
