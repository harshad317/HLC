from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np


def h50(steps: Sequence[int], survival: Sequence[float]) -> tuple[Optional[int], bool]:
    """Return first step where survival <= 0.5 and whether the value is censored."""
    if len(steps) != len(survival):
        raise ValueError("steps and survival must have the same length")
    for step, value in zip(steps, survival):
        if value <= 0.5:
            return int(step), False
    return None, True


def auc_survival(steps: Sequence[int], survival: Sequence[float]) -> float:
    """Normalized trapezoidal area under a survival curve."""
    if len(steps) != len(survival):
        raise ValueError("steps and survival must have the same length")
    if not steps:
        return 0.0
    if len(steps) == 1:
        return float(survival[0])
    x = np.asarray(steps, dtype=float)
    y = np.asarray(survival, dtype=float)
    span = float(x[-1] - x[0])
    if span <= 0:
        return float(np.mean(y))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / span)


def running_auc(steps: Sequence[int], survival: Sequence[float]) -> list[float]:
    values: list[float] = []
    for idx in range(len(steps)):
        values.append(auc_survival(steps[: idx + 1], survival[: idx + 1]))
    return values


def fraction(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(bool(v) for v in values) / len(values))
