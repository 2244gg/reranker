"""Effect size measures."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cohens_d(orig: Sequence[float], rer: Sequence[float]) -> float:
    """Cohen's d on paired differences (rer - orig).

    Defined here as ``mean(diff) / std(diff)`` (also known as d_z, the
    standardized effect for paired samples).
    """
    a = np.asarray(orig, dtype=float)
    b = np.asarray(rer, dtype=float)
    if len(a) < 2:
        return 0.0
    diff = b - a
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def label_effect_size(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def odds_ratio(orig_binary: Sequence[int], rer_binary: Sequence[int]) -> float:
    """Odds ratio for paired binary outcomes using McNemar's b/c counts.

    Returns ``c / b`` (improvements / worsenings). Adds 0.5 continuity correction
    when either b or c is zero.
    """
    a = np.asarray(orig_binary, dtype=int)
    b_arr = np.asarray(rer_binary, dtype=int)
    n10 = int(np.sum((a == 1) & (b_arr == 0)))
    n01 = int(np.sum((a == 0) & (b_arr == 1)))
    if n10 == 0 or n01 == 0:
        return float((n01 + 0.5) / (n10 + 0.5))
    return float(n01 / n10)
