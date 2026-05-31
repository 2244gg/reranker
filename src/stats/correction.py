"""Multiple comparison corrections (Holm-Bonferroni)."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Sorted by raw p ascending; adjusted p_i = max( (m - rank_i) * p_(rank_i),
    cumulative max of preceding adjusted ). Result returned in input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, raw in enumerate(sorted_p):
        scale = m - i
        adj = min(1.0, scale * raw)
        running_max = max(running_max, adj)
        adjusted_sorted[i] = running_max
    out = np.empty(m, dtype=float)
    out[order] = adjusted_sorted
    return out.tolist()


def label_significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
