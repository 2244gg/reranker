"""Statistical significance tests for paired metric values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    n: int
    extras: dict


def paired_wilcoxon(orig: Sequence[float], rer: Sequence[float], alternative: str = "two-sided") -> TestResult:
    """Wilcoxon signed-rank for paired samples (rer - orig)."""
    from scipy.stats import wilcoxon

    a = np.asarray(orig, dtype=float)
    b = np.asarray(rer, dtype=float)
    assert len(a) == len(b), "orig and rer must have equal length"
    diff = b - a
    if not np.any(diff != 0):
        return TestResult(
            name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            n=len(diff),
            extras={"alternative": alternative, "all_zero_diff": True},
        )
    try:
        result = wilcoxon(b, a, alternative=alternative, zero_method="wilcox")
        stat = float(result.statistic)
        p = float(result.pvalue)
    except ValueError as e:
        return TestResult(
            name="wilcoxon",
            statistic=float("nan"),
            p_value=1.0,
            n=len(diff),
            extras={"alternative": alternative, "error": str(e)},
        )
    return TestResult("wilcoxon", stat, p, len(diff), {"alternative": alternative})


def paired_t_test(orig: Sequence[float], rer: Sequence[float], alternative: str = "two-sided") -> TestResult:
    """Paired Student's t-test."""
    from scipy.stats import ttest_rel

    a = np.asarray(orig, dtype=float)
    b = np.asarray(rer, dtype=float)
    assert len(a) == len(b)
    if len(a) < 2:
        return TestResult("t_test", float("nan"), 1.0, len(a), {"alternative": alternative})
    res = ttest_rel(b, a, alternative=alternative)
    return TestResult("t_test", float(res.statistic), float(res.pvalue), len(a), {"alternative": alternative})


def mcnemar_test(orig_binary: Sequence[int], rer_binary: Sequence[int]) -> TestResult:
    """McNemar's test for paired binary outcomes (e.g. HitRate@k indicators).

    Builds the 2x2 contingency table:
        b = (orig=1, rer=0)  rer worsened the user
        c = (orig=0, rer=1)  rer improved the user
    """
    from statsmodels.stats.contingency_tables import mcnemar

    a = np.asarray(orig_binary, dtype=int)
    b_arr = np.asarray(rer_binary, dtype=int)
    assert len(a) == len(b_arr)
    n11 = int(np.sum((a == 1) & (b_arr == 1)))
    n10 = int(np.sum((a == 1) & (b_arr == 0)))
    n01 = int(np.sum((a == 0) & (b_arr == 1)))
    n00 = int(np.sum((a == 0) & (b_arr == 0)))
    table = [[n11, n10], [n01, n00]]
    if (n10 + n01) == 0:
        return TestResult(
            "mcnemar",
            0.0,
            1.0,
            len(a),
            {"b": n10, "c": n01, "table": table, "all_concordant": True},
        )
    res = mcnemar(table, exact=False, correction=True)
    return TestResult(
        "mcnemar",
        float(res.statistic),
        float(res.pvalue),
        len(a),
        {"b": n10, "c": n01, "table": table},
    )


def bootstrap_ci(
    deltas: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the mean of ``deltas`` (rer - orig).

    Returns ``(mean_delta, ci_low, ci_high)``.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    # Sample indices in batches for memory.
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    low_p = 100.0 * (alpha / 2.0)
    high_p = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(means, [low_p, high_p])
    return float(arr.mean()), float(ci_low), float(ci_high)
