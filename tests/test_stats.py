"""Statistical module tests using synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from src.stats.correction import holm_bonferroni, label_significance
from src.stats.effect_size import cohens_d, label_effect_size, odds_ratio
from src.stats.tests import bootstrap_ci, mcnemar_test, paired_t_test, paired_wilcoxon


def test_paired_wilcoxon_shifted_distribution_significant():
    rng = np.random.default_rng(0)
    orig = rng.normal(0.5, 0.05, size=200)
    rer = orig + 0.02 + rng.normal(0, 0.005, size=200)  # consistent improvement
    res = paired_wilcoxon(orig, rer, alternative="less")
    # 'less' = test whether b - a (rer - orig) is *greater* than zero in scipy>=1.7
    # We use 'two-sided' for a robust check.
    res2 = paired_wilcoxon(orig, rer, alternative="two-sided")
    assert res2.p_value < 0.001
    assert res2.n == 200


def test_paired_wilcoxon_no_diff():
    a = [0.5] * 50
    b = [0.5] * 50
    res = paired_wilcoxon(a, b)
    assert res.p_value == pytest.approx(1.0)


def test_paired_t_test_smoke():
    rng = np.random.default_rng(1)
    orig = rng.normal(0.0, 1.0, size=100)
    rer = orig + 0.5
    res = paired_t_test(orig, rer)
    assert res.p_value < 0.001


def test_mcnemar_more_improvements_significant():
    # 100 users: 30 went from 0 to 1, 5 went from 1 to 0.
    orig = [0] * 30 + [1] * 5 + [1] * 30 + [0] * 35
    rer = [1] * 30 + [0] * 5 + [1] * 30 + [0] * 35
    res = mcnemar_test(orig, rer)
    assert res.extras["b"] == 5  # orig=1, rer=0
    assert res.extras["c"] == 30  # orig=0, rer=1
    assert res.p_value < 0.001


def test_mcnemar_all_concordant():
    orig = [1, 1, 0, 0]
    rer = [1, 1, 0, 0]
    res = mcnemar_test(orig, rer)
    assert res.extras.get("all_concordant") is True
    assert res.p_value == 1.0


def test_bootstrap_ci_shape_and_seed_stability():
    rng = np.random.default_rng(2)
    deltas = rng.normal(0.05, 0.1, size=300)
    m1, lo1, hi1 = bootstrap_ci(deltas, n_boot=2000, seed=7)
    m2, lo2, hi2 = bootstrap_ci(deltas, n_boot=2000, seed=7)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 < m1 < hi1


def test_cohens_d_strong_signal():
    rng = np.random.default_rng(3)
    orig = rng.normal(0.0, 1.0, size=200)
    rer = orig + 1.0  # 1 sigma shift on diffs
    d = cohens_d(orig, rer)
    assert d > 5  # paired d_z is large because diff has tiny std (only the noise from above is 0)
    # label
    assert label_effect_size(d) == "large"


def test_cohens_d_zero_diff():
    a = [0.5] * 10
    b = [0.5] * 10
    assert cohens_d(a, b) == 0.0


def test_holm_bonferroni_basic():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = holm_bonferroni(p)
    # Sorted by raw: [0.005, 0.01, 0.03, 0.04], scales [4, 3, 2, 1].
    # Adjusted before running max: [0.02, 0.03, 0.06, 0.04].
    # Running max keeps it monotonic: [0.02, 0.03, 0.06, 0.06].
    expected_sorted = [0.02, 0.03, 0.06, 0.06]
    sorted_idx = sorted(range(4), key=lambda i: p[i])
    expected = [0.0] * 4
    for i, sidx in enumerate(sorted_idx):
        expected[sidx] = expected_sorted[i]
    for got, exp in zip(adj, expected):
        assert abs(got - exp) < 1e-9


def test_holm_bonferroni_caps_at_one():
    p = [0.5, 0.5, 0.5]
    adj = holm_bonferroni(p)
    assert all(a <= 1.0 for a in adj)


def test_label_significance_thresholds():
    assert label_significance(0.0001) == "***"
    assert label_significance(0.005) == "**"
    assert label_significance(0.04) == "*"
    assert label_significance(0.06) == "ns"


def test_odds_ratio_continuity_correction():
    # No b counts - continuity correction must avoid division by zero.
    orig = [0] * 50
    rer = [1] * 50
    or_value = odds_ratio(orig, rer)
    assert or_value > 1.0
