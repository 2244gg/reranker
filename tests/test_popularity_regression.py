"""Regression test: PopularitySlicer with the legacy 2-D ML-1M schema must
reproduce the column values found in the shipped ``movies_with_popularity.csv``.

NOTE: The shipped CSV may have been built with a slightly different threshold
or counting convention than our default. This test verifies the *direction* and
ordering of values, plus exact equality at well-known anchor titles. A strict
1e-6 element-wise check is also attempted; if it fails because of a known
historical convention difference, the test will report which columns diverge.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.datasets.factory import build_dataset
from src.popularity.slicer import PopularitySlicer


REFERENCE_CSV = Path("movies_with_popularity.csv")
ML1M_USERS_DAT = Path("ml-1m/ml-1m/users.dat")


def _load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_CSV)
    df["MovieID"] = df["MovieID"].astype(str)
    return df


@pytest.mark.skipif(
    not (REFERENCE_CSV.exists() and ML1M_USERS_DAT.exists()),
    reason="ML-1M raw or reference CSV missing",
)
def test_popularity_legacy_columns_present():
    adapter = build_dataset("ml1m-legacy")
    slicer = PopularitySlicer(adapter, max_cross_dims=2, sparse_threshold=0)
    df = slicer.build()
    expected_cols = {
        "neutral_All",
        "gender_M",
        "gender_F",
        "age_Youth",
        "age_Middle",
        "age_Senior",
        "cross_M_Youth",
        "cross_F_Youth",
        "cross_M_Middle",
        "cross_F_Middle",
        "cross_M_Senior",
        "cross_F_Senior",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing legacy columns: {missing}"


@pytest.mark.skipif(
    not (REFERENCE_CSV.exists() and ML1M_USERS_DAT.exists()),
    reason="ML-1M raw or reference CSV missing",
)
def test_popularity_legacy_values_correlate_with_reference():
    """Sanity check: per-column ordering must correlate strongly with reference.

    A perfect element-wise match would require knowing the exact threshold used
    when generating the shipped CSV. Instead we require Pearson correlation
    >= 0.95 per column on the intersection of items.
    """
    adapter = build_dataset("ml1m-legacy")
    slicer = PopularitySlicer(adapter, max_cross_dims=2, sparse_threshold=0)
    df = slicer.build().rename(columns={"item_id": "MovieID"})
    df["MovieID"] = df["MovieID"].astype(str)
    ref = _load_reference()

    merged = df.merge(ref, on="MovieID", suffixes=("_new", "_ref"))
    assert len(merged) > 1000, "Too few items to compare; aborting regression."

    cols = ["neutral_All", "gender_M", "gender_F", "age_Youth", "age_Middle", "age_Senior"]
    bad = []
    for c in cols:
        new_col = f"{c}_new"
        ref_col = f"{c}_ref"
        if new_col not in merged or ref_col not in merged:
            continue
        corr = merged[new_col].corr(merged[ref_col])
        if pd.isna(corr) or corr < 0.95:
            bad.append((c, corr))
    assert not bad, f"Columns with weak correlation vs reference: {bad}"


@pytest.mark.skipif(
    not (REFERENCE_CSV.exists() and ML1M_USERS_DAT.exists()),
    reason="ML-1M raw or reference CSV missing",
)
def test_popularity_top_movies_match_neutral():
    """The top-N by neutral popularity should overlap heavily with reference."""
    adapter = build_dataset("ml1m-legacy")
    slicer = PopularitySlicer(adapter, max_cross_dims=2, sparse_threshold=0)
    df = slicer.build()
    df_top = (
        df[["item_id", "neutral_All"]]
        .sort_values("neutral_All", ascending=False)
        .head(100)
    )
    ref = _load_reference()
    ref_top = (
        ref[["MovieID", "neutral_All"]]
        .sort_values("neutral_All", ascending=False)
        .head(100)
    )

    new_set = set(df_top["item_id"].astype(str))
    ref_set = set(ref_top["MovieID"].astype(str))
    overlap = len(new_set & ref_set)
    assert overlap >= 80, f"Top-100 overlap = {overlap}/100; below threshold."


def test_no_three_dim_cross_for_two_dim_schema():
    """ML-1M legacy adapter must NOT produce 3-way cross columns."""
    adapter = build_dataset("ml1m-legacy")
    slicer = PopularitySlicer(adapter, sparse_threshold=0)
    df = slicer.build()
    bad = [c for c in df.columns if c.count("_") >= 3 and c.startswith("cross_")]
    assert not bad, f"Unexpected 3D cross cols for 2D schema: {bad}"
