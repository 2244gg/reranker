"""Difficulty-weighted hit scoring for popularity-debias evaluation.

Given a precomputed group-conditional popularity table B (see
``src.popularity.slicer``), this module derives a parallel
*difficulty score table* that quantifies how "hard" it is to correctly
recommend each item within each demographic slice.

Intuition
---------
Recommending an item that is already mainstream in a user's group is
trivial — many baselines could do it. Surfacing a niche-but-loved item
is genuinely informative. We therefore weight every test-set hit by an
inverse-popularity score:

    score(i, k) = f(B[i, k])    where f is monotone-decreasing.

Three schemes are provided:

  * ``linear_x10`` (default, matches the existing
    ``unpopularity_score`` formula):  ``(1 - B) * 10``     in [0, 10]
  * ``log``:                          ``-log2(B + 1e-3)``  in [0, ~10]
  * ``bucket``:                        coarse 1/2/3/5 scoring

Use cases
---------

* Persisted lookup table: ``build_score_table(popularity_df)`` returns a
  DataFrame with the same shape as the popularity table but holding
  difficulty scores. Saved next to ``items_with_popularity.csv``.

* Per-(user, scenario) evaluation: ``difficulty_weighted_hit_score`` sums
  the difficulty scores over the items that hit the user's test set,
  optionally with a rank bonus.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd


DEFAULT_SCHEME = "linear_x10"


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------


def difficulty_score(b_norm: float, scheme: str = DEFAULT_SCHEME) -> float:
    """Convert a slice-conditional popularity ``B in [0, 1]`` to a difficulty score.

    Higher score means the item is harder to surface (more niche).

    Parameters
    ----------
    b_norm:
        Normalized slice popularity in [0, 1]. Values outside the range are
        clipped to [0, 1].
    scheme:
        ``"linear_x10"`` (default), ``"log"``, or ``"bucket"``.
    """
    try:
        b_norm = max(0.0, min(1.0, float(b_norm)))
    except (TypeError, ValueError):
        b_norm = 0.0

    if scheme == "linear_x10":
        return (1.0 - b_norm) * 10.0

    if scheme == "log":
        # -log2(B + eps); for eps = 1e-3, range is approximately [0, 9.97].
        return -math.log2(b_norm + 1e-3)

    if scheme == "bucket":
        if b_norm >= 0.8:
            return 1.0
        if b_norm >= 0.5:
            return 2.0
        if b_norm >= 0.2:
            return 3.0
        return 5.0

    raise ValueError(f"Unknown scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# Precomputed lookup table
# ---------------------------------------------------------------------------


# Columns that are metadata, not popularity slices.
_META_COLS = frozenset({"item_id", "title", "MovieID", "Title", "Genres"})


def slice_columns(df: pd.DataFrame) -> List[str]:
    """Return the popularity slice column names (everything except metadata)."""
    return [c for c in df.columns if c not in _META_COLS]


def build_score_table(
    popularity_df: pd.DataFrame,
    scheme: str = DEFAULT_SCHEME,
) -> pd.DataFrame:
    """Precompute the (item x slice) difficulty score table.

    Same row/column shape as ``popularity_df``: metadata columns are kept
    verbatim; numeric slice cells become difficulty scores.
    """
    out = popularity_df.copy()
    pop_cols = slice_columns(out)
    for col in pop_cols:
        out[col] = popularity_df[col].apply(
            lambda x, _s=scheme: difficulty_score(x, _s)
        )
    out["_score_scheme"] = scheme  # provenance metadata
    return out


def write_score_table(
    popularity_csv: str | Path,
    output_csv: str | Path,
    scheme: str = DEFAULT_SCHEME,
) -> Path:
    """Read a popularity CSV, derive the score table, and write it to disk."""
    df = pd.read_csv(popularity_csv)
    score_df = build_score_table(df, scheme=scheme)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Score lookup at evaluation time
# ---------------------------------------------------------------------------


class ItemScoreTable:
    """Wraps a precomputed score DataFrame for fast (title, slice) lookup.

    Title matching uses the same canonicalization as the popularity table
    (see ``src.experiment.metrics.normalize_title``).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        from src.experiment.metrics import normalize_title

        title_col = "title" if "title" in df.columns else "Title"
        id_col = "item_id" if "item_id" in df.columns else "MovieID"
        self._df = df
        self._scheme = (
            df["_score_scheme"].iloc[0] if "_score_scheme" in df.columns else DEFAULT_SCHEME
        )
        self._exact: Dict[str, Dict[str, float]] = {}
        self._norm: Dict[str, Dict[str, float]] = {}
        slice_cols = [
            c for c in df.columns
            if c not in _META_COLS and c != "_score_scheme"
        ]
        for _, row in df.iterrows():
            title = str(row[title_col])
            payload = {c: float(row[c]) for c in slice_cols if c in row}
            self._exact[title] = payload
            self._norm[normalize_title(title)] = payload

    @classmethod
    def from_csv(cls, path: str | Path) -> "ItemScoreTable":
        return cls(pd.read_csv(path))

    @classmethod
    def from_popularity(
        cls,
        popularity_df: pd.DataFrame,
        scheme: str = DEFAULT_SCHEME,
    ) -> "ItemScoreTable":
        return cls(build_score_table(popularity_df, scheme=scheme))

    def lookup(self, title: str, slice_column: str) -> float:
        from src.experiment.metrics import normalize_title

        row = self._exact.get(title) or self._norm.get(normalize_title(title))
        if row is None:
            return 0.0
        return float(row.get(slice_column, 0.0))

    @property
    def scheme(self) -> str:
        return self._scheme


# ---------------------------------------------------------------------------
# Aggregate metric: difficulty-weighted hit score
# ---------------------------------------------------------------------------


def difficulty_weighted_hit_score(
    rec_list: Sequence[str],
    test_set: Sequence[str],
    score_table: ItemScoreTable,
    slice_column: str,
    *,
    rank_weight: float = 0.0,
    list_size: int = 20,
) -> Dict[str, Any]:
    """Sum difficulty scores over the items in ``rec_list`` that hit ``test_set``.

    Each hit at 0-indexed rank ``r`` contributes:

        s_i = score(B[i, slice]) * (1 + rank_weight * (list_size - r) / list_size)

    so that earlier hits (higher rank) get an additional bonus when
    ``rank_weight > 0``.

    Title matching delegates to ``src.experiment.metrics.titles_match`` so
    fuzzy variants (HTML entities, parenthetical metadata, leading articles)
    still count.
    """
    from src.experiment.metrics import title_key, titles_match

    rec_top = list(rec_list)[:list_size]
    test_keys = [title_key(t) for t in test_set]
    test_keys = [k for k in test_keys if k]
    if not test_keys:
        return {
            "total_score": 0.0,
            "average_score": 0.0,
            "hits": 0,
            "items": [],
            "scheme": score_table.scheme,
            "rank_weight": rank_weight,
            "slice_column": slice_column,
        }

    items: List[Dict[str, Any]] = []
    total = 0.0
    for r0, title in enumerate(rec_top):
        kt = title_key(title)
        if not kt:
            continue
        # Fuzzy match against any test-set title.
        is_hit = any(
            kt == rk
            or (
                len(min(kt, rk, key=len)) >= 8
                and min(kt, rk, key=len) in max(kt, rk, key=len)
            )
            for rk in test_keys
        )
        if not is_hit:
            continue
        score = score_table.lookup(title, slice_column)
        rank_score = (list_size - r0) / list_size if list_size else 0.0
        weighted = score * (1.0 + rank_weight * rank_score)
        weighted = round(weighted, 4)
        total += weighted
        items.append(
            {
                "title": title,
                "rank": r0 + 1,
                "raw_score": round(float(score), 4),
                "weighted_score": weighted,
            }
        )

    avg = round(total / len(items), 4) if items else 0.0
    return {
        "total_score": round(total, 4),
        "average_score": avg,
        "hits": len(items),
        "items": items,
        "scheme": score_table.scheme,
        "rank_weight": rank_weight,
        "slice_column": slice_column,
    }
