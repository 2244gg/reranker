"""Group-conditional popularity slicing.

For each slice (combination of demographic values), compute a popularity
score per item normalized to [0, 1]. The score is the fraction of positive
interactions that fall on this item, normalized by the maximum item count
within the slice (so the most popular item per slice == 1.0).

This mirrors the convention used by the existing
``movies_with_popularity.csv`` shipped with the project, which is required
for regression tests (see ``tests/test_popularity_regression.py``).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.datasets.base import BaseDatasetAdapter, InteractionItem, UserRecord


# Default mapping of demographic values to short suffixes used in column names.
# Designed to be backwards-compatible with the legacy ML-1M CSV.
_DEFAULT_VALUE_SUFFIX: Dict[str, Dict[str, str]] = {
    "gender": {
        "male": "M",
        "female": "F",
        "unknown": "U",
    },
    "age_group": {
        "young": "Youth",
        "middle-aged": "Middle",
        "elderly": "Senior",
        "young-adult": "YoungAdult",
        "unknown": "Unknown",
    },
    "occupation_group": {
        "student": "Student",
        "professional": "Pro",
        "service": "Service",
        "other": "Other",
    },
}

# Sparse-slice protection threshold.
SPARSE_THRESHOLD = 30


def _suffix_for(dim: str, value: str) -> str:
    """Map a (dim, value) pair to a column suffix string."""
    table = _DEFAULT_VALUE_SUFFIX.get(dim, {})
    if value in table:
        return table[value]
    # Fallback: capitalize / sanitize the raw value.
    return str(value).replace(" ", "").replace("-", "_") or "Unknown"


def _column_name(slice_key: Tuple[Tuple[str, str], ...]) -> str:
    """Build a column name from a slice key.

    Examples:
        ()                                      -> ``neutral_All``
        (('gender', 'male'),)                   -> ``gender_M``
        (('age_group', 'young'),)               -> ``age_Youth``
        (('gender', 'male'), ('age_group', 'young')) -> ``cross_M_Youth``
        Three-dim cross uses prefix ``cross_``.
    """
    if not slice_key:
        return "neutral_All"
    if len(slice_key) == 1:
        dim, val = slice_key[0]
        prefix = "age" if dim == "age_group" else dim.split("_")[0]
        return f"{prefix}_{_suffix_for(dim, val)}"
    # 2D / 3D cross slices.
    parts = [_suffix_for(dim, val) for dim, val in slice_key]
    return "cross_" + "_".join(parts)


class PopularitySlicer:
    """Build a per-item popularity table sliced by demographic combinations."""

    def __init__(
        self,
        adapter: BaseDatasetAdapter,
        max_cross_dims: Optional[int] = None,
        sparse_threshold: int = SPARSE_THRESHOLD,
        positive_only: bool = True,
    ) -> None:
        self.adapter = adapter
        self.max_cross_dims = max_cross_dims
        self.sparse_threshold = sparse_threshold
        self.positive_only = positive_only

    # ------------------------------------------------------------------

    def _enumerate_slice_keys(self) -> List[Tuple[Tuple[str, str], ...]]:
        """All slice keys: neutral, single dims, and cross combinations."""
        users = self.adapter.load()
        dims = self.adapter.demographic_schema()
        # Collect the unique values observed per dimension.
        dim_values: Dict[str, List[str]] = {}
        for dim in dims:
            values = sorted({u.demographic_value(dim) for u in users})
            dim_values[dim] = values

        max_r = self.max_cross_dims if self.max_cross_dims is not None else len(dims)
        max_r = min(max_r, len(dims))

        keys: List[Tuple[Tuple[str, str], ...]] = [()]  # neutral
        for r in range(1, max_r + 1):
            for combo_dims in combinations(dims, r):
                # For each combination of dimensions, expand all value tuples.
                value_lists = [dim_values[d] for d in combo_dims]
                for vals in _cartesian(value_lists):
                    keys.append(tuple(zip(combo_dims, vals)))
        return keys

    def _user_in_slice(self, user: UserRecord, key: Tuple[Tuple[str, str], ...]) -> bool:
        for dim, val in key:
            if user.demographic_value(dim) != val:
                return False
        return True

    def _positive_items(self, user: UserRecord) -> Iterable[InteractionItem]:
        if self.positive_only:
            return [
                it for it in user.history
                if self.adapter.positive_feedback_predicate(it)
            ]
        return user.history

    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        users = self.adapter.load()
        slice_keys = self._enumerate_slice_keys()

        # 1. For each slice, count item occurrences across positive feedback.
        slice_counts: Dict[Tuple[Tuple[str, str], ...], Counter] = {
            k: Counter() for k in slice_keys
        }
        slice_user_count: Dict[Tuple[Tuple[str, str], ...], int] = {
            k: 0 for k in slice_keys
        }
        # Pre-compute positive items per user.
        per_user_pos: Dict[str, List[InteractionItem]] = {
            u.user_id: list(self._positive_items(u)) for u in users
        }
        # Item id -> title (last-write-wins; titles are stable in practice).
        item_titles: Dict[str, str] = {}

        for k in slice_keys:
            members = [u for u in users if self._user_in_slice(u, k)]
            slice_user_count[k] = len(members)
            for u in members:
                seen = set()
                for it in per_user_pos[u.user_id]:
                    if it.item_id in seen:
                        continue
                    seen.add(it.item_id)
                    slice_counts[k][it.item_id] += 1
                    item_titles[it.item_id] = it.title

        # 2. Normalize per slice: raw_count / max_count_in_slice (in [0, 1]).
        slice_norm: Dict[Tuple[Tuple[str, str], ...], Dict[str, float]] = {}
        for k, counter in slice_counts.items():
            if not counter:
                slice_norm[k] = {}
                continue
            max_c = max(counter.values())
            if max_c <= 0:
                slice_norm[k] = {iid: 0.0 for iid in counter}
            else:
                slice_norm[k] = {iid: c / max_c for iid, c in counter.items()}

        # 3. Sparse-slice protection: blend with neutral when slice users < threshold.
        neutral_key = ()
        neutral_norm = slice_norm.get(neutral_key, {})
        for k in slice_keys:
            if not k:
                continue
            n = slice_user_count[k]
            if n < self.sparse_threshold and self.sparse_threshold > 0:
                w = max(0.0, min(1.0, n / float(self.sparse_threshold)))
                blended: Dict[str, float] = {}
                # Union of items in slice and neutral.
                items = set(slice_norm[k].keys()) | set(neutral_norm.keys())
                for iid in items:
                    raw = slice_norm[k].get(iid, 0.0)
                    base = neutral_norm.get(iid, 0.0)
                    blended[iid] = w * raw + (1.0 - w) * base
                slice_norm[k] = blended

        # 4. Assemble DataFrame: one row per item, one column per slice key.
        all_items = set()
        for k in slice_keys:
            all_items.update(slice_norm[k].keys())
        all_items = sorted(all_items)

        column_order = [_column_name(k) for k in slice_keys]
        rows = []
        for iid in all_items:
            row = {"item_id": iid, "title": item_titles.get(iid, iid)}
            for k, col in zip(slice_keys, column_order):
                row[col] = float(slice_norm[k].get(iid, 0.0))
            rows.append(row)

        df = pd.DataFrame(rows, columns=["item_id", "title"] + column_order)
        return df

    # ------------------------------------------------------------------

    def to_csv(self, output_path: str | Path) -> Path:
        df = self.build()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return out


def _cartesian(value_lists: List[List[str]]) -> Iterable[Tuple[str, ...]]:
    if not value_lists:
        yield ()
        return
    head, *tail = value_lists
    for h in head:
        for rest in _cartesian(tail):
            yield (h,) + tuple(rest)
