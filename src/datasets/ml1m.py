"""MovieLens-1M dataset adapter.

Adds an ``occupation_group`` demographic dimension on top of the existing
gender + age fields, enabling 3-way intersectional analysis. A
``Ml1mLegacyAdapter`` reproduces the original 2-dimension schema for
regression testing against existing ``movies_with_popularity.csv``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .base import BaseDatasetAdapter, InteractionItem, UserRecord


# ML-1M occupation codes (per the dataset README).
# Source: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
_OCCUPATION_NAMES = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

# Compress 21 occupations to 4 groups to keep slice sizes >= ~30.
_OCCUPATION_GROUP = {
    0: "other",
    1: "professional",  # academic/educator
    2: "professional",  # artist (creative knowledge worker)
    3: "service",  # clerical/admin
    4: "student",
    5: "service",
    6: "professional",
    7: "professional",
    8: "service",  # farmer (manual / non-knowledge)
    9: "service",  # homemaker
    10: "student",
    11: "professional",
    12: "professional",
    13: "other",  # retired
    14: "service",
    15: "professional",
    16: "professional",
    17: "professional",
    18: "service",
    19: "other",
    20: "professional",
}


def _map_age_to_group(age_code: int) -> str:
    """Compress 7 ML-1M age codes into 3 groups (matches existing pipeline)."""
    if age_code in (1, 18):
        return "young"
    if age_code in (25, 35, 45):
        return "middle-aged"
    if age_code in (50, 56):
        return "elderly"
    return "unknown"


class Ml1mAdapter(BaseDatasetAdapter):
    """ML-1M adapter.

    By default exposes only ``gender + age_group`` (2 demographic dimensions),
    which matches the most-published convention and reuses the shipped
    ``movies_with_popularity.csv``.

    Pass ``include_occupation=True`` to additionally expose
    ``occupation_group`` for 3-way intersectional analysis. In that case you
    must build a fresh popularity table via
    ``python scripts/build_popularity.py --dataset ml1m`` so the new
    occupation/cross columns exist.
    """

    name = "ml1m"

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        rating_threshold: float = 3.0,
        min_interactions: int = 10,
        include_occupation: bool = False,
    ) -> None:
        if raw_dir is None:
            # Default to the existing in-repo path.
            raw_dir = str(Path("ml-1m") / "ml-1m")
        super().__init__(
            raw_dir=raw_dir,
            rating_threshold=rating_threshold,
            min_interactions=min_interactions,
        )
        self.include_occupation = include_occupation

    # ------------------------------------------------------------------

    def domain(self) -> str:
        return "movie"

    def demographic_schema(self) -> List[str]:
        if self.include_occupation:
            return ["gender", "age_group", "occupation_group"]
        return ["gender", "age_group"]

    # ------------------------------------------------------------------

    def _read_files(self) -> tuple:
        users_path = Path(self.raw_dir) / "users.dat"
        items_path = Path(self.raw_dir) / "movies.dat"
        ratings_path = Path(self.raw_dir) / "ratings.dat"
        if not users_path.exists():
            raise FileNotFoundError(f"ML-1M users.dat not found at {users_path}")

        users = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        )
        items = pd.read_csv(
            items_path,
            sep="::",
            engine="python",
            names=["MovieID", "Title", "Genres"],
            encoding="latin-1",
        )
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )
        return users, items, ratings

    def _build_demographics(self, row) -> Dict[str, str]:
        out = {
            "gender": "male" if row["Gender"] == "M" else "female",
            "age_group": _map_age_to_group(int(row["Age"])),
        }
        if self.include_occupation:
            out["occupation_group"] = _OCCUPATION_GROUP.get(
                int(row["Occupation"]), "other"
            )
        return out

    def _load_users(self) -> List[UserRecord]:
        users_df, items_df, ratings_df = self._read_files()
        item_lookup = {
            int(row["MovieID"]): (str(row["Title"]), str(row["Genres"]))
            for _, row in items_df.iterrows()
        }

        records: Dict[int, UserRecord] = {}
        for _, urow in users_df.iterrows():
            uid = int(urow["UserID"])
            records[uid] = UserRecord(
                user_id=str(uid),
                demographics=self._build_demographics(urow),
                history=[],
            )

        for _, r in ratings_df.iterrows():
            uid = int(r["UserID"])
            mid = int(r["MovieID"])
            if uid not in records or mid not in item_lookup:
                continue
            title, genres = item_lookup[mid]
            records[uid].history.append(
                InteractionItem(
                    item_id=str(mid),
                    title=title,
                    metadata={"genres": genres},
                    rating=float(r["Rating"]),
                    timestamp=int(r["Timestamp"]),
                )
            )
        return list(records.values())


class Ml1mLegacyAdapter(Ml1mAdapter):
    """Backward-compatibility alias.

    Identical to ``Ml1mAdapter`` with its default ``include_occupation=False``.
    Kept so existing CLI invocations (``--dataset ml1m-legacy``) keep working.
    """

    name = "ml1m-legacy"

    def __init__(self, **kwargs) -> None:
        kwargs.pop("include_occupation", None)
        super().__init__(include_occupation=False, **kwargs)
