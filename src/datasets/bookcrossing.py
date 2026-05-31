"""BookCrossing dataset adapter (age + country demographics)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .base import BaseDatasetAdapter, InteractionItem, UserRecord


def _age_to_group(age: Optional[float]) -> str:
    if age is None or pd.isna(age):
        return "unknown"
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if age <= 0 or age > 100:
        return "unknown"
    if age < 18:
        return "young"
    if age < 35:
        return "young-adult"
    if age < 55:
        return "middle-aged"
    return "elderly"


def _parse_country(location: Optional[str]) -> str:
    if not isinstance(location, str):
        return "unknown"
    parts = [p.strip() for p in location.split(",")]
    parts = [p for p in parts if p and p.lower() not in ("n/a", "na", "")]
    return parts[-1].lower() if parts else "unknown"


class BookCrossingAdapter(BaseDatasetAdapter):
    """BookCrossing adapter.

    Expected raw files (CSV with ``;`` separator and Latin-1 encoding):
        - ``BX-Users.csv`` columns: User-ID;Location;Age
        - ``BX-Books.csv`` columns: ISBN;Book-Title;Book-Author;...
        - ``BX-Book-Ratings.csv`` columns: User-ID;ISBN;Book-Rating

    Set ``implicit=True`` to treat any interaction (rating>=0) as positive,
    otherwise rating > 5 (on the 1-10 explicit scale).
    """

    name = "bookcrossing"

    DEFAULT_RAW_DIR = "data/raw/bookcrossing"
    TOP_COUNTRIES_K = 10

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        implicit: bool = False,
        rating_threshold: float = 5.0,
        min_interactions: int = 10,
    ) -> None:
        super().__init__(
            raw_dir=raw_dir or self.DEFAULT_RAW_DIR,
            rating_threshold=None if implicit else rating_threshold,
            min_interactions=min_interactions,
        )
        self.implicit = implicit
        self._top_countries: Optional[set] = None

    # ------------------------------------------------------------------

    def domain(self) -> str:
        return "book"

    def demographic_schema(self) -> List[str]:
        return ["age_group", "country"]

    # ------------------------------------------------------------------

    def positive_feedback_predicate(self, item: InteractionItem) -> bool:
        if self.implicit:
            return True
        return super().positive_feedback_predicate(item)

    def _country_or_other(self, country: str) -> str:
        if self._top_countries is None or country == "unknown":
            return country
        return country if country in self._top_countries else "other"

    def _load_users(self) -> List[UserRecord]:
        raw = Path(self.raw_dir)
        users_path = raw / "BX-Users.csv"
        books_path = raw / "BX-Books.csv"
        ratings_path = raw / "BX-Book-Ratings.csv"

        if not users_path.exists():
            raise FileNotFoundError(
                f"BookCrossing files not found in {raw}. "
                f"See data/README.md for download instructions."
            )

        users_df = pd.read_csv(
            users_path,
            sep=";",
            encoding="latin-1",
            on_bad_lines="skip",
        )
        books_df = pd.read_csv(
            books_path,
            sep=";",
            encoding="latin-1",
            on_bad_lines="skip",
            usecols=["ISBN", "Book-Title", "Book-Author"],
        )
        ratings_df = pd.read_csv(
            ratings_path,
            sep=";",
            encoding="latin-1",
            on_bad_lines="skip",
        )

        # Compute top-K countries on the user table (before filtering).
        countries = users_df["Location"].apply(_parse_country)
        country_counts = Counter(countries)
        top = [c for c, _ in country_counts.most_common(self.TOP_COUNTRIES_K) if c != "unknown"]
        self._top_countries = set(top)

        # Vectorized item lookup (avoid 270k-row iterrows).
        books_df["ISBN"] = books_df["ISBN"].astype(str)
        item_lookup: Dict[str, tuple] = dict(
            zip(
                books_df["ISBN"],
                zip(
                    books_df["Book-Title"].astype(str),
                    books_df.get("Book-Author", pd.Series([""] * len(books_df))).astype(str),
                ),
            )
        )

        # Vectorized user records build (avoid 280k-row iterrows).
        users_df["UID_STR"] = users_df["User-ID"].astype(str)
        users_df["country_norm"] = users_df["Location"].apply(
            lambda loc: self._country_or_other(_parse_country(loc))
        )
        users_df["age_group_norm"] = users_df["Age"].apply(_age_to_group)
        records: Dict[str, UserRecord] = {
            row.UID_STR: UserRecord(
                user_id=row.UID_STR,
                demographics={
                    "age_group": row.age_group_norm,
                    "country": row.country_norm,
                },
                history=[],
            )
            for row in users_df.itertuples(index=False)
        }

        # Vectorized ratings filter + groupby for batched assignment.
        ratings_df["UID_STR"] = ratings_df["User-ID"].astype(str)
        ratings_df["ISBN_STR"] = ratings_df["ISBN"].astype(str)
        # Drop rows whose user or ISBN we don't know about.
        valid_users = ratings_df["UID_STR"].isin(records)
        valid_items = ratings_df["ISBN_STR"].isin(item_lookup)
        ratings_df = ratings_df[valid_users & valid_items]

        # Build interactions in chunks per user.
        # Convert columns to numpy first to avoid per-row pandas overhead.
        uids = ratings_df["UID_STR"].to_numpy()
        isbns = ratings_df["ISBN_STR"].to_numpy()
        rats = ratings_df["Book-Rating"].to_numpy(dtype=float)
        for uid, isbn, rating in zip(uids, isbns, rats):
            title, author = item_lookup[isbn]
            records[uid].history.append(
                InteractionItem(
                    item_id=isbn,
                    title=title,
                    metadata={"author": author},
                    rating=float(rating),
                    timestamp=None,
                )
            )

        return list(records.values())
