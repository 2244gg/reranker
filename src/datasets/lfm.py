"""LFM-2b (LastFM) dataset adapter (gender + age + country)."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
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
    if age < 25:
        return "young"
    if age < 45:
        return "middle-aged"
    return "elderly"


def _normalize_gender(g: Optional[str]) -> str:
    if not isinstance(g, str):
        return "unknown"
    g = g.strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    return "unknown"


class LfmAdapter(BaseDatasetAdapter):
    """LFM-2b adapter (also compatible with LFM-1b user-track listening events).

    Expected raw files (LFM-2b distribution):
        - ``users.tsv`` with columns: user_id, country, age, gender, ...
        - ``listening_events.tsv`` (or ``user_artist_playcount.tsv``) with
          columns including user_id, track/artist id, playcount.

    Per user we treat playcount above the user's median as positive feedback.
    """

    name = "lfm"

    DEFAULT_RAW_DIR = "data/raw/lfm-2b"
    TOP_COUNTRIES_K = 10

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        users_file: str = "users.tsv",
        events_file: str = "user_artist_playcount.tsv",
        items_file: str = "artists.tsv",
        min_interactions: int = 20,
    ) -> None:
        super().__init__(
            raw_dir=raw_dir or self.DEFAULT_RAW_DIR,
            rating_threshold=None,
            min_interactions=min_interactions,
        )
        self.users_file = users_file
        self.events_file = events_file
        self.items_file = items_file
        self._top_countries: Optional[set] = None

    def domain(self) -> str:
        return "music"

    def demographic_schema(self) -> List[str]:
        return ["gender", "age_group", "country"]

    # ------------------------------------------------------------------

    def positive_feedback_predicate(self, item: InteractionItem) -> bool:
        thr = item.metadata.get("user_median_playcount") if item.metadata else None
        pc = item.rating  # we store playcount in rating slot
        if pc is None or thr is None:
            return True
        return float(pc) > float(thr)

    # ------------------------------------------------------------------

    def _country_or_other(self, country: str) -> str:
        if self._top_countries is None or country == "unknown":
            return country
        return country if country in self._top_countries else "other"

    def _load_users(self) -> List[UserRecord]:
        raw = Path(self.raw_dir)
        users_path = raw / self.users_file
        events_path = raw / self.events_file
        items_path = raw / self.items_file

        if not users_path.exists():
            raise FileNotFoundError(
                f"LFM users file not found at {users_path}. "
                f"See data/README.md for download instructions."
            )

        users_df = pd.read_csv(users_path, sep="\t")
        events_df = pd.read_csv(events_path, sep="\t")
        items_df = (
            pd.read_csv(items_path, sep="\t") if items_path.exists() else None
        )

        # Top-K countries.
        country_lower = users_df["country"].fillna("unknown").astype(str).str.lower()
        country_counts = Counter(country_lower)
        top = [
            c for c, _ in country_counts.most_common(self.TOP_COUNTRIES_K)
            if c != "unknown"
        ]
        self._top_countries = set(top)

        item_lookup: Dict[str, str] = {}
        if items_df is not None:
            id_col = "artist_id" if "artist_id" in items_df.columns else items_df.columns[0]
            name_col = (
                "artist_name"
                if "artist_name" in items_df.columns
                else (items_df.columns[1] if len(items_df.columns) > 1 else id_col)
            )
            for _, r in items_df.iterrows():
                item_lookup[str(r[id_col])] = str(r[name_col])

        records: Dict[str, UserRecord] = {}
        for _, urow in users_df.iterrows():
            uid = str(urow["user_id"])
            records[uid] = UserRecord(
                user_id=uid,
                demographics={
                    "gender": _normalize_gender(urow.get("gender")),
                    "age_group": _age_to_group(urow.get("age")),
                    "country": self._country_or_other(
                        str(urow.get("country", "unknown")).lower()
                    ),
                },
                history=[],
            )

        # First pass: compute per-user median playcount.
        per_user_pc: Dict[str, List[float]] = defaultdict(list)
        for _, r in events_df.iterrows():
            uid = str(r["user_id"])
            if uid not in records:
                continue
            per_user_pc[uid].append(float(r.get("playcount", r.get("count", 0)) or 0))
        user_median: Dict[str, float] = {
            uid: float(median(pcs)) if pcs else 0.0
            for uid, pcs in per_user_pc.items()
        }

        # Second pass: build interactions.
        item_id_col = "artist_id" if "artist_id" in events_df.columns else (
            "track_id" if "track_id" in events_df.columns else events_df.columns[1]
        )
        for _, r in events_df.iterrows():
            uid = str(r["user_id"])
            if uid not in records:
                continue
            iid = str(r[item_id_col])
            pc = float(r.get("playcount", r.get("count", 0)) or 0)
            title = item_lookup.get(iid, iid)
            records[uid].history.append(
                InteractionItem(
                    item_id=iid,
                    title=title,
                    metadata={"user_median_playcount": user_median.get(uid, 0.0)},
                    rating=pc,
                    timestamp=None,
                )
            )

        return list(records.values())
