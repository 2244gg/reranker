"""Tenrec short-video dataset adapter (gender + age demographics, Chinese domain)."""
from __future__ import annotations

from collections import defaultdict
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
    if age < 25:
        return "young"
    if age < 45:
        return "middle-aged"
    return "elderly"


def _gender_norm(g) -> str:
    if g in (1, "1", "M", "m", "male", "男"):
        return "male"
    if g in (0, "0", "F", "f", "female", "女"):
        return "female"
    if g in (2, "2"):
        return "unknown"
    return "unknown"


class TenrecAdapter(BaseDatasetAdapter):
    """Tenrec adapter (Chinese short-video).

    Tenrec ships several CSVs for different sub-tasks (QB-video, QK-video, ...).
    We use the user-feature CSV (with gender, age) joined with a video impression
    log that contains ``watch_time`` and ``video_length`` (or ``duration``).

    Expected raw files (paths relative to ``raw_dir``):
        - ``user_features.csv``: user_id, gender, age, ...
        - ``video_impressions.csv``: user_id, item_id, watch_time, video_length, ...
        - ``items.csv`` (optional): item_id, title

    Positive feedback: ``watch_time / max(video_length, 1) > 0.5``.
    """

    name = "tenrec"

    DEFAULT_RAW_DIR = "data/raw/tenrec"

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        users_file: str = "user_features.csv",
        events_file: str = "video_impressions.csv",
        items_file: str = "items.csv",
        min_interactions: int = 10,
        watch_ratio_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            raw_dir=raw_dir or self.DEFAULT_RAW_DIR,
            rating_threshold=None,
            min_interactions=min_interactions,
        )
        self.users_file = users_file
        self.events_file = events_file
        self.items_file = items_file
        self.watch_ratio_threshold = watch_ratio_threshold

    def domain(self) -> str:
        return "video"

    def demographic_schema(self) -> List[str]:
        return ["gender", "age_group"]

    # ------------------------------------------------------------------

    def positive_feedback_predicate(self, item: InteractionItem) -> bool:
        ratio = (item.metadata or {}).get("watch_ratio")
        if ratio is None:
            return False
        return float(ratio) > self.watch_ratio_threshold

    def _load_users(self) -> List[UserRecord]:
        raw = Path(self.raw_dir)
        users_path = raw / self.users_file
        events_path = raw / self.events_file
        items_path = raw / self.items_file

        if not users_path.exists():
            raise FileNotFoundError(
                f"Tenrec users file not found at {users_path}. "
                f"See data/README.md for download instructions."
            )

        users_df = pd.read_csv(users_path)
        events_df = pd.read_csv(events_path)
        items_df = pd.read_csv(items_path) if items_path.exists() else None

        item_lookup: Dict[str, str] = {}
        if items_df is not None:
            id_col = "item_id" if "item_id" in items_df.columns else items_df.columns[0]
            title_col = (
                "title"
                if "title" in items_df.columns
                else (items_df.columns[1] if len(items_df.columns) > 1 else id_col)
            )
            for _, r in items_df.iterrows():
                item_lookup[str(r[id_col])] = str(r[title_col])

        records: Dict[str, UserRecord] = {}
        for _, urow in users_df.iterrows():
            uid = str(urow["user_id"])
            records[uid] = UserRecord(
                user_id=uid,
                demographics={
                    "gender": _gender_norm(urow.get("gender")),
                    "age_group": _age_to_group(urow.get("age")),
                },
                history=[],
            )

        wt_col = "watch_time" if "watch_time" in events_df.columns else None
        len_col = (
            "video_length"
            if "video_length" in events_df.columns
            else ("duration" if "duration" in events_df.columns else None)
        )
        ratio_col = (
            "watch_ratio" if "watch_ratio" in events_df.columns else None
        )

        for _, r in events_df.iterrows():
            uid = str(r["user_id"])
            if uid not in records:
                continue
            iid = str(r["item_id"])
            if ratio_col:
                ratio = float(r[ratio_col])
            else:
                wt = float(r[wt_col]) if wt_col else 0.0
                ln = float(r[len_col]) if len_col else 0.0
                ratio = wt / ln if ln > 0 else 0.0
            title = item_lookup.get(iid, iid)
            records[uid].history.append(
                InteractionItem(
                    item_id=iid,
                    title=title,
                    metadata={"watch_ratio": ratio},
                    rating=ratio,  # store ratio in rating slot for convenience
                    timestamp=int(r["timestamp"]) if "timestamp" in r else None,
                )
            )

        return list(records.values())
