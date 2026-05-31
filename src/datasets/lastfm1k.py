"""LastFM-1K (Celma 2010) adapter.

Differs from ``LfmAdapter`` (LFM-2b) in the on-disk format:
  * ``userid-profile.tsv`` columns: userid, gender, age, country, signup
  * ``userid-timestamp-artid-artname-traid-traname.tsv``: 19M rows of
    per-track listening events. We stream this file in chunks and aggregate
    to per-(user, item) playcounts in memory.

Granularity (``granularity`` constructor flag):
  * ``"artist"`` (default) — backward-compatible: one ``InteractionItem`` per
    distinct artist the user listened to. ``item_id == artist_name``.
  * ``"track"`` — one ``InteractionItem`` per distinct (artist, track) pair.
    ``item_id = "ARTIST ||| TRACK"`` (pipe-separated to disambiguate
    same-titled songs by different artists). ``title = "TRACK - Artist"``
    so both the LLM prompt and the popularity table preserve disambiguation.
    ``domain()`` returns ``"music_track"`` so a dedicated prompt template
    (``prompts/music_track_en.j2``) is selected.

Per the user's research scope, this adapter exposes ``gender + age_group``
demographics by default (2 dims). Set ``include_country=True`` to also
expose ``country`` for 3-way intersectional analysis.

Pickle cache
------------
The events file is 2.5 GB and streaming + aggregation costs 4-6 minutes
on every run. To keep the rerank-iteration loop snappy, ``_load_users``
memoizes the materialized ``List[UserRecord]`` to
``data/cache/lastfm1k_<key>.pkl`` where ``<key>`` is a hash of the
(raw_dir, events file mtime, granularity, min_interactions,
min_track_listeners, include_country) tuple. Set the env var
``LASTFM1K_DISABLE_CACHE=1`` to bypass.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .base import BaseDatasetAdapter, InteractionItem, UserRecord
from .lfm import _normalize_gender


def _age_to_group_2bucket(age) -> str:
    """LFM-1K-specific age binning.

    The dataset has only 285 users with valid age (mean=25, std=7); the
    >=35 long tail is single digits. We collapse ``[0, 25)`` -> ``young``
    and ``[25, 100]`` -> ``middle-aged`` (absorbing what would have been
    ``elderly``). This keeps both age slices around 120-160 users so
    popularity B_norm has signal in both columns.

    Returns the same string vocabulary the slicer / rerank lookup tables
    already understand (``young`` / ``middle-aged`` / ``unknown``), so
    column names land on ``age_Youth`` / ``age_Middle``.
    """
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if pd.isna(a) or a <= 0 or a > 100:
        return "unknown"
    if a < 25:
        return "young"
    return "middle-aged"


# Sentinel separator for composite track keys. Pipe-space chosen because it
# does not appear in any artist or track name observed in the LFM-1K dump.
TRACK_KEY_SEP = " ||| "
# Display separator between track and artist. We use a plain ASCII hyphen
# (with surrounding spaces) so the popularity table key matches the format
# the LLM is asked to emit ("Track - Artist"). The runner disables the
# default "strip trailing - decoration" behaviour for the ``music_track``
# domain so the artist suffix survives parsing.
TRACK_TITLE_SEP = " - "


class LastFm1KAdapter(BaseDatasetAdapter):
    """LastFM-1K listening dataset (Celma 2010).

    Default demographic schema: ``gender + age_group``.
    Pass ``include_country=True`` to add ``country`` (3 dims).
    Pass ``granularity="track"`` to switch from artist-level to song-level
    items; this also flips ``domain()`` to ``"music_track"`` so the prompt
    builder picks ``prompts/music_track_en.j2``.
    """

    name = "lastfm1k"

    DEFAULT_RAW_DIR = "data/raw/lastfm-1k/lastfm-dataset-1K"
    PROFILE_FILE = "userid-profile.tsv"
    EVENTS_FILE = "userid-timestamp-artid-artname-traid-traname.tsv"
    EVENTS_CHUNK_SIZE = 1_000_000  # rows per chunk during aggregation
    TOP_COUNTRIES_K = 10

    SUPPORTED_GRANULARITIES = ("artist", "track")

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        min_interactions: int = 20,
        include_country: bool = False,
        granularity: str = "artist",
        min_track_listeners: int = 1,
    ) -> None:
        super().__init__(
            raw_dir=raw_dir or self.DEFAULT_RAW_DIR,
            rating_threshold=None,
            min_interactions=min_interactions,
        )
        if granularity not in self.SUPPORTED_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {self.SUPPORTED_GRANULARITIES}, "
                f"got {granularity!r}"
            )
        self.include_country = include_country
        self.granularity = granularity
        # Optional cold-track filter: drop tracks listened by fewer than
        # ``min_track_listeners`` distinct users (track mode only). Helps
        # avoid the long-tail collapse where almost every B_norm == 0.
        self.min_track_listeners = max(1, int(min_track_listeners))
        self._top_countries: Optional[set] = None

    # ------------------------------------------------------------------

    def domain(self) -> str:
        # Prompt builder picks ``{domain}_{lang}.j2``; track mode opts into
        # the dedicated, song-aware template.
        return "music_track" if self.granularity == "track" else "music"

    def demographic_schema(self) -> List[str]:
        if self.include_country:
            return ["gender", "age_group", "country"]
        return ["gender", "age_group"]

    # ------------------------------------------------------------------

    def positive_feedback_predicate(self, item: InteractionItem) -> bool:
        thr = (item.metadata or {}).get("user_median_playcount")
        pc = item.rating
        if pc is None or thr is None:
            return True
        if self.granularity == "track":
            # Track-level data is much sparser (most tracks played only
            # once). A strict ``> median`` would be too permissive when the
            # median is 1 (most users) yet too aggressive when a power user
            # has a high median. Combine both: require at least 2 plays AND
            # exceed the user's median.
            return float(pc) >= 2.0 and float(pc) > float(thr)
        return float(pc) > float(thr)

    def _country_or_other(self, country: str) -> str:
        if not self.include_country:
            return country
        if self._top_countries is None or country in ("", "unknown"):
            return country or "unknown"
        return country if country in self._top_countries else "other"

    # ------------------------------------------------------------------

    def _load_profile(self) -> pd.DataFrame:
        path = Path(self.raw_dir) / self.PROFILE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"LastFM-1K profile file not found at {path}. "
                f"See data/README.md for download instructions."
            )
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["user_id", "gender", "age", "country", "signup"],
            engine="python",
            on_bad_lines="skip",
        )
        return df

    def _stream_aggregate_plays(
        self,
        valid_user_ids: Iterable[str],
    ) -> Tuple[Dict[str, Counter], Dict[str, str]]:
        """Stream the 2.5GB events file in chunks.

        Returns ``(per_user, item_titles)`` where:
          * ``per_user[uid]`` is a ``Counter`` over item keys; keys are
            artist names in artist mode and ``"ARTIST ||| TRACK"`` in track
            mode.
          * ``item_titles[item_key]`` is the human-readable display title.
        """
        path = Path(self.raw_dir) / self.EVENTS_FILE
        if not path.exists():
            raise FileNotFoundError(f"LastFM-1K events file not found at {path}")

        valid = set(valid_user_ids)
        per_user: Dict[str, Counter] = defaultdict(Counter)
        item_titles: Dict[str, str] = {}

        # Tab-separated, no header.
        # Columns: 0=userid, 1=ts, 2=artid, 3=artname, 4=traid, 5=traname
        if self.granularity == "track":
            usecols = [0, 3, 5]
            names = ["user_id", "artist_name", "track_name"]
        else:
            usecols = [0, 3]
            names = ["user_id", "artist_name"]

        reader = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=usecols,
            names=names,
            chunksize=self.EVENTS_CHUNK_SIZE,
            on_bad_lines="skip",
            dtype=str,
            engine="c",
            quoting=3,  # QUOTE_NONE: do not interpret " as quote char
        )
        for chunk in reader:
            chunk = chunk[chunk["user_id"].isin(valid)]
            if chunk.empty:
                continue
            chunk = chunk.dropna(subset=["artist_name"])
            if self.granularity == "track":
                chunk = chunk.dropna(subset=["track_name"])
                # Drop rows with empty track_name after stripping.
                chunk = chunk[chunk["track_name"].astype(str).str.len() > 0]
                if chunk.empty:
                    continue
                for uid, an, tn in zip(
                    chunk["user_id"].values,
                    chunk["artist_name"].values,
                    chunk["track_name"].values,
                ):
                    an_s = str(an)
                    tn_s = str(tn)
                    key = f"{an_s}{TRACK_KEY_SEP}{tn_s}"
                    per_user[str(uid)][key] += 1
                    if key not in item_titles:
                        item_titles[key] = f"{tn_s}{TRACK_TITLE_SEP}{an_s}"
            else:
                for uid, an in zip(chunk["user_id"].values, chunk["artist_name"].values):
                    an_s = str(an)
                    per_user[str(uid)][an_s] += 1
                    item_titles.setdefault(an_s, an_s)
        return per_user, item_titles

    def _filter_cold_tracks(
        self,
        per_user: Dict[str, Counter],
    ) -> None:
        """Drop tracks listened by fewer than ``min_track_listeners`` users.

        Operates in-place; only meaningful for track-mode. Leaves artist-mode
        untouched so the legacy regression baseline stays bit-identical.
        """
        if self.granularity != "track" or self.min_track_listeners <= 1:
            return
        listener_counts: Counter = Counter()
        for counter in per_user.values():
            listener_counts.update(counter.keys())
        cold = {
            k for k, n in listener_counts.items()
            if n < self.min_track_listeners
        }
        if not cold:
            return
        for uid, counter in per_user.items():
            for k in cold:
                if k in counter:
                    del counter[k]

    def _cache_path(self) -> Path:
        """Path to the pickled UserRecord list for this configuration."""
        events_path = Path(self.raw_dir) / self.EVENTS_FILE
        try:
            mtime = int(events_path.stat().st_mtime)
        except OSError:
            mtime = 0
        sig = "|".join([
            str(Path(self.raw_dir).resolve()),
            str(mtime),
            str(self.granularity),
            str(self.min_interactions),
            str(self.min_track_listeners),
            str(self.include_country),
        ])
        digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
        return Path("data/cache") / f"lastfm1k_{self.granularity}_{digest}.pkl"

    def _load_users(self) -> List[UserRecord]:
        # Pickle-cache short-circuit (skip the 4-6 minute event stream).
        cache_path = self._cache_path()
        if cache_path.exists() and not os.environ.get("LASTFM1K_DISABLE_CACHE"):
            try:
                with cache_path.open("rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, list) and cached and isinstance(cached[0], UserRecord):
                    return cached
            except (pickle.UnpicklingError, EOFError, OSError):
                # Corrupt cache: fall through and rebuild.
                pass

        profile = self._load_profile()

        # Top-K countries (only computed if needed for country dim).
        if self.include_country:
            countries = profile["country"].fillna("unknown").astype(str).str.lower()
            country_counts = Counter(countries)
            top = [
                c
                for c, _ in country_counts.most_common(self.TOP_COUNTRIES_K)
                if c not in ("unknown", "")
            ]
            self._top_countries = set(top)

        # Aggregate per-user, per-item playcounts from events stream.
        valid_user_ids = profile["user_id"].astype(str).tolist()
        per_user, item_titles = self._stream_aggregate_plays(valid_user_ids)

        # Optional cold-track removal (track mode only).
        self._filter_cold_tracks(per_user)

        # Build UserRecord objects.
        records: List[UserRecord] = []
        for _, urow in profile.iterrows():
            uid = str(urow["user_id"])
            counter = per_user.get(uid)
            if not counter:
                continue
            # Per-user median playcount as positive-feedback threshold.
            user_median_pc = float(median(counter.values()))
            history: List[InteractionItem] = []
            for item_key, pc in counter.most_common():
                history.append(
                    InteractionItem(
                        item_id=item_key,
                        title=item_titles.get(item_key, item_key),
                        metadata={"user_median_playcount": user_median_pc},
                        rating=float(pc),
                        timestamp=None,
                    )
                )
            country = self._country_or_other(
                str(urow.get("country", "unknown") or "unknown").lower()
            )
            demographics = {
                "gender": _normalize_gender(urow.get("gender")),
                "age_group": _age_to_group_2bucket(urow.get("age")),
            }
            if self.include_country:
                demographics["country"] = country
            records.append(
                UserRecord(
                    user_id=uid,
                    demographics=demographics,
                    history=history,
                )
            )

        # Persist for next run (best-effort; ignore write failures).
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass

        return records
