"""Factory: build a dataset adapter by short name."""
from __future__ import annotations

from typing import Any

from .base import BaseDatasetAdapter


def build_dataset(name: str, **kwargs: Any) -> BaseDatasetAdapter:
    """Return a dataset adapter for the given short name.

    Supported names: ``ml1m``, ``ml1m-legacy``, ``bookcrossing``, ``lfm``, ``tenrec``.
    """
    n = name.lower()
    if n == "ml1m":
        from .ml1m import Ml1mAdapter

        return Ml1mAdapter(**kwargs)
    if n in ("ml1m-legacy", "ml1m_legacy"):
        from .ml1m import Ml1mLegacyAdapter

        return Ml1mLegacyAdapter(**kwargs)
    if n in ("bookcrossing", "bx", "book-crossing"):
        from .bookcrossing import BookCrossingAdapter

        return BookCrossingAdapter(**kwargs)
    if n in ("lfm", "lfm-2b"):
        from .lfm import LfmAdapter

        return LfmAdapter(**kwargs)
    if n in ("lastfm1k", "lastfm-1k", "lfm-1k"):
        from .lastfm1k import LastFm1KAdapter

        return LastFm1KAdapter(**kwargs)
    if n == "tenrec":
        from .tenrec import TenrecAdapter

        return TenrecAdapter(**kwargs)
    raise ValueError(f"Unknown dataset name: {name}")
