"""SQLite-backed LLM response cache."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import LLMResponse


_CACHE_KEY_FIELDS = ("temperature", "top_p", "max_tokens", "system")


def _stable_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Project gen kwargs to a deterministic subset used for hashing.

    Only fields that affect the output are included. Unknown kwargs that may
    be model-specific are also retained but JSON-serialized canonically.
    """
    base = {k: kwargs.get(k) for k in _CACHE_KEY_FIELDS}
    extras = {k: v for k, v in kwargs.items() if k not in _CACHE_KEY_FIELDS}
    return {"_": base, "extras": extras}


def make_cache_key(model: str, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "kwargs": _stable_kwargs(gen_kwargs),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    """Thread-safe SQLite cache for ``LLMResponse`` objects."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        text TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        finish_reason TEXT,
        latency_ms REAL,
        created_at REAL NOT NULL,
        meta TEXT
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self._conn:
            self._conn.execute(self._SCHEMA)

    # ------------------------------------------------------------------

    def get(
        self,
        model: str,
        prompt: str,
        gen_kwargs: Dict[str, Any],
    ) -> Optional[LLMResponse]:
        key = make_cache_key(model, prompt, gen_kwargs)
        with self._lock:
            row = self._conn.execute(
                "SELECT text, prompt_tokens, completion_tokens, finish_reason, "
                "latency_ms, model FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        text, p_tokens, c_tokens, finish, latency, cached_model = row
        return LLMResponse(
            text=text,
            model=cached_model,
            prompt_tokens=int(p_tokens),
            completion_tokens=int(c_tokens),
            latency_ms=float(latency or 0.0),
            retries=0,
            finish_reason=finish,
            cached=True,
        )

    def set(
        self,
        model: str,
        prompt: str,
        gen_kwargs: Dict[str, Any],
        response: LLMResponse,
    ) -> None:
        key = make_cache_key(model, prompt, gen_kwargs)
        meta = json.dumps(
            {"retries": response.retries, "kwargs": _stable_kwargs(gen_kwargs)},
            ensure_ascii=False,
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(key, model, text, prompt_tokens, completion_tokens, "
                " finish_reason, latency_ms, created_at, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    response.model,
                    response.text,
                    int(response.prompt_tokens),
                    int(response.completion_tokens),
                    response.finish_reason,
                    float(response.latency_ms),
                    time.time(),
                    meta,
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "LLMCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()
