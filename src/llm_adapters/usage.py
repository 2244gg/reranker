"""JSONL usage logger for LLM API call accounting."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional


class UsageLogger:
    """Append-only JSONL logger.

    Each ``log`` call writes one line with model + token counts + timestamp.
    Cache hits should NOT call this (they don't incur API cost).
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        experiment_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        record = {
            "ts": time.time(),
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "experiment_id": experiment_id,
        }
        if extra:
            record["extra"] = extra
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def total_tokens(self, experiment_id: Optional[str] = None) -> dict:
        """Aggregate tokens (optionally filtered by experiment_id)."""
        if not self.log_path.exists():
            return {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        p = c = n = 0
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if experiment_id is not None and rec.get("experiment_id") != experiment_id:
                    continue
                p += int(rec.get("prompt_tokens", 0))
                c += int(rec.get("completion_tokens", 0))
                n += 1
        return {"prompt_tokens": p, "completion_tokens": c, "calls": n}
