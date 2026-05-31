"""SiliconFlow / Together adapter (OpenAI-compatible protocol)."""
from __future__ import annotations

import os
from typing import Any, Optional

from .base import AuthError
from .openai_adapter import OpenAIAdapter


class TogetherAdapter(OpenAIAdapter):
    """OpenAI-compatible adapter for SiliconFlow (default) or Together AI.

    Both providers implement the OpenAI Chat Completions API; we only need
    a different base_url + API key.
    """

    provider = "siliconflow"

    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        base_url = (
            base_url
            or os.getenv("SILICONFLOW_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        if not api_key:
            raise AuthError(
                "SILICONFLOW_API_KEY missing. Set it via .env or environment variable."
            )
        # Bypass OpenAIAdapter's env lookups: pass key/base_url explicitly.
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
