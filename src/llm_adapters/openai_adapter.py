"""OpenAI-compatible adapter (works with official OpenAI and proxies like laozhang.ai)."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from .base import (
    AuthError,
    BaseLLMAdapter,
    LLMResponse,
    RateLimitError,
    TransientError,
)

try:
    from openai import OpenAI
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        InternalServerError,
        RateLimitError as OpenAIRateLimitError,
    )
except ImportError as exc:  # pragma: no cover - import error path
    raise ImportError(
        "openai>=1.30 is required. Install via `pip install -r requirements.txt`."
    ) from exc

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
        RetryError,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("tenacity is required. Install via requirements.txt.") from exc


class OpenAIAdapter(BaseLLMAdapter):
    """Calls OpenAI-compatible Chat Completions API."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise AuthError(
                "OPENAI_API_KEY missing. Set it via .env or environment variable."
            )
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        client_kwargs: dict = {"api_key": api_key, "timeout": self.timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------

    def _call_provider(self, prompt: str, **gen_kwargs: Any) -> LLMResponse:
        system = gen_kwargs.pop("system", None)
        temperature = gen_kwargs.pop("temperature", 0.7)
        top_p = gen_kwargs.pop("top_p", 1.0)
        max_tokens = gen_kwargs.pop("max_tokens", 1000)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        retries_seen = {"count": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(
                (
                    OpenAIRateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    TransientError,
                )
            ),
        )
        def _do_call() -> Any:
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except AuthenticationError as e:
                raise AuthError(str(e)) from e
            except OpenAIRateLimitError as e:
                retries_seen["count"] += 1
                raise
            except APIStatusError as e:
                # Treat 5xx as transient; non-5xx (4xx other than auth/rate) as fatal.
                status = getattr(e, "status_code", None)
                if status and 500 <= int(status) < 600:
                    retries_seen["count"] += 1
                    raise TransientError(str(e)) from e
                raise

        start = time.time()
        try:
            response = _do_call()
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e)) from e

        latency_ms = (time.time() - start) * 1000.0

        choice = response.choices[0]
        text = choice.message.content or ""
        finish = choice.finish_reason
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            retries=retries_seen["count"],
            finish_reason=finish,
            cached=False,
            raw={"id": getattr(response, "id", None)},
        )
