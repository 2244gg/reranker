"""Anthropic Messages API adapter."""
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
    import anthropic
    from anthropic import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        InternalServerError,
        RateLimitError as AnthropicRateLimitError,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "anthropic>=0.34 is required. Install via `pip install -r requirements.txt`."
    ) from exc

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("tenacity is required.") from exc


class AnthropicAdapter(BaseLLMAdapter):
    """Calls Anthropic Messages API (Claude family)."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        if not api_key:
            raise AuthError(
                "ANTHROPIC_API_KEY missing. Set it via .env or environment variable."
            )
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        client_kwargs: dict = {"api_key": api_key, "timeout": self.timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)

    # ------------------------------------------------------------------

    def _call_provider(self, prompt: str, **gen_kwargs: Any) -> LLMResponse:
        system = gen_kwargs.pop("system", None)
        temperature = gen_kwargs.pop("temperature", 0.7)
        top_p = gen_kwargs.pop("top_p", 1.0)
        max_tokens = gen_kwargs.pop("max_tokens", 1000)

        messages = [{"role": "user", "content": prompt}]
        retries_seen = {"count": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(
                (
                    AnthropicRateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    TransientError,
                )
            ),
        )
        def _do_call() -> Any:
            try:
                kwargs: dict = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                if system:
                    kwargs["system"] = system
                return self._client.messages.create(**kwargs)
            except AuthenticationError as e:
                raise AuthError(str(e)) from e
            except AnthropicRateLimitError:
                retries_seen["count"] += 1
                raise
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status and 500 <= int(status) < 600:
                    retries_seen["count"] += 1
                    raise TransientError(str(e)) from e
                raise

        start = time.time()
        try:
            response = _do_call()
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e)) from e

        latency_ms = (time.time() - start) * 1000.0

        # Anthropic returns content as a list of blocks; concatenate text blocks.
        text_chunks = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(getattr(block, "text", "") or "")
        text = "".join(text_chunks)

        finish = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)

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
