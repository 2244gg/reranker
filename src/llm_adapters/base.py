"""Base classes and shared types for LLM adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """Unified response object across providers."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    finish_reason: Optional[str] = None
    cached: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop raw to keep serialized payloads small.
        d.pop("raw", None)
        return d


class LLMError(Exception):
    """Base exception for LLM adapter errors."""


class AuthError(LLMError):
    """Raised when the provider rejects credentials (HTTP 401/403)."""


class RateLimitError(LLMError):
    """Raised when the provider rate-limits us (HTTP 429)."""


class TransientError(LLMError):
    """Retryable transient error (network, 5xx, timeout)."""


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM provider adapters.

    Subclasses must implement ``_call_provider`` returning a ``LLMResponse``.
    The public ``generate`` method handles caching, retries, and usage logging.
    """

    provider: str = "base"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache: Optional[Any] = None,
        usage_logger: Optional[Any] = None,
        max_retries: int = 6,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.cache = cache
        self.usage_logger = usage_logger
        self.max_retries = max_retries
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1000,
        system: Optional[str] = None,
        experiment_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion. Cached & retried automatically."""

        gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "system": system,
        }
        gen_kwargs.update(kwargs)

        # Cache lookup ----------------------------------------------------
        if self.cache is not None:
            cached = self.cache.get(self.model, prompt, gen_kwargs)
            if cached is not None:
                cached.cached = True
                return cached

        # Provider call ---------------------------------------------------
        response = self._call_provider(prompt, **gen_kwargs)

        # Cache write -----------------------------------------------------
        if self.cache is not None:
            self.cache.set(self.model, prompt, gen_kwargs, response)

        # Usage logging ---------------------------------------------------
        if self.usage_logger is not None:
            self.usage_logger.log(
                model=self.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                experiment_id=experiment_id,
            )

        return response

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _call_provider(self, prompt: str, **gen_kwargs: Any) -> LLMResponse:
        """Provider-specific call. Must return a populated ``LLMResponse``."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} model={self.model!r} provider={self.provider!r}>"
