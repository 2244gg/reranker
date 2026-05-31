"""Factory to build LLM adapters by provider/model name."""
from __future__ import annotations

from typing import Any, Optional

from .base import BaseLLMAdapter
from .cache import LLMCache
from .usage import UsageLogger


# Heuristic auto-routing for common model name prefixes when provider is unset.
_MODEL_PREFIX_TO_PROVIDER = {
    "gpt-": "openai",
    "o1-": "openai",
    "claude-": "anthropic",
    "Qwen/": "siliconflow",
    "qwen": "siliconflow",
    "meta-llama/": "siliconflow",
    "Llama-": "siliconflow",
    "deepseek": "siliconflow",
}


def _infer_provider(model: str) -> str:
    for prefix, provider in _MODEL_PREFIX_TO_PROVIDER.items():
        if model.startswith(prefix) or model.lower().startswith(prefix.lower()):
            return provider
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        f"Pass provider explicitly."
    )


def build_adapter(
    *,
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    cache: Optional[LLMCache] = None,
    usage_logger: Optional[UsageLogger] = None,
    **kwargs: Any,
) -> BaseLLMAdapter:
    """Build an LLM adapter instance.

    Parameters
    ----------
    model:
        Provider-specific model identifier (e.g. ``gpt-4.1-mini``,
        ``claude-3-5-sonnet-20241022``, ``Qwen/Qwen2.5-72B-Instruct``).
    provider:
        One of ``openai`` / ``anthropic`` / ``siliconflow`` / ``together``.
        If omitted, inferred from ``model``.
    """
    provider = (provider or _infer_provider(model)).lower()

    common: dict = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        cache=cache,
        usage_logger=usage_logger,
    )
    common.update(kwargs)

    if provider in ("openai", "openai-compatible", "laozhang"):
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(**common)
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(**common)
    if provider in ("siliconflow", "together"):
        from .together_adapter import TogetherAdapter

        return TogetherAdapter(**common)

    raise ValueError(f"Unknown provider '{provider}'.")
