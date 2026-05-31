"""Smoke tests for LLM adapters using monkeypatched provider clients.

These tests do NOT make real network calls. They verify:
- Successful generate path returns a unified ``LLMResponse``.
- Auth errors are surfaced as ``AuthError`` without retries.
- Rate-limit errors are retried, then re-raised as ``RateLimitError`` after
  exhausting attempts.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from src.llm_adapters.base import AuthError, LLMResponse, RateLimitError


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class _FakeOpenAIUsage:
    def __init__(self, p=12, c=34) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeOpenAIChoice:
    def __init__(self, text: str, finish: str = "stop") -> None:
        self.message = mock.SimpleNamespace(content=text)
        self.finish_reason = finish


class _FakeOpenAIResponse:
    def __init__(self, text: str = "ok") -> None:
        self.id = "resp-1"
        self.choices = [_FakeOpenAIChoice(text)]
        self.usage = _FakeOpenAIUsage()


class _FakeOpenAICompletions:
    def __init__(self, response_or_exc) -> None:
        self._response_or_exc = response_or_exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if isinstance(self._response_or_exc, Exception):
            raise self._response_or_exc
        return self._response_or_exc


class _FakeOpenAIChat:
    def __init__(self, completions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, response_or_exc) -> None:
        self._completions = _FakeOpenAICompletions(response_or_exc)
        self.chat = _FakeOpenAIChat(self._completions)


def test_openai_adapter_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from src.llm_adapters import openai_adapter as oa

    fake_client = _FakeOpenAIClient(_FakeOpenAIResponse("hello world"))
    monkeypatch.setattr(oa, "OpenAI", lambda **kw: fake_client)

    adapter = oa.OpenAIAdapter(model="gpt-4.1-mini")
    resp = adapter.generate("hi")

    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 34
    assert resp.finish_reason == "stop"
    assert resp.cached is False


def test_openai_adapter_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from src.llm_adapters.openai_adapter import OpenAIAdapter

    with pytest.raises(AuthError):
        OpenAIAdapter(model="gpt-4.1-mini")


def test_openai_adapter_auth_error_no_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from src.llm_adapters import openai_adapter as oa
    from openai import AuthenticationError

    # Build a minimal AuthenticationError-like exception.
    class _AuthExc(AuthenticationError):
        def __init__(self):
            pass

    fake_client = _FakeOpenAIClient(_AuthExc())
    monkeypatch.setattr(oa, "OpenAI", lambda **kw: fake_client)

    adapter = oa.OpenAIAdapter(model="gpt-4.1-mini", max_retries=4)
    with pytest.raises(AuthError):
        adapter.generate("hi")
    assert fake_client._completions.calls == 1, "auth error should not retry"


def test_openai_adapter_rate_limit_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from src.llm_adapters import openai_adapter as oa
    from openai import RateLimitError as OpenAIRateLimitError

    class _RLExc(OpenAIRateLimitError):
        def __init__(self):
            pass

    fake_client = _FakeOpenAIClient(_RLExc())
    monkeypatch.setattr(oa, "OpenAI", lambda **kw: fake_client)

    # Drastically reduce wait so test is fast.
    import tenacity

    monkeypatch.setattr(
        tenacity, "wait_exponential", lambda **kw: tenacity.wait_none()
    )
    # Re-import adapter module to pick up the patched wait? Adapter already imported
    # closures use the original wait_exponential; instead lower max_retries.
    adapter = oa.OpenAIAdapter(model="gpt-4.1-mini", max_retries=2)
    with pytest.raises(RateLimitError):
        adapter.generate("hi")
    assert fake_client._completions.calls >= 2, "rate-limit should be retried"


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


def test_factory_infers_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from src.llm_adapters import openai_adapter as oa
    from src.llm_adapters.factory import build_adapter

    monkeypatch.setattr(oa, "OpenAI", lambda **kw: _FakeOpenAIClient(_FakeOpenAIResponse()))
    adapter = build_adapter(model="gpt-4.1-mini")
    assert adapter.provider == "openai"


def test_factory_routes_qwen_to_siliconflow(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    from src.llm_adapters import openai_adapter as oa
    from src.llm_adapters.factory import build_adapter

    monkeypatch.setattr(oa, "OpenAI", lambda **kw: _FakeOpenAIClient(_FakeOpenAIResponse()))
    adapter = build_adapter(model="Qwen/Qwen2.5-72B-Instruct")
    assert adapter.provider == "siliconflow"


def test_factory_unknown_model_raises():
    from src.llm_adapters.factory import build_adapter

    with pytest.raises(ValueError):
        build_adapter(model="mystery-model-xyz")
