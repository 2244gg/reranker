"""Tests for the LLM response cache."""
from __future__ import annotations

import pytest

from src.llm_adapters.base import BaseLLMAdapter, LLMResponse
from src.llm_adapters.cache import LLMCache, make_cache_key


class _FakeAdapter(BaseLLMAdapter):
    provider = "fake"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _call_provider(self, prompt: str, **gen_kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"reply-{self.calls}-{prompt}",
            model=self.model,
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=12.0,
            retries=0,
            finish_reason="stop",
        )


def test_cache_key_stable():
    k1 = make_cache_key("m", "hello", {"temperature": 0.7})
    k2 = make_cache_key("m", "hello", {"temperature": 0.7})
    assert k1 == k2


def test_cache_key_changes_on_prompt():
    k1 = make_cache_key("m", "hello", {"temperature": 0.7})
    k2 = make_cache_key("m", "hello!", {"temperature": 0.7})
    assert k1 != k2


def test_cache_key_changes_on_model():
    k1 = make_cache_key("m1", "hi", {})
    k2 = make_cache_key("m2", "hi", {})
    assert k1 != k2


def test_cache_key_changes_on_temperature():
    k1 = make_cache_key("m", "hi", {"temperature": 0.7})
    k2 = make_cache_key("m", "hi", {"temperature": 0.8})
    assert k1 != k2


def test_cache_hit_after_set(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    adapter = _FakeAdapter(model="m1", cache=cache)

    r1 = adapter.generate("hello", temperature=0.7, max_tokens=100)
    assert adapter.calls == 1
    assert r1.cached is False

    r2 = adapter.generate("hello", temperature=0.7, max_tokens=100)
    assert adapter.calls == 1, "second call should be cached"
    assert r2.cached is True
    assert r2.text == r1.text
    cache.close()


def test_cache_miss_on_different_prompt(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    adapter = _FakeAdapter(model="m1", cache=cache)
    adapter.generate("hello")
    adapter.generate("hello!")
    assert adapter.calls == 2
    cache.close()


def test_cache_miss_on_different_temperature(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    adapter = _FakeAdapter(model="m1", cache=cache)
    adapter.generate("hi", temperature=0.7)
    adapter.generate("hi", temperature=0.8)
    assert adapter.calls == 2
    cache.close()


def test_usage_logger_skipped_on_cache_hit(tmp_path):
    from src.llm_adapters.usage import UsageLogger

    cache = LLMCache(tmp_path / "c.sqlite")
    log = UsageLogger(tmp_path / "usage.jsonl")
    adapter = _FakeAdapter(model="m1", cache=cache, usage_logger=log)

    adapter.generate("p", experiment_id="exp-1")
    adapter.generate("p", experiment_id="exp-1")  # cache hit, should not log

    totals = log.total_tokens(experiment_id="exp-1")
    assert totals["calls"] == 1
    cache.close()
