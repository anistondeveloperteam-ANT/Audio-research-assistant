"""
Tracing must be a zero-overhead no-op unless LANGFUSE_ENABLED=true AND all keys are set
(the default is off). These run fully offline — no Langfuse server, no client, no network —
and prove the "byte-identical when disabled" contract.
"""
import sys

import pytest

from backend.observability import tracing


def test_enabled_requires_flag_and_all_three_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing._enabled() is False            # a missing key keeps it off
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert tracing._enabled() is True
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert tracing._enabled() is False            # flag off keeps it off


def test_disabled_returns_shared_noops(monkeypatch):
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    trace = tracing.start_trace("chat_request", mode="Default", top_k=8)
    assert trace is tracing._NOOP_TRACE
    assert trace.span("local_rag") is tracing._NOOP_SPAN


def test_disabled_span_pipeline_does_nothing_and_never_raises(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    trace = tracing.start_trace("chat_request")
    with trace.span("llm_stream", round=1) as sp:
        sp.set(model="x", output_len=42)          # accepted + ignored
    trace.set(cached=False).end()
    tracing.flush()                               # no-op, returns cleanly


def test_disabled_never_imports_langfuse(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    sys.modules.pop("langfuse", None)
    trace = tracing.start_trace("chat_request")
    with trace.span("cache_check"):
        pass
    assert "langfuse" not in sys.modules          # client is never even imported
    assert tracing._client is None


def test_span_context_manager_never_swallows_exceptions(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    trace = tracing.start_trace("chat_request")
    with pytest.raises(ValueError):
        with trace.span("boom"):
            raise ValueError("must propagate")
