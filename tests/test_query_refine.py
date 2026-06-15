"""Query refinement: silently fix spelling/grammar BEFORE search, without ever
breaking a request. Pure-unit — the LLM provider is mocked, no network."""
import pytest

import backend.answering.query_refine as qr


class _FakeProvider:
    """Minimal stand-in for the streaming LLM provider."""

    def __init__(self, *, available=True, out="", calls=None, raise_exc=None, sleep=0.0):
        self._available = available
        self._out = out
        self._calls = calls
        self._raise = raise_exc
        self._sleep = sleep

    @property
    def is_available(self):
        return self._available

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3):
        if self._calls is not None:
            self._calls.append(messages[0]["content"])
        if self._sleep:
            import time
            time.sleep(self._sleep)
        if self._raise is not None:
            raise self._raise
        for ch in self._out:
            yield ch


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr("backend.llm.streaming_provider.get_provider", lambda *a, **k: provider)


@pytest.fixture(autouse=True)
def _enable_refine(monkeypatch):
    # conftest disables query-refine suite-wide for deterministic offline chat tests; these unit
    # tests exercise the enabled path (and mock the provider), so opt back in.
    monkeypatch.setenv("QUERY_REFINE", "true")
    qr.clear_cache()


def test_clean_query_skips_llm_entirely(monkeypatch):
    # Every word is recognizable -> the gate must NOT spend an LLM call.
    calls = []
    _patch_provider(monkeypatch, _FakeProvider(out="SHOULD NOT BE USED", calls=calls))
    q = "what is the best method to improve the system"
    assert qr.refine_query(q) == q
    assert calls == []


def test_typo_query_is_corrected(monkeypatch):
    corrected = "i want to explore delhi give best place"
    _patch_provider(monkeypatch, _FakeProvider(out=corrected))
    assert qr.refine_query("i want to exploer delhi give best place") == corrected


def test_provider_unavailable_returns_original(monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider(available=False, out="whatever"))
    q = "best devps resorces for kubernets"
    assert qr.refine_query(q) == q


def test_exception_returns_original(monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider(raise_exc=RuntimeError("boom")))
    q = "explan me huffman codeing"
    assert qr.refine_query(q) == q


def test_timeout_returns_original(monkeypatch):
    monkeypatch.setenv("QUERY_REFINE_TIMEOUT", "0.2")
    _patch_provider(monkeypatch, _FakeProvider(out="too late", sleep=1.0))
    q = "explan me huffman codeing"
    assert qr.refine_query(q) == q


def test_kill_switch_disables_refinement(monkeypatch):
    monkeypatch.setenv("QUERY_REFINE", "false")
    calls = []
    _patch_provider(monkeypatch, _FakeProvider(out="corrected", calls=calls))
    q = "explan me huffman codeing"
    assert qr.refine_query(q) == q
    assert calls == []


def test_sanitizes_label_quotes_and_extra_lines(monkeypatch):
    raw = 'Corrected query: "Explain me Huffman coding"\nHere is why I changed it...'
    _patch_provider(monkeypatch, _FakeProvider(out=raw))
    assert qr.refine_query("explan me huffman codeing") == "Explain me Huffman coding"


def test_rambling_output_falls_back_to_original(monkeypatch):
    q = "explan huffman"
    _patch_provider(monkeypatch, _FakeProvider(out="x " * 500))  # model rambled / answered
    assert qr.refine_query(q) == q


def test_result_is_cached_no_second_llm_call(monkeypatch):
    calls = []
    _patch_provider(monkeypatch, _FakeProvider(out="explore delhi best place", calls=calls))
    q = "exploer delhi best place"
    first = qr.refine_query(q)
    second = qr.refine_query(q)
    assert first == second == "explore delhi best place"
    assert len(calls) == 1  # second call served from cache


def test_short_or_empty_input_is_returned_unchanged(monkeypatch):
    calls = []
    _patch_provider(monkeypatch, _FakeProvider(out="nope", calls=calls))
    assert qr.refine_query("") == ""
    assert qr.refine_query("hi") == "hi"
    assert calls == []
