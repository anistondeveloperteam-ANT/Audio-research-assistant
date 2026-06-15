"""Compact-memory assembly in chat_logic: recent + rolling summary + relevant facts under a
token budget, summary caching, and summarization-failure fallback. The summarizer LLM is mocked
(no network)."""
import json

import pytest

import webapp.chat_logic as cl
from backend.memory.store import MemoryStore, estimate_tokens


class _FakeProvider:
    def __init__(self, payload, *, available=True, raise_exc=None, calls=None):
        self._payload = payload
        self._available = available
        self._raise = raise_exc
        self._calls = calls

    @property
    def is_available(self):
        return self._available

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0):
        if self._calls is not None:
            self._calls.append(1)
        if self._raise is not None:
            raise self._raise
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        for ch in text:
            yield ch


def _patch(monkeypatch, provider):
    monkeypatch.setattr("backend.llm.streaming_provider.get_provider", lambda *a, **k: provider)


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(tmp_path / "m.db", conversations_path=tmp_path / "c.db")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("COMPACT_MEMORY", "true")
    monkeypatch.setenv("MEMORY_RECENT_TURNS", "2")     # last 1 Q&A verbatim
    monkeypatch.setenv("MEMORY_MAX_TOKENS", "3000")
    monkeypatch.setenv("MEMORY_SUMMARY_STALE", "2")
    monkeypatch.setenv("MEMORY_MAX_FACTS", "6")


def _seed(mem, sid, pairs):
    """`pairs` completed Q&A pairs, then a trailing 'current question' (no answer yet)."""
    for i in range(pairs):
        mem.append_turn(sid, "user", f"question {i} about audio beamforming and noise")
        mem.append_turn(sid, "assistant", f"answer {i} with technical detail about it")
    mem.append_turn(sid, "user", "the current question being asked right now")


def test_assembles_recent_summary_and_relevant_facts(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 4)
    calls = []
    _patch(monkeypatch, _FakeProvider(
        {"summary": "User is building an MVDR beamformer in Python and asked several setup questions.",
         "facts": [{"key": "goal", "value": "build an MVDR beamformer"}]}, calls=calls))

    ctx = cl._build_compact_context(mem, sid, "how do I compute the MVDR weight vector?")

    assert len(ctx["history"]) == 2 and ctx["history"][-1]["role"] == "assistant"  # recent only
    assert "MVDR beamformer" in ctx["summary"]
    assert "[Conversation memory]" in ctx["system_extra"]
    assert "goal" in ctx["system_extra"]                       # extracted, relevant fact injected
    assert len(calls) == 1                                     # summarized once (older was stale)
    assert mem.get_session_summary(sid)["summary"].startswith("User is building")  # persisted
    assert any(f["key"] == "goal" for f in mem.list_facts("session", sid))         # fact persisted


def test_summary_is_cached_not_regenerated_when_not_stale(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 4)
    calls = []
    _patch(monkeypatch, _FakeProvider({"summary": "S1", "facts": []}, calls=calls))
    cl._build_compact_context(mem, sid, "q one")               # generates summary (1 call)
    assert len(calls) == 1
    # No new turns since -> nothing newly older -> must reuse the cached summary (no 2nd call).
    ctx = cl._build_compact_context(mem, sid, "q two")
    assert len(calls) == 1 and ctx["summary"] == "S1"


def test_summary_regenerates_when_more_older_turns_accumulate(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 4)
    calls = []
    _patch(monkeypatch, _FakeProvider({"summary": "S1", "facts": []}, calls=calls))
    cl._build_compact_context(mem, sid, "q")
    assert len(calls) == 1
    # The current question gets answered and a new one asked -> a full pair ages into the older
    # window -> stale -> regenerate once.
    mem.append_turn(sid, "assistant", "the answer to the current question")
    mem.append_turn(sid, "user", "a brand new follow-up question")
    cl._build_compact_context(mem, sid, "q again")
    assert len(calls) == 2


def test_token_budget_caps_assembled_context(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 6)
    monkeypatch.setenv("MEMORY_MAX_TOKENS", "120")
    _patch(monkeypatch, _FakeProvider({"summary": "X" * 4000, "facts": []}))   # ~1000-token summary
    ctx = cl._build_compact_context(mem, sid, "audio question about beamforming")
    assert ctx["tokens"] <= 120                               # hard cap honored
    assert estimate_tokens(ctx["summary"]) < 1000             # older summary was compressed
    assert len(ctx["history"]) >= 1                           # recent turns kept


def test_summarization_failure_falls_back_to_recent_only(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 4)
    _patch(monkeypatch, _FakeProvider(None, raise_exc=RuntimeError("LLM down")))
    ctx = cl._build_compact_context(mem, sid, "a question")
    assert ctx["summary"] == ""                               # no summary produced
    assert len(ctx["history"]) == 2                           # recent turns still sent
    assert mem.get_session_summary(sid)["summary"] == ""      # unchanged on failure


def test_provider_unavailable_is_safe(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 4)
    _patch(monkeypatch, _FakeProvider({"summary": "nope"}, available=False))
    ctx = cl._build_compact_context(mem, sid, "q")
    assert ctx["summary"] == "" and len(ctx["history"]) == 2


def test_compact_memory_disabled_is_recent_only(mem, monkeypatch):
    sid = mem.create_session(user_id="anjan")
    _seed(mem, sid, 5)
    monkeypatch.setenv("COMPACT_MEMORY", "false")
    calls = []
    _patch(monkeypatch, _FakeProvider({"summary": "S", "facts": []}, calls=calls))
    ctx = cl._build_compact_context(mem, sid, "q")
    assert ctx["system_extra"] == "" and ctx["summary"] == ""
    assert len(calls) == 0                                    # no summary call when disabled
    assert len(ctx["history"]) == 2
