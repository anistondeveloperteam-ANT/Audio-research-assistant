"""CACHE RE-VALIDATION. A stored answer is a speed optimization for VERIFIED answers only — never a
way to skip the correctness checks a fresh answer must pass. On a cache hit the stored answer is
re-checked (lightweight conclusion-matches-work) before serving; a suspect or stale-logic entry is
retired and re-answered fresh, and the better answer replaces it. The cache improves over time and
never keeps serving a worse answer.

Proves:
  (a) a hit on an INCONSISTENT stored answer triggers fresh re-answering, not a blind serve;
  (b) the fresh better answer REPLACES the stale cache entry;
  (c) a stored answer failing the lightweight consistency check is not served as-is;
  (d) a genuinely verified + consistent + current entry is FAST-PATH served (no search);
  (e) time-sensitive queries bypass the cache entirely;
  plus: a STALE logic_version entry is re-answered (deploy fixes take effect), and store-level unit
  tests for logic_version round-tripping and mark_cache_unverified.

Deterministic + offline: provider/memory mocked. No network, no Docker, no real LLM.
"""
import pytest

import webapp.chat_logic as cl
from backend.answering.agentic_answer import answer_logic_version
from backend.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _enable_revalidation(monkeypatch):
    # conftest disables the consistency gate suite-wide; the serve-time re-check IS that gate, so opt
    # back in. The provider is mocked, so still offline.
    monkeypatch.setenv("AGENTIC_CONSISTENCY_CHECK", "true")
    monkeypatch.setenv("ANSWER_CACHE_REVALIDATE", "true")


# ======================================================================================
# Store-level units: logic_version + mark_cache_unverified.
# ======================================================================================
def test_logic_version_round_trips_and_gates_reuse(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    mem.cache_answer(user_id="u", session_id="s", question="What is X?",
                     answer="X is a thing, explained at sufficient length to be cacheable here. " * 2,
                     verified=True, logic_version=3)
    hit = mem.find_cached_answer(user_id="u", question="What is X?", min_logic_version=3)
    assert hit is not None and hit["logic_version"] == 3
    # An entry below the required minimum is excluded (a deploy bumps the floor -> re-answer).
    assert mem.find_cached_answer(user_id="u", question="What is X?", min_logic_version=4) is None


def test_mark_cache_unverified_by_id_blocks_reuse(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    cid = mem.cache_answer(user_id="u", session_id="s", question="What is Y?",
                           answer="Y is another thing, explained at length so it clears the cache bar. " * 2,
                           verified=True, logic_version=answer_logic_version())
    assert mem.find_cached_answer(user_id="u", question="What is Y?") is not None
    assert mem.mark_cache_unverified(int(cid)) is True
    assert mem.find_cached_answer(user_id="u", question="What is Y?") is None   # retired -> not served


# ======================================================================================
# End-to-end serve path.
# ======================================================================================
_FRESH = "FRESH-REASONED answer: the correct total is 30 units, based on the full worked steps shown here."


class _P:
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        content = messages[-1]["content"] if messages else ""
        if "internal-consistency checker" in s:                  # the serve-time / fresh re-check
            bad = "STALE-CACHED" in content                      # the stored answer is inconsistent
            return ['{"consistent": %s, "derived_result": "30", "stated_result": "35", "issues": []}'
                    % ("false" if bad else "true")]
        if "independent checker" in s:
            return ['{"agrees": true, "issues": []}']
        if "answer-quality judge" in s:
            return ['{"ok": true, "score": 95}']
        return [_FRESH]                                          # the fresh reasoning draft

    def unavailable_message(self):
        return "n/a"


def _seed(mem, sid, question, answer, *, verified=True, logic_version=None):
    mem.cache_answer(user_id="local", session_id=sid, question=question, answer=answer,
                     verified=verified,
                     logic_version=answer_logic_version() if logic_version is None else logic_version)


def _setup(monkeypatch, mem, *, web=False, local=False):
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "true", "ENABLE_LOCAL_RAG": "true" if local else "false",
                 "ENABLE_WEB_SEARCH": "true" if web else "false", "CODE_INTENT_SEMANTIC": "false",
                 "AGENTIC_INDEPENDENT_VERIFY": "true", "AUTO_REVIEW": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _P())


def _run(mem, sid, question):
    events = list(cl.stream_chat_events(sid, question))
    return events[-1], events


_Q = "What is the total of six times five?"


def test_a_inconsistent_cached_answer_triggers_fresh_reanswer(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed(mem, sid, _Q, "STALE-CACHED: 6 x 5 = 30 in the work, but the total is 35 units overall.")
    monkeypatch.setattr(cl, "_memory", mem)
    _setup(monkeypatch, mem)

    done, _events = _run(mem, sid, _Q)
    assert "FRESH-REASONED" in done["answer"]              # re-answered, not the stored answer
    assert "STALE-CACHED" not in done["answer"]
    assert done.get("cached") is not True                 # not a blind cache serve


def test_b_fresh_better_answer_replaces_the_cache_entry(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed(mem, sid, _Q, "STALE-CACHED: the total is 35 units (contradicts the work).")
    monkeypatch.setattr(cl, "_memory", mem)
    _setup(monkeypatch, mem)

    _run(mem, sid, _Q)
    # The cache now holds the fresh, verified answer (the stale entry was replaced/retired).
    hit = mem.find_cached_answer(user_id="local", question=_Q, min_logic_version=answer_logic_version())
    assert hit is not None and "FRESH-REASONED" in hit["answer"] and "STALE-CACHED" not in hit["answer"]


def test_d_verified_consistent_current_entry_is_fast_path_served(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    good = "CONSISTENT cached answer: 6 x 5 = 30, so the total is 30 units. Shown in full here for length."
    _seed(mem, sid, _Q, good)
    monkeypatch.setattr(cl, "_memory", mem)
    _setup(monkeypatch, mem)
    # Search must NOT run on a (re-validated) cache hit.
    fail = lambda *a, **k: (_ for _ in ()).throw(AssertionError("search ran on a cache hit"))
    monkeypatch.setattr(cl, "_gather_local_items", fail)
    monkeypatch.setattr(cl, "_gather_external_items", fail)

    done, _events = _run(mem, sid, _Q)
    assert done.get("cached") is True and done["answer"] == good   # fast-path served after passing re-check


def test_stale_logic_version_entry_is_reanswered(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    # A CONSISTENT but OLD-logic entry: excluded by min_logic_version -> re-answered (fix takes effect).
    _seed(mem, sid, _Q, "CONSISTENT but old-logic answer: the total is 30 units, at cacheable length here.",
          logic_version=0)
    monkeypatch.setattr(cl, "_memory", mem)
    _setup(monkeypatch, mem)

    done, _events = _run(mem, sid, _Q)
    assert "FRESH-REASONED" in done["answer"] and done.get("cached") is not True


def test_e_time_sensitive_query_bypasses_cache(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    fresh_q = "What is the latest total this year?"
    _seed(mem, sid, fresh_q, "CONSISTENT cached answer about the total, long enough to be cacheable here.")
    monkeypatch.setattr(cl, "_memory", mem)
    _setup(monkeypatch, mem)

    done, _events = _run(mem, sid, fresh_q)
    assert done.get("cached") is not True                 # freshness bypasses cache regardless of quality
