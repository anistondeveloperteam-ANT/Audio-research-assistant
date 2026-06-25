"""TIME-SENSITIVE QUESTIONS NEVER SERVED FROM CACHE. A "latest / current / newest / now / this year"
question is stale by definition if replayed from a stored answer, so the time-sensitivity decision runs
BEFORE the cache lookup and a time-sensitive question bypasses the cache entirely — it always re-answers
fresh. Stable, non-time-sensitive questions keep verified-answer reuse exactly as before.

Proves:
  (a) the freshness/time-sensitivity check runs BEFORE the cache lookup — a time-sensitive question never
      calls find_cached_answer, even with a matching stored entry;
  (b) a time-sensitive question with a matching stored entry re-answers fresh (not served from cache);
  (c) two different wordings of the same time-sensitive question both re-answer fresh;
  (d) a stable, non-time-sensitive question still reuses its verified cached answer;
  (e) a time-sensitive answer is not stored for durable reuse;
  plus: _freshness_sensitive has NO escape hatch (recency is always time-sensitive).

Deterministic + offline: provider/memory mocked; recency cues route deterministically to 'web' (no LLM).
"""
import webapp.chat_logic as cl
from backend.answering.agentic_answer import answer_logic_version
from backend.memory.store import MemoryStore


# A recency question routes to 'web' DETERMINISTICALLY (decide_source checks freshness before the LLM),
# so these tests hold even with the source router disabled suite-wide.
_FRESH_Q = "What is the latest state-of-the-art model this year?"
_FRESH_Q2 = "Which model is the newest and best right now?"
_STABLE_Q = "Explain how a hash map works."


class _P:
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "answer-quality judge" in s:
            return ['{"ok": true, "score": 95}']
        if "own knowledge and step-by-step reasoning" in s:                 # reasoning draft
            return ["FRESH reasoning answer worked out directly, long enough to be a real answer. " * 2]
        if "meticulous, broad-domain research assistant" in s:              # evidence/web draft
            return ["FRESH answer from current web sources [1], as of this year. " * 2]
        return [messages[-1]["content"] if messages else ""]

    def unavailable_message(self):
        return "n/a"


def _env(monkeypatch, mem):
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "true", "ENABLE_LOCAL_RAG": "false", "ENABLE_WEB_SEARCH": "true",
                 "CRAG_ENABLED": "true", "AUTO_REVIEW": "false", "CODE_INTENT_SEMANTIC": "false",
                 "ENABLE_AGENTIC_ANSWER_LOOP": "false", "AGENTIC_INDEPENDENT_VERIFY": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _P())
    # The web route searches externally; return a current web source so it can answer.
    monkeypatch.setattr(cl, "_gather_external_items", lambda q, k: (
        [{"source_type": "web", "title": "News this year", "url": "http://x/n",
          "text": "the current state of the art as of this year", "score": 0.8}], []))


def _seed(mem, sid, question, answer):
    mem.cache_answer(user_id="local", session_id=sid, question=question, answer=answer,
                     verified=True, logic_version=answer_logic_version())


def _drive(sid, q):
    done = None
    for ev in cl.stream_chat_events(sid, q):
        if ev["type"] in ("done", "error", "sanity"):
            done = ev
            break
    return done


def test_a_freshness_check_precedes_the_cache_lookup(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _env(monkeypatch, mem)
    _seed(mem, sid, _FRESH_Q, "STALE stored answer naming an old model. " * 4)   # a matching entry EXISTS
    calls = {"n": 0}
    orig = mem.find_cached_answer

    def tracked(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(mem, "find_cached_answer", tracked)
    done = _drive(sid, _FRESH_Q)
    assert done and done["type"] == "done"
    assert calls["n"] == 0                          # cache lookup was SKIPPED for the time-sensitive query


def test_b_time_sensitive_with_stored_entry_reanswers_fresh(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _env(monkeypatch, mem)
    _seed(mem, sid, _FRESH_Q, "STALE stored answer naming an old model. " * 4)
    done = _drive(sid, _FRESH_Q)
    assert done.get("cached") is not True           # not served from cache
    assert "STALE" not in done["answer"] and "FRESH" in done["answer"]   # re-answered fresh


def test_c_two_wordings_both_bypass_the_cache(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _env(monkeypatch, mem)
    _seed(mem, sid, _FRESH_Q, "STALE stored answer naming an old model. " * 4)
    for q in (_FRESH_Q, _FRESH_Q2):
        done = _drive(sid, q)
        assert done.get("cached") is not True and "FRESH" in done["answer"], q


def test_d_stable_question_still_reuses_its_verified_cache(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _env(monkeypatch, mem)
    stable_answer = "A hash map stores key/value pairs with O(1) average lookup via hashing. " * 3
    _seed(mem, sid, _STABLE_Q, stable_answer)
    fail = lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieval ran for a cached stable query"))
    monkeypatch.setattr(cl, "_gather_local_items", fail)
    monkeypatch.setattr(cl, "_gather_external_items", fail)
    done = _drive(sid, _STABLE_Q)
    assert done.get("cached") is True and done["answer"] == stable_answer   # verified reuse still works


def test_e_time_sensitive_answer_is_not_stored(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _env(monkeypatch, mem)
    _drive(sid, _FRESH_Q)
    # The fresh web answer must NOT be written for later durable reuse.
    assert mem.find_cached_answer(user_id="local", question=_FRESH_Q) is None


def test_freshness_has_no_escape_hatch(monkeypatch):
    # Even if someone sets the old opt-out, a recency question is ALWAYS time-sensitive now.
    monkeypatch.setenv("ANSWER_CACHE_ALLOW_FRESHNESS_QUERIES", "true")
    assert cl._freshness_sensitive("what is the latest gpt model") is True
    assert cl._freshness_sensitive("current best framework in 2026") is True
    assert cl._freshness_sensitive("explain how a hash map works") is False
