"""ANSWER-QUALITY layer: an answer is trusted/shown/stored/reused on its CORRECTNESS, not its SOURCE.

Proves:
  (a) a reasoning-answerable question is answered from REASONING, not refused for lack of corpus;
  (c) quality is judged ORIGIN-INDEPENDENTLY (reasoning basis judges correctness, not citations);
  (d) a stored HIGH-QUALITY answer is reused; a stored LOW-QUALITY/unverified one is NOT reused;
  (e) a freshly-verified answer UPGRADES the stored record (and dissatisfaction downgrades it);
  (f) a low-quality reasoning answer gets an honest note / refinement, never a refusal or empty;
  (g) time-sensitive queries bypass reuse;
  plus: a high-quality REASONING answer (no sources) is cacheable and reusable.

Deterministic: no network, no Docker, no real LLM.
"""
import webapp.chat_logic as cl
from backend.answering import agentic_answer as aa
from backend.memory.store import MemoryStore


# ======================================================================================
# (d)+(e)+downgrade — quality governs reuse (store level, deterministic).
# ======================================================================================
def test_high_quality_answer_is_reused(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    mem.cache_answer(user_id="u", session_id="s", question="What is the capital of France?",
                     answer="The capital of France is Paris. " * 5, verified=True)
    hit = mem.find_cached_answer(user_id="u", question="What is the capital of France?")
    assert hit and "Paris" in hit["answer"]


def test_low_quality_answer_is_not_reused(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    mem.cache_answer(user_id="u", session_id="s", question="Explain widget calibration",
                     answer="A weak, unverified answer about widget calibration. " * 4, verified=False)
    # Not reused -> the caller re-answers fresh.
    assert mem.find_cached_answer(user_id="u", question="Explain widget calibration") is None


def test_fresh_verified_answer_upgrades_the_record(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    q = "Explain the bias-variance tradeoff in detail"
    mem.cache_answer(user_id="u", session_id="s", question=q,
                     answer="Weak first attempt about the tradeoff. " * 4, verified=False)
    assert mem.find_cached_answer(user_id="u", question=q) is None         # low-quality -> not reused
    mem.cache_answer(user_id="u", session_id="s", question=q,
                     answer="A correct, verified explanation of the tradeoff. " * 4, verified=True)
    hit = mem.find_cached_answer(user_id="u", question=q)
    assert hit and "verified explanation" in hit["answer"]                # upgraded -> now reusable


def test_dissatisfaction_downgrade_blocks_reuse(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    q = "What does the Adam optimizer do?"
    mem.cache_answer(user_id="u", session_id="s", question=q,
                     answer="Adam combines momentum and RMSProp. " * 4, verified=True)
    assert mem.find_cached_answer(user_id="u", question=q) is not None
    assert mem.downgrade_cached_answer("u", q) is True
    assert mem.find_cached_answer(user_id="u", question=q) is None         # regenerate -> not replayed


# ======================================================================================
# (c) — quality judged independently of ORIGIN.
# ======================================================================================
class _VProvider:
    is_available = True
    model = "fake"

    def __init__(self, verdict_json, capture):
        self.verdict_json, self.capture = verdict_json, capture

    def stream_chat(self, messages, system="", **k):
        self.capture["system"] = system
        return [self.verdict_json]


def test_reasoning_basis_judges_correctness_not_citations():
    cap = {}
    v = aa.verify_answer(_VProvider('{"ok": true, "score": 92}', cap),
                         question="What is 2+2?", evidence="", answer="2+2 = 4 by arithmetic.",
                         basis="reasoning")
    assert aa.verification_passed(v)                              # a correct reasoning answer passes...
    s = cap["system"].lower()
    assert "answer-quality judge" in s and "do not require citations" in s   # ...origin-independent judge


def test_evidence_basis_uses_the_evidence_verifier():
    cap = {}
    aa.verify_answer(_VProvider('{"ok": true, "score": 90}', cap),
                     question="Q", evidence="1. relevant evidence", answer="A [1]", basis="evidence")
    assert "evidence verifier" in cap["system"].lower()          # evidence basis -> grounding/citations


# ======================================================================================
# (a)+(f)+reasoning-cache — the reasoning fallback (origin-independent answering).
# ======================================================================================
class _FakeTrace:
    def set(self, **k):
        return self

    def end(self):
        pass


class _FakeProvider:
    is_available = True
    model = "fake"

    def __init__(self, verdict='{"ok": true, "score": 95}', answer=None):
        self.verdict = verdict
        self.answer = answer or ["17 times 23 equals 391. ",
                                 "Derivation: 17*23 = 17*20 + 17*3 = 340 + 51 = 391. " * 2]

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "independent checker" in s:                           # the INDEPENDENT confirmation pass
            return ['{"agrees": true, "confidence": 90}']
        if "answer-quality judge" in s:                          # the (dependent) reasoning verify
            return [self.verdict]
        return self.answer                                       # the reasoning draft

    def unavailable_message(self):
        return "n/a"


def _start(mem, q):
    info = mem.start_question(mem.create_session(user_id="local"), q)
    return info["turn_id"], info["node_id"]


def test_reasoning_fallback_answers_a_solvable_question(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, "What is 17 times 23?")
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setattr(cl, "get_provider", lambda: _FakeProvider())
    events = list(cl._reasoning_fallback("What is 17 times 23?", mem, sid, qid["turn_id"],
                                         qid["node_id"], "local", False, None, None, _FakeTrace()))
    done = [e for e in events if e["type"] == "done"][0]
    assert "391" in done["answer"]                               # answered from reasoning
    assert "couldn't find" not in done["answer"].lower()         # NOT refused for lack of corpus
    assert any(e["type"] == "token" for e in events)             # it streamed a real answer


def test_low_quality_reasoning_gets_honest_note_not_refusal(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, "Give the exact population of a future Mars colony")
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setattr(cl, "get_provider",
                        lambda: _FakeProvider(verdict='{"ok": false, "score": 35, "needs_more_search": true}'))
    events = list(cl._reasoning_fallback("Give the exact population of a future Mars colony", mem, sid,
                                         qid["turn_id"], qid["node_id"], "local", False, None, None,
                                         _FakeTrace(), no_sources_enabled=True))
    done = [e for e in events if e["type"] == "done"][0]
    assert "best-effort reasoning" in done["answer"]             # honest note, not a hard refusal
    assert done["answer"].strip()                                # never empty


def test_verified_reasoning_answer_is_cached_for_reuse(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, "Explain why the sky is blue")
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "true")
    monkeypatch.setattr(cl, "get_provider",
                        lambda: _FakeProvider(answer=["The sky is blue due to Rayleigh scattering. " * 6]))
    list(cl._reasoning_fallback("Explain why the sky is blue", mem, sid, qid["turn_id"],
                                qid["node_id"], "uX", True, None, None, _FakeTrace()))
    hit = mem.find_cached_answer(user_id="uX", question="Explain why the sky is blue")
    assert hit and "Rayleigh" in hit["answer"]                   # a REASONING answer (no sources) is reusable


# ======================================================================================
# (g) — time-sensitive queries bypass reuse / caching.
# ======================================================================================
def test_time_sensitive_query_bypasses_cache():
    assert cl._freshness_sensitive("what is the latest version of python")
    assert cl._freshness_sensitive("current state of the art in 2025")
    # A freshness query is never cached (so it always re-searches), regardless of answer quality.
    assert not cl._cacheable_answer("what is the latest framework",
                                    "A long answer about the latest framework. " * 5, [])
    # A normal question with a good answer IS cacheable (quality, not source presence).
    assert cl._cacheable_answer("explain gradient descent",
                                "Gradient descent iteratively steps down the loss. " * 5, [])
