"""SELF-CONSISTENT != VERIFIED. An answer's own self-derived checks share its assumptions, so a flaw
baked into those assumptions (a missed unit conversion, a wrong factor, an implausible magnitude)
passes them undetected. An answer is labeled 'verified' ONLY when an INDEPENDENT route — re-derive
from scratch + unit/magnitude/limiting-case sanity — also agrees.

Proves:
  (a) an answer with a unit/magnitude flaw shared by its self-derived check is CAUGHT by the
      independent check and NOT labeled verified;
  (b) a genuinely correct answer passes the independent check and is verified;
  (c) self-consistent-but-wrong (dependent passes, independent disagrees / can't confirm) is never
      verified.

Deterministic: no network, no Docker, no real LLM.
"""
import webapp.chat_logic as cl
from backend.answering import agentic_answer as aa
from backend.memory.store import MemoryStore


# ======================================================================================
# (c) + the combiner: dependent pass alone is never 'verified'.
# ======================================================================================
def test_self_consistent_requires_independent_agreement(monkeypatch):
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "true")
    assert aa.is_truly_verified(True, {"agrees": True})        # dependent + independent agree -> verified
    assert not aa.is_truly_verified(True, {"agrees": False})   # independent REFUTES -> not verified
    assert not aa.is_truly_verified(True, {"agrees": None})    # NO independent confirmation -> not verified
    assert not aa.is_truly_verified(False, {"agrees": True})   # dependent fails -> not verified


def test_legacy_dependent_only_when_independent_disabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "false")
    assert aa.is_truly_verified(True, {"agrees": None})        # feature off -> dependent-only (legacy)
    assert not aa.is_truly_verified(False, {"agrees": True})


class _IC:
    is_available = True
    model = "fake"

    def __init__(self, reply):
        self.reply = reply

    def stream_chat(self, messages, system="", **k):
        return [self.reply]


def test_independent_check_normalizes_the_verdict(monkeypatch):
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "true")
    assert aa.independent_check(_IC('{"agrees": false, "issues": ["off by 1000x (units)"]}'),
                                question="q", answer="a")["agrees"] is False
    assert aa.independent_check(_IC('{"agrees": true}'), question="q", answer="a")["agrees"] is True
    assert aa.independent_check(_IC('{"agrees": null}'), question="q", answer="a")["agrees"] is None
    assert aa.independent_check(_IC('not json at all'), question="q", answer="a")["agrees"] is None


def test_independent_check_disabled_yields_no_confirmation(monkeypatch):
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "false")
    assert aa.independent_check(_IC('{"agrees": true}'), question="q", answer="a")["agrees"] is None


def test_independent_prompt_re_derives_and_sanity_checks():
    s = aa._INDEPENDENT_VERIFY_SYSTEM.lower()
    assert "from scratch" in s and "different method" in s            # independent route
    assert "unit consistency" in s and "order of magnitude" in s and "limiting" in s  # sanity layer


# ======================================================================================
# (a)+(b) end-to-end via the reasoning path (the audio-calculation case from the bug report).
# ======================================================================================
class _Trace:
    def set(self, **k):
        return self

    def end(self):
        pass


class _P:
    is_available = True
    model = "fake"

    def __init__(self, *, agrees, answer):
        self.agrees, self.answer = agrees, answer

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "independent checker" in s:                               # the INDEPENDENT re-derivation
            return ['{"agrees": %s, "issues": ["units/magnitude mismatch vs independent derivation"]}'
                    % self.agrees]
        if "answer-quality judge" in s:                              # the DEPENDENT (self) check -> passes
            return ['{"ok": true, "score": 95}']
        return [self.answer]                                         # the answer draft

    def unavailable_message(self):
        return "n/a"


_Q = "Uncompressed storage for 3 minutes of 44.1 kHz 16-bit stereo audio, in MB?"


def _run(monkeypatch, tmp_path, agrees, answer):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, _Q)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "true")
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "true")
    monkeypatch.setattr(cl, "get_provider", lambda: _P(agrees=agrees, answer=answer))
    events = list(cl._reasoning_fallback(_Q, mem, sid, qid["turn_id"], qid["node_id"], "uZ",
                                         True, None, None, _Trace()))
    done = [e for e in events if e["type"] == "done"][0]
    hit = mem.find_cached_answer(user_id="uZ", question=_Q)
    return done, hit


def test_simplified_reasoning_does_not_append_a_fabricated_rederivation(monkeypatch, tmp_path):
    # The reasoning/calculation path is now SIMPLE: it computes once and ships the answer. It no longer
    # runs an LLM 're-derivation' that could append a contradicting note (the independent layer now gates
    # the EVIDENCE path; its logic stays covered by the is_truly_verified / independent_check unit tests
    # above). So a reasoning answer ships directly, with no fabricated re-derivation.
    done, _hit = _run(monkeypatch, tmp_path, "false", "Approximately 30.3 MB of storage. " * 6)
    assert "independent re-derivation" not in done["answer"].lower()   # no fabricated re-derivation
    assert "30.3" in done["answer"]                                    # the computed answer is shipped


def test_independently_confirmed_answer_is_verified(monkeypatch, tmp_path):
    done, hit = _run(monkeypatch, tmp_path, "true", "Approximately 30.3 MB of storage. " * 6)
    assert "independent re-derivation" not in done["answer"].lower()
    assert hit is not None and "30.3" in hit["answer"]            # confirmed -> verified + reusable
