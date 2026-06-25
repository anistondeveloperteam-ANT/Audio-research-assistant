"""SIMPLE, CORRECT, DIRECT reasoning/calculation answers. A calculation is computed ONCE and shipped:
no LLM 're-derivation' or 'reconcile' pass that could override correct work with a wrong value, no
forced citations, no 'state of the art'/'why this matters' padding. The only post-processing is a
DETERMINISTIC arithmetic check that can never introduce a wrong value.

Proves:
  (a) a calculation answer states the value its arithmetic yields, with NO self-override to a different
      value (no 'correction'/'reconciled'/'independent re-derivation' passage);
  (b) no correction passage flips a correct result;
  (c) no citations on a self-contained calculation;
  (d) every stated numeric equality is literally true (a planted false one is silently corrected);
  (e) the answer is concise (no padding sections);
  plus arithmetic_check unit behaviour and the routing of a calc question to the reasoning path.

Deterministic + offline: provider/memory mocked. No network.
"""
import pytest

import webapp.chat_logic as cl
from backend.answering import agentic_answer as aa
from backend.answering.arithmetic_check import false_equalities, fix_false_equalities
from backend.answering.code_intent import is_self_contained_calculation
from backend.memory.store import MemoryStore


# ======================================================================================
# Routing: a self-contained CALCULATION is answered directly; a FACTUAL lookup still retrieves.
# ======================================================================================
@pytest.mark.parametrize("q", [
    # Factual / research questions that happen to contain a number (often a year) must RETRIEVE — a
    # bare 'how much/many' is NOT a self-contained calculation (regression for the adversarial review).
    "how much funding did DeepMind get in 2023",
    "how many people attended NeurIPS 2023",
    "how much revenue did Apple make in Q3 2024",
    "explain step by step why the sky is blue",
    "explain the bias-variance tradeoff",
])
def test_factual_lookups_are_not_treated_as_self_contained_calculations(q):
    assert is_self_contained_calculation(q) is False, q


@pytest.mark.parametrize("q", [
    "How many MB does a 3-minute 44.1 kHz 16-bit stereo audio file need? Show your reasoning.",
    "what is 17*23 step by step",
    "derive 12 times 11 showing your work",
    "estimate 7 factorial step by step",
])
def test_genuine_self_contained_calculations_are_routed_directly(q):
    assert is_self_contained_calculation(q) is True, q


# ======================================================================================
# Deterministic arithmetic checker (the only post-processing on a reasoning answer).
# ======================================================================================
def test_arithmetic_checker_corrects_only_genuinely_false_equalities():
    assert fix_false_equalities("The area is 12 x 12 = 100 sq.") == "The area is 12 x 12 = 144 sq."
    assert fix_false_equalities("8 * 9 = 71 here") == "8 * 9 = 72 here"
    assert fix_false_equalities("100 / 4 = 30") == "100 / 4 = 25"
    # rounding the answer legitimately shows is accepted (not "false")
    assert false_equalities("10 / 3 = 3.33") == []
    # a CORRECT chained calculation is never mangled (the '=' applies to the whole chain)
    assert fix_false_equalities("44100 * 16 * 2 * 180 / 8 = 254016000 bytes") == \
        "44100 * 16 * 2 * 180 / 8 = 254016000 bytes"
    assert false_equalities("6 x 5 = 30, so the total is 30") == []


def test_arithmetic_checker_never_corrupts_ambiguous_or_correct_lines():
    # Regression for the adversarial review: these are correct/innocent and must be left UNTOUCHED.
    for s in ("3 / 4 = 75%",                       # fraction -> percent (not 3/4=75)
              "The success rate is 1 / 2 = 50%.",  # percent
              "Efficiency: 90 / 100 = 90%",        # percent
              "The gap 3 - 5 = 2 points",          # |a-b| magnitude (subtraction is excluded)
              "Difference of 100 - 250 = 150",     # magnitude
              "1,5 * 2 = 3",                       # European decimal comma
              "see fig 3.2 * 4 = 12",              # '3.2' is a figure number, not an operand
              "1000 * 2 = 2,000",                  # comma-grouped result
              "5 - 8 = -3"):                       # correct subtraction, left alone
        assert fix_false_equalities(s) == s, s


def test_reasoning_prompt_is_simple_compute_once_no_padding():
    s = aa.REASONING_ANSWER_SYSTEM.lower()
    assert "compute once" in s
    assert "no citations" in s and ("no padding" in s or "no unrelated sections" in s)
    assert "literally" in s                       # arithmetic must be literally true
    assert "own knowledge and step-by-step reasoning" in s   # mocks route the draft on this


# ======================================================================================
# End-to-end via the reasoning path.
# ======================================================================================
class _Trace:
    def set(self, **k):
        return self

    def end(self):
        pass


class _P:
    is_available = True
    model = "fake"

    def __init__(self, draft, verdict='{"ok": true, "score": 95}'):
        self.draft, self.verdict = draft, verdict

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "answer-quality judge" in s:                       # the reasoning verify (dependent)
            return [self.verdict]
        return [self.draft]                                   # the reasoning draft

    def unavailable_message(self):
        return "n/a"


def _run(monkeypatch, tmp_path, draft, *, verdict='{"ok": true, "score": 95}', question="Compute it."):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, question)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setattr(cl, "get_provider", lambda: _P(draft, verdict))
    events = list(cl._reasoning_fallback(question, mem, sid, qid["turn_id"], qid["node_id"], "u",
                                         False, None, None, _Trace()))
    return [e for e in events if e["type"] == "done"][0]


# A genuinely CORRECT, self-consistent calculation: 44100 x 16 x 2 x 180 = 254016000 bits, / 8 =
# 31752000 bytes ≈ 30 MB. (Every shown equality is literally true, so the deterministic arithmetic
# source-of-truth leaves it byte-for-byte unchanged.)
_GOOD_CALC = ("Parameters: 44.1 kHz, 16-bit, stereo, 180 s. Bits = rate x depth x channels x seconds "
              "= 44100 x 16 x 2 x 180 = 254016000 bits. Bytes = 254016000 / 8 = 31752000 bytes "
              "= about 30 MB. Final answer: about 30 MB.")


def test_a_states_the_arithmetic_result_with_no_self_override(monkeypatch, tmp_path):
    done = _run(monkeypatch, tmp_path, _GOOD_CALC)
    assert done["answer"] == _GOOD_CALC                       # shipped exactly as computed (no rewrite)


def test_b_no_correction_passage_flips_a_correct_result(monkeypatch, tmp_path):
    done = _run(monkeypatch, tmp_path, _GOOD_CALC)
    low = done["answer"].lower()
    assert "reconciled" not in low                            # no conclusion-matches-work override
    assert "independent re-derivation" not in low             # no fabricated re-derivation
    assert "correction" not in low and "on second thought" not in low


def test_c_no_citations_on_a_self_contained_calculation(monkeypatch, tmp_path):
    done = _run(monkeypatch, tmp_path, _GOOD_CALC)
    assert "[1]" not in done["answer"] and "sources:" not in done["answer"].lower()


def test_d_false_equality_is_silently_corrected(monkeypatch, tmp_path):
    done = _run(monkeypatch, tmp_path, "The area of a 12 by 12 square is 12 x 12 = 100 square units.")
    assert "12 x 12 = 144" in done["answer"]                  # the literally-false result is fixed
    assert "= 100" not in done["answer"]


def test_e_answer_is_concise_no_padding(monkeypatch, tmp_path):
    done = _run(monkeypatch, tmp_path, _GOOD_CALC)
    low = done["answer"].lower()
    assert "state of the art" not in low and "why this matters" not in low


# ======================================================================================
# Routing: a calculation question goes to the reasoning path (no retrieval, no code agent).
# ======================================================================================
def test_calc_question_routes_to_reasoning_without_retrieval(monkeypatch, tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "false", "ENABLE_LOCAL_RAG": "true", "ENABLE_WEB_SEARCH": "true",
                 "CODE_INTENT_SEMANTIC": "false"}.items():
        monkeypatch.setenv(k, v)
    draft = "Storage = 44100 x 16 x 2 x 180 / 8 = 31752000 bytes = about 30 MB."
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _P(draft))
    fail = lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieval ran for a reasoning question"))
    monkeypatch.setattr(cl, "_gather_local_items", fail)
    monkeypatch.setattr(cl, "_gather_external_items", fail)

    done = None
    for ev in cl.stream_chat_events(
            sid, "How many MB does a 3-minute 44.1 kHz 16-bit stereo audio file need? Show your reasoning."):
        if ev["type"] in ("done", "error", "sanity"):
            done = ev
            break
    assert done and done["type"] == "done"
    assert "30 MB" in done["answer"] and done.get("cached") is not True   # answered by reasoning, no search
