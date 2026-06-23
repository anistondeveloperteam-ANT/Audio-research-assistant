"""ROUTING: a calculation / reasoning question is ANSWERED BY REASONING, never sent to the code agent
— even when the semantic LLM classifier misfires and says "code". A deterministic reasoning veto
(strong reasoning cue + NO code-production signal) overrides the model. The mere presence of numbers,
a formula, or a needed numeric result must not route to code.

Proves:
  (a) a quantitative word problem with "show your reasoning" routes to DIRECT REASONING even when the
      semantic classifier is mocked to return code_task=true (the veto short-circuits BEFORE the LLM);
  (b) a genuine "write a function / program" request still routes to the code agent (incl. when
      "show your reasoning" is appended — a code word keeps the veto from firing);
  (c) numeric / formula phrasing alone does not trigger code routing;
  (d) the regex backstop does not over-trigger on calculation phrasing;
  plus is_reasoning_question unit behavior.

Deterministic + offline: the LLM provider is mocked (content-routed). No network.
"""
import json
import threading

import pytest

import backend.answering.task_classifier as tc
from backend.answering.code_intent import is_code_intent, is_reasoning_question


CALC = [
    "How much storage is needed for 3 minutes of 44.1 kHz 16-bit stereo audio? Give your answer in MB and show your reasoning.",
    "How many MB does a 3-minute 44.1 kHz 16-bit stereo audio file need? Show your reasoning.",
    "What is the value of 17 times 23? Show your reasoning.",
    "Explain why the sky is blue.",
    "How many seconds are in a leap year? Show your working.",
]


class _FakeProvider:
    """Content-routed mock. Default verdict is code_task=TRUE, so any question that reaches the LLM is
    called 'code' — the veto must keep calculation questions away from it entirely."""

    def __init__(self, mapping=None, calls=None):
        self._mapping = mapping or {}
        self._calls = calls
        self._lock = threading.Lock()

    @property
    def is_available(self):
        return True

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3):
        query = messages[-1]["content"] if messages else ""
        if self._calls is not None:
            with self._lock:
                self._calls.append(query)
        payload = self._mapping.get(query, {"code_task": True, "task_type": "deterministic"})
        for ch in json.dumps(payload):
            yield ch


def _patch_llm(monkeypatch, provider):
    monkeypatch.setattr("backend.llm.streaming_provider.get_provider", lambda *a, **k: provider)


# ======================================================================================
# (a) the veto OVERRIDES a semantic "code" verdict for calculation/reasoning questions.
# ======================================================================================
def test_calc_questions_vetoed_even_when_semantic_says_code(monkeypatch):
    monkeypatch.setenv("CODE_INTENT_SEMANTIC", "true")
    tc.clear_cache()
    calls = []
    _patch_llm(monkeypatch, _FakeProvider(calls=calls))     # LLM would call EVERYTHING code

    for q in CALC:
        assert tc.classify(q).code_task is False, q          # answered by reasoning, not the agent
    assert calls == []                                       # veto short-circuited BEFORE the LLM call


# ======================================================================================
# (b) genuine code intent still routes to the code agent (the veto does not fire).
# ======================================================================================
@pytest.mark.parametrize("q", [
    "write a python function to compute audio storage",
    "implement a function that calculates MB, show your reasoning",
    "write a program to simulate a pendulum",
    "give me code for the FFT",
])
def test_genuine_code_requests_still_route_to_code(monkeypatch, q):
    monkeypatch.setenv("CODE_INTENT_SEMANTIC", "true")
    tc.clear_cache()
    _patch_llm(monkeypatch, _FakeProvider())
    assert tc.classify(q).code_task is True, q


# ======================================================================================
# (c) numeric / formula phrasing alone does not trigger code routing (regex-only path).
# ======================================================================================
@pytest.mark.parametrize("q", [
    "compute 12% of 350",
    "what is 44100 * 16 * 2 * 180 / 8 expressed in MB",
    "the formula is sample_rate times bit_depth times channels times seconds over 8",
    "convert 5 miles to kilometers",
])
def test_numbers_and_formulas_alone_do_not_route_to_code(q):
    # conftest disables the semantic layer -> regex + veto only, fully deterministic.
    assert is_code_intent(q) is False, q
    assert tc.classify(q).code_task is False, q


# ======================================================================================
# (d) the regex backstop does not over-trigger on calculation phrasing.
# ======================================================================================
@pytest.mark.parametrize("q", CALC)
def test_regex_backstop_not_overtriggered_on_calculations(q):
    assert is_code_intent(q) is False, q


# ======================================================================================
# is_reasoning_question unit behavior.
# ======================================================================================
def test_is_reasoning_question_fires_only_on_calc_reasoning_without_code_signals():
    assert is_reasoning_question("How many MB does it need? Show your reasoning.")
    assert is_reasoning_question("Explain the bias-variance tradeoff in detail")
    assert is_reasoning_question("What is the value of 6 times 5? Show the steps.")
    # A code word anywhere keeps the veto from firing (the user wants software).
    assert not is_reasoning_question("write a python script and show your reasoning")
    assert not is_reasoning_question("simulate a pendulum and show your work")
    assert not is_reasoning_question("implement a function, show your steps")
    # No reasoning cue -> not a reasoning question (let the normal classifier decide).
    assert not is_reasoning_question("What is MVDR beamforming?")
    # "show X working" (demonstrate) is NOT "show your work" (steps) -> a code demo isn't vetoed.
    assert not is_reasoning_question("show MVDR working on synthetic signals")
