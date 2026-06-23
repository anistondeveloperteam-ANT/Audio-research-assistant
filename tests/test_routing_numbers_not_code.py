"""ROUTER: route by what the user WANTS (a program vs. an answer), not by whether math is
involved. A quantitative word problem ('compute / calculate / how much ... show your reasoning')
is a REASONING question -> direct answer, NOT the code agent. Only a genuine request to
write/run/simulate software (or a computation that truly needs execution) routes to the code agent.

Proves:
  (a) calculation word problems route to REASONING (code_task=False) -- at the regex layer
      (real, semantic off) AND the semantic layer (mocked LLM returning the corrected verdict);
  (b) a genuine 'write a program / simulation' request still routes to the CODE agent;
  (c) numbers / formulas / the words 'compute'|'calculate'|'function'|'formula' alone do NOT
      trigger the code path;
  plus prompt-content assertions pinning the new 'wants a program vs. an answer' framing.

Deterministic + offline: conftest sets CODE_INTENT_SEMANTIC=false (pure regex); the semantic
test opts back in and mocks the provider. No network, no LLM.
"""
import json
import threading

import pytest

import backend.answering.task_classifier as tc
from backend.answering.code_intent import is_code_intent

# ----------------------------------------------------------------------------------------------
# Quantitative REASONING questions -- numbers/formulas/'compute' present, but the user wants an
# ANSWER, not a program. Every one must route to reasoning (code_task=False).
# ----------------------------------------------------------------------------------------------
CALC_REASONING = [
    "How much storage is needed for 3 minutes of 44.1 kHz 16-bit stereo audio? "
    "Give your answer in MB and show your reasoning.",
    "Compute the value of the function f(x) = 3x^2 + 2 at x = 4, showing each step.",
    "Calculate the compound interest on $1000 at 5% for 3 years and show the steps.",
    "What is 17 times 23? Show your reasoning.",
    "Derive the formula for the kinetic energy of a rotating disk.",
    "Convert 5 miles to kilometers and explain the conversion factor.",
    "Estimate how many tennis balls fit in a school bus, showing your assumptions.",
    "What is the area of a circle with radius 7? Show the calculation.",
]

# Genuine CODE intent -- the user wants software written/run/simulated. (All are regex hits, so
# they route to the agent even with the semantic layer off.)
GENUINE_CODE = [
    "implement quicksort",
    "write a function that sorts a list",
    "give me python code for the FFT",
    "simulate a damped pendulum",
    "write a Python script to process a CSV dataset",
    "benchmark mergesort vs quicksort",
]


# ----------------------------------------------------------------------------------------------
# (a)+(c) regex layer (semantic OFF via conftest): numbers/calculation never trigger code.
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", CALC_REASONING)
def test_calculation_word_problems_route_to_reasoning(q):
    assert is_code_intent(q) is False, q          # the regex backstop is innocent on numbers
    assert tc.classify(q).code_task is False, q    # ...and the router agrees -> reasoning


@pytest.mark.parametrize("q", GENUINE_CODE)
def test_genuine_code_requests_route_to_code(q):
    assert tc.classify(q).code_task is True, q     # a real 'write/run/simulate' request -> agent


def test_numbers_and_calc_words_alone_do_not_trigger_code():
    # The trigger words 'compute', 'calculate', 'function', 'formula' and a required numeric
    # answer must NOT, by themselves, route to the code agent.
    for q in ("compute the average of 4, 8, 15 and 16",
              "calculate 12% of 350",
              "what is the numerical value of the function at x=2",
              "give the formula for kinetic energy"):
        assert is_code_intent(q) is False, q
        assert tc.classify(q).code_task is False, q


# ----------------------------------------------------------------------------------------------
# (a)+(b) semantic layer: with the corrected prompt, the LLM keeps calculations as reasoning and
# still routes genuine code/simulation to the agent. Provider is mocked (offline, deterministic).
# ----------------------------------------------------------------------------------------------
class _FakeProvider:
    """Content-routed, thread-safe mock returning JSON based on the user message."""

    def __init__(self, mapping):
        self._mapping = mapping
        self._default = {"code_task": False, "task_type": "none", "confidence": 0.9}
        self._lock = threading.Lock()

    @property
    def is_available(self):
        return True

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3):
        query = messages[-1]["content"] if messages else ""
        with self._lock:
            payload = self._mapping.get(query, self._default)
        for ch in json.dumps(payload):
            yield ch


@pytest.fixture
def _semantic_on(monkeypatch):
    monkeypatch.setenv("CODE_INTENT_SEMANTIC", "true")
    tc.clear_cache()


def _patch(monkeypatch, provider):
    monkeypatch.setattr("backend.llm.streaming_provider.get_provider", lambda *a, **k: provider)


def test_semantic_layer_keeps_calculation_as_reasoning(monkeypatch, _semantic_on):
    q = ("How much storage for 3 minutes of 44.1 kHz 16-bit stereo audio, in MB? "
         "Show your reasoning.")
    # The corrected prompt makes the LLM return code_task=false for a calculation; the regex
    # misses it too -> the union stays reasoning.
    _patch(monkeypatch, _FakeProvider({q: {"code_task": False, "task_type": "none"}}))
    assert is_code_intent(q) is False
    assert tc.classify(q).code_task is False


def test_semantic_layer_routes_write_program_to_code(monkeypatch, _semantic_on):
    # A 'write Python to ...' request is a regex MISS, so the semantic layer must carry it to code.
    q = "write python to FFT a chirp"
    assert is_code_intent(q) is False                                  # regex alone misses it
    _patch(monkeypatch, _FakeProvider({q: {"code_task": True, "task_type": "numeric_algorithm"}}))
    res = tc.classify(q)
    assert res.code_task is True and res.task_type == "numeric_algorithm"


def test_semantic_layer_keeps_document_question_as_retrieval(monkeypatch, _semantic_on):
    q = "What is MVDR beamforming?"
    _patch(monkeypatch, _FakeProvider({}))                            # default -> code_task False
    assert tc.classify(q).code_task is False


# ----------------------------------------------------------------------------------------------
# Prompt content: the deterministic proxy for the model's behavior (matches how this repo tests
# LLM prompts). The router prompt must decide by intent, and must say numbers != code.
# ----------------------------------------------------------------------------------------------
def test_router_prompt_decides_by_intent_not_by_math():
    s = tc._SYSTEM_PROMPT.lower()
    assert "decide by what the user wants" in s                       # intent, not surface form
    assert "the mere presence of numbers" in s and "is not a code task" in s   # numbers != code
    assert "show your reasoning" in s                                 # the reasoning signal
    assert "route to the code agent" in s                             # only genuine code intent
    assert "program produced or run" in s                             # 'wants a program' framing
    # 'compute' is now listed as a REASONING signal, not a code trigger.
    assert "signals: 'compute'" in s
