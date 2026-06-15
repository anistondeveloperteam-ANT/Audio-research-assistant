"""Semantic code-intent + task-type classification. Pure-unit — the LLM provider
is mocked (content-routed, thread-safe), no network."""
import json
import threading

import pytest

import backend.answering.task_classifier as tc
from backend.answering.code_intent import is_code_intent

# The 6 diverse phrasings from the spec — ALL must route to the agent, regardless
# of wording or domain. Four of them are regex misses today (proving the gap).
DIVERSE = [
    ("simulate a damped pendulum", "simulation"),
    ("benchmark quicksort vs mergesort", "deterministic"),
    ("show MVDR working on synthetic signals", "numeric_algorithm"),
    ("price a European option", "numeric_algorithm"),
    ("model SIR epidemic spread", "simulation"),
    ("write python to FFT a chirp", "numeric_algorithm"),
]

NOT_CODE = [
    "What is MVDR beamforming?",
    "Explain the Black-Scholes model",
    "Compare Raft and Paxos",
    "Give me the latest papers on speech enhancement",
]


class _FakeProvider:
    """Content-routed, thread-safe mock: returns JSON based on the user message."""

    def __init__(self, mapping, *, available=True, default=None, raise_exc=None, calls=None):
        self._mapping = mapping
        self._available = available
        self._default = default or {"code_task": False, "task_type": "none", "confidence": 0.9}
        self._raise = raise_exc
        self._calls = calls
        self._lock = threading.Lock()

    @property
    def is_available(self):
        return self._available

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3):
        query = messages[-1]["content"] if messages else ""
        if self._calls is not None:
            with self._lock:
                self._calls.append(query)
        if self._raise is not None:
            raise self._raise
        payload = self._mapping.get(query, self._default)
        for ch in json.dumps(payload):
            yield ch


def _patch(monkeypatch, provider):
    monkeypatch.setattr("backend.llm.streaming_provider.get_provider", lambda *a, **k: provider)


@pytest.fixture(autouse=True)
def _enable_semantic(monkeypatch):
    # conftest disables the semantic classifier suite-wide for deterministic routing;
    # these unit tests exercise the LLM path, so opt back in (tests mock the provider).
    monkeypatch.setenv("CODE_INTENT_SEMANTIC", "true")
    tc.clear_cache()


@pytest.mark.parametrize("query,task_type", DIVERSE)
def test_diverse_phrasings_route_to_agent(monkeypatch, query, task_type):
    mapping = {query: {"code_task": True, "task_type": task_type, "confidence": 0.95}}
    _patch(monkeypatch, _FakeProvider(mapping))
    res = tc.classify(query)
    assert res.code_task is True, query
    assert res.task_type == task_type, query


def test_four_of_six_are_regex_misses():
    # Documents WHY the semantic layer is needed: regex alone misroutes these.
    misses = [q for q, _ in DIVERSE if not is_code_intent(q)]
    assert set(misses) == {
        "show MVDR working on synthetic signals",
        "price a European option",
        "model SIR epidemic spread",
        "write python to FFT a chirp",
    }


@pytest.mark.parametrize("query", NOT_CODE)
def test_explanations_stay_prose(monkeypatch, query):
    _patch(monkeypatch, _FakeProvider({}))  # default => code_task False
    assert tc.classify(query).code_task is False, query


def test_regex_hit_forces_code_task_even_if_llm_says_false(monkeypatch):
    # "implement quicksort" is an obvious regex hit; the union must keep it code.
    q = "implement quicksort"
    _patch(monkeypatch, _FakeProvider({q: {"code_task": False, "task_type": "none"}}))
    res = tc.classify(q)
    assert res.code_task is True
    assert res.task_type in tc.TASK_TYPES and res.task_type != "none"


def test_llm_unavailable_falls_back_to_regex(monkeypatch):
    _patch(monkeypatch, _FakeProvider({}, available=False))
    hit = tc.classify("implement quicksort")          # regex catches this
    miss = tc.classify("price a European option")     # regex misses this
    assert hit.code_task is True and hit.source == "regex"
    assert miss.code_task is False and miss.source == "regex"


def test_malformed_json_falls_back_to_regex(monkeypatch):
    _patch(monkeypatch, _FakeProvider({}, default="not json at all"))
    assert tc.classify("implement quicksort").code_task is True   # regex still wins
    assert tc.classify("price a European option").code_task is False


def test_exception_falls_back_to_regex(monkeypatch):
    _patch(monkeypatch, _FakeProvider({}, raise_exc=RuntimeError("boom")))
    assert tc.classify("simulate a damped pendulum").code_task is True  # regex: 'simulate'


def test_result_is_cached_no_second_llm_call(monkeypatch):
    calls = []
    q = "price a European option"
    _patch(monkeypatch, _FakeProvider(
        {q: {"code_task": True, "task_type": "numeric_algorithm"}}, calls=calls))
    first = tc.classify(q)
    second = tc.classify(q)
    assert first == second and first.code_task is True
    assert len(calls) == 1


def test_disabled_env_uses_regex_only(monkeypatch):
    monkeypatch.setenv("CODE_INTENT_SEMANTIC", "false")
    calls = []
    _patch(monkeypatch, _FakeProvider({}, calls=calls))
    assert tc.classify("price a European option").code_task is False  # regex miss, no LLM
    assert tc.classify("implement quicksort").code_task is True       # regex hit
    assert calls == []


def test_infer_task_type_resolves_concrete_type(monkeypatch):
    mapping = {"model SIR epidemic spread": {"code_task": True, "task_type": "simulation"}}
    _patch(monkeypatch, _FakeProvider(mapping))
    assert tc.infer_task_type("model SIR epidemic spread") == "simulation"


def test_infer_task_type_regex_heuristic_when_unavailable(monkeypatch):
    _patch(monkeypatch, _FakeProvider({}, available=False))
    assert tc.infer_task_type("simulate a damped pendulum") == "simulation"
    assert tc.infer_task_type("compute the FFT of a chirp") == "numeric_algorithm"
    assert tc.infer_task_type("implement quicksort") == "deterministic"
