"""Task-type-aware verification: the agent classifies each task and steers test/invariant
generation accordingly (deterministic exact-output vs simulation/numeric invariants). Mocked —
no Docker, no LLM, no network. (conftest disables the semantic classifier, so infer_task_type
uses the offline regex heuristic here.)"""
import pytest

from backend.agent import loop
from test_agent import _FakeProvider, _fence, _ok, _HIDDEN_SRC, _INV_SRC


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "true")
    monkeypatch.setenv("AGENT_ANTICHEAT_SCAN", "true")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "1")
    monkeypatch.setenv("AGENT_PARALLEL_N", "1")           # deterministic; type-steering is the focus
    # These tests probe task-type steering, not the delivery gates; the stub sandbox returns no real
    # demo stdout, so the execution gate (covered by tests/test_agent_gates.py) is out of scope here.
    monkeypatch.setenv("AGENT_DELIVERY_GATES", "false")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)


def test_task_type_hint_contents():
    sim = loop._task_type_hint("simulation")
    assert "INVARIANT" in sim and "REPRODUCIBILITY" in sim.upper()
    num = loop._task_type_hint("numeric_algorithm")
    assert "put-call parity" in num and "Parseval" in num
    det = loop._task_type_hint("deterministic")
    assert "EXACT" in det
    assert loop._task_type_hint("none") == "" and loop._task_type_hint("") == ""


def _gen_user_prompts(provider, system):
    return [u for (s, u) in provider.calls if s == system]


def test_simulation_task_steers_invariants(monkeypatch):
    provider = _FakeProvider(
        requirements="- simulate a damped pendulum; return the period",
        tests=_fence("def test_period():\n    assert True"),
        first=[_fence("def simulate_pendulum(length, damping, steps):\n"
                      "    return [length]  # damped pendulum simulation")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    events = []
    res = loop.run_agent("simulate a damped pendulum and report the period",
                         use_search=False, max_iters=1, on_event=events.append)

    assert any(e.get("type") == "task_type" and e.get("task_type") == "simulation" for e in events)
    # The simulation guidance reached BOTH the visible-test and the invariant generators.
    assert any("SIMULATION / STOCHASTIC" in u for u in _gen_user_prompts(provider, loop._TESTS_SYSTEM))
    assert any("SIMULATION / STOCHASTIC" in u
               for u in _gen_user_prompts(provider, loop._INVARIANTS_SYSTEM))
    assert res.verification == "verified"


def test_numeric_algorithm_task_steers_domain_invariants(monkeypatch):
    provider = _FakeProvider(
        requirements="- price a European call with Black-Scholes",
        tests=_fence("def test_price():\n    assert True"),
        first=[_fence("def black_scholes(S, K, T, r, sigma):\n    return S - K + T")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    events = []
    loop.run_agent("price a European call with Black-Scholes and verify put-call parity",
                   use_search=False, max_iters=1, on_event=events.append)

    assert any(e.get("type") == "task_type" and e.get("task_type") == "numeric_algorithm"
               for e in events)
    assert any("NUMERIC ALGORITHM" in u for u in _gen_user_prompts(provider, loop._INVARIANTS_SYSTEM))


def test_deterministic_task_steers_exact_outputs(monkeypatch):
    provider = _FakeProvider(
        requirements="- implement quicksort(a)",
        tests=_fence("def test_q():\n    assert True"),
        first=[_fence("def quicksort(a):\n    return sorted(a)")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    events = []
    loop.run_agent("implement quicksort", use_search=False, max_iters=1, on_event=events.append)

    assert any(e.get("type") == "task_type" and e.get("task_type") == "deterministic"
               for e in events)
    assert any("DETERMINISTIC" in u for u in _gen_user_prompts(provider, loop._TESTS_SYSTEM))
