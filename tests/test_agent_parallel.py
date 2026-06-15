"""Parallel best-of-N specifics for the agent loop: candidates run concurrently, the best
genuine passer is kept from a diverse pool, anti-cheat still holds under parallelism, and an
all-fail round still escalates. Fully mocked (no Docker, no LLM, no network)."""
import threading
import time

import pytest

from backend.agent import loop
from test_agent import _FakeProvider, _fence, _ok, _runner, _HIDDEN_SRC, _INV_SRC


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "true")
    monkeypatch.setenv("AGENT_ANTICHEAT_SCAN", "true")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "1")          # keep held-out runs cheap
    monkeypatch.setenv("AGENT_PARALLEL_N", "4")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SANDBOXES", "4")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)


def test_candidates_run_concurrently(monkeypatch):
    """With AGENT_PARALLEL_N=4, the four candidate sandbox runs overlap in time."""
    provider = _FakeProvider(
        requirements="- implement widget()",
        tests=_fence("def test_w():\n    assert widget() == 1"),
        first=[_fence("def widget():\n    return 0  # partial")],
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)

    peak = {"cur": 0, "peak": 0}
    lock = threading.Lock()

    def run(code, **k):
        with lock:
            peak["cur"] += 1
            peak["peak"] = max(peak["peak"], peak["cur"])
        time.sleep(0.04)        # hold so candidates overlap
        with lock:
            peak["cur"] -= 1
        return _ok(1, 2)        # partial -> no held-out, just the 4 visible runs

    monkeypatch.setattr(loop, "run_python_auto", run)
    events = []
    loop.run_agent("Implement widget", use_search=False, max_iters=1, on_event=events.append)

    assert peak["peak"] >= 2, f"candidates did not overlap (peak={peak['peak']})"
    assert sum(1 for e in events if e.get("type") == "code") == 4   # four candidates attempted


def test_best_of_n_selects_passer_from_a_diverse_pool(monkeypatch):
    """Four DIFFERENT candidates; only one fully passes — it must be the one kept + verified."""
    provider = _FakeProvider(
        requirements="- implement widget()",
        tests=_fence("def test_w():\n    assert widget() == 7"),
        first=[_fence("def widget():\n    return 0  # bad-a"),
               _fence("def widget():\n    return 1  # bad-b"),
               _fence("def widget():\n    return 2  # bad-c"),
               _fence("def widget():\n    return 7  # winner")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        _runner(lambda c: _ok(1, 1) if "winner" in c else _ok(0, 1),
                                lambda c: _ok(2, 2)))
    events = []
    res = loop.run_agent("Implement widget", use_search=False, max_iters=1, on_event=events.append)

    assert res.success is True and res.verification == "verified"
    assert "winner" in res.best_code
    assert sum(1 for e in events if e.get("type") == "code") == 4   # all four were tried


def test_anticheat_enforced_under_parallelism(monkeypatch):
    """A round mixing 3 cheating candidates with 1 genuine: the genuine one is kept, cheats flagged."""
    cheat = "def black_scholes(S, K, T, r, sigma):\n    return 10.4506"
    genuine = "def black_scholes(S, K, T, r, sigma):\n    return S - K + T"
    provider = _FakeProvider(
        requirements="- implement black_scholes(S,K,T,r,sigma)",
        tests=_fence("def test_c():\n    assert abs(black_scholes(100,100,1,0.05,0.2) - 10.4506) < 1e-2"),
        first=[_fence(cheat), _fence(cheat), _fence(cheat), _fence(genuine)],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    events = []
    res = loop.run_agent("Give me Black-Scholes code", use_search=False, max_iters=1,
                         on_event=events.append)

    assert res.verification == "verified" and res.success is True
    assert "10.4506" not in res.best_code                       # the gaming code is never returned
    assert any(e.get("type") == "run_result" and e.get("cheating") for e in events)  # cheats caught


def test_all_candidates_fail_escalates_to_strong_model(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_STRONG", "strong-x")
    models = []
    wrong = _fence("def add(a, b):\n    return a - b")
    provider = _FakeProvider(
        requirements="- implement add(a, b)",
        tests=_fence("def test_a():\n    assert add(1, 1) == 2"),
        first=[wrong], refined=[wrong],
    )

    def fake_get_provider(model=None, *a, **k):
        models.append(model)
        return provider

    monkeypatch.setattr(loop, "get_provider", fake_get_provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(0, 1))

    res = loop.run_agent("Implement add", use_search=False, max_iters=3)
    assert "strong-x" in models                                # all-fail rounds escalate
    assert res.success is False
