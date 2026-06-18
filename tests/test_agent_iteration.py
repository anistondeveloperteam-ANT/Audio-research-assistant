"""Continuous iteration loop: the agent keeps generating + fixing until a solution is FULLY
verified, the attempt cap is hit, or progress stalls — never silently settling for a partial pass,
and never gaming a flawed test to reach a full pass.

Reference pattern only (AlphaCodium iterative flow + OpenHands-style retry-until-fixed); no new deps.
Fully offline: the LLM provider and the Docker sandbox runner are monkeypatched.

Proves: (a) the loop continues past a partial pass to full verification when the failures are
genuine + fixable; (b) a FALSE failure (correct code, flawed test) is resolved by DISCARDING the
flawed test (quarantine), not by changing the code; (c) it stops at AGENT_MAX_ATTEMPTS and at the
stall limit, labelling honestly; (d) it never hardcodes/games outputs to reach a full pass."""
import types

import backend.agent.loop as loop
from backend.agent.code_runner import RunResult


# ----------------------------------------------------------------------
# Toggles + the stall metric.
# ----------------------------------------------------------------------
def test_max_attempts_default_fallback_and_clamp(monkeypatch):
    monkeypatch.delenv("AGENT_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("AGENT_MAX_ITERS", raising=False)
    assert loop.max_attempts() == 10                        # default
    monkeypatch.setenv("AGENT_MAX_ATTEMPTS", "5")
    assert loop.max_attempts() == 5
    monkeypatch.delenv("AGENT_MAX_ATTEMPTS", raising=False)
    monkeypatch.setenv("AGENT_MAX_ITERS", "7")
    assert loop.max_attempts() == 7                         # legacy fallback
    monkeypatch.setenv("AGENT_MAX_ATTEMPTS", "0")
    assert loop.max_attempts() == 1                         # clamped >= 1
    monkeypatch.setenv("AGENT_MAX_ATTEMPTS", "nan")
    assert loop.max_attempts() == 10                        # bad value -> default


def test_stall_limit_default_and_clamp(monkeypatch):
    monkeypatch.delenv("AGENT_STALL_LIMIT", raising=False)
    assert loop.stall_limit() == 3
    monkeypatch.setenv("AGENT_STALL_LIMIT", "2")
    assert loop.stall_limit() == 2
    monkeypatch.setenv("AGENT_STALL_LIMIT", "0")
    assert loop.stall_limit() == 1                          # clamped >= 1


def test_remaining_failures_counts_only_genuine_shortfall():
    assert loop._remaining_failures({"verified": True}) == 0
    assert loop._remaining_failures({"verified": False, "passed": 1, "total": 2}) == 1
    assert loop._remaining_failures(
        {"verified": False, "passed": 2, "total": 2, "hidden_passed": 1, "hidden_total": 3}) == 2
    assert loop._remaining_failures(
        {"verified": False, "passed": 2, "total": 2, "gate_fail": "no output"}) == 1
    # not verified but every visible test passes -> still at least one outstanding failure
    assert loop._remaining_failures({"verified": False, "passed": 2, "total": 2}) == 1


# ----------------------------------------------------------------------
# (a) The loop CONTINUES past a partial pass and reaches full verification.
# ----------------------------------------------------------------------
def test_loop_continues_past_partial_to_full_verification(monkeypatch):
    from test_agent import _FakeProvider, _fence, _ok, _runner, _TESTS_SRC, _HIDDEN_SRC, _INV_SRC

    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)",
        tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return a")],            # round 1: partial (1/2)
        refined=[_fence("def bubble_sort(a):\n    return sorted(a)")],  # round 2: genuinely fixed
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        _runner(lambda c: _ok(2, 2) if "sorted(a)" in c else _ok(1, 2),
                                lambda c: _ok(2, 2)))
    monkeypatch.setenv("AGENT_REFERENCE_TESTS", "false")    # keep it to the visible+held-out gates
    monkeypatch.setenv("AGENT_PARALLEL_N", "1")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "1")
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)

    res = loop.run_agent("Implement bubble sort", use_search=False)     # no cap -> AGENT_MAX_ATTEMPTS
    assert res.verification == "verified" and res.success is True
    assert res.attempts_taken == 2                          # did NOT settle for the partial round 1
    assert res.stop_reason == "verified"
    assert "sorted(a)" in res.best_code


# ----------------------------------------------------------------------
# (b) A FALSE failure (correct code, flawed test) is fixed by DISCARDING the test, NOT the code.
# ----------------------------------------------------------------------
def _res_lines(good: bool, flawed: bool) -> RunResult:
    g, f = ("PASS" if good else "FAIL"), ("PASS" if flawed else "FAIL")
    n = int(good) + int(flawed)
    return RunResult(True, 0, f"TEST test_good {g}\nTEST test_flawed {f}\nTESTS_PASSED {n}/2\n", "", 0.1)


def test_false_failure_resolved_by_quarantine_not_by_changing_code(monkeypatch):
    from test_agent import _FakeProvider, _fence

    oracle = "def double(n):\n    return n * 2  # ORACLE_MARKER"
    candidate = "def double(n):\n    return n * 2  # CANDIDATE_MARKER"
    provider = _FakeProvider(
        requirements="- double(n): return n*2; report double(3)",
        tests=_fence("def test_good():\n    assert True\ndef test_flawed():\n    assert True"),
        first=[_fence(candidate)],
        reference=oracle,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    # The flawed test FAILS for everyone, including the known-correct oracle -> it is the TEST that is
    # wrong, so it gets quarantined; the correct candidate is then a full pass on the remaining test.
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res_lines(good=True, flawed=False))
    monkeypatch.setenv("AGENT_REFERENCE_TESTS", "true")
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "true")
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "false")       # isolate the visible-test quarantine
    monkeypatch.setenv("AGENT_DEFINITION_GATE", "false")
    monkeypatch.setenv("AGENT_DELIVERY_GATES", "false")
    monkeypatch.setenv("AGENT_ANTICHEAT_SCAN", "false")
    monkeypatch.setenv("AGENT_PARALLEL_N", "1")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "1")
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)

    res = loop.run_agent("implement double(n) returning n*2 and report double(3)", use_search=False)
    assert res.verification == "verified" and res.success is True
    assert res.attempts_taken == 1                          # no regeneration was needed
    assert res.best_code.strip() == candidate              # the CODE was left exactly as written...
    assert "CANDIDATE_MARKER" in res.best_code             # ...not edited/hardcoded to pass the test
    assert res.tests_total == 1                            # the flawed test was DISCARDED, not counted


# ----------------------------------------------------------------------
# (c) Stop conditions: AGENT_MAX_ATTEMPTS and stall -> honest partial, never a fake verified.
# ----------------------------------------------------------------------
def _partial_provider(monkeypatch):
    from test_agent import _FakeProvider, _fence, _ok, _TESTS_SRC
    partial = _fence("def bubble_sort(a):\n    return a")
    provider = _FakeProvider(requirements="- bubble_sort(a)", tests=_fence(_TESTS_SRC),
                             first=[partial], refined=[partial])      # never improves
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(1, 2))
    monkeypatch.setenv("AGENT_REFERENCE_TESTS", "false")
    monkeypatch.setenv("AGENT_PARALLEL_N", "1")
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)


def test_stops_at_max_attempts_with_honest_partial(monkeypatch):
    _partial_provider(monkeypatch)
    monkeypatch.setenv("AGENT_STALL_LIMIT", "9")            # high, so the cap is what stops it
    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=2)
    assert res.verification == "partial" and res.success is False
    assert res.attempts_taken == 2 and res.stop_reason == "max_attempts"
    md = loop.result_to_markdown(res)
    assert "Partially verified — 1/2 genuine checks passing" in md and "2 attempts" in md


def test_stops_on_stall_with_honest_partial(monkeypatch):
    _partial_provider(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_ATTEMPTS", "10")
    monkeypatch.setenv("AGENT_STALL_LIMIT", "2")           # bail after 2 no-progress rounds
    res = loop.run_agent("Implement bubble sort", use_search=False)   # budget 10, but stalls first
    assert res.verification == "partial" and res.success is False
    assert res.stop_reason == "stall"
    assert res.attempts_taken == 3                         # round 1 + 2 no-progress rounds, then stop


# ----------------------------------------------------------------------
# (d) Never reward-hack: a gamed candidate that "passes" the tests is rejected, never verified.
# ----------------------------------------------------------------------
def test_never_games_to_full_pass(monkeypatch):
    from test_agent import _FakeProvider, _fence, _ok, _TESTS_SRC

    gamed = _fence("def bubble_sort(a):\n    return [1, 2, 3]")        # hardcoded output
    provider = _FakeProvider(requirements="- bubble_sort(a)", tests=_fence(_TESTS_SRC),
                             first=[gamed], refined=[gamed])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))   # tests "pass" 2/2
    monkeypatch.setattr(loop, "scan_for_cheating",
                        lambda code, tests, task: types.SimpleNamespace(
                            flagged=True, reasons=["hardcoded outputs"]))
    monkeypatch.setenv("AGENT_REFERENCE_TESTS", "false")
    monkeypatch.setenv("AGENT_PARALLEL_N", "1")
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)

    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=3)
    # A "2/2 passing" result that was gamed is NEVER labelled verified — honesty over a fake full pass.
    assert res.verification == "rejected_cheating"
    assert res.success is False and res.best_code == ""
