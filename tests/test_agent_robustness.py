"""Robust-code standards + the reject-fragile-code mechanism for the code agent
(backend/agent/loop.py). Fully offline: no LLM, no Docker — the sandbox runner (_run_against_tests)
is monkeypatched, so these assert the harness BEHAVIOR, not a model."""
import types

from backend.agent import loop


# ----------------------------------------------------------------------
# The generation / test prompts actually demand the robustness standards.
# ----------------------------------------------------------------------
def test_gen_system_demands_correct_by_design():
    s = loop._GEN_SYSTEM.lower()
    assert "any valid input" in s                       # not just the demo value
    assert "convert" in s and "raise" in s              # convert-or-assert, never silently assume
    assert "units" in s                                 # explicit unit assumption (radians/degrees)
    assert "edge" in s and "boundary" in s              # edge/boundary handling
    assert "magic constants" in s                       # no example-only constants


def test_req_system_surfaces_input_contract():
    s = loop._REQ_SYSTEM.lower()
    assert "input contract" in s
    assert "units" in s and "ranges" in s


def test_reference_uses_independent_method():
    s = loop._REFERENCE_SYSTEM.lower()
    assert "independent method" in s
    assert "closed-form" in s and "numerical" in s      # the canonical independent-method example


def test_hidden_tests_demand_edge_varied_and_contract():
    s = loop._HIDDEN_SYSTEM.lower()
    assert "edge" in s and "boundary" in s
    assert "different parameter regime" in s
    assert "input contract" in s
    assert "radians" in s and "degrees" in s            # the unit-contract probe example


# ----------------------------------------------------------------------
# (a) A candidate right on the demo input but wrong on another valid input is REJECTED.
# ----------------------------------------------------------------------
def _ok_result(stdout="TESTS_PASSED", stderr=""):
    return types.SimpleNamespace(ok=True, stdout=stdout, stderr=stderr, error="")


def test_verify_heldout_rejects_when_a_different_input_fails(monkeypatch):
    # 4 held-out tests; the first input regime (seed 0) fully passes, a DIFFERENT valid input
    # (seed 1) fails — i.e. correct on the demo, fragile elsewhere.
    heldout = ("def test_hidden_a():\n    pass\n"
               "def test_hidden_b():\n    pass\n"
               "def test_hidden_c():\n    pass\n"
               "def test_hidden_d():\n    pass\n")
    calls = {"n": 0}

    def fake_run(sol, tests, footer, reference_src=""):
        calls["n"] += 1
        return (_ok_result(), 4, 4) if calls["n"] == 1 else (_ok_result(), 3, 4)

    monkeypatch.setattr(loop, "_run_against_tests", fake_run)
    ok, passed, total, _last = loop._verify_heldout("SOLUTION", heldout, seeds=3)
    assert ok is False                                  # rejected -> the loop regenerates
    assert calls["n"] == 2                               # stopped at the failing input, never accepted


def test_verify_heldout_accepts_only_when_every_input_passes(monkeypatch):
    heldout = "def test_hidden_a():\n    pass\ndef test_hidden_b():\n    pass\n"
    monkeypatch.setattr(loop, "_run_against_tests", lambda *a, **k: (_ok_result(), 2, 2))
    ok, passed, total, _ = loop._verify_heldout("SOLUTION", heldout, seeds=3)
    assert ok is True and passed == total


# ----------------------------------------------------------------------
# (b) An input-contract violation is CAUGHT (surfaced), not silently passed.
# ----------------------------------------------------------------------
def test_contract_violation_is_caught_not_passed():
    bad = _ok_result(stdout="TESTS_PASSED 5/6", stderr="AssertionError: angle unit mismatch")
    verdict = loop._verdict_from_tests(passed=5, total=6, relevant=True, result=bad)
    assert verdict["done"] is False                     # not accepted
    assert verdict["success"] is False
    assert "angle unit mismatch" in verdict["feedback"]  # the mismatch surfaces to the rewrite

    good = _ok_result(stdout="TESTS_PASSED 6/6")
    ok_verdict = loop._verdict_from_tests(passed=6, total=6, relevant=True, result=good)
    assert ok_verdict["done"] is True and ok_verdict["success"] is True
