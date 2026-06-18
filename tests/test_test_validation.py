"""Oracle test-validation — the general 'test validity before blame' safety net.

A generated test may only fail code when the code is genuinely wrong. Before a test is allowed to
fail a candidate it is run against the KNOWN-CORRECT reference oracle; any test the oracle itself
fails is invalid (a guessed expected value, a tolerance too tight for the method, or the wrong
quantity) and is QUARANTINED — excluded from the pass/total — so correct code is never falsely
failed, while every oracle-passing test still gates genuinely wrong code.

Fully offline: the sandbox runner is monkeypatched; no LLM, no Docker. The four behaviors the task
demands are proven directly: (a) a result within a justified band passes (the too-tight test is
quarantined), (b) expected values are derived, (c) tolerances are derived from the math, (d) a
genuinely wrong result still fails (an oracle-passing test the candidate fails is counted)."""
import types

import backend.agent.loop as loop


def _res(stdout: str):
    return types.SimpleNamespace(ok=True, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ----------------------------------------------------------------------
# Toggle
# ----------------------------------------------------------------------
def test_test_validation_toggle(monkeypatch):
    monkeypatch.delenv("AGENT_TEST_VALIDATION", raising=False)
    assert loop.test_validation_enabled() is True
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "false")
    assert loop.test_validation_enabled() is False
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "off")
    assert loop.test_validation_enabled() is False


# ----------------------------------------------------------------------
# Parser + count helpers
# ----------------------------------------------------------------------
def test_test_results_parses_per_test_lines():
    out = "noise\nTEST test_a PASS\nTEST test_b FAIL\nTEST test_c PASS\nTESTS_PASSED 2/3\n"
    assert loop._test_results(out) == {"test_a": True, "test_b": False, "test_c": True}


def test_test_results_empty_when_no_lines():
    assert loop._test_results("TESTS_PASSED 0/0\n") == {}
    assert loop._test_results("") == {}


def test_valid_counts_excludes_quarantined():
    out = "TEST test_a PASS\nTEST test_b FAIL\nTEST test_c PASS\n"
    # test_b is quarantined (invalid) -> only the two valid tests are counted, both pass.
    assert loop._valid_counts(out, {"test_b"}) == (2, 2)
    # nothing quarantined -> the failing test counts -> 2/3.
    assert loop._valid_counts(out, set()) == (2, 3)


def test_valid_counts_none_when_unparseable_or_all_quarantined():
    assert loop._valid_counts("TESTS_PASSED 1/2\n", {"x"}) is None     # no per-test lines -> keep tally
    assert loop._valid_counts("TEST test_a PASS\n", {"test_a"}) is None  # every test quarantined


# ----------------------------------------------------------------------
# _invalid_tests: the oracle decides which tests are valid.
# ----------------------------------------------------------------------
def test_invalid_tests_quarantines_tests_the_oracle_fails(monkeypatch):
    # The oracle (known-correct) fails test_tight -> that test is invalid and quarantined.
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_band PASS\nTEST test_tight FAIL\n"
                                               "TESTS_PASSED 1/2\n"))
    assert loop._invalid_tests("def f(): pass", "TESTS") == {"test_tight"}


def test_invalid_tests_fails_open(monkeypatch):
    assert loop._invalid_tests("", "TESTS") == set()                   # no oracle -> nothing quarantined
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res("TESTS_PASSED 0/0\n"))
    assert loop._invalid_tests("def f(): pass", "TESTS") == set()      # no per-test lines -> fail open
    def boom(*a, **k):
        raise RuntimeError("sandbox down")
    monkeypatch.setattr(loop, "run_python_auto", boom)
    assert loop._invalid_tests("def f(): pass", "TESTS") == set()      # crash -> fail open, no quarantine


# ----------------------------------------------------------------------
# (a) A correct result within a justified band PASSES even though a naive too-tight test rejects it.
#     The oracle (correct) also fails that too-tight test, so it is quarantined and never blames code.
# ----------------------------------------------------------------------
def test_a_result_in_justified_band_passes_when_naive_test_quarantined(monkeypatch):
    # Stdout from running the CANDIDATE: it passes the SE-band test, fails the too-tight one.
    cand_out = "TEST test_se_band PASS\nTEST test_too_tight FAIL\nTESTS_PASSED 1/2\n"
    # The oracle fails the same too-tight test (a correct estimate is ~1 SE off the mean).
    oracle_quarantine = {"test_too_tight"}
    vc = loop._valid_counts(cand_out, oracle_quarantine)
    assert vc == (1, 1)                                # judged ONLY on the valid SE-band test -> full pass


# ----------------------------------------------------------------------
# (b) Expected values are DERIVED (oracle / closed-form), never a guessed literal.
# (c) Tolerances are DERIVED FROM THE MATH (standard error / method error), never a constant.
# ----------------------------------------------------------------------
def test_b_prompts_demand_derived_expected_not_literals():
    for sys_prompt in (loop._TESTS_SYSTEM, loop._HIDDEN_SYSTEM):
        s = sys_prompt.lower()
        assert "never" in s
        assert "guessed literal" in s or "literal expected" in s or "guessed expected" in s
    # The oracle clause makes expected come from `ref`, executed, never imagined.
    oc = loop._ORACLE_CLAUSE.lower()
    assert "ref" in oc and "never write an expected literal" in oc


def test_c_prompts_demand_tolerance_derived_from_the_math():
    for sys_prompt in (loop._TESTS_SYSTEM, loop._HIDDEN_SYSTEM, loop._INVARIANTS_SYSTEM):
        s = sys_prompt.lower()
        assert "standard error" in s                  # stochastic -> SE band
        assert "tight" in s or "method's error" in s or "arbitrary" in s
    sim = loop._TASK_TYPE_GUIDANCE["simulation"].lower()
    assert "standard error" in sim and "sqrt(n)" in sim
    num = loop._TASK_TYPE_GUIDANCE["numeric_algorithm"].lower()
    assert "method's error" in num or "step size" in num


# ----------------------------------------------------------------------
# (d) A genuinely wrong result STILL FAILS — an oracle-PASSING (valid) test the candidate fails is
#     counted, so quarantine never hides a real defect.
# ----------------------------------------------------------------------
def test_d_genuinely_wrong_result_still_fails(monkeypatch):
    # The oracle passes both tests -> nothing quarantined.
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_value PASS\nTEST test_band PASS\n"
                                               "TESTS_PASSED 2/2\n"))
    quarantine = loop._invalid_tests("ORACLE", "TESTS")
    assert quarantine == set()                         # both tests are valid
    # A WRONG candidate fails the valid test_value -> it is counted, candidate is not a full pass.
    wrong_out = "TEST test_value FAIL\nTEST test_band PASS\n"
    assert loop._valid_counts(wrong_out, quarantine) == (1, 2)   # real failure survives, 1/2


# ----------------------------------------------------------------------
# Held-out quarantine: an invalid held-out check the oracle fails is excluded across seeds, but a
# valid one the CANDIDATE fails still rejects the candidate.
# ----------------------------------------------------------------------
def test_verify_heldout_quarantines_oracle_failing_check(monkeypatch):
    heldout = ("def test_hidden_a():\n    pass\n"
               "def test_invariant_b():\n    pass\n")
    # Candidate passes test_hidden_a, fails the (invalid) test_invariant_b on every seed.
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_hidden_a PASS\nTEST test_invariant_b FAIL\n"
                                               "TESTS_PASSED 1/2\n"))
    ok, p, t, _ = loop._verify_heldout("SOL", heldout, seeds=2, quarantine={"test_invariant_b"})
    assert ok is True and (p, t) == (1, 1)             # judged only on the valid held-out check


def test_verify_heldout_still_rejects_failure_on_a_valid_check(monkeypatch):
    heldout = "def test_hidden_a():\n    pass\ndef test_invariant_b():\n    pass\n"
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_hidden_a FAIL\nTEST test_invariant_b PASS\n"
                                               "TESTS_PASSED 1/2\n"))
    # test_invariant_b is quarantined, but the candidate fails the VALID test_hidden_a -> rejected.
    ok, _p, _t, _ = loop._verify_heldout("SOL", heldout, seeds=2, quarantine={"test_invariant_b"})
    assert ok is False


# ----------------------------------------------------------------------
# Definition checks are oracle-INDEPENDENT: the held-out quarantine must NEVER drop a
# test_definition_* (its job is to catch a wrong-definition oracle), even if the oracle fails it.
# ----------------------------------------------------------------------
def test_held_out_quarantine_exempts_definition_checks(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "true")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "1")
    heldout = ("def test_hidden_a():\n    pass\n"
               "def test_invariant_b():\n    pass\n"
               "def test_definition_value():\n    pass\n"
               "def test_hidden_ok():\n    pass\n")
    # The oracle fails the definition check (a wrong-definition oracle would) AND two others, but
    # passes test_hidden_ok — so it is NOT the all-fail (fail-open) case. The definition check must
    # still be EXEMPT from quarantine so it can gate the candidate.
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_hidden_a FAIL\nTEST test_invariant_b FAIL\n"
                                               "TEST test_definition_value FAIL\nTEST test_hidden_ok "
                                               "PASS\nTESTS_PASSED 1/4\n"))
    q = loop._heldout_quarantine("ORACLE", heldout)
    assert q == {"test_hidden_a", "test_invariant_b"}     # hidden/invariant quarantined...
    assert "test_definition_value" not in q              # ...but the definition check is EXEMPT


def test_invalid_tests_fails_open_when_oracle_fails_entire_suite(monkeypatch):
    # A reference that fails EVERY test is unreliable (e.g. wrong function names) — quarantine nothing
    # rather than nuke the suite and leave candidates judged on the very tests proven invalid.
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_a FAIL\nTEST test_b FAIL\nTESTS_PASSED 0/2\n"))
    assert loop._invalid_tests("def f(): pass", "TESTS") == set()
    # But a PARTIAL failure (some pass) is a real signal -> quarantine just the failing ones.
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_a FAIL\nTEST test_b PASS\nTESTS_PASSED 1/2\n"))
    assert loop._invalid_tests("def f(): pass", "TESTS") == {"test_a"}


def test_held_out_quarantine_off_when_disabled(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "false")
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_hidden_a FAIL\nTESTS_PASSED 0/1\n"))
    assert loop._heldout_quarantine("ORACLE", "def test_hidden_a():\n    pass\n") == set()
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "true")
    assert loop._heldout_quarantine("", "def test_hidden_a():\n    pass\n") == set()   # no oracle


# ----------------------------------------------------------------------
# SIMULATION tasks: there is no exact-value oracle, but an INDEPENDENT reference still validates the
# property/invariant checks — a flawed invariant the correct reference also fails is quarantined,
# while a REAL violation the reference passes still fails the candidate.
# ----------------------------------------------------------------------
def _sim_env(monkeypatch):
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type", lambda task: "simulation")
    for k, v in {"AGENT_REFERENCE_TESTS": "true", "AGENT_TEST_VALIDATION": "true",
                 "AGENT_HIDDEN_TESTS": "false", "AGENT_DEFINITION_GATE": "false",
                 "AGENT_DELIVERY_GATES": "false", "AGENT_ANTICHEAT_SCAN": "false",
                 "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)


def test_simulation_false_invariant_is_quarantined_not_blamed(monkeypatch):
    from test_agent import _FakeProvider, _fence

    _sim_env(monkeypatch)
    provider = _FakeProvider(
        requirements="- simulate(): return per-step energy; energy is conserved",
        tests=_fence("def test_conserved():\n    assert True\ndef test_flawed():\n    assert True"),
        first=[_fence("def simulate():\n    return [1.0, 1.0]  # CANDIDATE conserved")],
        reference="def simulate():\n    return [1.0, 1.0]  # independent reference",
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    # The flawed invariant FAILS for the correct reference too -> quarantined; the candidate then
    # passes the one valid invariant. (Same stdout for the validation run and the candidate run.)
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _res("TEST test_conserved PASS\nTEST test_flawed FAIL\n"
                                               "TESTS_PASSED 1/2\n"))
    res = loop.run_agent("simulate a system and report the conserved energy", use_search=False)
    assert res.verification == "verified"      # false invariant quarantined, correct sim not blamed
    assert res.attempts_taken == 1
    assert res.tests_total == 1                # the flawed invariant was discarded, not counted
    assert "CANDIDATE" in res.best_code        # code left unchanged (no hardcoding to pass it)


def test_simulation_real_violation_still_fails(monkeypatch):
    from test_agent import _FakeProvider, _fence

    _sim_env(monkeypatch)
    buggy = "def simulate():\n    return [1.0, 9.0]  # BUGGY: energy not conserved"
    provider = _FakeProvider(
        requirements="- simulate(): energy is conserved",
        tests=_fence("def test_conserved():\n    assert True"),
        first=[_fence(buggy)], refined=[_fence(buggy)],
        reference="def simulate():\n    return [1.0, 1.0]  # correct reference",
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    # The correct reference PASSES the conservation invariant (so it is NOT quarantined); the buggy
    # candidate FAILS it -> a real violation, correctly rejected, never hidden.
    def fake_run(code, **k):
        if "BUGGY" in code:                    # the candidate run
            return _res("TEST test_conserved FAIL\nTESTS_PASSED 0/1\n")
        return _res("TEST test_conserved PASS\nTESTS_PASSED 1/1\n")   # the reference validation run
    monkeypatch.setattr(loop, "run_python_auto", fake_run)
    res = loop.run_agent("simulate a system where energy is conserved", use_search=False, max_iters=2)
    assert res.verification == "partial" and res.success is False     # the real bug is not masked
    assert res.tests_passed == 0 and res.tests_total == 1
