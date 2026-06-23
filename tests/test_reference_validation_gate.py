"""THE UNIVERSAL TEST-ADMISSION GATE: a generated test may judge a candidate ONLY IF the known-correct
reference passes it. Any test the reference fails — for ANY reason, enumerated or not — is quarantined
by construction and never counted as a code failure. Tests the reference passes still gate genuinely
wrong code.

Proves:
  (a) any test the reference fails (any reason, incl. an UN-enumerated defect) is quarantined;
  (b) a correct solution is never failed by such a test (a wrong DEFINITION check the reference fails is
      now quarantined too) -> verified;
  (c) a test the reference PASSES is admitted and still fails a genuinely wrong solution;
  (d) the mechanism is the SINGLE reference-passing gate, not a list of defect types.

Deterministic: no network, no Docker, no real LLM.
"""
import types

from backend.agent import loop


def _res(stdout, ok=True):
    return types.SimpleNamespace(ok=ok, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ======================================================================================
# (a)+(d) Unit: the gate quarantines EVERY reference-failing test, regardless of WHY — including
# defects not in the enumerated list — and admits the ones the reference passes.
# ======================================================================================
def test_any_reference_failing_test_is_quarantined_whatever_the_cause(monkeypatch):
    # The reference (run as the candidate by _invalid_tests) FAILS three checks for three DIFFERENT
    # reasons — two of them NOT in the enumerated defect set — and PASSES one valid check.
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res(
        "TEST test_guessed_value FAIL\n"        # enumerated: a guessed expected value
        "TEST test_phase_of_moon FAIL\n"        # UN-enumerated: an arbitrary, unrelated assertion
        "TEST test_malformed FAIL\n"            # UN-enumerated: raises -> the runner marks it FAIL
        "TEST test_valid PASS\n"
        "TESTS_PASSED 1/4\n"))
    bad = loop._invalid_tests(
        "def solve():\n    return 5\n",
        ("def test_guessed_value():\n    assert solve() == 999\n"
         "def test_phase_of_moon():\n    assert solve() == 42  # unrelated to the task\n"
         "def test_malformed():\n    raise RuntimeError('boom')\n"
         "def test_valid():\n    assert solve() == 5\n"))
    assert bad == {"test_guessed_value", "test_phase_of_moon", "test_malformed"}  # all three quarantined
    assert "test_valid" not in bad                            # the reference-passing test is admitted


def test_reference_passing_suite_quarantines_nothing(monkeypatch):
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res(
        "TEST test_a PASS\nTEST test_b PASS\nTESTS_PASSED 2/2\n"))
    assert loop._invalid_tests("def solve():\n    return 5\n",
                               "def test_a():\n    assert True\ndef test_b():\n    assert True\n") == set()


# ======================================================================================
# Integration harness.
# ======================================================================================
def _env(monkeypatch, **over):
    base = {"AGENT_REFERENCE_TESTS": "true", "AGENT_TEST_VALIDATION": "true",
            "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "true",
            "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
            "AGENT_ROOT_CAUSE_DIAGNOSIS": "false", "AGENT_ANTICHEAT_SCAN": "false",
            "AGENT_MASKING_SCAN": "false", "AGENT_TEST_CRITIC": "false", "AGENT_PARALLEL_N": "1",
            "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "1",
            "AGENT_STALL_LIMIT": "1"}
    base.update(over)
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


class _P:
    is_available = True
    name, model = "openai", "test"

    def __init__(self, *, gen, invariants="", definitions="",
                 hidden="def test_hidden_step():\n    assert step() is not None\n",
                 oracle="def step():\n    return 5  # USES_GOOD\n"):
        self.gen, self.invariants, self.definitions = gen, invariants, definitions
        self.hidden, self.oracle = hidden, oracle

    def stream_chat(self, messages, system="", **k):
        if system == loop._REFERENCE_SYSTEM:
            return [self.oracle]
        if system == loop._REQ_SYSTEM:
            return ["- step(): simulate the system and report the result"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert step() is not None\n"]
        if system == loop._HIDDEN_SYSTEM:
            return [self.hidden]
        if system == loop._INVARIANTS_SYSTEM:
            return [self.invariants]
        if system == loop._DEFINITION_SYSTEM:
            return [self.definitions]
        if system == loop._GEN_SYSTEM:
            return [self.gen]
        return [""]


# ======================================================================================
# (b) A wrong DEFINITION check the reference fails is now quarantined (universal gate) -> correct code
# is verified. Before the fix, definition checks were exempt and this would falsely fail.
# ======================================================================================
class _RunnerB:
    def __call__(self, code, **k):
        if "held-out runner (seeded)" in code:    # the bad definition fails for everyone; valid pass
            return _res("TEST test_definition_wrong FAIL\nTEST test_invariant_ok PASS\n"
                        "TEST test_hidden_step PASS\nTESTS_PASSED 2/3\n")
        if "ModuleType('_sol')" in code:
            return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")
        return _res("TESTS_PASSED 0/0\n")


def test_bad_definition_check_is_quarantined_and_correct_code_verifies(monkeypatch):
    _env(monkeypatch, AGENT_DEFINITION_GATE="true")
    prov = _P(gen="def step():\n    return 5  # simulate the system\n",
              invariants="def test_invariant_ok():\n    assert step() is not None\n",
              definitions="def test_definition_wrong():\n    assert step() == 999  # wrong expected\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _RunnerB())

    res = loop.run_agent("simulate step() for the system", use_search=False)
    assert res.verification == "verified"        # the reference-failing definition check was quarantined


# ======================================================================================
# (c) A check the reference PASSES is admitted and still fails a genuinely wrong solution (the gate does
# not let wrong code through); a correct solution passes it.
# ======================================================================================
class _RunnerC:
    def __call__(self, code, **k):
        if "held-out runner (seeded)" in code:    # reference + correct pass; the wrong one fails
            ok = "USES_BAD" not in code
            return _res(f"TEST test_invariant_real {'PASS' if ok else 'FAIL'}\n"
                        f"TESTS_PASSED {1 if ok else 0}/1\n")
        if "ModuleType('_sol')" in code:
            return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")
        return _res("TESTS_PASSED 0/0\n")


def test_valid_check_admitted_and_fails_wrong_code(monkeypatch):
    _env(monkeypatch)
    prov = _P(gen="def step():\n    return 4  # USES_BAD: simulate the system (wrong)\n",
              invariants="def test_invariant_real():\n    assert step() == 5\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _RunnerC())

    res = loop.run_agent("simulate step() for the system", use_search=False)
    assert res.verification != "verified"        # a valid, reference-passing check still catches the bug


def test_valid_check_admitted_and_passes_correct_code(monkeypatch):
    _env(monkeypatch)
    prov = _P(gen="def step():\n    return 5  # USES_GOOD: simulate the system\n",
              invariants="def test_invariant_real():\n    assert step() == 5\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _RunnerC())

    res = loop.run_agent("simulate step() for the system", use_search=False)
    assert res.verification == "verified"        # the gate does not over-quarantine a valid check
