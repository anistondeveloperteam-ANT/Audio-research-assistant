"""TEST-CRITIC: a separate reviewer role that audits every generated test BEFORE it can judge a
candidate, and REWRITES invalid tests (guessed value / hardcoded tolerance / exact-match on a
non-unique quantity / wrong operational definition / requirement-not-implied-by-task / wrong entity)
into checks of the true requirement — so an invalid test never fails correct code, while a real bug
still fails the repaired (strict) suite.

Proves:
  - the critic prompt enumerates all six invalid-test reasons + the role/no-weakening guardrails;
  - _critique_tests returns the repaired suite and fail-opens (disabled / empty output);
  - end-to-end: with the critic ON a correct solution that an invalid test would falsely fail is
    VERIFIED; with the critic OFF the same correct solution is NOT verified; a real bug still fails the
    repaired suite; and the critic's rewrite is still validated against the reference (a still-wrong
    rewrite is quarantined by the execution backstop).

Deterministic: no network, no Docker, no real LLM.
"""
import types

from backend.agent import loop


def _res(stdout, ok=True):
    return types.SimpleNamespace(ok=ok, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ======================================================================================
# Prompt + helper unit tests.
# ======================================================================================
def test_critic_prompt_enumerates_all_invalid_reasons():
    s = loop._CRITIC_SYSTEM.lower()
    assert "test-critic" in s and "do not write solution code" in s        # distinct, tests-only role
    assert "guessed expected value" in s                                   # (1)
    assert "tolerance" in s and "sqrt(n)" in s                             # (2) math-derived tolerance
    assert "non-unique" in s and "defining propert" in s                  # (3) -> property check
    assert "operational definition" in s                                   # (4)
    assert "not implied by the task" in s                                  # (5)
    assert "wrong quantity" in s                                           # (6)
    assert "do not weaken" in s and "lenient" in s                        # no-weakening guardrail


class _CriticP:
    def __init__(self, reply):
        self.reply = reply

    def stream_chat(self, messages, system="", **k):
        assert system == loop._CRITIC_SYSTEM
        return [self.reply]


def test_critique_tests_returns_repaired_suite(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_CRITIC", "true")
    out = loop._critique_tests(_CriticP("def test_fixed():\n    assert solve() == 5\n"),
                               "task", "reqs", "def test_bad():\n    assert solve() == 999\n", "ref")
    assert "test_fixed" in out and "test_bad" not in out


def test_critique_tests_fail_open_on_empty(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_CRITIC", "true")
    orig = "def test_x():\n    assert True\n"
    assert loop._critique_tests(_CriticP(""), "t", "r", orig, "ref") == orig          # no test funcs
    assert loop._critique_tests(_CriticP("just prose, no code"), "t", "r", orig, "ref") == orig


def test_critique_tests_disabled_returns_original(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_CRITIC", "false")
    orig = "def test_x():\n    assert True\n"
    assert loop._critique_tests(object(), "t", "r", orig, "ref") == orig             # no call made


# ======================================================================================
# End-to-end: the critic's repaired suite judges the candidate.
# ======================================================================================
def _env(monkeypatch, **over):
    base = {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
            "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
            "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
            "AGENT_ROOT_CAUSE_DIAGNOSIS": "false", "AGENT_ANTICHEAT_SCAN": "false",
            "AGENT_MASKING_SCAN": "false", "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1",
            "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "1", "AGENT_STALL_LIMIT": "1",
            "AGENT_TEST_CRITIC": "true"}
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

    def __init__(self, gen,
                 original="def test_x():\n    # ORIGINAL_SUITE (invalid: guessed value)\n    assert solve() == 999\n",
                 repaired="def test_x():\n    # REPAIRED_SUITE (valid)\n    assert solve() == 5\n",
                 oracle="def solve():\n    return 5\n"):
        self.gen, self.original, self.repaired, self.oracle = gen, original, repaired, oracle

    def stream_chat(self, messages, system="", **k):
        if system == loop._REFERENCE_SYSTEM:
            return [self.oracle]
        if system == loop._REQ_SYSTEM:
            return ["- solve(): return the count"]
        if system == loop._TESTS_SYSTEM:
            return [self.original]
        if system == loop._CRITIC_SYSTEM:
            return [self.repaired]
        if system == loop._GEN_SYSTEM:
            return [self.gen]
        return [""]


class _Runner:
    """The ORIGINAL (invalid) suite asserts ==999 -> FAILS every candidate. The REPAIRED suite asserts
    ==5 -> only the correct candidate (SOLVE_OK) passes."""
    def __call__(self, code, **k):
        if "held-out runner (seeded)" in code:
            return _res("TESTS_PASSED 1/1\n")
        if "ModuleType('_sol')" in code:
            passed = ("REPAIRED_SUITE" in code) and ("SOLVE_OK" in code)
            return _res(f"TEST test_x {'PASS' if passed else 'FAIL'}\n"
                        f"TESTS_PASSED {1 if passed else 0}/1\n")
        return _res("TESTS_PASSED 0/0\n")


def test_critic_repairs_invalid_test_so_correct_code_verifies(monkeypatch):
    _env(monkeypatch, AGENT_TEST_CRITIC="true")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _P(gen="def solve():\n    return 5  # SOLVE_OK\n"))
    monkeypatch.setattr(loop, "run_python_auto", _Runner())
    events = []
    res = loop.run_agent("count via solve()", use_search=False, on_event=events.append)
    assert res.verification == "verified"                              # the repaired valid test passes
    assert any(e.get("type") == "test_critic" and e.get("scope") == "visible" for e in events)


def test_without_critic_the_invalid_test_falsely_fails_correct_code(monkeypatch):
    _env(monkeypatch, AGENT_TEST_CRITIC="false")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _P(gen="def solve():\n    return 5  # SOLVE_OK\n"))
    monkeypatch.setattr(loop, "run_python_auto", _Runner())
    res = loop.run_agent("count via solve()", use_search=False)
    assert res.verification != "verified"                              # un-repaired invalid test fails it


def test_repaired_suite_still_fails_a_real_bug(monkeypatch):
    _env(monkeypatch, AGENT_TEST_CRITIC="true")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _P(gen="def solve():\n    return 4  # SOLVE_BUG\n"))
    monkeypatch.setattr(loop, "run_python_auto", _Runner())
    res = loop.run_agent("count via solve()", use_search=False)
    assert res.verification != "verified"                              # repaired suite is still strict


def test_critic_rewrite_is_validated_against_reference(monkeypatch):
    # The critic rewrites to a suite containing a STILL-WRONG test; the reference fails it, so the
    # execution backstop (_invalid_tests, run on the REPAIRED suite) quarantines it and the correct
    # code is still verified on the valid remainder.
    _env(monkeypatch, AGENT_TEST_CRITIC="true", AGENT_REFERENCE_TESTS="true",
         AGENT_TEST_VALIDATION="true")
    monkeypatch.setattr(loop, "_invalid_tests", lambda *a, **k: {"test_bad"})   # reference fails the rewrite
    prov = _P(gen="def solve():\n    return 5  # SOLVE_OK\n",
              repaired=("def test_keep():\n    assert solve() == 5\n"
                        "def test_bad():\n    assert solve() == 888  # still wrong\n"))
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: (
        _res("TEST test_keep PASS\nTEST test_bad FAIL\nTESTS_PASSED 1/2\n")
        if "ModuleType('_sol')" in code else _res("TESTS_PASSED 1/1\n")))
    res = loop.run_agent("count via solve()", use_search=False)
    assert res.verification == "verified"            # still-wrong rewrite quarantined; valid test passes
