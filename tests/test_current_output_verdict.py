"""Every check's pass/fail must reflect the CURRENT/latest attempt's REAL output, and the final label
must name only the genuinely-failing checks — never a check the present code satisfies, and never a
check the system itself proved INVALID (quarantined). Universal across tasks/domains.

Proves:
  (a) a check satisfied by the current output is marked pass even if an earlier attempt failed it;
  (b) no failure flag from a prior attempt persists once the current code resolves it;
  (c) when only some checks genuinely fail, the label names EXACTLY those (and excludes quarantined
      invalid checks the reference oracle itself fails);
  (d) a correct, complete solution is labeled verified, not partial;
plus the selection prefers the latest, better-generalising attempt so a stale held-out tally is not
reported.

Deterministic: no network, no Docker, no real LLM.
"""
import types

from backend.agent import loop
from backend.agent.loop import Attempt


def _res(stdout, ok=True):
    return types.SimpleNamespace(ok=ok, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ======================================================================================
# Unit: the two pure helpers behind the fix.
# ======================================================================================
def test_genuine_failing_excludes_quarantined():
    stdout = "TEST test_a PASS\nTEST test_b FAIL\nTEST test_definition_x FAIL\n"
    # test_definition_x is quarantined (the reference oracle itself fails it) -> NOT a real failure.
    assert loop._genuine_failing(stdout, {"test_definition_x"}) == ["test_b"]
    # no quarantine -> raw failures (unchanged behaviour for valid suites).
    assert loop._genuine_failing(stdout, set()) == ["test_b", "test_definition_x"]
    assert loop._genuine_failing("", {"test_x"}) == []


def test_heldout_frac_orders_by_generalization():
    mk = lambda hp, ht: Attempt(1, "c", _res(""), {"hidden_passed": hp, "hidden_total": ht})
    assert loop._heldout_frac(mk(4, 5)) > loop._heldout_frac(mk(2, 5))   # better held-out ranks higher
    assert loop._heldout_frac(mk(0, 0)) == 0.0                           # none ran -> 0
    assert loop._heldout_frac(mk(5, 5)) == 1.0


# ======================================================================================
# Integration harness.
# ======================================================================================
def _env(monkeypatch, **over):
    base = {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
            "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
            "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
            "AGENT_ROOT_CAUSE_DIAGNOSIS": "true", "AGENT_ANTICHEAT_SCAN": "false",
            "AGENT_MASKING_SCAN": "false", "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1",
            "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "3", "AGENT_STALL_LIMIT": "2"}
    base.update(over)
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


class _P:
    """Routes by system prompt; emits generated solutions from `gens` one per round."""
    is_available = True
    name, model = "openai", "test"

    def __init__(self, *, requirements, tests, gens, oracle="def solve():\n    return 5\n",
                 hidden="def test_hidden():\n    assert solve() is not None\n"):
        self.requirements, self.tests, self.gens = requirements, tests, list(gens)
        self.oracle, self.hidden, self.gi, self.calls = oracle, hidden, 0, []

    def stream_chat(self, messages, system="", **k):
        u = messages[-1]["content"] if messages else ""
        self.calls.append((system, u))
        if system == loop._REFERENCE_SYSTEM:
            return [self.oracle]
        if system == loop._REQ_SYSTEM:
            return [self.requirements]
        if system == loop._TESTS_SYSTEM:
            return [self.tests]
        if system == loop._HIDDEN_SYSTEM:
            return [self.hidden]
        if system == loop._DIAGNOSE_SYSTEM:
            return ["ROOT CAUSE: a governing rule is wrong; fix the failing check at its source."]
        if system == loop._GEN_SYSTEM:
            g = self.gens[min(self.gi, len(self.gens) - 1)]
            self.gi += 1
            return [g]
        return [""]


class _Runner:
    """Routes by content: held-out runner, visible-test harness (ModuleType wrapper), keyed on the
    solution marker. `visible[marker]` / `heldout[marker]` map a candidate to its stdout."""
    def __init__(self, visible, heldout=None):
        self.visible, self.heldout = visible, (heldout or {})

    def __call__(self, code, **k):
        if "held-out runner (seeded)" in code:
            for marker, out in self.heldout.items():
                if marker in code:
                    return _res(out)
            return _res("TESTS_PASSED 1/1\n")
        if "ModuleType('_sol')" in code:                 # visible-test harness
            for marker, out in self.visible.items():
                if marker in code:
                    return _res(out)
        return _res("TESTS_PASSED 0/0\n")


# ======================================================================================
# (c) + quarantine: a quarantined invalid check is NEVER reported as failing on correct code.
# ======================================================================================
def test_quarantined_invalid_check_not_reported_and_code_verified(monkeypatch):
    _env(monkeypatch, AGENT_REFERENCE_TESTS="true", AGENT_TEST_VALIDATION="true")
    monkeypatch.setattr(loop, "_invalid_tests", lambda *a, **k: {"test_wrong"})   # oracle fails it
    prov = _P(requirements="- solve(): return the count",
              tests=("def test_good():\n    assert solve() == 5\n"
                     "def test_wrong():\n    assert solve() == 999\n"),
              gens=["def solve():\n    return 5  # CORRECT_V1\n"])
    runner = _Runner({"CORRECT_V1": "TEST test_good PASS\nTEST test_wrong FAIL\nTESTS_PASSED 1/2\n"})
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", runner)

    res = loop.run_agent("compute the count via solve()", use_search=False)

    assert res.verification == "verified"                       # the VALID checks pass -> verified
    assert res.attempts[-1].verdict.get("failing_checks") == []  # the invalid quarantined check is gone


# ======================================================================================
# (a) + (b) + (d): an early attempt fails a genuine check; the latest attempt fixes it -> verified,
# with no stale failure carried forward.
# ======================================================================================
def test_fixed_later_check_not_reported_and_verified(monkeypatch):
    _env(monkeypatch)
    prov = _P(requirements="- solve(): return 5",
              tests="def test_a():\n    assert solve() == 5\n",
              gens=["def solve():\n    return 4  # BUG_V1\n", "def solve():\n    return 5  # FIX_V2\n"])
    runner = _Runner({"BUG_V1": "TEST test_a FAIL\nTESTS_PASSED 0/1\n",
                      "FIX_V2": "TEST test_a PASS\nTESTS_PASSED 1/1\n"})
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", runner)

    res = loop.run_agent("solve() returns 5", use_search=False)

    assert res.verification == "verified"                       # the LATEST attempt satisfies test_a
    assert "FIX_V2" in res.best_code
    assert res.attempts[-1].verdict.get("failing_checks") == []  # no stale 'test_a failing'


# ======================================================================================
# (c): only some checks genuinely fail -> the label names EXACTLY those (quarantined excluded).
# ======================================================================================
def test_partial_names_only_genuine_current_failures(monkeypatch):
    _env(monkeypatch, AGENT_REFERENCE_TESTS="true", AGENT_TEST_VALIDATION="true",
         AGENT_MAX_ATTEMPTS="1")
    monkeypatch.setattr(loop, "_invalid_tests", lambda *a, **k: {"test_wrong"})
    prov = _P(requirements="- solve(): return an odd 5",
              tests=("def test_a():\n    assert solve() == 5\n"
                     "def test_b():\n    assert solve() % 2 == 1\n"
                     "def test_wrong():\n    assert solve() == 999\n"),
              gens=["def solve():\n    return 4  # PARTIAL_V1\n"])
    runner = _Runner({"PARTIAL_V1": ("TEST test_a FAIL\nTEST test_b FAIL\nTEST test_wrong FAIL\n"
                                     "TESTS_PASSED 0/2\n")})
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", runner)

    res = loop.run_agent("solve() returns an odd 5", use_search=False)

    assert res.verification == "partial"
    fc = set(res.attempts[-1].verdict.get("failing_checks") or [])
    assert fc == {"test_a", "test_b"}                            # exactly the genuine current failures
    assert "test_wrong" not in fc                                # the quarantined invalid check excluded


# ======================================================================================
# Held-out staleness: on a visible-score TIE the later, better-generalising attempt is selected, so
# the reported held-out tally reflects the LATEST attempt — not a stale earlier one.
# ======================================================================================
def test_selection_prefers_latest_better_heldout(monkeypatch):
    _env(monkeypatch, AGENT_HIDDEN_TESTS="true", AGENT_MAX_ATTEMPTS="2")
    prov = _P(requirements="- solve(): a sampler with invariant checks",
              tests="def test_a():\n    assert solve() is not None\n",
              gens=["def solve():\n    return 1  # GEN_V1\n", "def solve():\n    return 1  # GEN_V2\n"])
    # both pass all visible (score ties at 100); held-out: V1 2/5, V2 4/5 (both still fail -> partial)
    runner = _Runner(
        visible={"GEN_V1": "TEST test_a PASS\nTESTS_PASSED 1/1\n",
                 "GEN_V2": "TEST test_a PASS\nTESTS_PASSED 1/1\n"},
        heldout={"GEN_V1": "TESTS_PASSED 2/5\n", "GEN_V2": "TESTS_PASSED 4/5\n"})
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", runner)

    res = loop.run_agent("implement solve() with invariant checks", use_search=False)

    assert res.verification == "partial"                        # held-out genuinely fails on both
    assert "GEN_V2" in res.best_code                            # the LATER, better-generalising attempt
    assert res.hidden_passed == 4                               # the latest tally, not the stale 2/5
