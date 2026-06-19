"""Non-unique-quantity test validation.

A generated test may fail code only when the code is genuinely WRONG. Many correct answers are
defined only UP TO scaling / sign / ordering / phase / basis / representation (or come from an
underdetermined procedure): asserting exact equality to ONE reference output is invalid — it fails
valid ALTERNATIVE solutions. The single oracle agrees with itself, so such a test slips past the
normal quarantine. We detect it by EXECUTION: an independent cross-reference that returns a DIFFERENT
valid representation fails exactly those exact tests while still satisfying every property check, so
they are quarantined. Fully offline — the sandbox runner + reference generation are monkeypatched.

Proves: (a) an exact test on a non-unique quantity is quarantined (a valid alternative then passes);
(b) exact checks on UNIQUE quantities are NOT quarantined and still catch wrong answers; (c) property
checks are never dropped, and a cross-reference that violates a property is not trusted (fail-open),
so wrong answers stay catchable; (d) the verdict is DERIVED from the executed differential, not
guessed; plus the end-to-end wiring through run_agent.
"""
import types

import backend.agent.loop as loop


def _res(stdout):
    return types.SimpleNamespace(ok=True, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


ORACLE = "def solve(x):\n    return 1  # ORACLE_MARK"
ALT = "def solve(x):\n    return 1  # ALT_MARK"     # a DIFFERENT but equally-valid representation
TESTS = ("def test_exact_value():\n    assert solve(0) == ref.solve(0)\n"
         "def test_invariant_prop():\n    assert True\n"
         "def test_definition_x():\n    assert True\n")


def _router(*, oracle_stdout, alt_stdout):
    """run_python_auto mock: routes by whether the candidate SOLUTION is the cross-reference (ALT_MARK)
    or the primary oracle (the _invalid_tests self-run, which has no ALT_MARK)."""
    def run(code, **k):
        return _res(alt_stdout) if "ALT_MARK" in code else _res(oracle_stdout)
    return run


_ALL_PASS = ("TEST test_exact_value PASS\nTEST test_invariant_prop PASS\n"
             "TEST test_definition_x PASS\nTESTS_PASSED 3/3\n")
_ALT_DIVERGES = ("TEST test_exact_value FAIL\nTEST test_invariant_prop PASS\n"
                 "TEST test_definition_x PASS\nTESTS_PASSED 2/3\n")


def _nu(seeds=1, task_type="numeric_algorithm"):
    return loop._nonunique_exact_tests(object(), "task", "reqs", task_type, ORACLE, TESTS, seeds)


# ----------------------------------------------------------------------
# Toggle + the divergent cross-reference prompt
# ----------------------------------------------------------------------
def test_toggle(monkeypatch):
    monkeypatch.delenv("AGENT_NONUNIQUE_VALIDATION", raising=False)
    assert loop.nonunique_validation_enabled() is True
    monkeypatch.setenv("AGENT_NONUNIQUE_VALIDATION", "false")
    assert loop.nonunique_validation_enabled() is False


def test_divergent_clause_asks_for_a_different_valid_representation():
    seen = {}

    class P:
        is_available = True
        name, model = "openai", "t"

        def stream_chat(self, messages, system="", **k):
            seen["user"] = messages[-1]["content"]
            return ["def solve(x):\n    return 1"]

    loop._generate_reference(P(), "task", "reqs", "numeric_algorithm", divergent=True)
    u = seen["user"].lower()
    assert "non-unique" in u
    assert "scaling" in u and "sign" in u and "basis" in u
    assert "different but equally" in u or "opposite-sign" in u
    # Without divergent=True the clause is absent.
    loop._generate_reference(P(), "task", "reqs", "numeric_algorithm")
    assert "non-unique" not in seen["user"].lower()


# ----------------------------------------------------------------------
# (a) exact-on-non-unique is QUARANTINED (a valid alternative then passes)
# ----------------------------------------------------------------------
def test_a_nonunique_exact_test_is_quarantined(monkeypatch):
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: ALT)
    monkeypatch.setattr(loop, "run_python_auto", _router(oracle_stdout=_ALL_PASS, alt_stdout=_ALT_DIVERGES))
    assert _nu() == {"test_exact_value"}      # the valid alt fails ONLY the exact test -> quarantine it


# ----------------------------------------------------------------------
# (b) a UNIQUE quantity is NOT quarantined (the cross-reference returns the same value)
# ----------------------------------------------------------------------
def test_b_unique_exact_test_not_quarantined(monkeypatch):
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: ALT)
    monkeypatch.setattr(loop, "run_python_auto", _router(oracle_stdout=_ALL_PASS, alt_stdout=_ALL_PASS))
    assert _nu() == set()                     # alt agrees -> unique -> the exact check STAYS (catches wrong)


# ----------------------------------------------------------------------
# (c) property checks are protected; a cross-reference that VIOLATES a property is not trusted
# ----------------------------------------------------------------------
def test_c_property_violating_crossref_fails_open(monkeypatch):
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: ALT)
    alt_breaks_property = ("TEST test_exact_value FAIL\nTEST test_invariant_prop FAIL\n"
                           "TEST test_definition_x PASS\nTESTS_PASSED 1/3\n")
    monkeypatch.setattr(loop, "run_python_auto",
                        _router(oracle_stdout=_ALL_PASS, alt_stdout=alt_breaks_property))
    assert _nu() == set()                     # cross-ref violates a property -> untrusted -> drop nothing


def test_c_property_checks_are_never_quarantined(monkeypatch):
    # Even if the alt "fails" a property line, the differential never returns a test_invariant/
    # test_definition name (those are property checks, not exact-on-non-unique).
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: ALT)
    monkeypatch.setattr(loop, "run_python_auto", _router(oracle_stdout=_ALL_PASS, alt_stdout=_ALT_DIVERGES))
    nu = _nu()
    assert not any(n.startswith(("test_invariant", "test_definition")) for n in nu)


# ----------------------------------------------------------------------
# (d) the verdict is DERIVED from the executed differential, not guessed
# ----------------------------------------------------------------------
def test_d_classification_tracks_the_executed_differential(monkeypatch):
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: ALT)
    # alt DISAGREES on the exact value -> quarantined
    monkeypatch.setattr(loop, "run_python_auto", _router(oracle_stdout=_ALL_PASS, alt_stdout=_ALT_DIVERGES))
    assert _nu() == {"test_exact_value"}
    # SAME task/oracle/tests, but the alt now AGREES (different EXECUTION result) -> NOT quarantined
    monkeypatch.setattr(loop, "run_python_auto", _router(oracle_stdout=_ALL_PASS, alt_stdout=_ALL_PASS))
    assert _nu() == set()


# ----------------------------------------------------------------------
# Fail-open everywhere — a hiccup never drops a genuine exact test
# ----------------------------------------------------------------------
def test_fail_open(monkeypatch):
    monkeypatch.setenv("AGENT_NONUNIQUE_VALIDATION", "false")
    assert _nu() == set()                                            # disabled
    monkeypatch.setenv("AGENT_NONUNIQUE_VALIDATION", "true")
    assert loop._nonunique_exact_tests(object(), "t", "r", "n", "", TESTS, 1) == set()   # no oracle
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: "")                  # no cross-reference
    assert _nu() == set()
    monkeypatch.setattr(loop, "_generate_reference", lambda *a, **k: ALT)                 # cross-ref crashes
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res("TESTS_PASSED 0/0\n"))
    assert _nu() == set()
    # cross-ref produced but the suite has NO property check -> can't validate -> fail open
    monkeypatch.setattr(loop, "run_python_auto",
                        _router(oracle_stdout="TEST test_exact_value PASS\nTESTS_PASSED 1/1\n",
                                alt_stdout="TEST test_exact_value FAIL\nTESTS_PASSED 0/1\n"))
    no_props = "def test_exact_value():\n    assert solve(0) == ref.solve(0)\n"
    assert loop._nonunique_exact_tests(object(), "t", "r", "n", ORACLE, no_props, 1) == set()


# ----------------------------------------------------------------------
# End-to-end: run_agent accepts a VALID ALTERNATIVE for a non-unique result, and the exact test that
# only the oracle's representation satisfies is quarantined (not counted).
# ----------------------------------------------------------------------
class _NUProvider:
    """Returns a DIVERGENT cross-reference when the reference prompt asks for one (the 'NON-UNIQUE'
    clause); the primary reference otherwise."""
    is_available = True
    name, model = "openai", "test"

    def __init__(self, requirements, tests, primary_ref, alt_ref, solution):
        self.requirements, self.tests = requirements, tests
        self.primary_ref, self.alt_ref, self.solution = primary_ref, alt_ref, solution

    def stream_chat(self, messages, system="", **k):
        user = messages[-1]["content"] if messages else ""
        if system == loop._REQ_SYSTEM:
            return [self.requirements]
        if system == loop._REFERENCE_SYSTEM:
            return [self.alt_ref if "NON-UNIQUE" in user else self.primary_ref]
        if system == loop._TESTS_SYSTEM:
            return [self.tests]
        if system == loop._GEN_SYSTEM:
            return [self.solution]
        return [""]


def test_run_agent_accepts_valid_alternative_for_nonunique_result(monkeypatch):
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    for k, v in {"AGENT_REFERENCE_TESTS": "true", "AGENT_TEST_VALIDATION": "true",
                 "AGENT_NONUNIQUE_VALIDATION": "true", "AGENT_HIDDEN_TESTS": "false",
                 "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
                 "AGENT_ANTICHEAT_SCAN": "false", "AGENT_PARALLEL_N": "1",
                 "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)

    tests = ("def test_exact_value():\n    assert tuple(unit_eigvec()) == tuple(ref.unit_eigvec())\n"
             "def test_invariant_norm():\n    assert True\n")     # a property in the VISIBLE suite
    prov = _NUProvider(
        requirements="- unit_eigvec(): the dominant UNIT eigenvector",
        tests=tests,
        primary_ref="def unit_eigvec():\n    return (1.0, 0.0)  # PRIMARY_REF",
        alt_ref="def unit_eigvec():\n    return (-1.0, 0.0)  # ALT_REF (opposite sign, still valid)",
        solution="def unit_eigvec():\n    return (-1.0, 0.0)  # CAND_VALID (opposite sign)")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "docker_available", lambda: True)

    def run(code, **k):
        # Route by the SOLUTION (the candidate), which is the FIRST module assembled — every script
        # also embeds the oracle as the *reference*, so we key on the valid-alternative markers. Only
        # the oracle's own representation satisfies the exact test; every valid alternative (the
        # divergent cross-ref AND the candidate) returns the opposite sign and FAILS it, but all
        # satisfy the norm property.
        sol_is_oracle = ("ALT_REF" not in code) and ("CAND_VALID" not in code)
        ex = "PASS" if sol_is_oracle else "FAIL"
        n = 1 + (ex == "PASS")
        return _res(f"TEST test_exact_value {ex}\nTEST test_invariant_norm PASS\nTESTS_PASSED {n}/2\n")

    monkeypatch.setattr(loop, "run_python_auto", run)

    events = []
    res = loop.run_agent("compute the dominant unit eigenvector", use_search=False,
                         on_event=events.append)

    assert res.verification == "verified"          # the opposite-sign eigenvector is ACCEPTED as correct
    assert "CAND_VALID" in res.best_code           # code left as the valid alternative (not forced to match)
    nu = [e for e in events if e.get("type") == "test_validation" and e.get("scope") == "nonunique"]
    assert nu and "test_exact_value" in nu[0]["quarantined"]   # the non-unique exact test was quarantined

    # CONTRAST: with the fix OFF, the same valid alternative is WRONGLY rejected (the exact test
    # blames correct code) — proving the quarantine is what makes the alternative pass.
    monkeypatch.setenv("AGENT_NONUNIQUE_VALIDATION", "false")
    res_off = loop.run_agent("compute the dominant unit eigenvector", use_search=False, max_iters=1)
    assert res_off.verification != "verified"      # without the fix, the valid alternative fails the exact test
