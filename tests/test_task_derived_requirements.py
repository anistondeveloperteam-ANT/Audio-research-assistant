"""Requirements must be DERIVED FROM THE SPECIFIC TASK, never an idealised template. A property an
idealised system has but THIS task BREAKS (a dissipative / driving / open / injecting / asymmetric /
random term) must NOT be enforced: a FAITHFUL reference itself fails such an invariant, so it is
quarantined ('requirement-not-implied-by-task') and correct code is verified — not flagged for
violating a requirement the task never had.

Proves:
  (a) an invariant asserting a property THIS task breaks is quarantined (the faithful reference fails
      it) and correct, complete code is labeled verified — across 3 different domains;
  (b) an invariant the system genuinely has is KEPT (the reference passes it) so it still gates code;
  (c) definition checks stay oracle-exempt;
  (d) the four check-generators + the diagnosis carry the property-applicability rule.

Deterministic: no network, no Docker, no real LLM.
"""
import types

import pytest

from backend.agent import loop


def _res(stdout, ok=True):
    return types.SimpleNamespace(ok=ok, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ======================================================================================
# Unit: the reference backstop quarantines an INAPPLICABLE invariant, keeps an APPLICABLE one.
# ======================================================================================
def test_inapplicable_invariant_is_quarantined(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "true")
    # A faithful DAMPED-oscillator reference cannot conserve energy -> it FAILS the bogus invariant
    # but PASSES the applicable one (energy decays). The bogus check is 'not implied by the task'.
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res(
        "TEST test_invariant_energy_conserved FAIL\n"
        "TEST test_invariant_energy_decays PASS\n"
        "TESTS_PASSED 1/2\n"))
    q = loop._heldout_quarantine(
        "def step():\n    return 'damped'\n",
        ("def test_invariant_energy_conserved():\n    assert True\n"
         "def test_invariant_energy_decays():\n    assert True\n"), seeds=1)
    assert q == {"test_invariant_energy_conserved"}             # inapplicable property quarantined
    assert "test_invariant_energy_decays" not in q             # the real property is kept


def test_applicable_invariant_is_kept(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "true")
    # A genuinely CLOSED system conserves the total -> the reference PASSES the invariant -> kept.
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res(
        "TEST test_invariant_total_conserved PASS\nTESTS_PASSED 1/1\n"))
    q = loop._heldout_quarantine(
        "def step():\n    return 'closed'\n",
        "def test_invariant_total_conserved():\n    assert True\n", seeds=1)
    assert q == set()                                          # applicable invariant still gates code


def test_reference_validation_covers_definition_checks(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_VALIDATION", "true")
    # The reference passes one check (so it is not the "all-fail -> unreliable, trust nothing" case) and
    # fails one definition + one invariant. The UNIVERSAL gate quarantines BOTH — definition checks are
    # no longer exempt: any check the correct reference fails is invalid, whatever its type.
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _res(
        "TEST test_definition_x FAIL\nTEST test_invariant_y FAIL\n"
        "TEST test_invariant_ok PASS\nTESTS_PASSED 1/3\n"))
    q = loop._heldout_quarantine(
        "ref\n",
        ("def test_definition_x():\n    assert True\n"
         "def test_invariant_y():\n    assert True\n"
         "def test_invariant_ok():\n    assert True\n"), seeds=1)
    assert q == {"test_definition_x", "test_invariant_y"}      # definition is reference-validated too


# ======================================================================================
# (d) Prompt-content: the generators + diagnosis must reason about property APPLICABILITY.
# ======================================================================================
def test_generators_and_diagnosis_require_property_applicability():
    inv = loop._INVARIANTS_SYSTEM.lower()
    assert "dissipative" in inv and "driving" in inv and "open" in inv
    assert "must not assert" in inv and "damped" in inv                 # concrete breaking example
    hid = loop._HIDDEN_SYSTEM.lower()
    assert "do not assert" in hid and "dissipative" in hid
    ref = loop._REFERENCE_SYSTEM.lower()
    assert "dissipative" in ref and "textbook ideal" in ref            # model the real, non-ideal system
    diag = loop._DIAGNOSE_SYSTEM.lower()
    assert "genuinely" in diag and "dissipative" in diag and "drift is" in diag


# ======================================================================================
# (a)+(d) end-to-end: an inapplicable idealised invariant is quarantined -> correct code VERIFIED.
# Three DIFFERENT domains, each deliberately breaking a common idealised property.
# ======================================================================================
def _env(monkeypatch):
    for k, v in {"AGENT_REFERENCE_TESTS": "true", "AGENT_TEST_VALIDATION": "true",
                 "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "true",
                 "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
                 "AGENT_ROOT_CAUSE_DIAGNOSIS": "true", "AGENT_ANTICHEAT_SCAN": "false",
                 "AGENT_MASKING_SCAN": "false", "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1",
                 "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "2", "AGENT_STALL_LIMIT": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


class _P:
    is_available = True
    name, model = "openai", "test"

    def __init__(self, invariants, hidden):
        self.invariants, self.hidden = invariants, hidden

    def stream_chat(self, messages, system="", **k):
        if system == loop._REFERENCE_SYSTEM:
            return ["def step():\n    return 'faithful'  # models the REAL (non-ideal) system\n"]
        if system == loop._REQ_SYSTEM:
            return ["- step(): evolve the described system"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert step() is not None\n"]
        if system == loop._INVARIANTS_SYSTEM:
            return [self.invariants]
        if system == loop._HIDDEN_SYSTEM:
            return [self.hidden]
        if system == loop._GEN_SYSTEM:
            return ["def step():\n    return 'correct'  # correctly NON-ideal\n"]
        return [""]


class _DomainRunner:
    """Held-out run: the bogus idealised invariant FAILs for the faithful reference AND the correct
    candidate (both are correctly non-ideal); the applicable invariant + hidden test PASS. Visible
    run: passes."""
    def __init__(self, bogus, applicable):
        self.bogus, self.applicable = bogus, applicable

    def __call__(self, code, **k):
        if "held-out runner (seeded)" in code:
            return _res(f"TEST {self.bogus} FAIL\nTEST {self.applicable} PASS\n"
                        "TEST test_hidden_step PASS\nTESTS_PASSED 2/3\n")
        if "ModuleType('_sol')" in code:
            return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")
        return _res("TESTS_PASSED 0/0\n")


_DOMAINS = [
    ("physics: damped oscillator — energy NOT conserved",
     "test_invariant_energy_conserved", "test_invariant_energy_decays"),
    ("ecology: open population w/ injection+removal — total NOT conserved",
     "test_invariant_total_conserved", "test_invariant_total_tracks_net_flux"),
    ("signal: noisy series — NOT monotonic",
     "test_invariant_monotonic", "test_invariant_mean_within_band"),
]


@pytest.mark.parametrize("desc,bogus,applicable", _DOMAINS)
def test_inapplicable_idealized_property_not_enforced(monkeypatch, desc, bogus, applicable):
    _env(monkeypatch)
    prov = _P(invariants=(f"def {bogus}():\n    assert True\n"
                          f"def {applicable}():\n    assert True\n"),
              hidden="def test_hidden_step():\n    assert step() is not None\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _DomainRunner(bogus, applicable))

    res = loop.run_agent(f"simulate step() — {desc}", use_search=False)

    # The faithful reference fails the bogus invariant -> it is quarantined -> the correct, non-ideal
    # candidate is VERIFIED, not flagged for violating a requirement THIS task never had.
    assert res.verification == "verified", desc
