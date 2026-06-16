"""The four universal delivery gates for the code agent (backend/agent/loop.py): COMPLETE, CORRECT
(spec), ROBUST, RUNNABLE. Fully offline — no LLM, no Docker; the sandbox runner is monkeypatched."""
import types

from backend.agent import loop


def _ok(stdout="TESTS_PASSED", stderr=""):
    return types.SimpleNamespace(ok=True, stdout=stdout, stderr=stderr, error="")


# ----------------------------------------------------------------------
# The prompts/standards demand each gate.
# ----------------------------------------------------------------------
def test_prompts_demand_complete_and_spec():
    assert "complete" in loop._GEN_SYSTEM.lower() and "every result" in loop._GEN_SYSTEM.lower()
    assert "deliverable" in loop._REQ_SYSTEM.lower()
    assert "print every value" in loop._DRIVER_SYSTEM.lower()
    inv = loop._INVARIANTS_SYSTEM.lower()
    assert "spec" in inv and "request" in inv and "same-model reference" in inv


# ----------------------------------------------------------------------
# Pure helpers: deliverable parsing + completeness check.
# ----------------------------------------------------------------------
def test_parse_deliverables_cleans_and_dedups():
    text = "- Period\n* Amplitude\n1. Energy\n`Momentum`\nPeriod\nimport math\n" + "x" * 50
    out = loop._parse_deliverables(text)
    assert out == ["period", "amplitude", "energy", "momentum"]


def test_check_completeness_flags_missing_only():
    assert loop._check_completeness(["period", "kinetic energy"],
                                    "Period (s): 2.0\nKinetic energy: 5.0") == []
    assert loop._check_completeness(["period", "energy", "momentum"],
                                    "period: 2.0  energy: 5.0") == ["momentum"]
    assert loop._check_completeness([], "anything") == []
    assert loop._check_completeness(["put-call parity"], "Put-Call parity holds: True") == []


# ----------------------------------------------------------------------
# (a) COMPLETE — a solution missing a requested output is rejected.
# ----------------------------------------------------------------------
def test_completeness_gate_rejects_missing_output():
    verdict = {"verified": True, "done": True}
    loop._apply_output_gates(verdict, wants_output=True, output="period = 2.0", missing=["energy"])
    assert verdict["verified"] is False and verdict["done"] is False
    assert "completeness" in verdict["gate_fail"] and "energy" in verdict["gate_fail"]


# ----------------------------------------------------------------------
# (d) RUNNABLE — a code-intent task with no executed output is rejected.
# ----------------------------------------------------------------------
def test_execution_gate_rejects_empty_output():
    verdict = {"verified": True, "done": True}
    loop._apply_output_gates(verdict, wants_output=True, output="", missing=[])
    assert verdict["verified"] is False
    assert "execution" in verdict["gate_fail"]


def test_output_gates_pass_when_complete_and_runnable():
    verdict = {"verified": True, "done": True}
    loop._apply_output_gates(verdict, wants_output=True, output="period=2.0 energy=5.0", missing=[])
    assert verdict["verified"] is True and "gate_fail" not in verdict


def test_output_gates_never_resurrect_a_failed_verdict():
    verdict = {"verified": False, "done": False}     # an earlier gate already failed
    loop._apply_output_gates(verdict, wants_output=True, output="all good", missing=[])
    assert verdict["verified"] is False


def test_output_gates_skip_when_no_output_requested():
    verdict = {"verified": True, "done": True}
    loop._apply_output_gates(verdict, wants_output=False, output="", missing=[])
    assert verdict["verified"] is True


# ----------------------------------------------------------------------
# (b) CORRECT (spec) — passes its reference/visible tests but fails a spec-derived held-out check.
# ----------------------------------------------------------------------
def test_spec_gate_rejects_when_spec_check_fails(monkeypatch):
    heldout = ("def test_invariant_spec():\n    pass\n"
               "def test_hidden_a():\n    pass\n")
    # The very first input (the request's own values) violates a spec-derived identity -> fail.
    monkeypatch.setattr(loop, "_run_against_tests", lambda *a, **k: (_ok(), 1, 2))
    ok, passed, total, _ = loop._verify_heldout("SOLUTION", heldout, seeds=3)
    assert ok is False                               # rejected despite matching the reference


# ----------------------------------------------------------------------
# (c) ROBUST — right on the demo input, wrong on another valid input -> rejected.
# ----------------------------------------------------------------------
def test_robustness_gate_rejects_fragile_code(monkeypatch):
    heldout = ("def test_hidden_a():\n    pass\n"
               "def test_hidden_b():\n    pass\n")
    calls = {"n": 0}

    def fake_run(sol, tests, footer, reference_src=""):
        calls["n"] += 1
        return (_ok(), 2, 2) if calls["n"] == 1 else (_ok(), 1, 2)

    monkeypatch.setattr(loop, "_run_against_tests", fake_run)
    ok, _p, _t, _ = loop._verify_heldout("SOLUTION", heldout, seeds=3)
    assert ok is False and calls["n"] == 2


# ----------------------------------------------------------------------
# Toggle
# ----------------------------------------------------------------------
def test_delivery_gates_toggle(monkeypatch):
    monkeypatch.delenv("AGENT_DELIVERY_GATES", raising=False)
    assert loop.delivery_gates_enabled() is True
    monkeypatch.setenv("AGENT_DELIVERY_GATES", "false")
    assert loop.delivery_gates_enabled() is False
