"""Re-validate-against-the-latest-attempt: the code agent must not OVER-FLAG a failure that an
earlier attempt had but the LATEST attempt has since fixed. Before settling for 'partial', the
finalize step re-runs the most recent delivery-eligible attempt on FRESH stdout and, if it now
produces every requested deliverable, labels it VERIFIED — without ever resurrecting a held-out /
visible / cheating / off-topic failure.

Proves:
  (a) a delivery problem FIXED in a later attempt is re-validated and NOT reported as still-failing;
  (b) a genuinely still-missing deliverable on the latest code KEEPS the honest partial;
  (c) a complete final attempt ends 'verified', not 'partial' (end-to-end);
plus the safety rails: held-out / visible / cheating failures are never resurrected, no false
'verified' on an incomplete run, and the re-validation is bounded to ONE extra run.

Deterministic: no network, no Docker, no real LLM.
"""
import types

from backend.agent import loop
from backend.agent.loop import Attempt, _latest_revalidated_delivery


def _res(stdout, ok=True):
    return types.SimpleNamespace(ok=ok, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ======================================================================================
# Unit: _latest_revalidated_delivery — the bounded finalize re-validation.
# ======================================================================================
def _att(code, **verdict):
    """An Attempt whose verdict defaults to delivery-eligible (visible + held-out passed, only the
    delivery gate could have demoted it: verified=False)."""
    base = dict(verified=False, cheating=False, total=2, passed=2, hidden_total=1, hidden_passed=1,
                gate_fail="completeness: requested output(s) missing from stdout: energy",
                diagnosis="ROOT CAUSE: the program never prints the energy.")
    base.update(verdict)
    return Attempt(1, code, _res(""), base)


def _runner():
    """Demo-capture stub: prints the deliverable only for code marked COMPLETE; counts its calls."""
    calls = {"n": 0}

    def run(code, **k):
        calls["n"] += 1
        return _res("energy = 5.0\n" if "HASOUT" in code else "x = 5.0\n")
    run.calls = calls
    return run


def test_revalidate_upgrades_latest_complete(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    atts = [_att("NOOUT_v1"), _att("HASOUT_v2")]                  # latest prints the deliverable
    res = _latest_revalidated_delivery(atts, ["energy"], True)
    assert res is not None
    assert res.code == "HASOUT_v2"                               # the LATEST attempt was chosen
    assert res.verdict["verified"] is True and res.verdict["done"] is True
    assert res.verdict["gate_fail"] == "" and res.verdict["diagnosis"] == ""   # stale flags cleared
    assert "energy" in res.verdict["demo_output"]                # fresh stdout cached
    assert run.calls["n"] == 1                                   # bounded: exactly ONE re-run


def test_revalidate_keeps_partial_when_latest_incomplete(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    atts = [_att("NOOUT_v1")]                                     # latest genuinely lacks the value
    assert _latest_revalidated_delivery(atts, ["energy"], True) is None
    assert run.calls["n"] == 1                                   # re-ran the latest to CONFIRM, once


def test_revalidate_does_not_resurrect_heldout_failure(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    # latest prints the value but FAILED the held-out gate -> a genuine non-delivery failure
    atts = [_att("HASOUT_v1", hidden_fail=True, hidden_passed=0, hidden_total=1)]
    assert _latest_revalidated_delivery(atts, ["energy"], True) is None
    assert run.calls["n"] == 0                                   # never even re-run -> never upgraded


def test_revalidate_no_false_verify_on_visible_failure(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    atts = [_att("HASOUT_v1", passed=1, total=2)]                # a visible test failed
    assert _latest_revalidated_delivery(atts, ["energy"], True) is None
    assert run.calls["n"] == 0


def test_revalidate_noop_when_latest_already_verified(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    atts = [_att("HASOUT_v1", verified=True, gate_fail="", diagnosis="")]
    assert _latest_revalidated_delivery(atts, ["energy"], True) is None
    assert run.calls["n"] == 0


def test_revalidate_skips_cheating_and_uses_latest_legitimate(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    atts = [_att("HASOUT_v1"), _att("HASOUT_v2_cheat", cheating=True)]
    res = _latest_revalidated_delivery(atts, ["energy"], True)
    assert res is not None and res.code == "HASOUT_v1"           # the cheating latest is excluded


def test_revalidate_disabled_when_gate_off(monkeypatch):
    run = _runner()
    monkeypatch.setattr(loop, "run_python_auto", run)
    atts = [_att("HASOUT_v1")]
    assert _latest_revalidated_delivery(atts, ["energy"], False) is None
    assert run.calls["n"] == 0


# ======================================================================================
# Integration: the full run_agent loop. Early attempt misses the output, the LATEST attempt prints
# it; the per-round delivery capture went stale -> finalize re-validation upgrades to verified.
# ======================================================================================
def _env(monkeypatch, max_attempts="2"):
    for k, v in {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
                 "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
                 "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "true",
                 "AGENT_ROOT_CAUSE_DIAGNOSIS": "true", "AGENT_ANTICHEAT_SCAN": "false",
                 "AGENT_MASKING_SCAN": "false", "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1",
                 "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": max_attempts,
                 "AGENT_STALL_LIMIT": "3"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


_R1 = "def compute_energy():\n    return 5.0  # R1_INCOMPLETE: never prints\n"
_R2 = ("def compute_energy():\n    return 5.0  # R2_COMPLETE\n"
       "if __name__ == '__main__':\n    print('energy =', compute_energy())\n")


class _P:
    """Routes by system prompt. Returns R1 first; once the injected diagnosis ('ROOT CAUSE') reaches
    the solver, returns R2 (which prints the deliverable)."""
    is_available = True
    name, model = "openai", "test"

    def __init__(self):
        self.calls = []

    def stream_chat(self, messages, system="", **k):
        user = messages[-1]["content"] if messages else ""
        self.calls.append((system, user))
        if system == loop._REQ_SYSTEM:
            return ["- compute_energy(): compute and PRINT the energy"]
        if system == loop._DELIVERABLES_SYSTEM:
            return ["energy"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert compute_energy() is not None\n"]
        if system == loop._DIAGNOSE_SYSTEM:
            return ["ROOT CAUSE: the program never prints the energy; add a __main__ that prints it."]
        if system == loop._DRIVER_SYSTEM:
            return ["print('energy =', compute_energy())"]
        if system == loop._GEN_SYSTEM:
            return [_R2 if "ROOT CAUSE" in user else _R1]
        return [""]


class _StaleThenFreshRunner:
    """Visible test runs pass. The demo capture for the COMPLETE code returns NO deliverable the
    first time (a stale/transient per-round capture) and the REAL output on the finalize re-run."""
    def __init__(self, complete_eventually=True):
        self.complete_eventually = complete_eventually
        self.r2_demo = 0

    def __call__(self, code, **k):
        if "ModuleType('_sol')" in code:                 # the visible-test harness wrapper
            return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")
        # raw solution -> delivery/completeness capture
        if "R2_COMPLETE" in code:
            self.r2_demo += 1
            if self.complete_eventually and self.r2_demo >= 2:
                return _res("computing…\nenergy = 5.0\n")     # finalize re-validation: fresh+complete
            return _res("computing…\n")                       # per-round capture: stale (no value)
        return _res("computing…\n")                           # R1: never prints the deliverable


def test_complete_final_attempt_ends_verified_not_partial(monkeypatch):
    _env(monkeypatch)
    prov = _P()
    runner = _StaleThenFreshRunner(complete_eventually=True)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", runner)

    events = []
    res = loop.run_agent("compute and print the energy", use_search=False, on_event=events.append)

    assert res.verification == "verified"                # NOT a stale 'partial'
    assert "R2_COMPLETE" in res.best_code                # the LATEST attempt is presented
    assert "energy" in (res.best_output or "")           # its real output is shown
    assert loop._DIAGNOSE_SYSTEM in [s for s, _u in prov.calls]   # diagnosis still drove the rewrite


def test_genuinely_missing_output_stays_partial(monkeypatch):
    _env(monkeypatch)
    prov = _P()
    runner = _StaleThenFreshRunner(complete_eventually=False)   # the value is NEVER printed
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", runner)

    res = loop.run_agent("compute and print the energy", use_search=False)

    assert res.verification == "partial"                 # a genuine current miss is still honest
