"""The DEFINITION-MATCH gate: the reported value must be the EXACT quantity the user asked for
(quantity / point / aggregation / units), checked independently of the candidate AND its reference.
Fully offline — no LLM, no Docker; the sandbox runner is monkeypatched and the fakes are plain."""
import statistics
import types

import pytest

import backend.agent.loop as loop


# ----------------------------------------------------------------------
# Toggle + prompt demands the right, independent, per-output checks.
# ----------------------------------------------------------------------
def test_definition_gate_toggle(monkeypatch):
    monkeypatch.delenv("AGENT_DEFINITION_GATE", raising=False)
    assert loop.definition_gate_enabled() is True
    monkeypatch.setenv("AGENT_DEFINITION_GATE", "false")
    assert loop.definition_gate_enabled() is False


def test_definition_prompt_demands_independent_per_output_checks():
    s = loop._DEFINITION_SYSTEM.lower()
    assert "per explicitly requested output" in s
    assert "independently" in s
    assert "do not trust or call any reference" in s
    for kw in ("median", "aggregation", "point", "unit", "label"):    # the whole wrong-thing class
        assert kw in s, kw


def test_generate_definition_checks_uses_definition_system():
    class P:
        def __init__(self):
            self.sys = None

        def stream_chat(self, messages, system="", **k):
            self.sys = system
            return ["```python\ndef test_definition_x():\n    assert True\n```"]

    p = P()
    out = loop._generate_definition_checks(p, "task", "requirements")
    assert p.sys == loop._DEFINITION_SYSTEM
    assert "test_definition_x" in out


# ----------------------------------------------------------------------
# The gate DISCRIMINATES by definition — a wrong-but-related quantity is caught.
# (Plain functions stand in for what the gate generates + the candidate; sandbox-free.)
# ----------------------------------------------------------------------
def test_definition_check_catches_wrong_aggregation():
    # The request asked for the MEDIAN; the check (built from the request, not from the candidate)
    # must FAIL a candidate that reports the MEAN and PASS one that reports the median.
    def definition_check(report_fn):
        data = [1, 2, 3, 4, 100]                       # mean=22, median=3 -> clearly different
        assert abs(report_fn(data) - statistics.median(data)) < 1e-9, "reported value is not the median"

    with pytest.raises(AssertionError):
        definition_check(lambda xs: sum(xs) / len(xs))  # mean -> wrong quantity -> caught
    definition_check(statistics.median)                 # correct -> passes


# ----------------------------------------------------------------------
# The gate is ENFORCED: a failing definition check -> not verified -> regenerate.
# ----------------------------------------------------------------------
def test_verify_heldout_rejects_definition_mismatch(monkeypatch):
    heldout = ("def test_definition_value():\n    pass\n"
               "def test_hidden_a():\n    pass\n")
    monkeypatch.setattr(loop, "_run_against_tests", lambda *a, **k: (
        types.SimpleNamespace(ok=True, stdout="", stderr="reported value is not the median", error=""),
        1, 2))
    ok, _p, _t, _last = loop._verify_heldout("SOLUTION", heldout, seeds=2)
    assert ok is False                                  # the loop then regenerates with the reason


# ----------------------------------------------------------------------
# The definition gate runs as part of the held-out suite — even when hidden tests are OFF.
# ----------------------------------------------------------------------
def test_definition_gate_runs_even_when_hidden_tests_off(monkeypatch):
    from test_agent import _FakeProvider, _fence, _ok

    provider = _FakeProvider(
        requirements="- double(n): return n*2; report double(3)",
        tests=_fence("def test_double():\n    assert True"),
        first=[_fence("def double(n):\n    return n * 2")],
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "false")
    monkeypatch.setenv("AGENT_DEFINITION_GATE", "true")
    monkeypatch.setenv("AGENT_DELIVERY_GATES", "false")
    monkeypatch.setenv("AGENT_PARALLEL_N", "1")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "1")
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)

    loop.run_agent("implement double(n) returning n and report double(3)", use_search=False,
                   max_iters=1, on_event=lambda e: None)
    systems = [s for s, _u in provider.calls]
    assert loop._DEFINITION_SYSTEM in systems           # definition checks were generated
    assert loop._HIDDEN_SYSTEM not in systems           # hidden tests were off (gate is independent)
