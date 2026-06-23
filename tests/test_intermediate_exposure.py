"""Intermediate-value exposure: when a task asks to report an INTERMEDIATE / component quantity (an
envelope, a sub-result, a per-stage metric), the function must EXPOSE it (return it) and the report
must read THAT real value — never a fabricated stand-in re-derived from the final result.

Proves, across three domains:
  (a) the reported intermediate, taken from the function's REAL return, matches the reference;
  (b) a report that fabricates/substitutes the intermediate (because the function returned only the
      final result) produces a WRONG value that a definition-style check catches;
  (c) the exact named quantity is reported from its real source;
plus the loop: a solution exposing only the final result FAILS the held-out intermediate check and is
corrected to expose it; and the strengthened prompts require exposing intermediates and forbid
fabrication.

Deterministic: no network, no Docker, no real LLM.
"""
import types

import numpy as np
import pytest

from backend.agent import loop


# ======================================================================================
# Domain 1 — signal envelope (the reported example): report the intermediate envelope's peak.
# ======================================================================================
def _true_envelope(n):
    t = np.linspace(0.0, 1.0, n)
    return 1.0 + 0.5 * np.sin(2 * np.pi * t)                 # the intermediate envelope (peak 1.5)


def _final_signal(n):
    t = np.linspace(0.0, 1.0, n)
    return _true_envelope(n) * np.cos(2 * np.pi * 10 * t)     # carrier * envelope -> final signal


def report_envelope_peak_real(n):
    return float(np.max(_true_envelope(n)))                  # from the REAL exposed envelope


def report_envelope_peak_fabricated(n):
    sig = _final_signal(n)
    fake_env = sig * np.linspace(0.0, 1.0, n)                # invented stand-in (signal * a new ramp)
    return float(np.max(fake_env))


def check_envelope_peak(report_fn):
    n = 256
    expected = float(np.max(_true_envelope(n)))             # independent: the real envelope's peak
    assert abs(report_fn(n) - expected) < 1e-9, "reported peak must be the real envelope's peak"


def test_demo_envelope_real_passes_fabricated_fails():
    check_envelope_peak(report_envelope_peak_real)
    with pytest.raises(AssertionError):
        check_envelope_peak(report_envelope_peak_fabricated)


# ======================================================================================
# Domain 2 — moving-average -> slope: report the intermediate moving average's LAST value.
# ======================================================================================
def _moving_average(xs, w=3):
    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]


def report_ma_last_real(xs):
    return _moving_average(xs)[-1]                            # the real intermediate's last value


def report_ma_last_fabricated(xs):
    return float(xs[-1])                                      # substitute: the raw last input (wrong)


def check_ma_last(report_fn):
    xs = [1.0, 2.0, 9.0, 4.0, 5.0, 6.0]
    expected = _moving_average(xs)[-1]                       # (4+5+6)/3 = 5.0
    assert abs(report_fn(xs) - expected) < 1e-9


def test_demo_moving_average_real_passes_fabricated_fails():
    check_ma_last(report_ma_last_real)
    with pytest.raises(AssertionError):
        check_ma_last(report_ma_last_fabricated)             # xs[-1]=6.0 != 5.0


# ======================================================================================
# Domain 3 — normalize -> transform -> reduce: report the intermediate normalized vector's MAX.
# ======================================================================================
def _normalize(v):
    s = sum(x * x for x in v) ** 0.5
    return [x / s for x in v]


def report_norm_max_real(v):
    return max(_normalize(v))                                # the real normalized vector's max


def report_norm_max_fabricated(v):
    return float(max(v))                                     # substitute: the raw max (wrong)


def check_norm_max(report_fn):
    v = [3.0, 4.0]
    expected = max(_normalize(v))                            # 4/5 = 0.8
    assert abs(report_fn(v) - expected) < 1e-9


def test_demo_normalized_real_passes_fabricated_fails():
    check_norm_max(report_norm_max_real)
    with pytest.raises(AssertionError):
        check_norm_max(report_norm_max_fabricated)           # max(v)=4.0 != 0.8


# ======================================================================================
# Loop integration: a solution that exposes only the final result FAILS the held-out intermediate
# check; exposing the real intermediate passes.
# ======================================================================================
def _res(stdout):
    return types.SimpleNamespace(ok=True, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


def _runner(code, **k):
    """The held-out definition check (run with the seeded footer) verifies the intermediate; it passes
    ONLY for the solution that EXPOSES it. Visible tests pass for any candidate."""
    if "held-out runner" in code:
        ok = "EXPOSED" in code
        return _res(f"TEST test_definition_envelope_peak {'PASS' if ok else 'FAIL'}\n"
                    f"TESTS_PASSED {1 if ok else 0}/1\n")
    return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")


# Returns ONLY the final signal; the report re-derives a fabricated envelope peak.
_FABRICATED = ("def signal(n):\n    return [float(i) for i in range(n)]\n"
               "def report_peak(n):\n    s = signal(n)\n"
               "    return max(x * 0.1 for x in s)  # FABRICATED envelope peak (no real envelope)\n")

# EXPOSES the intermediate envelope; the report reads the real one.
_EXPOSED = ("def envelope(n):\n    return [1.0 + 0.5 * (i / n) for i in range(n)]\n"
            "def signal(n):\n    e = envelope(n)\n    return [e[i] * 1.0 for i in range(n)]\n"
            "def report_peak(n):\n    return max(envelope(n))  # EXPOSED: the real envelope peak\n")


class _IProvider:
    is_available = True
    name, model = "openai", "test"

    def __init__(self, fabricated, exposed=""):
        self.fabricated, self.exposed = fabricated, exposed

    def stream_chat(self, messages, system="", **k):
        user = messages[-1]["content"] if messages else ""
        if system == loop._REQ_SYSTEM:
            return ["- envelope(n): return the intermediate envelope\n"
                    "- signal(n): return the final signal\n"
                    "- report the envelope's peak and the signal's last value"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert signal(8) is not None\n"]
        if system == loop._DEFINITION_SYSTEM:
            return ["def test_definition_envelope_peak():\n    assert True\n"]
        if system == loop._GEN_SYSTEM:
            return [self.exposed if ("unseen" in user.lower() and self.exposed) else self.fabricated]
        return [""]


def _i_env(monkeypatch):
    for k, v in {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
                 "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
                 "AGENT_DEFINITION_GATE": "true", "AGENT_DELIVERY_GATES": "false",
                 "AGENT_ANTICHEAT_SCAN": "false", "AGENT_ROOT_CAUSE_DIAGNOSIS": "false",
                 "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false",
                 "AGENT_MAX_ATTEMPTS": "3", "AGENT_STALL_LIMIT": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type", lambda t: "deterministic")
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", _runner)


_TASK = "compute the signal and report the intermediate envelope's peak"


def test_final_only_solution_fails_intermediate_check(monkeypatch):
    _i_env(monkeypatch)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _IProvider(_FABRICATED))   # never exposes
    res = loop.run_agent(_TASK, use_search=False)
    assert res.verification != "verified"          # the fabricated intermediate is caught


def test_fabricated_intermediate_is_corrected_by_exposing_it(monkeypatch):
    _i_env(monkeypatch)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _IProvider(_FABRICATED, exposed=_EXPOSED))
    res = loop.run_agent(_TASK, use_search=False)
    assert res.verification == "verified"           # corrected by exposing the real intermediate
    assert "EXPOSED" in res.best_code and "envelope(" in res.best_code


# ======================================================================================
# Prompt wiring: expose intermediates + never fabricate.
# ======================================================================================
def test_gen_prompt_requires_exposing_intermediates_and_forbids_fabrication():
    s = loop._GEN_SYSTEM.lower()
    assert "expose every reported quantity" in s
    assert "intermediate" in s and "fabricat" in s
    assert "different-but-related quantity" in s


def test_definition_and_invariant_prompts_cover_intermediate_values():
    d = loop._DEFINITION_SYSTEM.lower()
    assert "intermediate" in d and "expose" in d and "must fail" in d
    inv = loop._INVARIANTS_SYSTEM.lower()
    assert "intermediate" in inv and "expose" in inv


def test_requirements_deliverables_driver_cover_intermediates():
    assert "intermediate" in loop._REQ_SYSTEM.lower() and "expose" in loop._REQ_SYSTEM.lower()
    assert "intermediate" in loop._DELIVERABLES_SYSTEM.lower()
    dr = loop._DRIVER_SYSTEM.lower()
    assert "real function return" in dr and "fabricat" in dr
