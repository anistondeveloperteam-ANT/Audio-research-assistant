"""Completeness / execution gate: a task that asks to PRINT results must produce those values in the
SOLUTION's OWN real stdout. Defining a function that *could* produce a value is not enough — the gate
runs the solution itself (its `if __name__ == "__main__":` block) and checks the captured stdout.

Proves:
  (a) a solution that DEFINES functions but never CALLS them (empty stdout) FAILS — never "verified";
  (b) a requested value ABSENT from the real stdout fails completeness;
  (c) a self-running solution that prints ALL requested values passes;
  (demo) an empty-output first attempt triggers a rewrite that adds the runnable __main__, then passes;
plus prompt wiring: the solver is told to add a runnable program for output tasks.

Deterministic: no network, no Docker, no real LLM. The mocked sandbox distinguishes a TEST-harness run
(contains the harness markers) from the GATE run (the solution alone) and returns the solution's own
__main__ output for the latter.
"""
import types

from backend.agent import loop


_BAD_NO_MAIN = (
    "def period():\n    return 6.2832\n"
    "def amplitude():\n    return 0.5\n"
    "# functions are defined but never called or printed\n")

_GOOD_FULL = (
    "def period():\n    return 6.2832\n"
    "def amplitude():\n    return 0.5\n"
    'if __name__ == "__main__":\n'
    '    print("period (s):", period())\n'
    '    print("amplitude:", amplitude())\n')

_PARTIAL_MAIN = (
    "def period():\n    return 6.2832\n"
    "def amplitude():\n    return 0.5\n"
    'if __name__ == "__main__":\n'
    '    print("period (s):", period())\n')

_TASK = "compute and PRINT the pendulum period and amplitude"


def _res(stdout):
    return types.SimpleNamespace(ok=True, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


def _runner(code, **k):
    """A TEST-harness run carries the harness markers; the GATE runs the SOLUTION ALONE as a script,
    so we return whatever its __main__ would print (keyed on the print statements present)."""
    if "ModuleType('_sol')" in code or "TESTS_PASSED" in code or "held-out runner" in code:
        return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")          # tests pass for any candidate
    if 'print("amplitude:"' in code:
        return _res("period (s): 6.2832\namplitude: 0.5\n")              # full __main__
    if 'print("period (s):"' in code:
        return _res("period (s): 6.2832\n")                             # partial __main__
    return _res("")                                                      # no call site -> empty stdout


class _CProvider:
    is_available = True
    name, model = "openai", "test"

    def __init__(self, bad, good=""):
        self.bad, self.good = bad, good

    def stream_chat(self, messages, system="", **k):
        user = messages[-1]["content"] if messages else ""
        if system == loop._REQ_SYSTEM:
            return ["- period(): the pendulum period\n- amplitude(): the amplitude"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert period() == 6.2832\n"]
        if system == loop._DELIVERABLES_SYSTEM:
            return ["period\namplitude"]
        if system == loop._GEN_SYSTEM:
            return [self.good if ("delivery gate" in user.lower() and self.good) else self.bad]
        return [""]


def _gate_env(monkeypatch):
    for k, v in {"AGENT_DELIVERY_GATES": "true", "AGENT_REFERENCE_TESTS": "false",
                 "AGENT_TEST_VALIDATION": "false", "AGENT_NONUNIQUE_VALIDATION": "false",
                 "AGENT_HIDDEN_TESTS": "false", "AGENT_DEFINITION_GATE": "false",
                 "AGENT_ANTICHEAT_SCAN": "false", "AGENT_ROOT_CAUSE_DIAGNOSIS": "false",
                 "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false",
                 "AGENT_MAX_ATTEMPTS": "3", "AGENT_STALL_LIMIT": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type", lambda t: "deterministic")
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", _runner)


# ---- (a) defines functions but never calls them -> empty stdout -> FAIL ----
def test_defines_functions_but_never_calls_them_fails(monkeypatch):
    _gate_env(monkeypatch)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _CProvider(_BAD_NO_MAIN))
    events = []
    res = loop.run_agent(_TASK, use_search=False, on_event=events.append)

    assert res.verification != "verified"                     # passing tests is NOT enough
    gate = [e for e in events if e.get("type") == "gate_fail"]
    assert gate and "output" in gate[0]["reason"].lower()     # empty real stdout -> execution gate fail
    assert "stdout" in gate[0]["reason"].lower()


# ---- (b) a requested value absent from real stdout -> completeness FAIL ----
def test_requested_value_absent_from_stdout_fails(monkeypatch):
    _gate_env(monkeypatch)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _CProvider(_PARTIAL_MAIN))
    events = []
    res = loop.run_agent(_TASK, use_search=False, on_event=events.append)

    assert res.verification != "verified"
    gate = [e for e in events if e.get("type") == "gate_fail"]
    assert gate and "completeness" in gate[0]["reason"].lower()
    assert "amplitude" in gate[0]["reason"].lower()           # the missing deliverable is named


# ---- (c) a self-running solution that prints all requested values -> PASS ----
def test_self_running_solution_that_prints_all_values_passes(monkeypatch):
    _gate_env(monkeypatch)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _CProvider(_GOOD_FULL))
    res = loop.run_agent(_TASK, use_search=False)

    assert res.verification == "verified"
    assert "6.2832" in res.best_output and "0.5" in res.best_output
    assert "period" in res.best_output.lower() and "amplitude" in res.best_output.lower()


# ---- demo: empty output is rejected, the rewrite adds the runnable __main__, then it passes ----
def test_empty_output_triggers_rewrite_then_passes(monkeypatch):
    _gate_env(monkeypatch)
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _CProvider(_BAD_NO_MAIN, good=_GOOD_FULL))
    res = loop.run_agent(_TASK, use_search=False)

    assert res.verification == "verified"                     # round 1 empty -> fail -> round 2 prints
    assert "amplitude" in res.best_output.lower() and "6.2832" in res.best_output


# ---- prompt wiring: the solver is required to add a runnable program for output tasks ----
def test_gen_prompt_requires_runnable_main_for_output_tasks():
    s = loop._GEN_SYSTEM.lower()
    assert "__main__" in s and "runnable program" in s and "incomplete" in s


def test_generate_solution_injects_runnable_requirement_only_when_output_wanted():
    seen = {}

    class P:
        is_available, name, model = True, "openai", "t"

        def stream_chat(self, messages, system="", **k):
            seen["user"] = messages[-1]["content"]
            return ["def f():\n    pass"]

    loop._generate_solution(P(), "print the period", "reqs", "tests", "", "", "", wants_output=True)
    assert "__main__" in seen["user"] and "RUNNABLE PROGRAM" in seen["user"]

    loop._generate_solution(P(), "implement f(x)", "reqs", "tests", "", "", "", wants_output=False)
    assert "RUNNABLE PROGRAM" not in seen["user"]
