"""Return-contract consistency: generated code must consume a function's return value by its
ACTUAL structure (type / shape / length / fields / order / units), and the agent's self-checks
must read the REAL returned object — not a wrong-shaped stand-in.

These tests prove, across three different domains, that:
  (a) a consumer that reads a function's return with the WRONG shape/type is CAUGHT by a check
      that asserts the output's true structure, while a correct consumer passes;
  (b) a self-check that validates a wrong-shaped stand-in MISSES the bug, whereas a check that
      reads the actual returned object catches it;
  (c) the agent's generation/verification prompts now instruct exactly this;
  (d) end to end, run_agent rejects a wrong-shape solution that a held-out structure check fails,
      and verifies the correct one.

Everything is deterministic real Python (no network, no Docker, no real LLM).
"""
import types

import numpy as np
import pytest

from backend.agent import loop


# ======================================================================================
# Domain 1 — a function returns a (mean, std) SUMMARY TUPLE; the task wants the two largest
# data values. The classic bug: slice the stats tuple as if it were the full data array.
# ======================================================================================
def summarize(data):
    """Return (mean, std) — a 2-tuple of summary statistics, NOT the data."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return (mean, std)


def top_two_right(data):
    """Correct: the two largest values come from the DATA."""
    return sorted(data)[-2:]


def top_two_wrong(data):
    """Buggy: slices the (mean, std) summary tuple as if it were the data array."""
    s = summarize(data)
    return sorted(s)[-2:]


def check_top_two(report_fn):
    """A structure+value check the agent now writes: read the ACTUAL output object and assert
    it is a length-2 sequence of real data values equal to the two largest."""
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    out = report_fn(data)
    assert len(out) == 2, "expected exactly the two largest values"
    assert all(v in data for v in out), "each reported value must be an actual data element"
    assert sorted(out) == sorted(data)[-2:], "must be the two largest data values"


def test_stats_tuple_misconsumed_is_caught():
    # ARRANGE / ACT / ASSERT: the wrong-shape consumer fails the structure check.
    with pytest.raises(AssertionError):
        check_top_two(top_two_wrong)


def test_stats_tuple_correct_consumer_passes():
    check_top_two(top_two_right)  # no error -> correct consumer accepted


# ======================================================================================
# Domain 2 — a function returns a CONFIG DICT; a consumer wrongly unpacks it as a tuple,
# silently binding the dict KEYS instead of the values.
# ======================================================================================
def parse_config():
    """Return a dict {host, port} — arity/fields matter."""
    return {"host": "localhost", "port": 8600}


def get_port_right():
    return parse_config()["port"]


def get_port_wrong():
    host, port = parse_config()  # BUG: unpacks dict KEYS -> port == "port" (a str)
    return port


def check_port(report_fn):
    """Structure check: the port must be the INTEGER value from the dict, not a key string."""
    port = report_fn()
    assert isinstance(port, int), "port must be an int (wrong-arity unpack yields the key string)"
    assert port == 8600


def test_dict_unpack_mismatch_is_caught():
    with pytest.raises(AssertionError):
        check_port(get_port_wrong)


def test_dict_correct_consumer_passes():
    check_port(get_port_right)


# ======================================================================================
# Domain 3 — a function returns a 2-D ARRAY; a consumer reduces along the WRONG AXIS,
# producing the wrong shape (one value per row instead of one per column).
# ======================================================================================
def column_means_right(matrix):
    return np.asarray(matrix, dtype=float).mean(axis=0)  # one mean per column


def column_means_wrong(matrix):
    return np.asarray(matrix, dtype=float).mean(axis=1)  # BUG: per-row mean (wrong axis)


def check_column_means(report_fn):
    """Structure+value check: one mean per COLUMN -> length == #columns."""
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 rows x 3 cols
    out = np.asarray(report_fn(matrix))
    assert out.shape == (3,), "expected one mean per column (length = number of columns)"
    assert np.allclose(out, [2.5, 3.5, 4.5])


def test_array_wrong_axis_is_caught():
    with pytest.raises(AssertionError):
        check_column_means(column_means_wrong)


def test_array_correct_consumer_passes():
    check_column_means(column_means_right)


# ======================================================================================
# (b) The self-check must read the ACTUAL returned object, not a wrong-shaped stand-in.
# ======================================================================================
def test_self_check_on_wrong_object_misses_bug_real_object_catches_it():
    data = [3, 1, 4, 1, 5, 9, 2, 6]

    # A self-check that validates a wrong-shaped STAND-IN (the stats tuple) never inspects the
    # consumer's real output, so it cannot catch the wrong-shape consumer.
    def check_wrong_object():
        s = summarize(data)
        assert len(s) == 2  # passes no matter what the consumer does

    check_wrong_object()  # no error -> the bug slips through a stand-in check

    # A self-check that reads the consumer's ACTUAL output catches the wrong-shape consumer.
    check_top_two(top_two_right)
    with pytest.raises(AssertionError):
        check_top_two(top_two_wrong)


# ======================================================================================
# (c) The agent's prompts now instruct return-contract consistency.
# ======================================================================================
def test_solver_prompt_requires_return_contract_consumption():
    s = loop._GEN_SYSTEM.lower()
    assert "consume return values by their true contract" in s
    assert "wrong axis" in s and "stats tuple" in s
    assert "fail loudly" in s


def test_requirements_prompt_states_return_contract():
    s = loop._REQ_SYSTEM.lower()
    assert "return contract" in s
    assert "arity" in s and "axis" in s


def test_definition_check_prompt_requires_structure_assertion():
    s = loop._DEFINITION_SYSTEM.lower()
    assert "structure" in s and "arity" in s
    assert "wrong-shaped stand-in" in s


def test_invariant_prompt_requires_structure_assertion():
    s = loop._INVARIANTS_SYSTEM.lower()
    assert "structure" in s and "arity" in s
    assert "real returned object" in s


def test_test_writer_prompts_require_true_contract_consumption():
    for prompt in (loop._TESTS_SYSTEM, loop._HIDDEN_SYSTEM):
        s = prompt.lower()
        assert "true" in s and "contract" in s or "true return structure" in s
        assert "shape" in s


# ======================================================================================
# (d) End to end: run_agent rejects a wrong-shape solution a held-out structure check fails,
#     and verifies the correct one. (Mirrors the non-unique-validation integration pattern.)
# ======================================================================================
def _res(stdout):
    return types.SimpleNamespace(ok=True, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


class _RCProvider:
    """Routes by system prompt; supplies a held-out definition (structure) check and a solution
    carrying a RIGHT_SHAPE / WRONG_SHAPE marker the sandbox mock keys off."""
    is_available = True
    name, model = "openai", "test"

    def __init__(self, requirements, tests, definitions, solution):
        self.requirements, self.tests = requirements, tests
        self.definitions, self.solution = definitions, solution

    def stream_chat(self, messages, system="", **k):
        if system == loop._REQ_SYSTEM:
            return [self.requirements]
        if system == loop._TESTS_SYSTEM:
            return [self.tests]
        if system == loop._DEFINITION_SYSTEM:
            return [self.definitions]
        if system == loop._GEN_SYSTEM:
            return [self.solution]
        return [""]


def _rc_env(monkeypatch):
    for k, v in {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
                 "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
                 "AGENT_DEFINITION_GATE": "true", "AGENT_DELIVERY_GATES": "false",
                 "AGENT_ANTICHEAT_SCAN": "false", "AGENT_PARALLEL_N": "1",
                 "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "deterministic")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


def _make_provider(solution):
    return _RCProvider(
        requirements="- top_two(data): return a list of the two largest values in data",
        tests="def test_basic():\n    assert top_two([1, 2, 3]) == [2, 3]\n",
        definitions="def test_definition_top_two():\n    assert True\n",
        solution=solution)


def test_run_agent_rejects_wrong_shape_consumer(monkeypatch):
    _rc_env(monkeypatch)

    def run(code, **k):
        held = "held-out runner (seeded)" in code
        bad = "WRONG_SHAPE" in code
        if held:  # the held-out structure check fails the wrong-shape solution, passes the right one
            ex, n = ("FAIL", 0) if bad else ("PASS", 1)
            return _res(f"TEST test_definition_top_two {ex}\nTESTS_PASSED {n}/1\n")
        return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")  # both pass the visible test

    monkeypatch.setattr(loop, "run_python_auto", run)

    # WRONG-SHAPE solution: consumes a summary tuple as if it were the data array.
    bad_sol = ("def top_two(data):\n"
               "    s = (len(data), sum(data), max(data))  # WRONG_SHAPE: a stats tuple\n"
               "    return sorted(s)[-2:]\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _make_provider(bad_sol))
    res_bad = loop.run_agent("implement top_two(data) returning the two largest values",
                             use_search=False, max_iters=2)
    assert res_bad.verification != "verified"  # caught by the held-out structure check

    # CORRECT solution: reads the data array per its true contract.
    good_sol = ("def top_two(data):\n"
                "    return sorted(data)[-2:]  # RIGHT_SHAPE\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _make_provider(good_sol))
    res_good = loop.run_agent("implement top_two(data) returning the two largest values",
                              use_search=False, max_iters=2)
    assert res_good.verification == "verified"
