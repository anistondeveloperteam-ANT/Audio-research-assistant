"""Tests for the research agent loop — fully mocked (no Docker, no LLM, no network)."""
import json

import pytest

from backend.agent import loop
from backend.agent import hooks
from backend.agent.code_runner import RunResult
from backend.agent.hooks import HookDecision
from backend.agent.memory import TwoTierMemory


@pytest.fixture(autouse=True)
def _disable_auto_review(monkeypatch):
    monkeypatch.setenv("AUTO_REVIEW", "false")


# ---- pre-execution lifecycle hook (kimi-code idea) -------------------
def test_hook_allows_by_default_and_audits(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(hooks, "AUDIT_LOG", str(log))
    monkeypatch.setattr(hooks, "BLOCK_PATTERNS", "")
    monkeypatch.setattr(hooks, "PRERUN_HOOK", "")
    d = hooks.pre_run("print(1)", task="demo")
    assert d.allowed is True
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["allowed"] is True and rec["code_len"] == len("print(1)")


def test_hook_blocks_on_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "AUDIT_LOG", str(tmp_path / "a.jsonl"))
    monkeypatch.setattr(hooks, "BLOCK_PATTERNS", r"os\.system,shutil\.rmtree")
    monkeypatch.setattr(hooks, "PRERUN_HOOK", "")
    d = hooks.pre_run("import shutil; shutil.rmtree('/x')", task="x")
    assert not d.allowed and "blocked pattern" in d.reason


def test_hook_blocks_via_prerun_command(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "AUDIT_LOG", str(tmp_path / "a.jsonl"))
    monkeypatch.setattr(hooks, "BLOCK_PATTERNS", "")
    monkeypatch.setattr(hooks, "PRERUN_HOOK", "mygate")
    monkeypatch.setattr(hooks.shutil, "which", lambda name: "/usr/bin/mygate")

    class _R:
        returncode = 1
        stdout = "rejected by gate"
        stderr = ""
    monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **k: _R())
    d = hooks.pre_run("print(1)", task="x")
    assert not d.allowed and "rejected" in d.reason


# ---- two-tier memory -------------------------------------------------
def test_memory_brief_is_clipped():
    mem = TwoTierMemory(brief="x" * 5000, brief_max=100)
    assert len(mem.brief) <= 130 and mem.brief.endswith("…[clipped]")


def test_memory_log_stays_bounded():
    mem = TwoTierMemory(brief="goal", log_max=200, keep_last=5)
    for i in range(50):
        mem.append(f"attempt {i}: some moderately long progress note about what happened")
    assert len(mem.log_entries) <= 5          # count cap
    assert mem._log_chars() <= 200            # char cap
    ctx = mem.context()
    assert ctx.startswith("goal")
    assert "attempt 49" in ctx                # newest kept
    assert "attempt 0" not in ctx             # oldest dropped


def test_build_brief_variants():
    assert loop._build_brief("do X", "", "") == "# Goal\ndo X"
    assert loop._build_brief("do X", "# Goal\ncustom", "") == "# Goal\ncustom"
    assert "Relevant approaches" in loop._build_brief("do X", "", "ctx text")


# ---- pure helpers ----------------------------------------------------
def test_extract_code_handles_fences_and_raw():
    fenced = "Here:\n```python\nprint(1)\n```\n"
    assert loop._extract_code(fenced) == "print(1)"
    assert loop._extract_code("print(2)") == "print(2)"


def test_parse_json_clean_embedded_and_garbage():
    assert loop._parse_json('{"a": 1}') == {"a": 1}
    assert loop._parse_json('noise {"ok": true} tail')["ok"] is True
    assert loop._parse_json("not json") == {}


def test_score_running_beats_non_running():
    ok = loop.Attempt(1, "", RunResult(True, 0, "", "", 0.1), {"score": 10})
    bad = loop.Attempt(1, "", RunResult(False, 1, "", "boom", 0.1), {"score": 99})
    assert loop._score(ok) > loop._score(bad)


# ---- fakes -----------------------------------------------------------
class _FakeProvider:
    """Returns queued responses, one per stream_chat call. The test-first flow calls in this
    order: requirements, tests, then one response per solution candidate."""
    name = "openai"
    model = "test"
    is_available = True

    def __init__(self, responses):
        self._responses = list(responses)

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0):
        return [self._responses.pop(0)]


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


# A tiny suite the agent "writes"; the real pass tally comes from the mocked sandbox stdout.
_TESTS_SRC = (
    "def test_sorted():\n    assert bubble_sort([3, 1, 2]) == [1, 2, 3]\n"
    "def test_empty():\n    assert bubble_sort([]) == []\n"
)


# ---- the test-first loop --------------------------------------------
def test_agent_succeeds_first_try(monkeypatch):
    provider = _FakeProvider([
        "- implement bubble_sort(a) returning a sorted list",       # (a) requirements
        _fence(_TESTS_SRC),                                         # (b) generated tests
        _fence("def bubble_sort(a):\n    return sorted(a)"),        # (c) solution: passes all
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: RunResult(True, 0, "TESTS_PASSED 2/2\n", "", 0.2))

    res = loop.run_agent("Implement bubble sort", use_search=False)
    assert res.success is True
    assert res.tests_passed == 2 and res.tests_total == 2
    assert "bubble" in res.answer.lower()
    assert len(res.attempts) == 1                       # accepted on the first round/candidate


def test_agent_refines_after_a_failure(monkeypatch):
    provider = _FakeProvider([
        "- implement bubble_sort(a)",                            # requirements
        _fence(_TESTS_SRC),                                     # tests
        _fence("def bubble_sort(a):\n    return a"),            # round 1, candidate 1 (fails)
        _fence("def bubble_sort(a):\n    return list(a)"),      # round 1, candidate 2 (fails)
        _fence("def bubble_sort(a):\n    return sorted(a)"),    # round 2, candidate 1 (passes)
    ])
    results = iter([
        RunResult(True, 0, "TESTS_PASSED 1/2\n", "", 0.1),     # r1 c1
        RunResult(True, 0, "TESTS_PASSED 1/2\n", "", 0.1),     # r1 c2
        RunResult(True, 0, "TESTS_PASSED 2/2\n", "", 0.1),     # r2 c1
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: next(results))

    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=3)
    assert res.success is True
    assert res.tests_passed == 2 and res.tests_total == 2
    assert len(res.attempts) == 2                       # one accepted attempt per round
    assert "sorted(a)" in res.best_code


def test_only_full_pass_is_accepted(monkeypatch):
    """A candidate that merely RAN (some tests fail) is never 'success' — honest partial label."""
    provider = _FakeProvider(
        ["- implement bubble_sort(a)", _fence(_TESTS_SRC)]
        + [_fence("def bubble_sort(a):\n    return a")] * 8     # partial every round
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: RunResult(True, 0, "TESTS_PASSED 1/2\n", "", 0.1))

    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=2)
    assert res.success is False
    assert res.tests_passed == 1 and res.tests_total == 2
    assert "Partially verified — 1/2" in loop.result_to_markdown(res)


def test_loop_blocks_without_running(monkeypatch):
    provider = _FakeProvider([
        "- requirements",
        _fence("def test_x():\n    assert True"),
        _fence("print(1)"),
        _fence("print(2)"),
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    ran = []
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: ran.append(1) or RunResult(True, 0, "", "", 0.1))
    monkeypatch.setattr(loop, "pre_run", lambda code, task="": HookDecision(False, "matched blocked pattern"))
    events = []
    res = loop.run_agent("x", use_search=False, max_iters=1, on_event=events.append)
    assert ran == []                                    # never executed in the sandbox
    assert any(e.get("type") == "blocked" for e in events)
    assert res.success is False


def test_agent_stops_clean_when_docker_missing(monkeypatch):
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _FakeProvider([]))
    monkeypatch.setattr(loop, "docker_available", lambda: False)
    events = []
    res = loop.run_agent("anything", use_search=False, on_event=events.append)
    assert res.success is False
    assert any(e.get("type") == "error" for e in events)


# ---- test-gate, best-of-2, relevance gate, escalation ---------------
def test_verdict_from_tests_gate():
    ok = loop._verdict_from_tests(8, 8, True, RunResult(True, 0, "", "", 0.1))
    assert ok["done"] is True and ok["success"] is True and ok["score"] == 100
    partial = loop._verdict_from_tests(5, 8, True, RunResult(True, 0, "", "", 0.1))
    assert partial["done"] is False and partial["success"] is False    # "ran" is not success
    offtopic = loop._verdict_from_tests(8, 8, False, RunResult(True, 0, "", "", 0.1))
    assert offtopic["done"] is False                                   # all pass but off-topic
    none_run = loop._verdict_from_tests(0, 0, True, RunResult(False, 1, "", "boom", 0.1))
    assert none_run["done"] is False and none_run["score"] == 0


def test_is_relevant_code_matches_requested_algorithm():
    assert loop._is_relevant_code("Give me RTF-MVDR code", "def mvdr(cov):\n    pass", "") is True
    assert loop._is_relevant_code("Give me RTF-MVDR code",
                                  "def foo():\n    return 1", "def test_foo():\n    pass") is False
    assert loop._is_relevant_code("write some code", "print(1)", "") is True   # no specific term


def test_best_of_two_keeps_higher_pass_rate(monkeypatch):
    provider = _FakeProvider([
        "- implement widget()",
        _fence("def test_w():\n    assert widget() == 1"),
        _fence("def widget():\n    return 0  # weak"),      # candidate 1: partial
        _fence("def widget():\n    return 1  # strong"),    # candidate 2: full pass
    ])
    results = iter([
        RunResult(True, 0, "TESTS_PASSED 1/3\n", "", 0.1),    # c1
        RunResult(True, 0, "TESTS_PASSED 3/3\n", "", 0.1),    # c2
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: next(results))

    res = loop.run_agent("Implement widget", use_search=False, max_iters=1)
    assert res.tests_passed == 3 and res.tests_total == 3
    assert "strong" in res.best_code
    assert res.success is True


def test_relevance_gate_blocks_offtopic_code(monkeypatch):
    """Code that never mentions the requested algorithm is not 'verified' even if its tests pass."""
    provider = _FakeProvider(
        ["- implement mvdr_beamformer(cov, steering)", _fence("def test_x():\n    assert True")]
        + [_fence("def unrelated():\n    return 42")] * 4
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: RunResult(True, 0, "TESTS_PASSED 1/1\n", "", 0.1))

    res = loop.run_agent("Give me RTF-MVDR code", use_search=False, max_iters=1)
    assert res.success is False                          # off-topic -> relevance gate fails it


def test_escalation_uses_strong_model_after_two_failures(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_STRONG", "strong-x")
    models = []
    provider = _FakeProvider(
        ["- implement add(a, b)", _fence("def test_a():\n    assert add(1, 1) == 2")]
        + [_fence("def add(a, b):\n    return 0")] * 6     # always wrong -> every round fails
    )

    def fake_get_provider(model=None, *a, **k):
        models.append(model)
        return provider

    monkeypatch.setattr(loop, "get_provider", fake_get_provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: RunResult(True, 0, "TESTS_PASSED 0/1\n", "", 0.1))

    res = loop.run_agent("Implement add", use_search=False, max_iters=3)
    assert "strong-x" in models                          # escalated after two failed rounds
    assert res.success is False
