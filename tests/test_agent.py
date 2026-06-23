"""Tests for the research agent loop — fully mocked (no Docker, no LLM, no network)."""
import json
import threading

import pytest

from backend.agent import loop
from backend.agent import hooks
from backend.agent.code_runner import RunResult
from backend.agent.hooks import HookDecision
from backend.agent.memory import TwoTierMemory


@pytest.fixture(autouse=True)
def _agent_env(monkeypatch):
    # Deterministic agent env regardless of the dev's shell: review off, hardening on, no escalation
    # unless a test opts in. AGENT_PARALLEL_N>1 exercises the real best-of-N parallel path.
    monkeypatch.setenv("AUTO_REVIEW", "false")
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "true")
    monkeypatch.setenv("AGENT_ANTICHEAT_SCAN", "true")
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "3")
    monkeypatch.setenv("AGENT_PARALLEL_N", "4")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SANDBOXES", "4")
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)


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


def test_parallel_n_reads_and_clamps_env(monkeypatch):
    monkeypatch.delenv("AGENT_PARALLEL_N", raising=False)
    assert loop.parallel_n() == 4                 # default
    monkeypatch.setenv("AGENT_PARALLEL_N", "2")
    assert loop.parallel_n() == 2
    monkeypatch.setenv("AGENT_PARALLEL_N", "99")
    assert loop.parallel_n() == 8                 # clamped to max 8
    monkeypatch.setenv("AGENT_PARALLEL_N", "0")
    assert loop.parallel_n() == 1                 # clamped to min 1
    monkeypatch.setenv("AGENT_PARALLEL_N", "nope")
    assert loop.parallel_n() == 4                 # bad value -> default


# ---- fakes -----------------------------------------------------------
class _FakeProvider:
    """Thread-safe, content-routed mock. Responses are keyed by the agent STEP (inferred from the
    system prompt) — NOT call order — so parallel best-of-N candidate generation and lazily-built
    held-out generation are deterministic. Solution responses route by attempt: `first` for a round
    with no failure feedback, `refined` once the round carries 'FAILED last time' feedback. Each
    queue hands out FIFO and REPEATS its last entry when exhausted, so N parallel candidates can
    draw from a 1- or 2-entry pool."""
    name = "openai"
    model = "test"
    is_available = True

    def __init__(self, *, requirements="- requirements", tests="", first=None, refined=None,
                 hidden="", invariants="", reference="", driver=""):
        self._req = requirements
        self._tests = tests
        self._first = list(first or [])
        self._refined = list(refined or [])
        self._hidden = hidden
        self._invariants = invariants
        self._reference = reference
        self._driver = driver
        self.calls = []          # (system, user) for every stream_chat call
        self._lock = threading.Lock()

    @staticmethod
    def _take(lst):
        if not lst:
            return ""
        return lst.pop(0) if len(lst) > 1 else lst[0]

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0, yield_reasoning=False):
        user = messages[-1]["content"] if messages else ""
        with self._lock:
            self.calls.append((system, user))
            if system == loop._REQ_SYSTEM:
                return [self._req]
            if system == loop._REFERENCE_SYSTEM:
                return [self._reference]
            if system == loop._DRIVER_SYSTEM:
                return [self._driver]
            if system == loop._TESTS_SYSTEM:
                return [self._tests]
            if system == loop._HIDDEN_SYSTEM:
                return [self._hidden]
            if system == loop._INVARIANTS_SYSTEM:
                return [self._invariants]
            if system == loop._GEN_SYSTEM:
                use_refined = ("FAILED last time" in user) and bool(self._refined)
                return [self._take(self._refined if use_refined else self._first)]
            return [""]


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


def _ok(passed: int, total: int) -> RunResult:
    return RunResult(True, 0, f"TESTS_PASSED {passed}/{total}\n", "", 0.1)


def _runner(visible_fn, heldout_fn):
    """A run_python_auto mock that returns different results for VISIBLE vs HELD-OUT scripts
    (the held-out runner is seeded, so its script carries a distinctive marker). Thread-safe:
    it routes purely by code content, so parallel candidates don't interfere."""
    def run(code, **k):
        return heldout_fn(code) if "held-out runner (seeded)" in code else visible_fn(code)
    return run


# A tiny suite the agent "writes"; the real pass tally comes from the mocked sandbox stdout.
_TESTS_SRC = (
    "def test_sorted():\n    assert bubble_sort([3, 1, 2]) == [1, 2, 3]\n"
    "def test_empty():\n    assert bubble_sort([]) == []\n"
)
# Fake held-out suites (hidden tests + invariants) the model "writes" on first acceptance.
_HIDDEN_SRC = "def test_hidden_a():\n    assert True\n"
_INV_SRC = "def test_invariant_b():\n    assert True\n"
_BS_TESTS = "def test_call():\n    assert abs(black_scholes(100, 100, 1, 0.05, 0.2) - 10.4506) < 1e-2\n"


# ---- the test-first loop --------------------------------------------
def test_agent_succeeds_first_try(monkeypatch):
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a) returning a sorted list",
        tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return sorted(a)")],   # every candidate passes
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))   # visible + held-out

    res = loop.run_agent("Implement bubble sort", use_search=False)
    assert res.success is True and res.verification == "verified"
    assert res.tests_passed == 2 and res.tests_total == 2
    assert res.hidden_total >= 1                          # held-out hidden/invariant checks ran
    assert "bubble" in res.answer.lower()
    assert len(res.attempts) == 1


def test_agent_refines_after_a_failure(monkeypatch):
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)",
        tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return a")],            # round 1: fails
        refined=[_fence("def bubble_sort(a):\n    return sorted(a)")],  # round 2+: passes
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    # Visible passes only for the real (sorted) solution; held-out always passes.
    monkeypatch.setattr(loop, "run_python_auto",
                        _runner(lambda c: _ok(2, 2) if "sorted(a)" in c else _ok(1, 2),
                                lambda c: _ok(2, 2)))

    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=3)
    assert res.success is True and res.verification == "verified"
    assert res.tests_passed == 2 and res.tests_total == 2
    assert len(res.attempts) == 2                       # round 1 (fail) + round 2 (refined, pass)
    assert "sorted(a)" in res.best_code


def test_only_full_pass_is_accepted(monkeypatch):
    """A candidate that merely RAN (some tests fail) is never 'success' — honest partial label."""
    partial = _fence("def bubble_sort(a):\n    return a")
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)", tests=_fence(_TESTS_SRC),
        first=[partial], refined=[partial],            # partial every round
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(1, 2))

    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=2)
    assert res.success is False and res.verification == "partial"
    assert res.tests_passed == 1 and res.tests_total == 2
    assert "Partially verified — 1/2" in loop.result_to_markdown(res)


def test_loop_blocks_without_running(monkeypatch):
    provider = _FakeProvider(
        requirements="- requirements",
        tests=_fence("def test_x():\n    assert True"),
        first=[_fence("print(1)")],
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    ran = []
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: ran.append(1) or _ok(0, 1))
    monkeypatch.setattr(loop, "pre_run", lambda code, task="": HookDecision(False, "matched blocked pattern"))
    events = []
    res = loop.run_agent("x", use_search=False, max_iters=1, on_event=events.append)
    assert ran == []                                    # never executed in the sandbox
    assert any(e.get("type") == "blocked" for e in events)
    assert res.success is False


def test_agent_stops_clean_when_docker_missing(monkeypatch):
    """Execution is mandatory: a sandbox outage yields a clear error, never a prose/fake answer."""
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(loop, "docker_available", lambda: False)
    events = []
    res = loop.run_agent("simulate a damped pendulum", use_search=False, on_event=events.append)
    assert res.success is False and res.best_code == ""
    assert any(e.get("type") == "error" for e in events)
    md = loop.result_to_markdown(res)
    assert "Sandbox unavailable" in md                   # explicit, honest error
    assert "```python" not in md                          # never a fabricated code answer


def test_print_request_returns_real_captured_stdout(monkeypatch):
    """A 'print/show the result' request must surface the REAL stdout from RUNNING the SOLUTION
    ITSELF (its __main__ prints the values), not test-runner noise or a 'when executed' claim."""
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)", tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return sorted(a)\n"
                      "if __name__ == '__main__':\n    print('sorted:', bubble_sort([3, 1, 2]))")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)

    def fake_run(code, **k):
        if "ModuleType('_sol')" in code or "TESTS_PASSED" in code:   # a test-harness run
            return _ok(2, 2)
        return RunResult(True, 0, "sorted: [1, 2, 3]\n", "", 0.1)     # the SOLUTION's own __main__

    monkeypatch.setattr(loop, "run_python_auto", fake_run)

    res = loop.run_agent("Sort a list and print the sorted result", use_search=False)
    assert res.verification == "verified"
    assert res.best_output == "sorted: [1, 2, 3]"         # REAL values from running the solution
    assert "TESTS_PASSED" not in res.best_output          # not the test-runner noise
    assert "**Output:**" in loop.result_to_markdown(res)  # shown to the user


def test_non_output_request_has_no_stdout_noise(monkeypatch):
    """A task that doesn't ask to print/show a result carries no test-runner noise as 'output'."""
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)", tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return sorted(a)")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))

    res = loop.run_agent("Implement bubble sort", use_search=False)
    assert res.verification == "verified" and res.best_output == ""   # no demo intent -> no output


# ---- test-gate, best-of-N, relevance gate, escalation ---------------
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


def test_best_of_n_keeps_higher_pass_rate(monkeypatch):
    provider = _FakeProvider(
        requirements="- implement widget()",
        tests=_fence("def test_w():\n    assert widget() == 1"),
        first=[_fence("def widget(n):\n    return n  # weak"),       # candidate: partial
               _fence("def widget(n):\n    return n + 1  # strong")],  # candidate: full pass
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        _runner(lambda c: _ok(3, 3) if "strong" in c else _ok(1, 3),
                                lambda c: _ok(3, 3)))

    res = loop.run_agent("Implement widget", use_search=False, max_iters=1)
    assert res.tests_passed == 3 and res.tests_total == 3
    assert "strong" in res.best_code
    assert res.success is True and res.verification == "verified"


def test_relevance_gate_blocks_offtopic_code(monkeypatch):
    """Code that never mentions the requested algorithm is not 'verified' even if its tests pass."""
    off = _fence("def unrelated(z):\n    return z + 1")
    provider = _FakeProvider(
        requirements="- implement mvdr_beamformer(cov, steering)",
        tests=_fence("def test_x():\n    assert True"),
        first=[off], refined=[off],
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(1, 1))

    res = loop.run_agent("Give me RTF-MVDR code", use_search=False, max_iters=1)
    assert res.success is False                          # off-topic -> relevance gate fails it


def test_escalation_uses_strong_model_after_two_failures(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_STRONG", "strong-x")
    models = []
    wrong = _fence("def add(a, b):\n    return a - b")   # uses args (not gaming), wrong result
    provider = _FakeProvider(
        requirements="- implement add(a, b)",
        tests=_fence("def test_a():\n    assert add(1, 1) == 2"),
        first=[wrong], refined=[wrong],
    )

    def fake_get_provider(model=None, *a, **k):
        models.append(model)
        return provider

    monkeypatch.setattr(loop, "get_provider", fake_get_provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(0, 1))

    res = loop.run_agent("Implement add", use_search=False, max_iters=3)
    assert "strong-x" in models                          # escalated after two failed rounds
    assert res.success is False


# ---- anti-reward-hacking hardening -----------------------------------
def test_verified_runs_hidden_tests_and_invariants(monkeypatch):
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)", tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return sorted(a)")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))

    res = loop.run_agent("Implement bubble sort", use_search=False)
    assert res.verification == "verified" and res.hidden_total >= 1
    systems = [s for (s, _u) in provider.calls]
    assert loop._HIDDEN_SYSTEM in systems and loop._INVARIANTS_SYSTEM in systems  # both generated


def test_hidden_tests_reject_visible_only_pass(monkeypatch):
    """Passing the VISIBLE tests is not enough: held-out hidden tests on fresh inputs must pass too."""
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)", tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return sorted(a)")],    # passes visible
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto",
                        _runner(lambda c: _ok(2, 2), lambda c: _ok(0, 2)))   # held-out FAILS
    events = []
    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=1, on_event=events.append)
    assert res.success is False and res.verification == "partial"
    assert any(e.get("type") == "run_result" and e.get("verified") is False
               and e.get("passed") == e.get("total") and e.get("hidden_total", 0) > 0
               for e in events)                          # visible fully passed, held-out ran & failed


def test_heldout_error_degrades_to_visible_acceptance(monkeypatch):
    """If the held-out machinery itself errors (provider hiccup), a genuine visible-passing
    solution is accepted on the visible tests, not silently discarded."""
    provider = _FakeProvider(
        requirements="- implement bubble_sort(a)", tests=_fence(_TESTS_SRC),
        first=[_fence("def bubble_sort(a):\n    return sorted(a)")],
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))

    def boom(*a, **k):
        raise RuntimeError("held-out provider down")
    monkeypatch.setattr(loop, "_verify_heldout", boom)

    events = []
    res = loop.run_agent("Implement bubble sort", use_search=False, max_iters=1, on_event=events.append)
    assert res.verification == "verified"               # visible-passing solution preserved
    assert "sorted(a)" in res.best_code
    assert any(e.get("type") == "warning" and "Held-out" in (e.get("message") or "")
               for e in events)


def test_multi_seed_rejection(monkeypatch):
    """A solution that passes on one seed but not another is a fluke, not verified."""
    heldout = _HIDDEN_SRC + _INV_SRC
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: _ok(0, 2) if "random.seed(1001)" in code else _ok(2, 2))
    ok, _p, _t, _r = loop._verify_heldout("def f():\n    pass", heldout, seeds=3)
    assert ok is False                                   # passes seed 1000, fails seed 1001
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    ok2, _p2, _t2, _r2 = loop._verify_heldout("def f():\n    pass", heldout, seeds=3)
    assert ok2 is True


def test_static_cheat_caught_then_regenerated_to_verified(monkeypatch):
    """A hardcoded-output solution is caught by the static scan, rejected, and regenerated into a
    genuine, verified one — and the cheat flag is carried forward in cross-attempt memory."""
    cheat = "def black_scholes(S, K, T, r, sigma):\n    return 10.4506"      # hardcodes test output
    genuine = "def black_scholes(S, K, T, r, sigma):\n    return S - K + T"  # uses args, not gaming
    provider = _FakeProvider(
        requirements="- implement black_scholes(S,K,T,r,sigma)", tests=_fence(_BS_TESTS),
        first=[_fence(cheat)],            # round 1: every candidate cheats
        refined=[_fence(genuine)],        # round 2: genuine
        hidden=_HIDDEN_SRC, invariants=_INV_SRC,
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))
    events = []
    res = loop.run_agent("Give me Black-Scholes code", use_search=False, max_iters=2,
                         on_event=events.append)
    assert res.verification == "verified" and res.success is True
    assert "10.4506" not in res.best_code                # the gaming code is never returned
    assert any(e.get("type") == "run_result" and e.get("cheating") for e in events)  # caught
    assert any("REJECTED for gaming" in u for (_s, u) in provider.calls)  # memory fed forward


def test_all_cheating_is_rejected_never_verified(monkeypatch):
    cheat = _fence("def black_scholes(S, K, T, r, sigma):\n    return 10.4506")
    provider = _FakeProvider(
        requirements="- implement black_scholes(S,K,T,r,sigma)", tests=_fence(_BS_TESTS),
        first=[cheat], refined=[cheat],
    )
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))

    res = loop.run_agent("Give me Black-Scholes code", use_search=False, max_iters=2)
    assert res.verification == "rejected_cheating"
    assert res.success is False and res.best_code == ""          # gaming code never presented
    assert "test gaming" in loop.result_to_markdown(res)


def test_escalation_on_two_cheats_uses_strong_model(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_STRONG", "strong-x")
    models = []
    cheat = _fence("def black_scholes(S, K, T, r, sigma):\n    return 10.4506")
    provider = _FakeProvider(
        requirements="- implement black_scholes(S,K,T,r,sigma)", tests=_fence(_BS_TESTS),
        first=[cheat], refined=[cheat],
    )

    def fake_get_provider(model=None, *a, **k):
        models.append(model)
        return provider

    monkeypatch.setattr(loop, "get_provider", fake_get_provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok(2, 2))

    res = loop.run_agent("Give me Black-Scholes code", use_search=False, max_iters=3)
    assert "strong-x" in models                          # escalated after two cheating rounds
    assert res.verification == "rejected_cheating"


def test_rejected_cheating_markdown_hides_code():
    res = loop.AgentResult("t", False, "", "", "x", verification="rejected_cheating")
    md = loop.result_to_markdown(res)
    assert "Rejected" in md and "test gaming" in md
    assert "```python" not in md                         # never present the gaming code


def test_attempt_memory_caps_and_records_cheat_flag():
    m = loop._AttemptMemory(max_notes=3, max_chars=500)
    for i in range(6):
        m.add(f"iter {i}: a note")
    s = m.summary()
    assert "iter 5" in s and "iter 0" not in s           # only the last 3 kept
    m2 = loop._AttemptMemory()
    m2.add("iter 1: REJECTED for gaming — hardcoded output")
    assert "REJECTED for gaming" in m2.summary()


def test_hardening_env_toggles(monkeypatch):
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "false")
    assert loop.hidden_tests_enabled() is False
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "true")
    assert loop.hidden_tests_enabled() is True
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "5")
    assert loop.verify_seeds() == 5
    monkeypatch.setenv("AGENT_VERIFY_SEEDS", "nope")
    assert loop.verify_seeds() == 3
