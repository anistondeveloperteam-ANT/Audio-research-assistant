"""Learning / memory layer for the code agent: record every run, REUSE only verified ones as a
starting point (never bypassing a gate), surface recurring failure patterns for a human, and suggest
config thresholds from measured eval (print-only). The agent never edits its own source.

Proves:
  (a) a verified solution is recorded and REUSED (seeded) on a near-identical task;
  (b) an unverified / failed run is NOT reused;
  (c) recurring failure patterns are aggregated;
  (d) reuse respects the reuse-safety gate (swaps/identifier changes) and the AGENT_RESULT_MEMORY flag;
  plus schema migration v5->v6 and the parameter-suggestion logic.

Deterministic: no network, no Docker, no real LLM.
"""
import sqlite3
import types
from pathlib import Path

from backend.agent import loop
from backend.memory.store import MemoryStore
from backend.evaluation.param_suggest import suggest_thresholds, format_suggestions


def _mem(dir_path) -> MemoryStore:
    return MemoryStore(Path(dir_path) / "memory.db")


# ======================================================================================
# Store: record / find (verified-only) / reuse-count / patterns.
# ======================================================================================
def test_verified_run_is_findable_and_unverified_is_not(tmp_path):
    mem = _mem(tmp_path)
    mem.record_agent_run(user_id="u", task="implement quicksort and print the sorted list",
                         code="def f():\n    return 1\n", verification="verified",
                         tests_passed=3, tests_total=3)
    mem.record_agent_run(user_id="u", task="implement mergesort and print the sorted list",
                         code="def g():\n    return 2\n", verification="partial")

    found = mem.find_verified_solution(user_id="u", task="implement quicksort and print sorted list")
    assert found and "return 1" in found["code"] and found["similarity"] >= 0.90      # (a)/verified
    assert mem.find_verified_solution(user_id="u",
                                      task="implement mergesort and print sorted list") is None  # (b)


def test_reuse_safety_blocks_swaps_and_identifier_changes(tmp_path):
    mem = _mem(tmp_path)
    mem.record_agent_run(user_id="u", task="convert miles to km", code="def f():\n    return 1\n",
                         verification="verified")
    assert mem.find_verified_solution(user_id="u", task="convert km to miles") is None   # swap blocked
    assert mem.find_verified_solution(user_id="u", task="convert miles to km") is not None  # exact ok


def test_record_run_reuse_bumps_count(tmp_path):
    mem = _mem(tmp_path)
    rid = mem.record_agent_run(user_id="u", task="t one", code="x", verification="verified")
    mem.record_agent_run_reuse(rid)
    mem.record_agent_run_reuse(rid)
    with mem._conn() as conn:
        row = conn.execute("SELECT reuse_count FROM agent_runs WHERE id = ?", (rid,)).fetchone()
    assert row["reuse_count"] == 2


def test_failure_patterns_aggregate(tmp_path):
    mem = _mem(tmp_path)
    mem.record_agent_run(user_id="u", task="evolve a wavefunction conserving the norm",
                         verification="partial",
                         cheat_reasons=["renormalises the state inside the evolution loop (masking)"])
    mem.record_agent_run(user_id="u", task="compute the fft and print the peak",
                         verification="partial", gate_fail="execution: no real stdout")
    mem.record_agent_run(user_id="u", task="report the median of the data", verification="partial",
                         failing_checks=["test_definition_median"])
    mem.record_agent_run(user_id="u", task="a solved task", verification="verified", code="x")

    rep = mem.agent_failure_patterns(user_id="u")
    assert rep["total_runs"] == 4 and rep["verified"] == 1 and rep["unverified"] == 3
    labels = " ".join(p["pattern"].lower() for p in rep["patterns"])
    assert "masking" in labels and ("output" in labels or "delivery" in labels)
    assert "definition" in labels or "quantity" in labels


def test_schema_upgrades_from_v5_creates_agent_runs(tmp_path):
    # A pre-existing v5 DB (no agent_runs) must upgrade cleanly and gain the table.
    p = tmp_path / "memory.db"
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA user_version = 5;")
    conn.commit()
    conn.close()

    store = MemoryStore(p)
    rid = store.record_agent_run(user_id="u", task="t", code="c", verification="verified")
    assert rid is not None
    assert store.find_verified_solution(user_id="u", task="t") is not None


# ======================================================================================
# Loop integration: verified solutions are recorded by run_agent and reused (seeded) next time.
# ======================================================================================
def _ok():
    return types.SimpleNamespace(ok=True, exit_code=0, stdout="TEST test_basic PASS\nTESTS_PASSED 1/1\n",
                                 stderr="", duration=0.1, error="", summary="ok")


class _CapProvider:
    """Captures the _GEN_SYSTEM user prompts so a test can confirm the seeded prior code reached the
    rewrite, and returns a fixed solution."""
    is_available = True
    name, model = "openai", "test"

    def __init__(self, solution):
        self.solution = solution
        self.gen_users = []

    def stream_chat(self, messages, system="", **k):
        user = messages[-1]["content"] if messages else ""
        if system == loop._REQ_SYSTEM:
            return ["- add_two(x): return x + 2"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert add_two(1) == 3\n"]
        if system == loop._GEN_SYSTEM:
            self.gen_users.append(user)
            return [self.solution]
        return [""]


def _verify_env(monkeypatch):
    for k, v in {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
                 "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
                 "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
                 "AGENT_ANTICHEAT_SCAN": "false", "AGENT_ROOT_CAUSE_DIAGNOSIS": "false",
                 "AGENT_RESULT_MEMORY": "true", "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1",
                 "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type", lambda t: "deterministic")
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python_auto", lambda code, **k: _ok())


def test_verified_solution_is_recorded_and_reused(monkeypatch, tmp_path):
    _verify_env(monkeypatch)
    mem = _mem(tmp_path)
    task = "implement add_two(x) that returns x plus two"

    prov1 = _CapProvider("def add_two(x):\n    return x + 2  # ORIGINAL\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov1)
    res1 = loop.run_agent(task, use_search=False, result_memory=mem, user_id="u")
    assert res1.verification == "verified"
    assert mem.find_verified_solution(user_id="u", task=task) is not None        # recorded

    # A near-identical NEW task reuses the prior verified code as the seed for the rewrite.
    prov2 = _CapProvider("def add_two(x):\n    return x + 2  # REGENERATED\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov2)
    events = []
    res2 = loop.run_agent("implement add_two(x) which returns x plus two", use_search=False,
                          result_memory=mem, user_id="u", on_event=events.append)
    assert res2.verification == "verified"
    assert any(e.get("type") == "reuse" for e in events)                         # reuse happened
    assert prov2.gen_users and "ORIGINAL" in prov2.gen_users[0]                  # prior code seeded it


def test_disabling_result_memory_skips_record_and_reuse(monkeypatch, tmp_path):
    _verify_env(monkeypatch)
    monkeypatch.setenv("AGENT_RESULT_MEMORY", "false")
    mem = _mem(tmp_path)
    prov = _CapProvider("def add_two(x):\n    return x + 2\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    res = loop.run_agent("implement add_two(x) that returns x plus two", use_search=False,
                         result_memory=mem, user_id="u")
    assert res.verification == "verified"
    assert mem.find_verified_solution(user_id="u",
                                      task="implement add_two(x) that returns x plus two") is None


def test_seeded_but_wrong_solution_is_still_downgraded_by_the_gate(monkeypatch, tmp_path):
    # Reuse seeds the first attempt, but the full gate stack still decides: a seed that fails the
    # held-out check is NOT accepted (reuse never overrides a gate).
    _verify_env(monkeypatch)
    monkeypatch.setenv("AGENT_HIDDEN_TESTS", "false")
    monkeypatch.setenv("AGENT_DEFINITION_GATE", "true")     # a held-out gate that will fail the seed
    mem = _mem(tmp_path)
    mem.record_agent_run(user_id="u", task="implement add_two(x) that returns x plus two",
                         code="def add_two(x):\n    return x + 2  # SEED\n", verification="verified")

    class _P(_CapProvider):
        def stream_chat(self, messages, system="", **k):
            if system == loop._DEFINITION_SYSTEM:
                return ["def test_definition_add(): assert True\n"]
            return super().stream_chat(messages, system=system, **k)

    prov = _P("def add_two(x):\n    return x + 2  # SEED\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)

    def run(code, **k):
        if "held-out runner" in code:
            return types.SimpleNamespace(ok=True, exit_code=0,
                                         stdout="TEST test_definition_add FAIL\nTESTS_PASSED 0/1\n",
                                         stderr="", duration=0.1, error="", summary="ok")
        return _ok()
    monkeypatch.setattr(loop, "run_python_auto", run)

    res = loop.run_agent("implement add_two(x) returning x plus 2", use_search=False,
                         result_memory=mem, user_id="u", max_iters=2)
    assert res.verification != "verified"        # the gate still rejects, despite the verified seed


# ======================================================================================
# Parameter suggestions (print-only; never auto-applied).
# ======================================================================================
def test_suggest_raises_strong_min_on_low_skip_precision():
    s = suggest_thresholds({"crag_strong_min": 0.55, "crag_skip_precision": 0.90})
    rec = next(x for x in s if x["var"] == "CRAG_STRONG_MIN")
    assert rec["recommended"] > 0.55 and "skip precision" in rec["evidence"].lower()


def test_suggest_lowers_partial_min_on_low_recall():
    s = suggest_thresholds({"crag_partial_min": 0.30, "crag_recall": 0.60})
    rec = next(x for x in s if x["var"] == "CRAG_PARTIAL_MIN")
    assert rec["recommended"] < 0.30 and "recall" in rec["evidence"].lower()


def test_suggest_no_change_when_healthy_and_never_auto_applies():
    s = suggest_thresholds({"crag_strong_min": 0.55, "crag_skip_precision": 0.97,
                            "crag_partial_min": 0.30, "crag_recall": 0.80})
    assert s and all(x["change"] == 0.0 for x in s)
    assert "not auto-applied" in format_suggestions(s).lower()
