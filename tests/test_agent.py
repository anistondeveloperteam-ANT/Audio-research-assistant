"""Tests for the research agent loop — fully mocked (no Docker, no LLM, no network)."""
import json

from backend.agent import loop
from backend.agent import hooks
from backend.agent.code_runner import RunResult
from backend.agent.hooks import HookDecision
from backend.agent.memory import TwoTierMemory


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
    """Returns queued responses, one per stream_chat call."""
    name = "openai"
    model = "test"
    is_available = True

    def __init__(self, responses):
        self._responses = list(responses)

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0):
        return [self._responses.pop(0)]


def _verdict(success, done, score, feedback="", answer=""):
    return json.dumps({"success": success, "done": done, "score": score,
                       "feedback": feedback, "answer": answer})


# ---- the loop --------------------------------------------------------
def test_agent_succeeds_first_try(monkeypatch):
    provider = _FakeProvider([
        "```python\nprint('answer=42')\n```",                    # generated code
        _verdict(True, True, 95, answer="The answer is 42."),    # review: done
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python",
                        lambda code, **k: RunResult(True, 0, "answer=42\n", "", 0.2))

    res = loop.run_agent("What is 6*7?", use_search=False)
    assert res.success is True
    assert res.answer == "The answer is 42."
    assert "42" in res.best_output
    assert len(res.attempts) == 1


def test_agent_refines_after_a_failure(monkeypatch):
    provider = _FakeProvider([
        "```python\nimport nope\n```",                # attempt 1: will "fail"
        _verdict(False, False, 0, feedback="ImportError — use stdlib"),
        "```python\nprint('ok')\n```",               # attempt 2: succeeds
        _verdict(True, True, 90, answer="Use the stdlib version."),
    ])
    results = iter([
        RunResult(False, 1, "", "ModuleNotFoundError: No module named 'nope'", 0.1),
        RunResult(True, 0, "ok\n", "", 0.1),
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python", lambda code, **k: next(results))

    res = loop.run_agent("Print ok", use_search=False)
    assert res.success is True
    assert len(res.attempts) == 2
    assert res.best_output.strip() == "ok"


def test_loop_blocks_without_running(monkeypatch):
    provider = _FakeProvider([
        "```python\nprint(1)\n```",
        _verdict(False, False, 0, feedback="was blocked"),
    ])
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    ran = []
    monkeypatch.setattr(loop, "run_python", lambda code, **k: ran.append(1) or RunResult(True, 0, "", "", 0.1))
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
