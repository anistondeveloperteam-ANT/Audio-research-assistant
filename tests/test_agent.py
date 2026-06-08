"""Tests for the research agent loop — fully mocked (no Docker, no LLM, no network)."""
import json

import pytest

from backend.agent import loop
from backend.agent.code_runner import RunResult


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
    monkeypatch.setattr(loop, "get_provider", lambda: provider)
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
    monkeypatch.setattr(loop, "get_provider", lambda: provider)
    monkeypatch.setattr(loop, "docker_available", lambda: True)
    monkeypatch.setattr(loop, "run_python", lambda code, **k: next(results))

    res = loop.run_agent("Print ok", use_search=False)
    assert res.success is True
    assert len(res.attempts) == 2
    assert res.best_output.strip() == "ok"


def test_agent_stops_clean_when_docker_missing(monkeypatch):
    monkeypatch.setattr(loop, "get_provider", lambda: _FakeProvider([]))
    monkeypatch.setattr(loop, "docker_available", lambda: False)
    events = []
    res = loop.run_agent("anything", use_search=False, on_event=events.append)
    assert res.success is False
    assert any(e.get("type") == "error" for e in events)
