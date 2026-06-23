"""A code-intent query routes to the agent: _run_code_agent streams the agent's events and
saves the final code as the assistant turn. The agent itself is mocked (offline, no Docker)."""
import types

import webapp.chat_logic as cl
from backend.memory.store import MemoryStore


def test_run_code_agent_streams_events_and_persists_turn(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")
    mem.append_turn(sid, "user", "Give me RTF-MVDR python code")

    def fake_run_agent(task, *, brief="", use_search=True, conversation="", on_event=None, **kwargs):
        on_event({"type": "think", "iteration": 1, "message": "designing"})
        on_event({"type": "final", "success": True, "answer": "MVDR weights",
                  "code": "print(1)", "output": "1"})
        return types.SimpleNamespace(answer="MVDR weights", best_code="print(1)",
                                     best_output="1", success=True, tests_total=2, tests_passed=2)

    monkeypatch.setattr("backend.agent.loop.run_agent", fake_run_agent)

    events = list(cl._run_code_agent("Give me RTF-MVDR python code", sid, mem))
    kinds = [e["type"] for e in events]
    assert "think" in kinds and "final" in kinds
    assert kinds[-1] == "done"

    turns = mem.get_turns(sid)
    assert turns[-1]["role"] == "assistant"
    assert "print(1)" in turns[-1]["content"]


def test_partially_verified_label_when_tests_fail(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")

    def fake_run_agent(task, *, brief="", use_search=True, conversation="", on_event=None, **kwargs):
        return types.SimpleNamespace(answer="", best_code="x=1", best_output="",
                                     success=False, tests_total=4, tests_passed=2)

    monkeypatch.setattr("backend.agent.loop.run_agent", fake_run_agent)
    list(cl._run_code_agent("implement quicksort", sid, mem))
    content = mem.get_turns(sid)[-1]["content"]
    assert "Partially verified" in content and "2/4" in content
