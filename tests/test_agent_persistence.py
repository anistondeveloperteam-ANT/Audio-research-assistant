"""A coding-agent run must be saved as a normal user+assistant turn pair so it reloads after
the page is closed and reopened (previously agent runs were view-only and vanished)."""
import types

from backend.memory.store import MemoryStore
from webapp.server import _persist_agent_run


def _fake_result(answer="", best_code="", best_output=""):
    return types.SimpleNamespace(answer=answer, best_code=best_code, best_output=best_output)


def test_agent_run_persists_as_user_and_assistant_turns(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")

    res = _fake_result(answer="Quicksort runs in O(n log n) on average.",
                       best_code="print(sorted([3, 1, 2]))",
                       best_output="[1, 2, 3]")
    _persist_agent_run(mem, sid, "write code to sort a list", res)

    turns = mem.get_turns(sid)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "write code to sort a list"
    assert turns[1]["role"] == "assistant"
    body = turns[1]["content"]
    assert "Quicksort runs in O(n log n)" in body      # explanation kept
    assert "print(sorted([3, 1, 2]))" in body           # code kept (in a fence)
    assert "[1, 2, 3]" in body                          # output kept


def test_agent_run_survives_reopen(tmp_path):
    # Simulate "reopen": a brand-new store pointed at the same file must see the saved turns.
    db = tmp_path / "mem.db"
    mem = MemoryStore(db)
    sid = mem.create_session(user_id="local")
    _persist_agent_run(mem, sid, "task", _fake_result(best_code="x = 1", best_output="ok"))

    reopened = MemoryStore(db)
    turns = reopened.get_turns(sid)
    assert len(turns) == 2
    assert "x = 1" in turns[1]["content"] and "ok" in turns[1]["content"]


def test_agent_run_with_no_result_still_saves_a_turn(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")
    _persist_agent_run(mem, sid, "impossible task", _fake_result())

    turns = mem.get_turns(sid)
    assert len(turns) == 2
    assert turns[1]["role"] == "assistant" and turns[1]["content"].strip()
