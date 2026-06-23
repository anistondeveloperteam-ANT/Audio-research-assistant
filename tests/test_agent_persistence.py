"""A coding-agent run must be saved as a normal user+assistant turn pair so it reloads after
the page is closed and reopened (previously agent runs were view-only and vanished)."""
import types

from backend.memory.store import MemoryStore
from webapp.server import _persist_agent_markdown, _persist_agent_run


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


def test_persisted_agent_turn_is_marked_kind_agent(tmp_path):
    # The saved run carries kind='agent' so a reloaded run is re-run via the AGENT on Regenerate.
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")
    _persist_agent_run(mem, sid, "write code", _fake_result(best_code="x = 1", best_output="ok"))
    ans = mem.get_conversation_tree(sid)[0]["versions"][0]["answers"][0]
    assert ans["kind"] == "agent"


def test_normal_chat_turn_kind_is_none(tmp_path):
    # The new column does not affect ordinary chat turns.
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, "what is RMS?")["turn_id"]
    mem.add_answer_version(qid, "Root mean square …")
    assert mem.get_conversation_tree(sid)[0]["versions"][0]["answers"][0]["kind"] is None


def test_regenerate_adds_answer_version_under_same_question(tmp_path):
    # Regenerate re-runs the agent and adds a NEW answer version under the SAME question (a switcher),
    # rather than a separate exchange.
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")
    _persist_agent_run(mem, sid, "build a FIR filter", _fake_result(best_code="v1"))
    qv = mem.get_conversation_tree(sid)[0]["versions"][0]
    _persist_agent_markdown(mem, sid, "build a FIR filter", "regenerated",
                            regen_qversion_id=qv["turn_id"])
    qv2 = mem.get_conversation_tree(sid)[0]["versions"][0]
    assert qv2["answer_total"] == 2 and qv2["active_answer_index"] == 2     # two versions, latest active
    assert all(a["kind"] == "agent" for a in qv2["answers"])


def test_interrupted_run_is_still_saved_and_regeneratable(tmp_path):
    # A run that drops mid-stream is still persisted (kind='agent') so it reloads and can be retried.
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="local")
    _persist_agent_markdown(mem, sid, "long task", "**The agent run was interrupted.** Regenerate.")
    ans = mem.get_conversation_tree(sid)[0]["versions"][0]["answers"][0]
    assert ans["kind"] == "agent"
    assert "interrupted" in (ans["content"] or "").lower()
