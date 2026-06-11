"""
B1: chats survive a page refresh. The frontend re-renders from /api/sessions/{id}/turns,
so the store's turns-restore path must return the full conversation WITH per-turn sources.
"""
from backend.memory.store import MemoryStore


def test_turns_restore_includes_sources(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db")
    sid = mem.create_session(user_id="anjan", title="MVDR chat")
    mem.append_turn(sid, "user", "What is MVDR beamforming?")
    mem.append_turn(sid, "assistant", "MVDR minimizes output power [1].", sources=[
        {"n": 1, "title": "MVDR primer", "url": "https://example.com/mvdr", "source_type": "web"},
    ])
    turns = mem.get_turns(sid)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "What is MVDR beamforming?"
    assert turns[0]["sources"] is None                       # user turns carry no sources
    assert turns[1]["sources"] and turns[1]["sources"][0]["title"] == "MVDR primer"
