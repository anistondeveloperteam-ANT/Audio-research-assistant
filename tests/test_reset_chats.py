"""reset_chat_dbs must back up each DB first, then empty the chat tables."""
import sqlite3

from backend.memory.store import MemoryStore
from backend.memory.reset_chats import reset_chat_dbs


def test_reset_backs_up_then_empties(tmp_path):
    mem = MemoryStore(tmp_path / "conversations.db")
    sid = mem.create_session(user_id="local")
    mem.append_turn(sid, "user", "hello")
    mem.append_turn(sid, "assistant", "hi")
    assert len(mem.get_turns(sid)) == 2

    lines = reset_chat_dbs(tmp_path, now_ts=123)

    # chat tables are now empty
    assert MemoryStore(tmp_path / "conversations.db").get_turns(sid) == []
    # a timestamped backup exists AND still holds the original turns
    bak = tmp_path / "conversations.db.bak-123"
    assert bak.exists()
    assert sqlite3.connect(str(bak)).execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 2
    assert any("cleared conversations.db" in ln for ln in lines)


def test_reset_skips_missing_dbs(tmp_path):
    lines = reset_chat_dbs(tmp_path, now_ts=1)
    assert lines and all("skip (missing)" in ln for ln in lines)
