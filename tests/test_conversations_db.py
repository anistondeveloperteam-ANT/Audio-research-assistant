"""
B2: conversations (sessions/turns/facts) live in their own SQLite file; the answer cache
stays in memory.db. Migration from the legacy single DB is one-time and idempotent, and
the old file is never deleted. All cross-table SQL still works via ATTACH.
"""
import sqlite3

from backend.memory.store import MemoryStore


def test_split_routes_tables_to_the_right_files(tmp_path):
    cache = tmp_path / "memory.db"
    conv = tmp_path / "conversations.db"
    mem = MemoryStore(cache, conversations_path=conv)
    sid = mem.create_session(user_id="anjan", title="t")
    mem.append_turn(sid, "user", "hi")
    mem.cache_answer(user_id="anjan", session_id=sid, question="hi", answer="hello")

    c = sqlite3.connect(conv)
    assert c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1   # conv db
    assert c.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1
    c.close()
    m = sqlite3.connect(cache)
    assert m.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 1  # cache db
    m.close()
    assert [t["content"] for t in mem.get_turns(sid)] == ["hi"]             # single interface


def test_migration_copies_legacy_and_is_idempotent(tmp_path):
    cache = tmp_path / "memory.db"
    conv = tmp_path / "conversations.db"
    # legacy: a single-file store (conv == cache) writes conversation rows into memory.db
    legacy = MemoryStore(cache, conversations_path=cache)
    sid = legacy.create_session(user_id="anjan", title="old chat")
    legacy.append_turn(sid, "user", "legacy question")

    m1 = MemoryStore(cache, conversations_path=conv)            # migrates legacy -> conv.db
    assert any(s["id"] == sid for s in m1.list_sessions(user_id="anjan"))
    assert [t["content"] for t in m1.get_turns(sid)] == ["legacy question"]

    m2 = MemoryStore(cache, conversations_path=conv)            # second open = no-op
    assert len([s for s in m2.list_sessions(user_id="anjan") if s["id"] == sid]) == 1
    assert len(m2.get_turns(sid)) == 1
    assert cache.exists()                                       # legacy file never deleted


def test_delete_session_cleans_both_dbs(tmp_path):
    mem = MemoryStore(tmp_path / "memory.db", conversations_path=tmp_path / "conversations.db")
    sid = mem.create_session(user_id="anjan", title="t")
    mem.append_turn(sid, "user", "q")
    mem.cache_answer(user_id="anjan", session_id=sid, question="q", answer="a")
    mem.delete_session(sid)
    assert mem.list_sessions(user_id="anjan") == []
    assert mem.get_turns(sid) == []
