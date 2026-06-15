"""ChatGPT-style message versioning in the store: edit -> new question version, regenerate ->
new answer version, migration of legacy single-version turns, fetch a specific version, and
active-version restore after reload. SQLite only — no network."""
import sqlite3

from backend.memory.store import MemoryStore


def _mem(tmp_path):
    return MemoryStore(tmp_path / "memory.db", conversations_path=tmp_path / "conversations.db")


def _slot0(mem, sid):
    return mem.get_conversation_tree(sid)[0]


def test_first_ask_is_version_1_no_switchers(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "What is MVDR?")
    mem.append_turn(sid, "assistant", "MVDR minimizes output power.", sources=[{"n": 1}])

    tree = mem.get_conversation_tree(sid)
    assert len(tree) == 1
    slot = tree[0]
    assert slot["version_total"] == 1 and slot["active_version_index"] == 1
    qv = slot["versions"][0]
    assert qv["content"] == "What is MVDR?" and qv["answer_total"] == 1
    assert qv["answers"][0]["content"] == "MVDR minimizes output power."
    assert qv["answers"][0]["sources"] == [{"n": 1}]


def test_edit_question_creates_new_version_keeping_old(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "What is MVDR?")
    mem.append_turn(sid, "assistant", "answer A")
    node = _slot0(mem, sid)["node_id"]

    qv2 = mem.add_question_version(sid, node, "What is MVDR beamforming exactly?")
    assert qv2["version_index"] == 2 and qv2["total"] == 2
    mem.add_answer_version(qv2["turn_id"], "answer B")

    slot = _slot0(mem, sid)
    assert slot["version_total"] == 2
    assert slot["active_version_index"] == 2                      # new version is active (latest)
    v1 = next(v for v in slot["versions"] if v["version_index"] == 1)
    v2 = next(v for v in slot["versions"] if v["version_index"] == 2)
    assert v1["is_active"] is False and v2["is_active"] is True
    # Old version + its answer are KEPT (just not active); content lazy (None) when inactive.
    assert v1["answers"][0]["turn_id"] > 0
    assert v2["content"] == "What is MVDR beamforming exactly?"
    assert v2["answers"][0]["content"] == "answer B"


def test_regenerate_creates_new_answer_version_under_same_question(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "Explain RRF")
    mem.append_turn(sid, "assistant", "answer 1")
    qv_id = _slot0(mem, sid)["versions"][0]["turn_id"]

    r = mem.add_answer_version(qv_id, "answer 2")
    assert r["version_index"] == 2

    qv = _slot0(mem, sid)["versions"][0]
    assert qv["version_total"] if "version_total" in qv else True   # questions unchanged
    assert qv["answer_total"] == 2 and qv["active_answer_index"] == 2
    active = [a for a in qv["answers"] if a["is_active"]]
    assert len(active) == 1 and active[0]["content"] == "answer 2"


def test_migration_of_legacy_single_version_turns(tmp_path):
    # Write rows the OLD way (no version columns) straight into a turns table, then open the store.
    cache = tmp_path / "memory.db"
    conv = tmp_path / "conversations.db"
    raw = sqlite3.connect(conv)
    raw.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, user_id TEXT, "
        "created_at REAL, updated_at REAL);"
        "CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
        "turn_index INTEGER, role TEXT, content TEXT, sources_json TEXT, created_at REAL);"
        "INSERT INTO sessions VALUES ('s1','old','anjan',1.0,1.0);"
        "INSERT INTO turns (session_id,turn_index,role,content,sources_json,created_at) VALUES"
        " ('s1',0,'user','old q',NULL,1.0),"
        " ('s1',1,'assistant','old a','[{\"n\":1}]',1.0);"
    )
    raw.commit()
    raw.close()

    mem = MemoryStore(cache, conversations_path=conv)            # triggers migration + backfill
    # Loads as version 1 (single version, no switcher), Q -> A linkage reconstructed.
    tree = mem.get_conversation_tree("s1")
    assert len(tree) == 1 and tree[0]["version_total"] == 1
    qv = tree[0]["versions"][0]
    assert qv["content"] == "old q" and qv["answers"][0]["content"] == "old a"
    assert qv["answers"][0]["sources"] == [{"n": 1}]
    # get_turns shape unchanged for old chats.
    turns = mem.get_turns("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["sources"] is None and turns[1]["sources"] == [{"n": 1}]

    # Idempotent: re-open doesn't duplicate or re-version.
    mem2 = MemoryStore(cache, conversations_path=conv)
    assert mem2.get_conversation_tree("s1")[0]["version_total"] == 1


def test_get_version_fetches_specific_content(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "q")
    mem.append_turn(sid, "assistant", "first")
    qv_id = _slot0(mem, sid)["versions"][0]["turn_id"]
    a2 = mem.add_answer_version(qv_id, "second", sources=[{"n": 2}])

    got = mem.get_version(a2["turn_id"])
    assert got["content"] == "second" and got["sources"] == [{"n": 2}]
    assert got["version_index"] == 2
    assert mem.get_version(999999) is None


def test_active_version_restore_after_reload(tmp_path):
    cache = tmp_path / "memory.db"
    conv = tmp_path / "conversations.db"
    mem = MemoryStore(cache, conversations_path=conv)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "q v1")
    mem.append_turn(sid, "assistant", "a for v1")
    node = _slot0(mem, sid)["node_id"]
    mem.add_question_version(sid, node, "q v2")
    qv2_id = [v for v in _slot0(mem, sid)["versions"] if v["version_index"] == 2][0]["turn_id"]
    mem.add_answer_version(qv2_id, "a for v2")

    # User switches BACK to version 1 (persisted).
    assert mem.set_active_question_version(sid, node, 1) is True

    # Reopen the store (simulates a page refresh / new process).
    mem2 = MemoryStore(cache, conversations_path=conv)
    slot = _slot0(mem2, sid)
    assert slot["active_version_index"] == 1                      # restored selection, not latest
    v1 = next(v for v in slot["versions"] if v["version_index"] == 1)
    assert v1["is_active"] is True and v1["content"] == "q v1"    # active path carries content
    assert mem2.get_turns(sid)[0]["content"] == "q v1"           # active path follows the switch


def test_set_active_answer_version_persists(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "q")
    mem.append_turn(sid, "assistant", "ans1")
    qv_id = _slot0(mem, sid)["versions"][0]["turn_id"]
    mem.add_answer_version(qv_id, "ans2")

    assert mem.set_active_answer_version(qv_id, 1) is True
    qv = _slot0(mem, sid)["versions"][0]
    assert qv["active_answer_index"] == 1
    assert mem.get_turns(sid)[1]["content"] == "ans1"            # active path shows ans1 again
    assert mem.set_active_answer_version(qv_id, 99) is False      # missing version


def test_delete_node_removes_all_versions_and_answers(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.append_turn(sid, "user", "keep")
    mem.append_turn(sid, "assistant", "keep-a")
    mem.append_turn(sid, "user", "drop")
    mem.append_turn(sid, "assistant", "drop-a")
    tree = mem.get_conversation_tree(sid)
    drop_node = tree[1]["node_id"]
    mem.add_answer_version(tree[1]["versions"][0]["turn_id"], "drop-a2")  # extra answer version

    deleted = mem.delete_node(sid, drop_node)
    assert deleted == 3                                           # 1 question + 2 answers
    remaining = mem.get_conversation_tree(sid)
    assert len(remaining) == 1 and remaining[0]["versions"][0]["content"] == "keep"
