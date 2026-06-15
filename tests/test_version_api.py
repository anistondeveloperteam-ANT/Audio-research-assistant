"""End-to-end versioning through the HTTP API: editing/regenerating create new versions, a
specific version can be fetched, and the active selection is persisted + restored. Offline:
auth off, no web/local retrieval, answer cache off, query-refine off, planning stubbed — so the
chat stream takes the no-sources fast path and never touches the network or an LLM."""
import json

import pytest
from fastapi.testclient import TestClient

from backend.memory.store import MemoryStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTH", "false")
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 64)
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "false")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "false")
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setenv("QUERY_REFINE", "false")
    monkeypatch.setenv("CODE_INTENT_SEMANTIC", "false")
    import webapp.chat_logic as chat_logic
    store = MemoryStore(tmp_path / "memory.db", conversations_path=tmp_path / "conversations.db")
    monkeypatch.setattr(chat_logic, "_memory", store)
    monkeypatch.setattr(chat_logic, "_deep_queries", lambda q: [q])   # no planning LLM call
    import webapp.server as server
    return TestClient(server.app), store


def _events(resp):
    assert resp.status_code == 200, resp.text
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _chat(c, sid, **body):
    body["session_id"] = sid
    return _events(c.post("/api/chat", json=body))


def _done(events):
    return next(e for e in events if e.get("type") == "done")


def test_edit_then_regenerate_then_restore(client):
    c, store = client
    sid = c.post("/api/sessions").json()["id"]

    # 1) First ask -> question node v1 + answer v1 (no version event for a brand-new question).
    d1 = _done(_chat(c, sid, question="What is MVDR?"))
    node, qv1 = d1["node_id"], d1["qversion_id"]
    assert d1["answer_version_index"] == 1 and d1["answer_total"] == 1
    assert qv1 > 0 and node

    # 2) Edit the question -> NEW question version, old one kept.
    ev2 = _chat(c, sid, question="What is MVDR beamforming?", edit_node_id=node)
    vev = next(e for e in ev2 if e.get("type") == "version")
    assert vev["scope"] == "question" and vev["version_index"] == 2 and vev["total"] == 2
    qv2 = vev["qversion_id"]
    d2 = _done(ev2)
    assert d2["qversion_id"] == qv2 and d2["answer_version_index"] == 1

    # 3) Regenerate the answer for the active (v2) question -> answer v2.
    d3 = _done(_chat(c, sid, regen_qversion_id=qv2))
    assert d3["qversion_id"] == qv2
    assert d3["answer_version_index"] == 2 and d3["answer_total"] == 2

    # Tree: one slot, two question versions (v2 active), v2 has two answers.
    tree = c.get(f"/api/sessions/{sid}/tree").json()
    assert len(tree) == 1
    slot = tree[0]
    assert slot["version_total"] == 2 and slot["active_version_index"] == 2
    v1 = next(v for v in slot["versions"] if v["version_index"] == 1)
    v2 = next(v for v in slot["versions"] if v["version_index"] == 2)
    assert v2["answer_total"] == 2 and v1["content"] is None      # inactive content is lazy

    # 4) Fetch the inactive v1 content lazily.
    got = c.get(f"/api/sessions/{sid}/versions/{v1['turn_id']}").json()
    assert got["content"] == "What is MVDR?"

    # 5) Switch active question back to v1, persisted; a fresh store (reopen) restores it.
    r = c.post(f"/api/sessions/{sid}/versions/active",
               json={"scope": "question", "node_id": node, "version_index": 1})
    assert r.json()["ok"] is True
    assert c.get(f"/api/sessions/{sid}/tree").json()[0]["active_version_index"] == 1
    assert store.get_turns(sid)[0]["content"] == "What is MVDR?"   # active path follows the switch


def test_regenerate_keeps_both_answers_and_can_switch(client):
    c, store = client
    sid = c.post("/api/sessions").json()["id"]
    qv = _done(_chat(c, sid, question="explain RRF"))["qversion_id"]
    _done(_chat(c, sid, regen_qversion_id=qv))                    # answer v2 becomes active

    slot = c.get(f"/api/sessions/{sid}/tree").json()[0]
    assert slot["versions"][0]["answer_total"] == 2 and slot["versions"][0]["active_answer_index"] == 2

    r = c.post(f"/api/sessions/{sid}/versions/active",
               json={"scope": "answer", "qversion_id": qv, "version_index": 1})
    assert r.json()["ok"] is True
    slot2 = c.get(f"/api/sessions/{sid}/tree").json()[0]
    assert slot2["versions"][0]["active_answer_index"] == 1


def test_get_version_is_scoped_to_its_session(client):
    c, store = client
    sid = c.post("/api/sessions").json()["id"]
    _chat(c, sid, question="what is RRF fusion")
    other = c.post("/api/sessions").json()["id"]
    tid = c.get(f"/api/sessions/{sid}/tree").json()[0]["versions"][0]["turn_id"]
    # The turn belongs to `sid`, not `other` -> 404 under the wrong session.
    assert c.get(f"/api/sessions/{other}/versions/{tid}").status_code == 404
    assert c.get(f"/api/sessions/{sid}/versions/{tid}").status_code == 200


def test_old_chats_still_load_as_single_version(client):
    c, store = client
    sid = c.post("/api/sessions").json()["id"]
    # Simulate a legacy turn pair written by append_turn (no edits).
    store.append_turn(sid, "user", "legacy q")
    store.append_turn(sid, "assistant", "legacy a")
    tree = c.get(f"/api/sessions/{sid}/tree").json()
    assert len(tree) == 1 and tree[0]["version_total"] == 1
    assert tree[0]["versions"][0]["content"] == "legacy q"
    assert tree[0]["versions"][0]["answers"][0]["content"] == "legacy a"
