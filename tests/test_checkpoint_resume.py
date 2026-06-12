"""
PHASE 3 #3: persistent SqliteSaver checkpoints (WAL). Verifies the backend switch in
graph._make_checkpointer and that a graph genuinely RESUMES from a checkpoint (does not
re-run completed nodes). Offline — no LLM/Docker/Redis.
"""
import operator
import sqlite3
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite")

import backend.agent.graph as G


def test_make_checkpointer_backend_switch_and_wal(tmp_path, monkeypatch):
    db = tmp_path / "agent_checkpoints.db"
    monkeypatch.setattr(G, "CHECKPOINT_DB", db)

    monkeypatch.setenv("CHECKPOINT_BACKEND", "postgres")
    assert G._make_checkpointer() is None              # not wired yet -> graceful None

    monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")
    saver = G._make_checkpointer()
    assert saver is not None
    assert db.exists()
    mode = sqlite3.connect(str(db)).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"                       # WAL enabled


def test_graph_resumes_from_checkpoint(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import START, END, StateGraph

    class S(TypedDict, total=False):
        steps: Annotated[list, operator.add]

    def node_a(state):
        return {"steps": ["a"]}

    def node_b(state):
        return {"steps": ["b"]}

    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    saver = SqliteSaver(conn)

    g = StateGraph(S)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    graph = g.compile(checkpointer=saver, interrupt_before=["b"])

    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"steps": []}, cfg)                   # runs 'a', pauses before 'b'
    paused = graph.get_state(cfg)
    assert paused.values["steps"] == ["a"]
    assert "b" in paused.next                          # 'b' is pending

    graph.invoke(None, cfg)                            # RESUME from the checkpoint
    final = graph.get_state(cfg)
    assert final.values["steps"] == ["a", "b"]         # resumed; 'a' was NOT re-run
