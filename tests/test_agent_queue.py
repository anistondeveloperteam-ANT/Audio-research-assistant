"""
PHASE 3 #4: the optional distributed queue. Redis is mocked (in-memory list), so these run
offline with no server. They cover the enqueue->publish->stream path (the real Celery task
body + the real stream consumer) and the fallback contract.
"""
import json

import pytest

from backend.agent import task_channel as TC


class FakeRedis:
    """Minimal in-memory stand-in for the redis list ops the channel uses."""

    def __init__(self):
        self.lists = {}

    def ping(self):
        return True

    def rpush(self, key, val):
        self.lists.setdefault(key, []).append(val)

    def expire(self, key, ttl):
        pass

    def blpop(self, key, timeout=0):
        lst = self.lists.get(key) or []
        return (key, lst.pop(0)) if lst else None


def test_publish_then_stream_roundtrip():
    fake = FakeRedis()
    publish = TC.make_publisher(fake, "t1")
    publish({"type": "code", "code": "print(1)"})
    publish({"type": "final", "success": True})
    TC.finish(fake, "t1")
    events = list(TC.stream_events(fake, "t1", idle_limit=1))
    assert [e["type"] for e in events] == ["code", "final"]      # DONE consumed, not yielded


def test_stream_stops_on_done_sentinel():
    fake = FakeRedis()
    TC.make_publisher(fake, "t2")({"type": "status", "message": "x"})
    TC.finish(fake, "t2")
    fake.rpush("agent:events:t2", json.dumps({"type": "late"}))  # arrives after DONE
    events = list(TC.stream_events(fake, "t2", idle_limit=1))
    assert [e["type"] for e in events] == ["status"]            # stops at DONE; 'late' ignored


def test_celery_task_publishes_and_stream_reads(monkeypatch):
    """The actual worker task body publishes to (mocked) Redis; the stream consumer reads it."""
    from backend.agent import celery_app as CA
    import backend.agent.graph as G

    fake = FakeRedis()
    monkeypatch.setattr(TC, "connect_redis", lambda: fake)

    def fake_graph(task, **kw):
        emit = kw.get("on_event") or (lambda e: None)
        emit({"type": "code", "code": "print(1)"})
        emit({"type": "final", "success": True, "code": "print(1)"})
        return G.AgentResult(task, True, "print(1)", "", "answer", [])

    monkeypatch.setattr(G, "run_agent_graph", fake_graph)
    CA.run_agent_graph_task("tid", "do x", "", "")               # run the task body inline

    events = list(TC.stream_events(fake, "tid", idle_limit=1))
    assert [e["type"] for e in events] == ["code", "final"]


def test_endpoint_enqueues_and_returns_task_id(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("ENABLE_AUTH", "false")
    import webapp.server as server

    fake = FakeRedis()
    monkeypatch.setattr(TC, "queue_enabled", lambda: True)
    monkeypatch.setattr(TC, "get_redis", lambda: fake)

    import backend.agent.celery_app as CA

    def fake_delay(task_id, task, brief="", conversation=""):
        pub = TC.make_publisher(fake, task_id)
        pub({"type": "code", "code": "print(1)"})
        pub({"type": "final", "success": True})
        TC.finish(fake, task_id)

    monkeypatch.setattr(CA.run_agent_graph_task, "delay", fake_delay)

    client = fastapi_testclient.TestClient(server.app)
    resp = client.post("/api/agent", json={"question": "do x"})
    assert resp.status_code == 200
    task_id = resp.json().get("task_id")
    assert task_id, f"expected a task_id, got {resp.text!r}"

    stream = client.get(f"/api/agent/{task_id}/stream")
    types = [json.loads(line)["type"] for line in stream.text.splitlines() if line.strip()]
    assert types == ["code", "final"]
