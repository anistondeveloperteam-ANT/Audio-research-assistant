"""
Tests for the autonomous agent module + the gated POST /agent/task endpoint.
The Claude Agent SDK (and its CLI/key) are never invoked — the run is mocked — so
these pass offline with no Anthropic credentials.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import webapp.server as srv
    import webapp.auth as webauth
    monkeypatch.setattr(webauth, "auth_enabled", lambda: False)   # bypass login for the test
    monkeypatch.setattr(srv, "_is_loopback", lambda r: True)
    try:
        srv._RATE_BUCKETS.clear()
    except Exception:
        pass
    return TestClient(srv.app)


def test_agent_task_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("ENABLE_AUTO_AGENT", raising=False)
    r = client.post("/agent/task", json={"task": "do x"})
    assert r.status_code == 403 and "disabled" in r.json()["error"]


def test_agent_task_requires_loopback(client, monkeypatch):
    import webapp.server as srv
    monkeypatch.setenv("ENABLE_AUTO_AGENT", "true")
    monkeypatch.setattr(srv, "_is_loopback", lambda r: False)
    r = client.post("/agent/task", json={"task": "do x"})
    assert r.status_code == 403 and "localhost" in r.json()["error"]


def test_agent_task_requires_task(client, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_AGENT", "true")
    r = client.post("/agent/task", json={"task": "   "})
    assert r.status_code == 400


def test_agent_task_runs_when_enabled(client, monkeypatch):
    import webapp.server as srv
    monkeypatch.setenv("ENABLE_AUTO_AGENT", "true")

    async def fake_run(task, **kw):
        return {"task": task, "steps": [{"type": "text", "text": "done"}],
                "result": {"num_turns": 3, "is_error": False}}

    monkeypatch.setattr(srv.auto_agent, "run_auto_agent", fake_run)
    r = client.post("/agent/task", json={"task": "add a helper"})
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "add a helper" and body["result"]["num_turns"] == 3


def test_agent_task_missing_key_returns_503(client, monkeypatch):
    import webapp.server as srv
    monkeypatch.setenv("ENABLE_AUTO_AGENT", "true")

    async def boom(task, **kw):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(srv.auto_agent, "run_auto_agent", boom)
    r = client.post("/agent/task", json={"task": "x"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["error"]


# ---- run_auto_agent unit (no SDK / CLI / key needed; checks happen first) ----
def test_run_auto_agent_rejects_empty_task():
    from backend.agent import auto_agent
    with pytest.raises(ValueError):
        asyncio.run(auto_agent.run_auto_agent(""))


def test_run_auto_agent_surfaces_cli_missing(monkeypatch):
    # No Anthropic key needed (Pro/Max subscription works); when the CLI is missing,
    # the SDK raises CLINotFoundError and we surface a clear, actionable message.
    import claude_agent_sdk as sdk
    from backend.agent import auto_agent

    async def boom_query(*a, **k):
        raise sdk.CLINotFoundError("claude not found")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sdk, "query", boom_query)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(auto_agent.run_auto_agent("implement something", "."))
    assert "Claude Code CLI" in str(exc.value)


def test_allowed_tools_and_turns_match_spec():
    from backend.agent import auto_agent
    assert auto_agent.ALLOWED_TOOLS == ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
    assert auto_agent.MAX_TURNS == 20


# ---- streaming endpoint + /api/me flag ----
def test_agent_stream_emits_ndjson(client, monkeypatch):
    import json
    import webapp.server as srv
    monkeypatch.setenv("ENABLE_AUTO_AGENT", "true")

    async def fake_stream(task, project_dir=None, **kw):
        yield {"type": "step", "kind": "tool", "name": "Write", "input": "x"}
        yield {"type": "result", "num_turns": 2, "is_error": False, "result": "ok"}

    monkeypatch.setattr(srv.auto_agent, "stream_auto_agent", fake_stream)
    r = client.post("/agent/task/stream", json={"task": "do x"})
    assert r.status_code == 200
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events[0]["type"] == "step" and events[-1]["type"] == "result"


def test_agent_stream_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("ENABLE_AUTO_AGENT", raising=False)
    r = client.post("/agent/task/stream", json={"task": "x"})
    assert r.status_code == 403


def test_api_me_exposes_auto_agent_flag(client, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_AGENT", "true")
    assert client.get("/api/me").json().get("auto_agent") is True


def test_stream_auto_agent_yields_error_event_on_empty_task():
    from backend.agent import auto_agent

    async def collect():
        return [ev async for ev in auto_agent.stream_auto_agent("")]

    events = asyncio.run(collect())
    assert events == [{"type": "error", "message": "task is required"}]
