"""
Unit tests for the Docker sandbox runner's image handling. All Docker calls are
mocked, so these run with no Docker daemon, no network, and no image build.
"""
import types

import backend.agent.code_runner as cr


def _reset():
    cr._image_ready = None


def test_ensure_sandbox_image_respects_user_override(monkeypatch):
    _reset()
    monkeypatch.setenv("AGENT_DOCKER_IMAGE", "myorg/myimage:latest")
    calls = []
    monkeypatch.setattr(cr.subprocess, "run", lambda *a, **k: calls.append(a))
    ready, err = cr.ensure_sandbox_image()
    assert ready is True and err == "" and calls == []   # trusted; no docker calls


def test_ensure_sandbox_image_uses_existing(monkeypatch):
    _reset()
    monkeypatch.delenv("AGENT_DOCKER_IMAGE", raising=False)
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    ready, err = cr.ensure_sandbox_image()
    assert ready is True and err == ""
    assert any("inspect" in c for c in cmds)             # inspected the image
    assert not any("build" in c for c in cmds)           # did NOT rebuild


def test_ensure_sandbox_image_builds_when_missing(monkeypatch):
    _reset()
    monkeypatch.delenv("AGENT_DOCKER_IMAGE", raising=False)
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if "inspect" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="No such image")
        return types.SimpleNamespace(returncode=0, stdout="built", stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    ready, err = cr.ensure_sandbox_image()
    assert ready is True and err == ""
    assert any("build" in c for c in seen)               # built the missing image


def test_ensure_sandbox_image_reports_build_failure(monkeypatch):
    _reset()
    monkeypatch.delenv("AGENT_DOCKER_IMAGE", raising=False)

    def fake_run(cmd, **kw):
        if "inspect" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="pip failed: boom")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    ready, err = cr.ensure_sandbox_image()
    assert ready is False and "build failed" in err


def test_default_image_is_the_sandbox_tag():
    # Without an override, the runner targets our prebuilt scientific image.
    assert cr.DEFAULT_IMAGE == cr.SANDBOX_TAG


def test_run_python_blocks_when_docker_missing(monkeypatch):
    monkeypatch.setattr(cr, "docker_available", lambda: False)
    res = cr.run_python("print(1)")
    assert res.ok is False and "Docker is not available" in res.error
