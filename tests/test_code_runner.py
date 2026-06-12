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


def test_run_python_pipes_unicode_code_as_utf8(monkeypatch):
    # Regression: AI code often contains Unicode math (λ, ∇, π). It must be piped to the
    # container as UTF-8 — not Windows cp1252 — or the run crashes with
    # "'charmap' codec can't encode character '\\u03bb'" and the code never runs.
    monkeypatch.setattr(cr, "docker_available", lambda: True)
    monkeypatch.setattr(cr, "ensure_sandbox_image", lambda: (True, ""))
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return types.SimpleNamespace(returncode=0, stdout="λ = 1.0, ∇L = 0", stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    code = "lam = 'λ'  # Lagrange multiplier λ, gradient ∇\nprint('λ =', 1.0)"
    res = cr.run_python(code)

    assert res.ok is True
    assert captured["kw"].get("encoding") == "utf-8"     # the actual fix (host side)
    assert captured["kw"].get("input") == code            # code piped through unchanged
    assert "PYTHONUTF8=1" in captured["cmd"]              # container runs in UTF-8 too
    assert "λ" in res.stdout and "∇" in res.stdout        # Unicode output survives the round-trip
