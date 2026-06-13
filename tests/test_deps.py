"""Dynamic sandbox dependencies: import parsing, module->package mapping, per-hash image
cache reuse / build / fallback, and auto-run image selection. Docker is always mocked."""
import types

import backend.agent.deps as deps
import backend.agent.code_runner as cr


# ---- deps.py -----------------------------------------------------------------
def test_parse_imports_top_level_and_skips_relative():
    code = ("import numpy as np\nimport cv2\nfrom sklearn.linear_model import Lasso\n"
            "import os, sys\nfrom . import helper\n")
    mods = deps.parse_imports(code)
    assert {"numpy", "cv2", "sklearn", "os", "sys"} <= mods
    assert "helper" not in mods   # relative import skipped


def test_parse_imports_bad_syntax_is_empty():
    assert deps.parse_imports("def (::: bad") == set()


def test_modules_to_packages_alias_passthrough_and_filters():
    pkgs = deps.modules_to_packages(
        ["numpy", "cv2", "sklearn", "PIL", "os", "requests", "bs4"])
    # numpy/sklearn = base (drop); os = stdlib (drop); cv2/PIL/bs4 aliased; requests passthrough.
    assert pkgs == sorted(["opencv-python-headless", "pillow", "requests", "beautifulsoup4"])


def test_requirements_hash_order_independent_and_distinct():
    assert deps.requirements_hash(["b", "a"]) == deps.requirements_hash(["a", "b"])
    assert deps.requirements_hash(["a"]) != deps.requirements_hash(["b"])


# ---- code_runner image cache -------------------------------------------------
def _mock_base_ready(monkeypatch):
    monkeypatch.setenv("AGENT_DYNAMIC_DEPS", "true")
    monkeypatch.setattr(cr, "ensure_sandbox_image", lambda: (True, ""))
    cr._built_images.clear()


def test_ensure_image_reuses_cached(monkeypatch):
    _mock_base_ready(monkeypatch)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")  # inspect hit

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    tag = cr.ensure_image_for(["requests"])
    assert tag.startswith("agent-sandbox:")
    assert any("inspect" in c for c in calls)
    assert not any("build" in c for c in calls)   # reused, never built


def test_ensure_image_builds_when_missing(monkeypatch):
    _mock_base_ready(monkeypatch)
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if "inspect" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="No such image")
        return types.SimpleNamespace(returncode=0, stdout="built", stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    tag = cr.ensure_image_for(["requests"])
    assert tag.startswith("agent-sandbox:")
    assert any("build" in c for c in seen)


def test_ensure_image_fallback_to_base_on_build_fail(monkeypatch):
    _mock_base_ready(monkeypatch)

    def fake_run(cmd, **kw):
        if "inspect" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="pip failed: boom")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    assert cr.ensure_image_for(["requests"]) == cr.DEFAULT_IMAGE


def test_ensure_image_rejects_unsafe_package_name(monkeypatch):
    _mock_base_ready(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("docker must not be invoked for an unsafe package name")

    monkeypatch.setattr(cr.subprocess, "run", boom)
    assert cr.ensure_image_for(["evil; rm -rf /"]) == cr.DEFAULT_IMAGE


def test_run_python_auto_selects_image_from_imports(monkeypatch):
    monkeypatch.setenv("AGENT_DYNAMIC_DEPS", "true")
    captured = {}
    monkeypatch.setattr(cr, "ensure_image_for", lambda pkgs: "img:" + "-".join(pkgs))
    monkeypatch.setattr(cr, "run_python",
                        lambda code, *, timeout=30, image=cr.DEFAULT_IMAGE:
                        (captured.update(image=image), cr.RunResult(True, 0, "", "", 0.1))[1])
    cr.run_python_auto("import requests\nprint(1)")
    assert captured["image"] == "img:requests"
