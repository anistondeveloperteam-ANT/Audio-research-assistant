"""Ingest subprocesses must emit UTF-8 logs. A PDF skip prints a '⚠' warning; on a Windows cp1252
pipe that raised UnicodeEncodeError mid-print and aborted the WHOLE ingest. The spawners now force
PYTHONIOENCODING=utf-8, and the parse module reconfigures its own stdio as a backstop.
"""
import os
import subprocess
import sys

import backend.ingestion.ingest_papers as ip
import webapp.ingest as ingest

UNICODE_LOGS = "parse ⚠ → done …"          # the ⚠ → … chars that crashed


def test_subprocess_env_forces_utf8():
    assert ingest._subprocess_env()["PYTHONIOENCODING"] == "utf-8"


def test_subprocess_env_inherits_os_environ(monkeypatch):
    monkeypatch.setenv("ARA_TEST_KEY", "value")
    assert ingest._subprocess_env().get("ARA_TEST_KEY") == "value"


def test_child_emits_unicode_with_utf8_env():
    """With UTF-8 forced, a child printing ⚠ → … exits cleanly and the parent reads the chars."""
    r = subprocess.run(
        [sys.executable, "-c", f"print({UNICODE_LOGS!r})"],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    assert "⚠" in r.stdout and "→" in r.stdout


def test_child_crashes_under_cp1252_proving_the_bug():
    """Control: the same print under cp1252 raises UnicodeEncodeError (non-zero exit) — the original
    failure. This is exactly what the UTF-8 env/reconfigure prevents."""
    r = subprocess.run(
        [sys.executable, "-c", "print('⚠')"],
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert r.returncode != 0
    assert "UnicodeEncodeError" in (r.stderr or "")


def test_force_utf8_stdio_reconfigures_streams(monkeypatch):
    calls = []

    class FakeStream:
        def reconfigure(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())
    ip._force_utf8_stdio()
    assert calls == [{"encoding": "utf-8", "errors": "replace"}] * 2


def test_force_utf8_stdio_is_safe_when_reconfigure_missing(monkeypatch):
    # Streams without .reconfigure (e.g. a plain buffer) must not raise.
    monkeypatch.setattr(sys, "stdout", object())
    monkeypatch.setattr(sys, "stderr", object())
    ip._force_utf8_stdio()                                # no exception == pass
