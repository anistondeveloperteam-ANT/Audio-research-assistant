"""Tests for run._run_server -- the launcher that fixes the 'starts then immediately stops' bug.

A stray console Ctrl+C right after launch was tearing the freshly started server down. _run_server
isolates uvicorn in its own process group and ignores ONE stray startup Ctrl+C, while a deliberate
Ctrl+C a moment later still stops it. No real server/network is started -- subprocess.Popen is mocked.
"""
import run as runpy


class _FakeProc:
    def __init__(self, wait_sequence):
        self._seq = list(wait_sequence)
        self.stopped = False
        self.killed = False
        self.signals = []

    def wait(self, timeout=None):
        item = self._seq.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item
        return item

    def send_signal(self, sig):
        self.signals.append(sig)

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.killed = True


def test_run_server_isolates_child_from_console_ctrl_c(monkeypatch):
    captured = {}

    def fake_popen(cmd, **kw):
        captured.update(kw)
        return _FakeProc([0])

    monkeypatch.setattr(runpy.subprocess, "Popen", fake_popen)
    assert runpy._run_server(["x"], ".") == 0
    # The child must NOT share the console's Ctrl+C group.
    if runpy.os.name == "nt":
        assert captured.get("creationflags") == runpy.subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert captured.get("start_new_session") is True


def test_run_server_ignores_one_stray_startup_ctrl_c(monkeypatch):
    # First wait() raises KeyboardInterrupt (the stray startup signal, within the grace window) ->
    # ignored; the loop continues and the second wait() returns cleanly. The server is never stopped.
    fake = _FakeProc([KeyboardInterrupt, 0])
    monkeypatch.setattr(runpy.subprocess, "Popen", lambda cmd, **kw: fake)
    assert runpy._run_server(["x"], ".") == 0
    assert not fake.stopped and not fake.killed
    assert not fake.signals          # no shutdown signal was sent for the stray Ctrl+C


def test_run_server_stops_on_ctrl_c_after_grace(monkeypatch):
    # A Ctrl+C AFTER the grace window stops the server cleanly (graceful shutdown signal / terminate).
    fake = _FakeProc([KeyboardInterrupt, 0])
    monkeypatch.setattr(runpy.subprocess, "Popen", lambda cmd, **kw: fake)
    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else runpy.STARTUP_GRACE_SECONDS + 1.0

    monkeypatch.setattr(runpy.time, "monotonic", fake_monotonic)
    assert runpy._run_server(["x"], ".") == 0
    # On Windows a CTRL_BREAK_EVENT is sent; elsewhere terminate() is called.
    assert fake.signals or fake.stopped
