"""
run.py port pre-flight — make `python run.py` never die with the cryptic Windows
"[Errno 10048] only one usage of each socket address" + a silent "Application shutdown".

The bug: `--lan`/`--share` bind uvicorn to 0.0.0.0 (all interfaces), but the "free the
port" pre-flight probed only 127.0.0.1. On Windows' weak address binding a 127.0.0.1
probe SUCCEEDS even while a leftover server holds 0.0.0.0:PORT, so the stale server is
missed and uvicorn then fails to bind 0.0.0.0. Fix: probe the SAME host uvicorn will
bind, so a wildcard leftover is detected (and a stale Python server is auto-stopped).

Everything here is deterministic and cross-platform: it uses real ephemeral sockets for
the bind checks and monkeypatches the OS process lookups for the kill/refuse paths.
"""
import os
import socket
import sys

import pytest

import run


# ---- the host uvicorn actually binds must drive the pre-flight probe ----
def test_bind_host_is_localhost_by_default():
    assert run._bind_host(lan=False, share=False) == "127.0.0.1"


def test_bind_host_is_wildcard_for_lan():
    assert run._bind_host(lan=True, share=False) == "0.0.0.0"


def test_bind_host_is_wildcard_for_share():
    assert run._bind_host(lan=False, share=True) == "0.0.0.0"


# ---- probing a genuinely free port reports it free ----
def test_port_in_use_false_for_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert run.port_in_use(port, "127.0.0.1") is False


# ---- THE regression: a leftover server on 0.0.0.0:PORT is detected when we probe 0.0.0.0 ----
def test_port_in_use_detects_wildcard_leftover():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("0.0.0.0", 0))         # simulate a previous `--lan` server on all interfaces
    port = listener.getsockname()[1]
    listener.listen(1)
    try:
        # Probing the SAME host uvicorn would bind (0.0.0.0) must see the conflict.
        assert run.port_in_use(port, "0.0.0.0") is True
    finally:
        listener.close()


# ---- ensure_port_free is host-aware ----
def test_ensure_port_free_true_when_free():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert run.ensure_port_free(port, "0.0.0.0") is True


def test_ensure_port_free_kills_stale_python_server(monkeypatch):
    """A leftover *Python* server squatting on the port is stopped, then the port is free —
    and the wildcard host is threaded into BOTH the initial probe and the post-kill re-probe."""
    states = iter([True, False])           # busy on first probe, free after the kill
    seen = []

    def fake_port_in_use(port, host="127.0.0.1"):
        seen.append(host)
        return next(states)

    monkeypatch.setattr(run, "port_in_use", fake_port_in_use)
    monkeypatch.setattr(run, "_pids_on_port", lambda port: {4321})
    monkeypatch.setattr(run, "_proc_name", lambda pid: "python.exe")
    killed = []
    monkeypatch.setattr(run, "_kill", lambda pid: killed.append(pid))

    assert run.ensure_port_free(8600, "0.0.0.0") is True
    assert killed == [4321]
    assert seen == ["0.0.0.0", "0.0.0.0"]  # same host uvicorn binds, both probes


def test_ensure_port_free_refuses_to_kill_non_python(monkeypatch):
    """A non-Python owner is never killed — we fail safe and tell the user to pick a port."""
    monkeypatch.setattr(run, "port_in_use", lambda port, host="127.0.0.1": True)
    monkeypatch.setattr(run, "_pids_on_port", lambda port: {777})
    monkeypatch.setattr(run, "_proc_name", lambda pid: "nginx.exe")
    killed = []
    monkeypatch.setattr(run, "_kill", lambda pid: killed.append(pid))

    assert run.ensure_port_free(8600, "0.0.0.0") is False
    assert killed == []                    # left the foreign process untouched


def test_ensure_port_free_refuses_when_owner_unidentified(monkeypatch):
    """Busy but no PID found (netstat couldn't attribute it) → never kill blindly; fail safe."""
    monkeypatch.setattr(run, "port_in_use", lambda port, host="127.0.0.1": True)
    monkeypatch.setattr(run, "_pids_on_port", lambda port: set())
    killed = []
    monkeypatch.setattr(run, "_kill", lambda pid: killed.append(pid))

    assert run.ensure_port_free(8600, "0.0.0.0") is False
    assert killed == []


def test_ensure_port_free_gives_up_when_kill_does_not_free_port(monkeypatch):
    """If the OS never releases the port after the kill, give up cleanly (don't hang/loop forever)."""
    monkeypatch.setattr(run.time, "sleep", lambda s: None)            # skip the ~6s real wait
    monkeypatch.setattr(run, "port_in_use", lambda port, host="127.0.0.1": True)  # stays busy
    monkeypatch.setattr(run, "_pids_on_port", lambda port: {4321})
    monkeypatch.setattr(run, "_proc_name", lambda pid: "python.exe")
    monkeypatch.setattr(run, "_kill", lambda pid: None)

    assert run.ensure_port_free(8600, "0.0.0.0") is False


# ---- the probe must NOT be stricter than uvicorn: a 127.0.0.1-only leftover must read FREE ----
@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows weak binding: 0.0.0.0 can bind beside a 127.0.0.1-only listener (POSIX rejects it)")
def test_port_in_use_matches_uvicorn_for_localhost_only_leftover():
    """A leftover bound only to 127.0.0.1 does NOT stop uvicorn binding 0.0.0.0 on Windows, so
    the probe must agree (return False). Guards against re-adding SO_EXCLUSIVEADDRUSE, which would
    falsely report busy and get a non-conflicting Python server killed."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.listen(1)
    try:
        assert run.port_in_use(port, "0.0.0.0") is False
    finally:
        listener.close()


# ---- main() wires the bind host into BOTH the pre-flight and uvicorn (the fix's integration point) ----
def _stub_launch(monkeypatch):
    """Stop main() before it really spawns uvicorn; capture the pre-flight host + uvicorn argv."""
    captured = {}
    monkeypatch.setattr(run, "ensure_port_free",
                        lambda port, host="127.0.0.1": captured.__setitem__("preflight_host", host) or True)
    monkeypatch.setattr(run.subprocess, "call",
                        lambda cmd, cwd=None: captured.__setitem__("cmd", cmd) or 0)
    monkeypatch.setattr(run, "_local_ip", lambda: "192.168.1.23")
    return captured


def test_main_threads_wildcard_host_for_lan(monkeypatch):
    captured = _stub_launch(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py", "--lan"])
    assert run.main() == 0
    assert captured["preflight_host"] == "0.0.0.0"
    cmd = captured["cmd"]
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"   # SAME host the pre-flight probed


def test_main_threads_localhost_host_by_default(monkeypatch):
    captured = _stub_launch(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    assert run.main() == 0
    assert captured["preflight_host"] == "127.0.0.1"
    cmd = captured["cmd"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_main_returns_1_and_never_launches_when_port_unfree(monkeypatch):
    """Known-unclearable port → fail BEFORE uvicorn, so the user never sees the confusing
    'startup complete' immediately followed by 'application shutdown'."""
    launched = []
    monkeypatch.setattr(run, "ensure_port_free", lambda port, host="127.0.0.1": False)
    monkeypatch.setattr(run.subprocess, "call", lambda cmd, cwd=None: launched.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["run.py", "--lan"])

    assert run.main() == 1
    assert launched == []
