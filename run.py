#!/usr/bin/env python
"""
run.py -- launch the Audio Research Assistant web app (FastAPI).

Usage:
    python run.py                # this PC only -> http://localhost:8600 (no prompts)
    python run.py --port 9000     # override the port
    python run.py --no-free-port  # do NOT auto-clear a stale server on the port

A local-only wrapper around uvicorn. Config is read from the local .env.

If a previous run is still holding the port (the classic
"[Errno 10048] only one usage of each socket address" on Windows), this script
detects it and stops that leftover server automatically before starting -- so
you never have to hunt down the process by hand. It only ever stops a *Python*
process that is squatting on the port; anything else is left alone with a clear
message.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Port helpers -- so a leftover server can never block startup again
# ----------------------------------------------------------------------
def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _pids_on_port(port: int) -> set[int]:
    """PIDs listening on the given TCP port (Windows netstat / POSIX lsof)."""
    pids: set[int] = set()
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True, text=True,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if (len(parts) >= 5 and parts[0].upper() == "TCP"
                        and parts[3].upper() == "LISTENING"
                        and parts[1].endswith(f":{port}")):
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        else:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True,
            ).stdout
            for tok in out.split():
                try:
                    pids.add(int(tok))
                except ValueError:
                    pass
    except Exception:
        pass
    return pids


def _proc_name(pid: int) -> str:
    """Best-effort process image name for a PID (lowercased)."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out and "," in out:
                return out.splitlines()[0].split(",")[0].strip().strip('"').lower()
        else:
            return subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True, text=True,
            ).stdout.strip().lower()
    except Exception:
        pass
    return ""


def _kill(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, text=True)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def ensure_port_free(port: int) -> bool:
    """Make sure `port` is free. If a leftover Python server holds it, stop that
    process. Returns True if the port is (now) free, False if we couldn't clear
    it safely."""
    if not port_in_use(port):
        return True

    pids = _pids_on_port(port)
    if not pids:
        print(f"  Port {port} is in use, but its owner couldn't be identified.")
        print(f"  Try a different port:  python run.py --port {port + 1}")
        return False

    named = {p: (_proc_name(p) or "?") for p in pids}
    ours = [p for p, n in named.items() if "python" in n]
    others = [p for p in pids if p not in ours]

    if others:
        listing = ", ".join(f"PID {p} ({named[p]})" for p in others)
        print(f"  Port {port} is held by a non-Python process: {listing}.")
        print(f"  Not touching it. Use another port:  python run.py --port {port + 1}")
        return False

    for p in ours:
        print(f"  Port {port} was held by a leftover server (PID {p}); stopping it...")
        _kill(p)

    for _ in range(24):                      # wait up to ~6s for the OS to release it
        if not port_in_use(port):
            print(f"  Port {port} is free now.")
            return True
        time.sleep(0.25)

    print(f"  Port {port} is still busy after stopping the old server.")
    print(f"  Use another port:  python run.py --port {port + 1}")
    return False


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Audio Research Assistant web app.")
    parser.add_argument("--port", type=int, default=8600, help="Server port (default: 8600).")
    parser.add_argument("--no-free-port", action="store_true",
                        help="Do not auto-stop a leftover server occupying the port.")
    args = parser.parse_args()

    host = "127.0.0.1"

    if not args.no_free_port:
        if not ensure_port_free(args.port):
            return 1

    print(f"Starting web UI (this PC only) on http://localhost:{args.port}  (Ctrl+C to stop)")

    cmd = [sys.executable, "-m", "uvicorn", "webapp.server:app",
           "--host", host, "--port", str(args.port)]
    # Run from the project root so `import backend.*` / `import webapp.*` resolve.
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
