#!/usr/bin/env python
"""
run.py -- launch the Audio Research Assistant web app (FastAPI).

Usage:
    python run.py                # SHARED on your Wi-Fi/LAN -> http://<your-ip>:8600
    python run.py --local        # this PC only -> http://localhost:8600
    python run.py --port 9000     # override the port
    python run.py --host 0.0.0.0  # bind a specific interface (advanced)
    python run.py --no-free-port  # do NOT auto-clear a stale server on the port

A thin wrapper around uvicorn. Config is read from the local .env.

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
def lan_ips() -> list[str]:
    """Best-effort list of this machine's LAN IPv4 addresses (no loopback)."""
    ips: set[str] = set()
    # The address used to reach the outside world is almost always the LAN IP.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254."))


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


def _firewall_rule_name(port: int) -> str:
    return f"Audio Research Assistant {port}"


def firewall_rule_exists(port: int) -> bool:
    if os.name != "nt":
        return True
    try:
        out = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={_firewall_rule_name(port)}"],
            capture_output=True, text=True,
        ).stdout
        return bool(out) and "No rules match" not in out
    except Exception:
        return False


def ensure_firewall_rule(port: int) -> None:
    """On Windows, make sure inbound TCP `port` is allowed so teammates can
    connect. If the rule is missing, ask Windows for permission (one UAC
    prompt) and add it. Non-fatal: if the user declines, we print how to do it
    by hand and carry on."""
    if os.name != "nt" or firewall_rule_exists(port):
        return

    name = _firewall_rule_name(port)
    print(f"  Opening Windows Firewall for port {port} (approve the prompt that pops up)...")
    # Run netsh elevated. -ArgumentList as an array lets PowerShell quote the
    # rule name (which contains spaces) correctly.
    arg_array = (
        "@('advfirewall','firewall','add','rule',"
        f"'name={name}','dir=in','action=allow','protocol=TCP',"
        f"'localport={port}','profile=any')"
    )
    ps = f"Start-Process netsh -Verb RunAs -WindowStyle Hidden -Wait -ArgumentList {arg_array}"
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=60)
    except Exception:
        pass

    if firewall_rule_exists(port):
        print(f"  Firewall opened for port {port}.  [OK]")
    else:
        print("  Could not add the firewall rule automatically. To allow it once,")
        print("  open PowerShell as Administrator and run:")
        print(f'    netsh advfirewall firewall add rule name="{name}" '
              f'dir=in action=allow protocol=TCP localport={port} profile=any')


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Audio Research Assistant web app.")
    parser.add_argument("--port", type=int, default=8600, help="Server port (default: 8600).")
    parser.add_argument("--local", action="store_true",
                        help="Restrict to THIS PC only (no network sharing).")
    parser.add_argument("--share", action="store_true", help=argparse.SUPPRESS)  # sharing is the default now
    parser.add_argument("--host", default=None,
                        help="Interface to bind (advanced). Default 0.0.0.0 (shared), or 127.0.0.1 with --local.")
    parser.add_argument("--no-free-port", action="store_true",
                        help="Do not auto-stop a leftover server occupying the port.")
    args = parser.parse_args()

    # Sharing is the DEFAULT: bind all interfaces so teammates on the same
    # network can reach it. Use --local to keep it to this PC only.
    if args.host:
        host = args.host
    elif args.local:
        host = "127.0.0.1"
    else:
        host = "0.0.0.0"
    shared = host not in ("127.0.0.1", "localhost")

    if not args.no_free_port:
        if not ensure_port_free(args.port):
            return 1

    if shared:
        ensure_firewall_rule(args.port)
        ips = lan_ips()
        print("=" * 62)
        print(" Audio Research Assistant is SHARED on your network.")
        print(" Your teammate (on the SAME Wi-Fi/LAN) opens this in a browser:")
        if ips:
            for ip in ips:
                print(f"     ->  http://{ip}:{args.port}")
        else:
            print("     (couldn't detect your LAN IP -- run 'ipconfig', use the IPv4 address)")
        print(f" On this PC you can also use:  http://localhost:{args.port}")
        print(" No password: anyone on the network with the URL can use it, and")
        print(" questions run on YOUR models/keys. Keep this window open; Ctrl+C to stop.")
        print(" Want it private?  ->  python run.py --local")
        print("=" * 62)
    else:
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
