#!/usr/bin/env python
"""
run.py -- launch the Research Assistant web app (FastAPI).

Usage:
    python run.py                # this PC only -> http://localhost:8600 (no prompts)
    python run.py --share        # PUBLIC https URL anyone can open (Cloudflare tunnel)
    python run.py --lan          # reachable by other devices on your network
    python run.py --port 9000     # override the port
    python run.py --no-free-port  # do NOT auto-clear a stale server on the port

A wrapper around uvicorn. Config is read from the local .env. `--share` downloads
the Cloudflare tunnel client once (into data/tools/) and prints a public
`https://…trycloudflare.com` link — no account needed; keep ENABLE_AUTH=true.

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

# Print UTF-8 regardless of the Windows console code page, so the startup banner's
# arrows/symbols (→, ✓, …) never crash with UnicodeEncodeError when stdout is a
# redirected pipe/log (cp1252). `reconfigure` lives on TextIOWrapper at runtime; it's
# fetched via getattr so this stays a safe no-op (and type-clean) when it's absent.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Load .env so --share can read CLOUDFLARE_TUNNEL_TOKEN / ENABLE_AUTH for warnings.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


# ----------------------------------------------------------------------
# Port helpers -- so a leftover server can never block startup again
# ----------------------------------------------------------------------
def _bind_host(lan: bool, share: bool) -> str:
    """The host uvicorn will ACTUALLY bind to. `--lan`/`--share` expose the app on every
    interface (0.0.0.0); otherwise it is localhost-only. The port pre-flight MUST probe
    this same host — otherwise a leftover server on 0.0.0.0 slips past a 127.0.0.1-only
    check (Windows lets a 127.0.0.1 probe bind even while 0.0.0.0 is held) and uvicorn then
    dies with '[Errno 10048] only one usage of each socket address' right after startup."""
    return "0.0.0.0" if (lan or share) else "127.0.0.1"


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if `host:port` can't be bound right now (something is already using it).

    This is a PLAIN bind — exactly what uvicorn/asyncio does on Windows (no SO_REUSEADDR,
    no SO_EXCLUSIVEADDRUSE), so the probe's verdict matches the real listener's. What makes
    a leftover server detectable is probing the SAME host uvicorn will bind (see `_bind_host`
    / `ensure_port_free`): a leftover on 0.0.0.0 is only seen when we probe 0.0.0.0 too."""
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


def ensure_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Make sure `port` is free ON THE HOST uvicorn will bind. If a leftover Python server
    holds it, stop that process. Returns True if the port is (now) free, False if we
    couldn't clear it safely. `host` must match the eventual uvicorn bind (0.0.0.0 for
    --lan/--share) so a wildcard leftover is actually detected."""
    if not port_in_use(port, host):
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
        if not port_in_use(port, host):
            print(f"  Port {port} is free now.")
            return True
        time.sleep(0.25)

    print(f"  Port {port} is still busy after stopping the old server.")
    print(f"  Use another port:  python run.py --port {port + 1}")
    return False


# ----------------------------------------------------------------------
# Sharing helpers (LAN + public Cloudflare quick tunnel)
# ----------------------------------------------------------------------
def _local_ip() -> str:
    """This machine's LAN IP (best effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _cloudflared_url() -> str | None:
    """Direct-download URL for the cloudflared binary on this platform."""
    import platform
    osname = platform.system().lower()
    arch = platform.machine().lower()
    amd64 = arch in ("amd64", "x86_64")
    arm64 = "arm" in arch or "aarch64" in arch
    base = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    if osname == "windows" and amd64:
        return base + "cloudflared-windows-amd64.exe"
    if osname == "linux":
        return base + ("cloudflared-linux-amd64" if amd64 else
                       "cloudflared-linux-arm64" if arm64 else "")
    return None  # macOS: `brew install cloudflared`


def _ensure_cloudflared() -> str | None:
    """Return a path to cloudflared, downloading it (once) into data/tools/ if needed."""
    from shutil import which
    found = which("cloudflared")
    if found:
        return found
    dest = ROOT / "data" / "tools" / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
    if dest.exists():
        return str(dest)
    url = _cloudflared_url()
    if not url:
        print("  Could not auto-download cloudflared for this OS.")
        print("  Install it (macOS: `brew install cloudflared`) and re-run --share.")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("  Downloading the tunnel client (cloudflared, one-time ~50 MB)…")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        if os.name != "nt":
            dest.chmod(0o755)
        return str(dest)
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return None


def _run_with_tunnel(port: int) -> int:
    """Start the server on 0.0.0.0 + a public Cloudflare quick tunnel, print the URL."""
    import re
    uv = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(ROOT))
    for _ in range(60):                      # wait up to ~15s for the server to come up
        if port_in_use(port, "127.0.0.1"):
            break
        time.sleep(0.25)

    cf = _ensure_cloudflared()
    if not cf:
        print(f"\n  Falling back to your network URL:  http://{_local_ip()}:{port}")
        try:
            uv.wait()
        except KeyboardInterrupt:
            uv.terminate()
        return 0

    # Permanent / custom-domain mode: a named tunnel (token from the Cloudflare
    # dashboard, which also maps your custom hostname -> http://localhost:PORT).
    token = (os.getenv("CLOUDFLARE_TUNNEL_TOKEN") or "").strip()
    if token:
        return _run_named_tunnel(cf, token, uv)

    tunnel = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    shown = False
    try:
        # tunnel.stdout can be None (e.g. if pipes were closed); guard against that
        import io
        stream = tunnel.stdout if tunnel.stdout is not None else io.StringIO()
        for line in stream:
            m = re.search(r"https://[a-z0-9.-]+\.trycloudflare\.com", line)
            if m and not shown:
                shown = True
                url = m.group(0)
                print("\n" + "=" * 64)
                print("  PUBLIC URL — share this link:")
                print("     " + url)
                print("  (Anyone with the link can open it. Keep ENABLE_AUTH=true so a")
                print("   login is required. Press Ctrl+C to stop sharing.)")
                print("=" * 64 + "\n")
    except KeyboardInterrupt:
        print("\nStopping the tunnel and server…")
    finally:
        for p in (tunnel, uv):
            try:
                p.terminate()
            except Exception:
                pass
    return 0


def _run_named_tunnel(cf: str, token: str, uv: "subprocess.Popen") -> int:
    """Run a PERMANENT Cloudflare named tunnel (stable URL / custom domain).

    The tunnel + public hostname (e.g. research.yourdomain.com -> http://localhost:PORT)
    are created once in the Cloudflare Zero Trust dashboard; this just runs it with the
    token. The hostname comes from the dashboard — set CLOUDFLARE_TUNNEL_HOSTNAME only
    so we can print it for you.
    """
    host = (os.getenv("CLOUDFLARE_TUNNEL_HOSTNAME") or "").strip()
    host = host.replace("https://", "").replace("http://", "").strip("/")
    print("\n" + "=" * 64)
    print("  PERMANENT URL (Cloudflare named tunnel):")
    print("     https://" + host if host else
          "     your dashboard hostname (set CLOUDFLARE_TUNNEL_HOSTNAME to show it here)")
    print("  Keep ENABLE_AUTH=true so visitors must sign in. Ctrl+C to stop.")
    print("=" * 64 + "\n")
    tunnel = subprocess.Popen([cf, "tunnel", "run", "--token", token], cwd=str(ROOT))
    try:
        tunnel.wait()
    except KeyboardInterrupt:
        print("\nStopping the tunnel and server…")
    finally:
        for p in (tunnel, uv):
            try:
                p.terminate()
            except Exception:
                pass
    return 0


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Research Assistant web app.")
    parser.add_argument("--port", type=int, default=8600, help="Server port (default: 8600).")
    parser.add_argument("--no-free-port", action="store_true",
                        help="Do not auto-stop a leftover server occupying the port.")
    parser.add_argument("--share", action="store_true",
                        help="Expose a PUBLIC https URL via a Cloudflare tunnel (sharable anywhere).")
    parser.add_argument("--lan", action="store_true",
                        help="Bind to your whole network so other devices here can reach it.")
    args = parser.parse_args()

    # The host uvicorn will bind — probe THIS host for a leftover server (0.0.0.0 for
    # --lan/--share), so a stale wildcard server is detected instead of slipping past a
    # localhost-only check and crashing uvicorn with "[Errno 10048]".
    host = _bind_host(args.lan, args.share)

    if not args.no_free_port:
        if not ensure_port_free(args.port, host):
            return 1

    if args.share:
        print("Starting the app and opening a public tunnel…")
        if (os.getenv("ENABLE_AUTH", "") or "").strip().lower() not in ("1", "true", "yes", "on"):
            print("  ⚠ Heads-up: ENABLE_AUTH is not on — anyone with the link could use the app.")
            print("    Set ENABLE_AUTH=true in .env (and EXTERNAL_ALLOW_UNSAFE_URLS=false) before sharing widely.")
        try:
            return _run_with_tunnel(args.port)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0

    # `host` was computed above (and already drove the port pre-flight) — reuse it.
    if args.lan:
        print(f"Starting on your network → http://{_local_ip()}:{args.port}  (other devices on this Wi-Fi)")
        print("  (Windows may ask to allow it through the firewall — choose Allow.)")
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
