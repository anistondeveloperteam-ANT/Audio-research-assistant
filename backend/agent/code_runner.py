"""
Sandboxed code execution for the research agent.

AI-generated Python is run inside a **throwaway Docker container** with:
  - no network            (--network none)
  - capped memory + CPU   (--memory, --cpus)
  - a process-count cap   (--pids-limit)
  - a wall-clock timeout   (kills the container)
  - no host filesystem     (the code is piped in on stdin; nothing is mounted)
  - automatic removal      (--rm)

Nothing the generated code does can touch the host. If Docker is unavailable the
runner returns a clear error instead of falling back to unsafe local execution.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

# Tunable via .env (sensible, safe defaults).
DEFAULT_IMAGE = os.getenv("AGENT_DOCKER_IMAGE", "python:3.11-slim")
RUN_TIMEOUT = int(os.getenv("AGENT_RUN_TIMEOUT", "30"))      # seconds the code may run
MEM_LIMIT = os.getenv("AGENT_MEM_LIMIT", "512m")
CPU_LIMIT = os.getenv("AGENT_CPUS", "1.0")
PIDS_LIMIT = os.getenv("AGENT_PIDS_LIMIT", "128")
OUTPUT_CAP = 20_000   # chars of stdout/stderr kept


@dataclass
class RunResult:
    ok: bool            # exited 0, no timeout, no harness error
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    error: str = ""     # harness-level problem (docker missing / timeout / pull fail)

    @property
    def summary(self) -> str:
        if self.error:
            return f"DID NOT RUN: {self.error}"
        status = "OK" if self.ok else f"FAILED (exit {self.exit_code})"
        return f"{status} in {self.seconds:.1f}s"


_docker_ok: bool | None = None


def docker_available() -> bool:
    """True if the Docker CLI exists and the daemon answers. Cached per process."""
    global _docker_ok
    if _docker_ok is not None:
        return _docker_ok
    _docker_ok = False
    if shutil.which("docker"):
        try:
            r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                               capture_output=True, text=True, timeout=15)
            _docker_ok = r.returncode == 0 and bool(r.stdout.strip())
        except Exception:
            _docker_ok = False
    return _docker_ok


def _cap(text: str) -> str:
    if text and len(text) > OUTPUT_CAP:
        return text[:OUTPUT_CAP] + f"\n... [truncated, {len(text) - OUTPUT_CAP} more chars]"
    return text or ""


def run_python(code: str, *, timeout: int = RUN_TIMEOUT, image: str = DEFAULT_IMAGE) -> RunResult:
    """Run `code` as a Python script inside a locked-down Docker container.

    The code is fed to `python` on stdin, so no file is written to the host and no
    directory is mounted into the container.
    """
    if not docker_available():
        return RunResult(False, -1, "", "", 0.0,
                         "Docker is not available. Start Docker Desktop and try again.")

    cmd = [
        "docker", "run", "--rm", "-i",
        "--network", "none",
        "--memory", MEM_LIMIT,
        "--cpus", str(CPU_LIMIT),
        "--pids-limit", str(PIDS_LIMIT),
        image, "python", "-",   # read the script from stdin
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, input=code, capture_output=True, text=True,
            timeout=timeout + 20,   # grace for container startup/pull
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, -1, _cap(exc.stdout or ""), _cap(exc.stderr or ""),
                         float(timeout), f"timed out after ~{timeout}s")
    except Exception as exc:
        return RunResult(False, -1, "", "", time.time() - start, f"could not start container: {exc}")

    secs = time.time() - start
    # A failed image pull surfaces on stderr with a non-zero exit before any Python runs.
    if proc.returncode != 0 and "Unable to find image" in (proc.stderr or "") \
            and "pull access denied" in (proc.stderr or "").lower():
        return RunResult(False, proc.returncode, "", _cap(proc.stderr), secs,
                         f"could not pull image {image!r}")
    return RunResult(proc.returncode == 0, proc.returncode,
                     _cap(proc.stdout), _cap(proc.stderr), secs)
