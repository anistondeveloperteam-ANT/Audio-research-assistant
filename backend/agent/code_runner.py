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
from pathlib import Path

# Our sandbox image bakes in the scientific stack (numpy/scipy/...) so generated
# simulations run under --network none. Built once on first use (see
# ensure_sandbox_image). Override with AGENT_DOCKER_IMAGE to pin your own image.
SANDBOX_TAG = "audio-research-sandbox:latest"
_SANDBOX_DOCKERFILE = Path(__file__).resolve().parent / "sandbox.Dockerfile"

# Tunable via .env (sensible, safe defaults).
DEFAULT_IMAGE = os.getenv("AGENT_DOCKER_IMAGE", SANDBOX_TAG)
RUN_TIMEOUT = int(os.getenv("AGENT_RUN_TIMEOUT", "30"))      # seconds the code may run
MEM_LIMIT = os.getenv("AGENT_MEM_LIMIT", "512m")
CPU_LIMIT = os.getenv("AGENT_CPUS", "1.0")
PIDS_LIMIT = os.getenv("AGENT_PIDS_LIMIT", "128")
BUILD_TIMEOUT = int(os.getenv("AGENT_BUILD_TIMEOUT", "1200"))  # first-run image build
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


_image_ready: bool | None = None


def ensure_sandbox_image() -> tuple[bool, str]:
    """Make sure the scientific sandbox image exists, building it once if needed.

    No-op when the user pinned their own AGENT_DOCKER_IMAGE (we trust it). The
    build needs network, but the containers that run user code never do.
    Returns (ready, error_message). Cached per process once ready.
    """
    global _image_ready
    if _image_ready:
        return True, ""
    if os.getenv("AGENT_DOCKER_IMAGE"):          # user pinned an image — trust it
        _image_ready = True
        return True, ""
    try:                                          # already built?
        r = subprocess.run(["docker", "image", "inspect", SANDBOX_TAG],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            _image_ready = True
            return True, ""
    except Exception:
        pass
    if not _SANDBOX_DOCKERFILE.exists():
        return False, "sandbox Dockerfile is missing"
    try:                                          # build it (one-time, then cached)
        r = subprocess.run(
            ["docker", "build", "-t", SANDBOX_TAG, "-f", str(_SANDBOX_DOCKERFILE),
             str(_SANDBOX_DOCKERFILE.parent)],
            capture_output=True, text=True, timeout=BUILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, "building the sandbox image timed out (first run only)"
    except Exception as exc:
        return False, f"could not build the sandbox image: {exc}"
    if r.returncode != 0:
        return False, f"sandbox image build failed: {_cap(r.stderr)[-600:]}"
    _image_ready = True
    return True, ""


def run_python(code: str, *, timeout: int = RUN_TIMEOUT, image: str = DEFAULT_IMAGE) -> RunResult:
    """Run `code` as a Python script inside a locked-down Docker container.

    The code is fed to `python` on stdin, so no file is written to the host and no
    directory is mounted into the container.
    """
    if not docker_available():
        return RunResult(False, -1, "", "", 0.0,
                         "Docker is not available. Start Docker Desktop and try again.")

    if image == SANDBOX_TAG:   # ensure our scientific image is built (numpy/scipy/...)
        ready, err = ensure_sandbox_image()
        if not ready:
            return RunResult(False, -1, "", "", 0.0, err)

    cmd = [
        "docker", "run", "--rm", "-i",
        "--network", "none",
        "--memory", MEM_LIMIT,
        "--cpus", str(CPU_LIMIT),
        "--pids-limit", str(PIDS_LIMIT),
        # Force UTF-8 inside the container so code that uses Unicode (λ, ∇, π, …) can be read
        # from stdin and printed to stdout regardless of the container's locale.
        "-e", "PYTHONUTF8=1",
        "-e", "PYTHONIOENCODING=utf-8",
        image, "python", "-",   # read the script from stdin
    ]
    start = time.time()
    try:
        # encoding="utf-8" is REQUIRED: the generated code often contains Unicode math symbols,
        # and without it Windows would encode stdin with cp1252 and crash ('charmap' codec can't
        # encode character). errors="replace" keeps any odd output byte from killing the capture.
        proc = subprocess.run(
            cmd, input=code, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
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
