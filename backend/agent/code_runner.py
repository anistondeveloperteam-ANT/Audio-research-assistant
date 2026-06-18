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
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

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

# Concurrency cap: when the agent runs best-of-N candidates in parallel, each one
# needs its own container. This bounds how many sandboxes run AT ONCE (extra runs
# queue, they don't fail) so we never overwhelm the Docker daemon. It only LIMITS
# concurrency — it never weakens a container's own limits (network/cpu/mem/pids/
# timeout/--rm are unchanged). Read live so .env / tests take effect.
_sandbox_sem: "threading.BoundedSemaphore | None" = None
_sandbox_sem_size: int = 0
_sandbox_sem_lock = threading.Lock()


def max_concurrent_sandboxes() -> int:
    try:
        return max(1, int(os.getenv("AGENT_MAX_CONCURRENT_SANDBOXES", "4")))
    except (TypeError, ValueError):
        return 4


def _sandbox_semaphore() -> "threading.BoundedSemaphore":
    """A process-wide bounded semaphore sized to AGENT_MAX_CONCURRENT_SANDBOXES,
    rebuilt if the configured size changes (e.g. in tests)."""
    global _sandbox_sem, _sandbox_sem_size
    size = max_concurrent_sandboxes()
    with _sandbox_sem_lock:
        if _sandbox_sem is None or size != _sandbox_sem_size:
            _sandbox_sem = threading.BoundedSemaphore(size)
            _sandbox_sem_size = size
        return _sandbox_sem


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


def clip_keep_ends(text: str, limit: int = OUTPUT_CAP) -> str:
    """Bound `text` to ~`limit` chars but keep BOTH ENDS — a small head for early context and a
    LARGER TAIL, with the middle elided. Requested results are typically printed LAST (after any
    intermediate dump), so head-only truncation drops exactly the values we must preserve; keeping
    the tail means a final labelled result block always survives. Finite by construction."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = max(0, limit // 3)            # favour the tail: the requested result is usually last
    tail = limit - head
    elided = len(text) - head - tail
    return text[:head] + f"\n... [truncated, {elided} chars elided] ...\n" + text[-tail:]


def _cap(text: str) -> str:
    return clip_keep_ends(text or "", OUTPUT_CAP)


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


def dynamic_deps_enabled() -> bool:
    return os.getenv("AGENT_DYNAMIC_DEPS", "true").strip().lower() == "true"


_SAFE_PKG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_built_images: dict = {}   # hash -> tag, in-process memo so repeat domains skip the inspect call


def ensure_image_for(packages: List[str]) -> str:
    """Return a sandbox image tag that has `packages` installed on top of the base scientific
    image. Reuses a cached `agent-sandbox:<hash>` image when present; otherwise builds one —
    network is allowed ONLY during this build step (the eventual `docker run` stays --network
    none). On ANY problem (bad package name, base image missing, build failure) it returns the
    base image, so a missing import surfaces as an ImportError that feeds the agent's REFLECT
    step rather than hard-failing."""
    packages = sorted({p for p in (packages or []) if _SAFE_PKG.match(p)})
    if not packages or not dynamic_deps_enabled():
        return DEFAULT_IMAGE

    from backend.agent.deps import requirements_hash
    digest = requirements_hash(packages)
    if digest in _built_images:
        return _built_images[digest]

    base_ready, _ = ensure_sandbox_image()
    if not base_ready:
        return DEFAULT_IMAGE

    tag = f"agent-sandbox:{digest}"
    try:
        r = subprocess.run(["docker", "image", "inspect", tag],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            _built_images[digest] = tag
            return tag
    except Exception:
        return DEFAULT_IMAGE

    # Build FROM the base image; install as root, then drop back to the non-root sandbox user
    # so the RUN limits and non-root execution are preserved.
    dockerfile = (
        f"FROM {SANDBOX_TAG}\n"
        f"USER root\n"
        f"RUN pip install --no-cache-dir {' '.join(packages)}\n"
        f"USER sandbox\n"
    )
    try:
        r = subprocess.run(["docker", "build", "-t", tag, "-"], input=dockerfile,
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        if r.returncode == 0:
            _built_images[digest] = tag
            return tag
    except Exception:
        pass
    return DEFAULT_IMAGE   # build failed -> base image; the ImportError feeds REFLECT


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
    # Bound how many containers run concurrently (best-of-N). The semaphore is held
    # ONLY for the duration of the container run; extra candidates queue here.
    sem = _sandbox_semaphore()
    sem.acquire()
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
    finally:
        sem.release()

    secs = time.time() - start
    # A failed image pull surfaces on stderr with a non-zero exit before any Python runs.
    if proc.returncode != 0 and "Unable to find image" in (proc.stderr or "") \
            and "pull access denied" in (proc.stderr or "").lower():
        return RunResult(False, proc.returncode, "", _cap(proc.stderr), secs,
                         f"could not pull image {image!r}")
    return RunResult(proc.returncode == 0, proc.returncode,
                     _cap(proc.stdout), _cap(proc.stderr), secs)


def run_python_auto(code: str, *, timeout: int = RUN_TIMEOUT) -> RunResult:
    """Run `code` in a sandbox image that has its third-party imports installed (built/cached by
    hash when AGENT_DYNAMIC_DEPS is on; network only during build). Runtime limits are unchanged:
    the actual run still uses --network none + memory/CPU/PID caps + timeout + non-root + --rm.
    Falls back to the base image if deps can't be resolved/built — the ImportError then drives the
    agent's REFLECT step to rewrite using available libraries."""
    image = DEFAULT_IMAGE
    if dynamic_deps_enabled():
        try:
            from backend.agent.deps import requirements_for
            pkgs = requirements_for(code)
            if pkgs:
                image = ensure_image_for(pkgs)
        except Exception:
            image = DEFAULT_IMAGE
    return run_python(code, timeout=timeout, image=image)
