"""
_chat_ui_utils.py  --  Batch 8B (Phase 2)

Non-Streamlit helpers used by chat_ui.py:
  - PDF hashing + dedup against existing library
  - .env updates (for the model switcher)
  - Ollama model listing
  - Subprocess-based ingestion with line-streaming progress

These are pure functions so they can be smoke-tested without Streamlit
or Oracle or Ollama running.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Hashing + dedup
# ----------------------------------------------------------------------

def compute_file_hash(data: bytes) -> str:
    """Stable SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def list_existing_pdf_hashes(papers_dir: Path) -> Dict[str, str]:
    """Return {sha256_hex: filename} for every PDF currently in papers_dir.
    Skips unreadable files. Empty dict if the directory doesn't exist."""
    result: Dict[str, str] = {}
    if not papers_dir.exists() or not papers_dir.is_dir():
        return result
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        try:
            data = pdf_path.read_bytes()
        except Exception:
            continue
        result[compute_file_hash(data)] = pdf_path.name
    return result


def safe_pdf_target(papers_dir: Path, original_name: str) -> Path:
    """Pick a non-colliding path inside papers_dir for a new upload.
    If `original_name` already exists, appends _1, _2, ... to the stem."""
    papers_dir.mkdir(parents=True, exist_ok=True)
    target = papers_dir / original_name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for i in range(1, 1000):
        cand = papers_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find a free filename for {original_name}")


# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------

def parse_ollama_tags_response(payload: dict) -> List[str]:
    """Pull model names out of /api/tags response. Safe against missing keys."""
    if not isinstance(payload, dict):
        return []
    out = []
    for m in payload.get("models", []) or []:
        if isinstance(m, dict):
            name = m.get("name") or ""
            if name:
                out.append(name)
    return out


def list_ollama_models(host: str = "http://localhost:11434", timeout: float = 2.0) -> List[str]:
    """Live-query Ollama. Returns [] if the service isn't reachable."""
    try:
        import requests
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return []
        return parse_ollama_tags_response(r.json())
    except Exception:
        return []


# ----------------------------------------------------------------------
# .env updates  (used by the model switcher)
# ----------------------------------------------------------------------

def patch_env_text(env_text: str, key: str, value: str) -> str:
    """Return new .env content with key=value. Replaces the first
    uncommented occurrence; if absent, appends. Other lines untouched."""
    lines = env_text.splitlines()
    new_lines = []
    found = False
    for line in lines:
        # Don't touch comment lines
        if line.strip().startswith("#"):
            new_lines.append(line)
            continue
        if not found and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")
    out = "\n".join(new_lines)
    if not out.endswith("\n"):
        out += "\n"
    return out


def update_env_var(env_path: Path, key: str, value: str) -> None:
    """Persist a single key=value into .env. Creates a rolling backup
    at .env.uiupdate.bak so the last-good state is always recoverable."""
    if env_path.exists():
        backup = env_path.with_name(".env.uiupdate.bak")
        shutil.copy2(env_path, backup)
        env_text = env_path.read_text(encoding="utf-8", errors="ignore")
    else:
        env_text = ""
    new_text = patch_env_text(env_text, key, value)
    env_path.write_text(new_text, encoding="utf-8")


# ----------------------------------------------------------------------
# Ingestion pipeline  (subprocess + line streaming)
# ----------------------------------------------------------------------

INGEST_MODULE = "backend.ingestion.ingest_papers"
EMBED_MODULE = "backend.ingestion.embed_chunks"


def _module_file(project_root: Path, module: str) -> Optional[Path]:
    """Resolve a dotted module to its .py file under project_root (or None)."""
    p = project_root / (module.replace(".", "/") + ".py")
    return p if p.exists() else None


def _stream_subprocess(
    module: str,
    cwd: Path,
    on_line: Optional[Callable[[str], None]] = None,
) -> Tuple[int, List[str]]:
    """Run `python -m <module>` from cwd; stream stdout+stderr.
    Returns (return_code, captured_lines)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=None,  # inherit
    )
    lines: List[str] = []
    if proc.stdout is not None:
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip()
            lines.append(line)
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:
                    pass
        proc.stdout.close()
    code = proc.wait()
    return code, lines


def run_ingestion(
    project_root: Path,
    on_line: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str, List[str]]:
    """Run the ingest then embed stages as `python -m`, streaming lines.
    Returns (return_code, summary_message, all_lines).
    summary_message describes which step failed (or success)."""
    if _module_file(project_root, INGEST_MODULE) is None:
        return 1, "Could not find backend/ingestion/ingest_papers.py", []
    if _module_file(project_root, EMBED_MODULE) is None:
        return 1, "Could not find backend/ingestion/embed_chunks.py", []

    ingest_name = INGEST_MODULE.rsplit(".", 1)[-1]
    if on_line:
        on_line(f"--- Running {ingest_name} ---")
    code, ingest_lines = _stream_subprocess(INGEST_MODULE, project_root, on_line)
    if code != 0:
        return code, f"{ingest_name} exited with code {code}", ingest_lines

    embed_name = EMBED_MODULE.rsplit(".", 1)[-1]
    if on_line:
        on_line(f"--- Running {embed_name} ---")
    code, embed_lines = _stream_subprocess(EMBED_MODULE, project_root, on_line)
    if code != 0:
        return code, f"{embed_name} exited with code {code}", ingest_lines + embed_lines

    return 0, "Ingestion + embedding completed", ingest_lines + embed_lines
