"""
OpenAI chat-model selection for the web UI.

Lists the OpenAI models the user can pick and switches the active one by updating
both the running process env and the on-disk .env, so the choice persists. OpenAI
is the only chat provider, so there is no provider concept to manage.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_PATH = ROOT / ".env"

# Ensure .env is loaded even if this module is imported before the app wires it up,
# so OPENAI_BASE_URL / OPENAI_MODEL are visible when listing models.
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except Exception:
    pass

DEFAULT_OPENAI_MODEL = "gpt-4o"

# The provider adapts API parameters per model, so GPT-5 / o-series and the older
# gpt-4o/4.1 models all work. Your account may not have every model listed.
OPENAI_MODELS = [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.1",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]


def _base_url() -> str:
    return (os.getenv("OPENAI_BASE_URL") or "").rstrip("/")


def _is_ollama() -> bool:
    b = _base_url().lower()
    return "11434" in b or "ollama" in b


def _label(model: str) -> str:
    return f"{'Ollama' if _is_ollama() else 'OpenAI'} · {model}"


def _local_models() -> List[str]:
    """Models installed on a local OpenAI-compatible server (Ollama). [] otherwise."""
    if not _is_ollama():
        return []
    try:
        req = urllib.request.Request(_base_url() + "/models",
                                     headers={"Authorization": "Bearer ollama"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.load(resp)
        return sorted({m.get("id") for m in (data.get("data") or []) if m.get("id")})
    except Exception:
        return []


def current() -> Dict[str, str]:
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return {"provider": "openai", "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    # On Ollama, list the actually-installed models; otherwise the cloud list.
    models = _local_models() or list(OPENAI_MODELS)
    # Always include the active model and the configured agent model (e.g. a coder).
    for extra in (cur["model"], os.getenv("AGENT_MODEL", "")):
        extra = (extra or "").strip()
        if extra and extra not in models:
            models.append(extra)
    options = [{"provider": "openai", "model": m, "label": _label(m)} for m in models]
    return {"current": cur, "options": options}


def _persist_env(updates: Dict[str, str]) -> None:
    """Write key=value pairs into .env (replace first occurrence or append)."""
    text = ENV_PATH.read_text(encoding="utf-8", errors="ignore") if ENV_PATH.exists() else ""
    lines = text.splitlines()
    for key, value in updates.items():
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                continue
            if re.match(rf"^\s*{re.escape(key)}\s*=", line):
                lines[i] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_model(provider: str, model: str) -> Dict[str, str]:
    """Switch the active OpenAI model. `provider` is accepted for API compatibility
    but ignored — OpenAI is the only chat provider."""
    model = (model or "").strip()
    if not model:
        raise ValueError("Model is required")

    # Update the running process immediately (get_provider reads os.environ with
    # override=False, so this wins) and persist to disk for next time.
    os.environ["OPENAI_MODEL"] = model
    _persist_env({"OPENAI_MODEL": model})
    return {"provider": "openai", "model": model, "label": _label(model)}
