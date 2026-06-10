"""
Chat-model selection for the web UI.

Two models only, routed by name:
  - `gemini-2.5-flash` -> Google Gemini (GEMINI_API_KEY)
  - `gpt-5.5`          -> OpenAI         (OPENAI_CLOUD_KEY)

Switching a model updates OPENAI_MODEL + OPENAI_BASE_URL + OPENAI_API_KEY in both the
running process and the on-disk .env, so the whole connection switches — not just the
model string.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_PATH = ROOT / ".env"

# Ensure .env is loaded even if this module is imported first.
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except Exception:
    pass

DEFAULT_OPENAI_MODEL = "gemini-2.5-flash"

# Single source of truth for model -> (endpoint, key) routing, shared with the
# provider factory so the dropdown and the code agent route models identically.
from backend.llm.streaming_provider import route_model as _route  # noqa: E402

# The only two models offered. Gemini reuses GEMINI_API_KEY; GPT-5.5 needs OPENAI_CLOUD_KEY.
CLOUD_MODELS = [
    "gemini-2.5-flash",   # Gemini — free
    "gpt-5.5",            # OpenAI — needs OPENAI_CLOUD_KEY
]


def _provider_name(model: str) -> str:
    return "Gemini" if (model or "").strip().lower().startswith("gemini") else "OpenAI"


def _label(model: str) -> str:
    return f"{_provider_name(model)} · {model}"


def _available(model: str) -> bool:
    _, key = _route(model)
    return bool(key)


def current() -> Dict[str, str]:
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return {"provider": "openai", "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    models: List[str] = list(CLOUD_MODELS)
    if cur["model"] not in models:
        models.insert(0, cur["model"])
    options = []
    for m in models:
        label = _label(m)
        if not _available(m):
            label += "  (add key)"
        options.append({"provider": "openai", "model": m, "label": label})
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
    """Switch the active model AND its endpoint/key (inferred from the model name),
    persisting to .env so the choice survives a restart."""
    model = (model or "").strip()
    if not model:
        raise ValueError("Model is required")
    base, key = _route(model)
    os.environ["OPENAI_MODEL"] = model
    os.environ["OPENAI_BASE_URL"] = base
    updates: Dict[str, str] = {"OPENAI_MODEL": model, "OPENAI_BASE_URL": base}
    if key:
        os.environ["OPENAI_API_KEY"] = key
        updates["OPENAI_API_KEY"] = key
    _persist_env(updates)
    return {"provider": "openai", "model": model, "label": _label(model)}
