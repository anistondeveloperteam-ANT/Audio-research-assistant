"""
LLM model selection for the web UI.

Lists the models the user can pick (local Ollama models + OpenAI models when a
key is present) and switches the active one by updating both the running
process env and the on-disk .env, so the choice persists across restarts.
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
VALID_PROVIDERS = ("ollama", "openai")


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_models() -> List[str]:
    try:
        import requests
        r = requests.get(f"{_ollama_host()}/api/tags", timeout=2.5)
        if r.status_code != 200:
            return []
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def _openai_models() -> List[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return []
    try:
        from backend.llm.fallback_provider import OPENAI_AVAILABLE_MODELS
        return list(OPENAI_AVAILABLE_MODELS)
    except Exception:
        return ["gpt-4o-mini", "gpt-4o"]


def current() -> Dict[str, str]:
    provider = (os.getenv("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
    if provider not in VALID_PROVIDERS:
        provider = "ollama"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if provider == "openai" \
        else os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    return {"provider": provider, "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    options: List[Dict[str, str]] = []
    for m in _ollama_models():
        options.append({"provider": "ollama", "model": m, "label": f"Ollama · {m}"})
    for m in _openai_models():
        options.append({"provider": "openai", "model": m, "label": f"OpenAI · {m}"})
    # Make sure the current selection is always present, even if Ollama is down.
    if not any(o["provider"] == cur["provider"] and o["model"] == cur["model"] for o in options):
        options.insert(0, {"provider": cur["provider"], "model": cur["model"],
                           "label": f"{cur['provider'].title()} · {cur['model']}"})
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
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}")
    if not model:
        raise ValueError("Model is required")

    model_key = "OPENAI_MODEL" if provider == "openai" else "OLLAMA_MODEL"
    # Update the running process immediately (get_provider reads os.environ
    # with override=False, so these values win) and persist to disk.
    os.environ["LLM_PROVIDER"] = provider
    os.environ[model_key] = model
    _persist_env({"LLM_PROVIDER": provider, model_key: model})
    return {"provider": provider, "model": model, "label": f"{provider} · {model}"}
