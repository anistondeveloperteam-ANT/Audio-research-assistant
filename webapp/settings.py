"""
LLM model selection for the web UI.

Lists the supported OpenAI chat models the user can pick and switches the active
one by updating both the running process env and the on-disk .env, so the choice
persists. OpenAI is the only chat provider.
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

DEFAULT_OPENAI_MODEL = "gpt-4o"

VALID_PROVIDERS = ("openai",)
PROVIDER_LABEL = {"openai": "OpenAI"}
MODEL_ENV = {"openai": "OPENAI_MODEL"}
DEFAULT_MODEL = {"openai": DEFAULT_OPENAI_MODEL}
PROVIDER_MODELS = {"openai": OPENAI_MODELS}


def _normalize_provider(provider: str | None) -> str:
    provider = (provider or "openai").strip().lower()
    return provider if provider in VALID_PROVIDERS else "openai"


def _label(provider: str, model: str) -> str:
    return f"{PROVIDER_LABEL.get(provider, provider.title())} · {model}"


def current() -> Dict[str, str]:
    provider = _normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model = os.getenv(MODEL_ENV[provider], DEFAULT_MODEL[provider])
    return {"provider": provider, "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    options: List[Dict[str, str]] = [
        {"provider": "openai", "model": model, "label": _label("openai", model)}
        for model in OPENAI_MODELS
    ]

    # Always include the current selection, even if it's a custom model not listed.
    if not any(o["model"] == cur["model"] for o in options):
        options.insert(0, {
            "provider": cur["provider"],
            "model": cur["model"],
            "label": _label(cur["provider"], cur["model"]),
        })
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
    provider = _normalize_provider(provider)
    model = (model or "").strip()
    if not model:
        raise ValueError("Model is required")

    # Update the running process immediately (get_provider reads os.environ with
    # override=False, so these win) and persist to disk for next time.
    os.environ["LLM_PROVIDER"] = provider
    os.environ[MODEL_ENV[provider]] = model
    _persist_env({"LLM_PROVIDER": provider, MODEL_ENV[provider]: model})
    return {"provider": provider, "model": model, "label": _label(provider, model)}
