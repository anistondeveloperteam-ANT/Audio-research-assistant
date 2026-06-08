"""
LLM model selection for the web UI (OpenAI).

Lists the OpenAI models the user can pick and switches the active one by updating
both the running process env and the on-disk .env, so the choice persists.
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

# The single provider: OpenAI. Models shown in the dropdown. The provider adapts
# its API parameters per model, so both the GPT-5 family and the gpt-4o/4.1 models
# work. Your account may not have every one of these — pick one it has access to.
OPENAI_MODELS = [
    "gpt-5.5",         # newest flagship -> default
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.1",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",     # fast + cheap
]
DEFAULT_OPENAI_MODEL = "gpt-5.5"

VALID_PROVIDERS = ("openai",)
MODEL_ENV = {"openai": "OPENAI_MODEL"}
DEFAULT_MODEL = {"openai": DEFAULT_OPENAI_MODEL}


def current() -> Dict[str, str]:
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return {"provider": "openai", "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    options: List[Dict[str, str]] = [
        {"provider": "openai", "model": m, "label": f"OpenAI · {m}"}
        for m in OPENAI_MODELS
    ]
    # Always include the current selection, even if it's a custom model not listed.
    if not any(o["model"] == cur["model"] for o in options):
        options.insert(0, {"provider": "openai", "model": cur["model"],
                           "label": f"OpenAI · {cur['model']}"})
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
    provider = (provider or "openai").strip().lower()
    model = (model or "").strip()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r} (only 'openai' is supported)")
    if not model:
        raise ValueError("Model is required")

    # Update the running process immediately (get_provider reads os.environ with
    # override=False, so these win) and persist to disk for next time.
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_MODEL"] = model
    _persist_env({"LLM_PROVIDER": "openai", "OPENAI_MODEL": model})
    return {"provider": "openai", "model": model, "label": f"OpenAI · {model}"}
