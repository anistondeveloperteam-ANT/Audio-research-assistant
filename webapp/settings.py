"""
Chat-model selection for the web UI.

One picker, several providers. A model is routed by its name:
  - `deepseek/...` or any `vendor/model` slug -> DeepSeek / OpenRouter (OPENROUTER_API_KEY)
  - `gpt-*` / `o*` / `chatgpt*`                -> OpenAI               (OPENAI_CLOUD_KEY)
  - anything else (e.g. `qwen3:8b`)            -> local Ollama         (no key)

Switching a model updates OPENAI_MODEL + OPENAI_BASE_URL + OPENAI_API_KEY in both the
running process and the on-disk .env, so the whole connection switches — not just the
model string.
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

# Ensure .env is loaded even if this module is imported first.
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except Exception:
    pass

DEFAULT_OPENAI_MODEL = "gpt-4o"

OLLAMA_BASE = "http://localhost:11434/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Single source of truth for model -> (endpoint, key) routing, shared with the
# provider factory so the dropdown and the code agent route models identically.
from backend.llm.streaming_provider import route_model as _route, GROQ_MODELS  # noqa: E402

# Cloud models always offered in the picker. Each needs its key in .env:
#   GROQ_API_KEY (Groq, free), GEMINI_API_KEY (Gemini, free),
#   OPENROUTER_API_KEY (DeepSeek), OPENAI_CLOUD_KEY (GPT/OpenAI).
CLOUD_MODELS = [
    # Free, good for agentic loops (free-llm-api-resources, 2026):
    "llama-3.3-70b-versatile",   # Groq — best free pick (~1,000 req/day, fast)
    "llama-3.1-8b-instant",      # Groq — fastest, for high-volume loops
    "gemini-2.5-flash",          # Gemini — free, strong reasoning
    "gemini-2.0-flash",          # Gemini — free, lighter
    # Paid:
    "gpt-5.5",
    "gpt-4o",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
]


def _provider_name(model: str) -> str:
    m = (model or "").strip()
    ml = m.lower()
    if ml.startswith("gemini"):
        return "Gemini"
    if m in GROQ_MODELS:
        return "Groq"
    if ml.startswith("deepseek"):
        return "DeepSeek"
    if "/" in m:
        return "OpenRouter"
    if ml.startswith(("gpt-", "chatgpt", "o1", "o3", "o4")):
        return "OpenAI"
    return "Ollama"


def _label(model: str) -> str:
    return f"{_provider_name(model)} · {model.split('/')[-1]}"


def _available(model: str) -> bool:
    _, key = _route(model)
    return bool(key)


def _local_models() -> List[str]:
    """Models installed on the local Ollama server, queried directly so you can
    always switch back to a local model regardless of the current selection. []
    if Ollama is not running."""
    try:
        req = urllib.request.Request(OLLAMA_BASE + "/models",
                                     headers={"Authorization": "Bearer ollama"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.load(resp)
        return sorted({m.get("id") for m in (data.get("data") or []) if m.get("id")})
    except Exception:
        return []


def current() -> Dict[str, str]:
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return {"provider": "openai", "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    models: List[str] = []
    for m in list(_local_models()) + CLOUD_MODELS + [cur["model"], os.getenv("AGENT_MODEL", "")]:
        m = (m or "").strip()
        if m and m not in models:
            models.append(m)
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
