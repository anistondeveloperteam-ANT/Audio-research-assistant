"""
LLM model selection for the web UI.

Lists the models the user can pick and switches the active one by updating both
the running process env and the on-disk .env, so the choice persists.

Providers:
  - ollama     : local models (auto-listed from the Ollama server)
  - openrouter : one key (OPENROUTER_API_KEY) -> DeepSeek, Qwen, GPT, Claude, 300+
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

# Cloud providers: API key + the env var that holds their chosen model + a
# curated model list shown in the dropdown (OpenAI's list is fetched separately).
CLOUD: Dict[str, Dict[str, Any]] = {
    # OpenRouter: one key (sk-or-v1-...) serves DeepSeek, Qwen, GPT, Claude and
    # 300+ others. Slugs are "vendor/model"; ":free" variants cost nothing.
    "openrouter": {"key_env": "OPENROUTER_API_KEY", "model_env": "OPENROUTER_MODEL", "label": "OpenRouter", "models": [
        "deepseek/deepseek-v4-flash",      # fast + accurate -> default
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3-32b",
        "qwen/qwen3.5-35b-a3b",
        "deepseek/deepseek-chat-v3-0324:free",
        "qwen/qwen3-235b-a22b:free",
    ]},
}
VALID_PROVIDERS = ("ollama",) + tuple(CLOUD.keys())
MODEL_ENV = {"ollama": "OLLAMA_MODEL", **{p: c["model_env"] for p, c in CLOUD.items()}}
DEFAULT_MODEL = {"ollama": "llama3.2:3b", "openrouter": "deepseek/deepseek-v4-flash"}


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


def _label(provider: str) -> str:
    return CLOUD.get(provider, {}).get("label") or provider.title()


def current() -> Dict[str, str]:
    provider = (os.getenv("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
    if provider not in VALID_PROVIDERS:
        provider = "ollama"
    model = os.getenv(MODEL_ENV[provider], DEFAULT_MODEL.get(provider, ""))
    return {"provider": provider, "model": model}


def list_models() -> Dict[str, Any]:
    cur = current()
    options: List[Dict[str, str]] = []

    for m in _ollama_models():
        options.append({"provider": "ollama", "model": m, "label": f"Ollama · {m}"})

    for prov, cfg in CLOUD.items():
        if not os.getenv(cfg["key_env"]):
            continue  # provider not configured -> hide it
        for m in cfg["models"]:
            options.append({"provider": prov, "model": m, "label": f"{cfg['label']} · {m}"})

    # Always include the current selection, even if its source is unavailable.
    if not any(o["provider"] == cur["provider"] and o["model"] == cur["model"] for o in options):
        options.insert(0, {"provider": cur["provider"], "model": cur["model"],
                           "label": f"{_label(cur['provider'])} · {cur['model']}"})
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

    key = MODEL_ENV[provider]
    # Update the running process immediately (get_provider reads os.environ with
    # override=False, so these win) and persist to disk for next time.
    os.environ["LLM_PROVIDER"] = provider
    os.environ[key] = model
    _persist_env({"LLM_PROVIDER": provider, key: model})
    return {"provider": provider, "model": model, "label": f"{_label(provider)} · {model}"}
