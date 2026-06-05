"""
llm_provider.py  --  Batch 8 (Phase 2)

Swappable LLM backend. The chat UI calls this; nothing else cares
which backend is active.

Two providers are supported. Choose at runtime via .env:

    LLM_PROVIDER=ollama       (default; runs locally, free, offline)
    LLM_PROVIDER=openrouter   (one key -> DeepSeek, Qwen, GPT, Claude, 300+)

Provider-specific env vars (only the one you use matters):

    Ollama:
        OLLAMA_HOST=http://localhost:11434   (default)
        OLLAMA_MODEL=llama3.2:3b             (default; fits 6GB GPU)

    OpenRouter (OpenAI-compatible gateway):
        OPENROUTER_API_KEY=sk-or-v1-...
        OPENROUTER_MODEL=deepseek/deepseek-v4-pro   (vendor/model; :free slugs are free)

Public API:

    provider = get_provider()
    if not provider.is_available:
        # show a helpful error
        ...
    for token in provider.stream_chat(messages, system="..."):
        print(token, end="", flush=True)

`messages` is a list of {"role": "user"|"assistant", "content": "..."}.
`stream_chat` yields raw text chunks (no JSON envelope).
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Dict


# ----------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------

class LLMProvider:
    """Abstract base. Subclasses override name, is_available, stream_chat."""

    @property
    def name(self) -> str:
        return "unknown"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def is_available(self) -> bool:
        """Quick liveness check. Should be cheap; no real LLM call."""
        return False

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Yield text tokens / chunks. Implementer-specific."""
        raise NotImplementedError


# ----------------------------------------------------------------------
# Ollama (local, free)
# ----------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    def __init__(self, model: str, host: str):
        self._model = model
        self.host = host.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            if r.status_code != 200:
                return False
            tags = r.json().get("models", [])
            names = {t.get("name", "") for t in tags}
            # ollama lists as "llama3.2:3b"; we accept exact match or short prefix
            for n in names:
                if n == self._model or n.startswith(self._model.split(":")[0]):
                    return True
            return len(names) > 0
        except Exception:
            return False

    def stream_chat(
        self,
        messages,
        system="",
        max_tokens=2048,
        temperature=0.3,
    ):
        import requests

        body_messages = []
        if system:
            body_messages.append({"role": "system", "content": system})
        body_messages.extend(messages)

        body = {
            "model": self._model,
            "messages": body_messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        with requests.post(
            f"{self.host}/api/chat",
            json=body,
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                content = (data.get("message") or {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break


# ----------------------------------------------------------------------
# OpenAI-compatible client (used for OpenRouter)
# ----------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Client for any OpenAI-compatible chat API via a custom base_url.
    Used here for OpenRouter (one key -> DeepSeek, Qwen, GPT, Claude, 300+)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, name: str = "openrouter"):
        self._model = model
        self.api_key = api_key
        self.base_url = base_url
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def stream_chat(
        self,
        messages,
        system="",
        max_tokens=2048,
        temperature=0.3,
    ):
        import openai
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url) if self.base_url \
            else openai.OpenAI(api_key=self.api_key)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        stream = client.chat.completions.create(
            model=self._model,
            messages=msgs,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield delta


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

VALID_PROVIDERS = ("ollama", "openrouter")

# OpenAI-compatible cloud providers: base URL + the env keys they read.
#
#   openrouter -> One gateway to DeepSeek, Qwen, GPT, Claude and 300+ others with
#                 a single `sk-or-v1-...` key. Models use "vendor/model" slugs and
#                 there are ":free" variants. This is the only cloud provider we
#                 keep, because it covers every model with one key.
OPENAI_COMPATIBLE = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "deepseek/deepseek-v4-flash",
    },
}


def get_provider() -> LLMProvider:
    """Construct the active provider based on .env. Always returns
    a usable provider object -- caller must check `.is_available`
    before use.

    If LLM_PROVIDER is unset or set to something unknown, we fall
    back to ollama and print a clear warning so the user notices.
    """
    from dotenv import load_dotenv
    load_dotenv(override=False)

    raw = os.getenv("LLM_PROVIDER", "ollama").strip()
    backend = raw.lower()

    if backend not in VALID_PROVIDERS:
        # Unknown value (e.g. typo like LLM_PROVIDER=local). Don't
        # break -- print a clear warning and fall back to ollama.
        if raw:
            print(
                f"[llm_provider] WARNING: LLM_PROVIDER={raw!r} is not "
                f"recognized. Valid values: {VALID_PROVIDERS}. "
                f"Falling back to 'ollama'."
            )
        backend = "ollama"

    if backend == "ollama":
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    if backend in OPENAI_COMPATIBLE:
        cfg = OPENAI_COMPATIBLE[backend]
        return OpenAIProvider(
            model=os.getenv(cfg["model_env"], cfg["default_model"]),
            api_key=os.getenv(cfg["key_env"], ""),
            base_url=cfg["base_url"],
            name=backend,
        )

    # Defensive -- shouldn't reach here after the fallback above
    return OllamaProvider(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


if __name__ == "__main__":
    p = get_provider()
    print(f"Provider:  {p.name}")
    print(f"Model:     {p.model}")
    print(f"Available: {p.is_available}")
    if p.is_available:
        print("\nQuick test:")
        for chunk in p.stream_chat(
            messages=[{"role": "user", "content": "Reply with exactly: hello"}],
            max_tokens=20,
        ):
            print(chunk, end="", flush=True)
        print()
