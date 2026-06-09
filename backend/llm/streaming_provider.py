"""
Streaming LLM backend: OpenAI, OpenRouter, and Gemini.

The chat UI calls this; nothing else cares about the details. Configure via .env:

    LLM_PROVIDER=openai                    (openai | openrouter | gemini)

    OPENAI_API_KEY=sk-...                  (required when LLM_PROVIDER=openai)
    OPENAI_MODEL=gpt-4o                    (default; e.g. gpt-4o-mini, gpt-4.1)
    OPENAI_BASE_URL=https://api.openai.com/v1   (optional; Azure/proxies)

    OPENROUTER_API_KEY=sk-or-v1-...        (required when LLM_PROVIDER=openrouter)
    OPENROUTER_MODEL=deepseek/deepseek-chat  (one key -> DeepSeek, GPT, Claude, 300+)
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  (optional override)

    GEMINI_API_KEY=...                     (required when LLM_PROVIDER=gemini)
    GEMINI_MODEL=gemini-2.5-flash
    GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

OpenRouter is OpenAI-compatible, so it reuses the OpenAI SDK with a custom base URL.
One OpenRouter key reaches DeepSeek, GPT, Qwen, Claude and 300+ models by slug
("vendor/model", e.g. deepseek/deepseek-chat) — a cheap way to use DeepSeek.

Public API:

    provider = get_provider()
    if not provider.is_available:
        ...                               # show a helpful error (missing key?)
    for token in provider.stream_chat(messages, system="..."):
        print(token, end="", flush=True)

`messages` is a list of {"role": "user"|"assistant", "content": "..."}.
`stream_chat` yields raw text chunks (no JSON envelope).
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-chat"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Gemini is FREE-tier (same key used for embeddings) and OpenAI-compatible.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SUPPORTED_PROVIDERS = ("openai", "openrouter", "gemini")


class LLMProvider:
    """Abstract base. Subclasses override name, model, is_available, stream_chat."""

    @property
    def name(self) -> str:
        return "unknown"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def is_available(self) -> bool:
        """Quick liveness check. Cheap; no real LLM call."""
        return False

    def unavailable_message(self) -> str:
        return "LLM not available - configure a supported provider in .env."

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions compatible client with token streaming.

    Both OpenAI and OpenRouter use the OpenAI Python SDK here; OpenRouter is just
    a different base URL + key, with "vendor/model" slugs.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        api_key_env: str,
        default_model: str,
        base_url: Optional[str] = None,
    ):
        self._name = name
        self._model = model or default_model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url

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
            __import__("openai")
            return True
        except ImportError:
            return False

    def unavailable_message(self) -> str:
        if not self.api_key:
            return f"LLM not available - set {self.api_key_env} in .env."
        try:
            __import__("openai")
        except ImportError:
            return "LLM not available - install dependencies with `pip install -r requirements.txt`."
        return "LLM not available - check the provider configuration in .env."

    def _request_variants(self, max_tokens: int, temperature: float) -> List[Dict[str, object]]:
        # Newer OpenAI models (GPT-5 family, o-series) require `max_completion_tokens`
        # and only allow the default temperature; everything else (gpt-4o/4.1 and all
        # OpenRouter slugs) uses `max_tokens` + a custom temperature. Try the right
        # shape first, then fall back so any current/future model just works.
        bare = self._model.split("/")[-1]   # OpenRouter slugs look like vendor/model
        newer = bare.startswith(("gpt-5", "o1", "o3", "o4"))
        if newer:
            return [
                {"max_completion_tokens": max_tokens},
                {"max_completion_tokens": max_tokens, "temperature": temperature},
                {"max_tokens": max_tokens, "temperature": temperature},
            ]
        return [
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens},
        ]

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3):
        import openai

        client = (openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                  if self.base_url else openai.OpenAI(api_key=self.api_key))
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        stream = None
        last_err = None
        for params in self._request_variants(max_tokens, temperature):
            try:
                stream = client.chat.completions.create(
                    model=self._model, messages=msgs, stream=True, **params)
                break
            except openai.BadRequestError as e:
                last_err = e
                low = str(e).lower()
                if not any(k in low for k in
                           ("max_tokens", "max_completion_tokens", "temperature", "unsupported")):
                    raise
        if stream is None:
            raise last_err

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield delta


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(
            name="openai", model=model, api_key=api_key,
            api_key_env="OPENAI_API_KEY", default_model=DEFAULT_OPENAI_MODEL,
            base_url=base_url,
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(
            name="openrouter", model=model, api_key=api_key,
            api_key_env="OPENROUTER_API_KEY", default_model=DEFAULT_OPENROUTER_MODEL,
            base_url=base_url or DEFAULT_OPENROUTER_BASE_URL,
        )


class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini via its OpenAI-compatible endpoint — free tier, reuses GEMINI_API_KEY."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(
            name="gemini", model=model, api_key=api_key,
            api_key_env="GEMINI_API_KEY", default_model=DEFAULT_GEMINI_MODEL,
            base_url=base_url or DEFAULT_GEMINI_BASE_URL,
        )


def get_provider() -> LLMProvider:
    """Construct the active provider from .env.

    Always returns a provider object; callers must check `.is_available` to catch
    a missing API key or missing dependency.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    provider = (os.getenv("LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider == "openrouter":
        return OpenRouterProvider(
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
        )
    if provider == "gemini":
        return GeminiProvider(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url=os.getenv("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE_URL,
        )

    return OpenAIProvider(
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


if __name__ == "__main__":
    p = get_provider()
    print(f"Provider:  {p.name}")
    print(f"Model:     {p.model}")
    print(f"Available: {p.is_available}")
    if not p.is_available:
        print(p.unavailable_message())
    else:
        print("\nQuick test:")
        for chunk in p.stream_chat(
            messages=[{"role": "user", "content": "Reply with exactly: hello"}],
            max_tokens=20,
        ):
            print(chunk, end="", flush=True)
        print()
