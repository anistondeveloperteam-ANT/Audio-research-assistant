"""
Streaming LLM backend: one OpenAI-compatible client for every provider.

The chat UI calls this; nothing else cares about the details. Configure via .env:

    OPENAI_API_KEY=...                    (required)
    OPENAI_MODEL=gemini-2.5-flash         (e.g. llama-3.3-70b-versatile, deepseek/deepseek-chat, qwen3:8b)
    OPENAI_BASE_URL=...                   (per provider: Gemini / Groq / OpenRouter / Ollama)

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
import re
from typing import Dict, Iterator, List, Optional

DEFAULT_OPENAI_MODEL = "gemini-2.5-flash"
OLLAMA_BASE = "http://localhost:11434/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE = "https://api.groq.com/openai/v1"

# Free, OpenAI-compatible models good for agentic loops (free-llm-api-resources, 2026).
# Groq gives ~1,000 requests/day on Llama 3.3 70B — the best free pick for chained calls.
GROQ_MODELS = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant"}

_AFFORD_RE = re.compile(r"can only afford (\d+)")


def route_model(model: str):
    """(base_url, api_key) for a model name, so a model override (e.g. the code
    agent's AGENT_MODEL) connects to its OWN provider, not the active chat one:
      gemini-*                     -> Google Gemini (GEMINI_API_KEY) [free tier]
      Groq free models             -> Groq          (GROQ_API_KEY)   [free, fast]
      deepseek/... or vendor/model -> OpenRouter     (OPENROUTER_API_KEY)
      anything else (qwen3:8b ...) -> local Ollama
    """
    m = (model or "").strip()
    ml = m.lower()
    if ml.startswith("gemini"):
        return GEMINI_BASE, os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if m in GROQ_MODELS:
        return GROQ_BASE, os.getenv("GROQ_API_KEY", "")
    if ml.startswith("deepseek") or "/" in m:
        return OPENROUTER_BASE, os.getenv("OPENROUTER_API_KEY", "")
    return OLLAMA_BASE, "ollama"


def _affordable_tokens(message: str) -> Optional[int]:
    """Parse OpenRouter's 402 'can only afford N tokens' hint, if present."""
    m = _AFFORD_RE.search(message or "")
    return int(m.group(1)) if m else None


class LLMProvider:
    """Abstract base for a streaming chat provider."""

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
        return "LLM not available - configure OpenAI in .env."

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions with token streaming."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self._model = model or DEFAULT_OPENAI_MODEL
        self.api_key = api_key
        self.api_key_env = "OPENAI_API_KEY"
        self.base_url = base_url

    @property
    def name(self) -> str:
        return "openai"

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
        return "LLM not available - check the OpenAI configuration in .env."

    def _request_variants(self, max_tokens: int, temperature: float) -> List[Dict[str, object]]:
        # Try the common shape first (max_tokens + temperature), then fall back to
        # max_completion_tokens / default temperature, so any OpenAI-compatible model
        # (Groq, Gemini, DeepSeek, Ollama, o-series, …) just works.
        return [
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens},
        ]

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3,
                    yield_reasoning=False):
        """Yield answer content as strings. When `yield_reasoning=True`, also yields
        the model's hidden reasoning/'thinking' as {"reasoning": "..."} dicts (for
        models like qwen3/DeepSeek-R1 that expose it) so the UI can show it."""
        import openai

        client = (openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                  if self.base_url else openai.OpenAI(api_key=self.api_key))
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        # Open the stream, shrinking the token budget if the provider (e.g. a
        # low-balance OpenRouter account) replies 402 "can only afford N tokens".
        budget = max_tokens
        stream = None
        last_err = None
        for _ in range(4):
            try:
                stream = self._open_stream(client, openai, msgs, budget, temperature)
                break
            except Exception as e:  # noqa: BLE001 - inspect the message for an affordable cap
                last_err = e
                afford = _affordable_tokens(str(e))
                # Shrink to what the account can actually afford (small floor, not 256,
                # so a near-empty OpenRouter balance still yields a short answer).
                new_budget = max(64, afford - 16) if afford else 0
                if new_budget and new_budget < budget:
                    budget = new_budget
                    continue
                raise
        if stream is None:
            # Never silently yield nothing — surface the failure so callers can show
            # a real message instead of a fake "(no answer)".
            if last_err is not None:
                raise last_err
            return

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
            except Exception:
                continue
            if yield_reasoning:
                think = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if think:
                    yield {"reasoning": think}
            content = getattr(delta, "content", None)
            if content:
                yield content

    def _open_stream(self, client, openai, msgs, max_tokens: int, temperature: float):
        """Create the streaming completion, adapting the token/temperature params."""
        last_err = None
        for params in self._request_variants(max_tokens, temperature):
            try:
                return client.chat.completions.create(
                    model=self._model, messages=msgs, stream=True, **params)
            except openai.BadRequestError as e:
                last_err = e
                low = str(e).lower()
                if not any(k in low for k in
                           ("max_tokens", "max_completion_tokens", "temperature", "unsupported")):
                    raise
        raise last_err


def get_provider(model: Optional[str] = None) -> LLMProvider:
    """Construct the OpenAI-compatible provider from .env.

    `model` overrides OPENAI_MODEL for this provider (e.g. the code agent can use a
    coder model while chat uses a general one) — the API key and base URL are shared,
    so it works the same against OpenAI, OpenRouter, or a local Ollama endpoint.

    Always returns a provider object; callers must check `.is_available` to catch
    a missing API key or a missing dependency.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    # A model override (e.g. the code agent's AGENT_MODEL) routes to its own
    # endpoint/key by name, so it works even when the active chat model is on a
    # different provider (e.g. coder on local Ollama while chat is on OpenRouter).
    if model:
        base, key = route_model(model)
        return OpenAIProvider(model=model, api_key=key, base_url=base or None)

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
