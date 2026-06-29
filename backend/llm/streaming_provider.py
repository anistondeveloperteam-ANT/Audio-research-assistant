"""
Streaming LLM backend: one OpenAI-compatible client for every provider.

The chat UI calls this; nothing else cares about the details. Configure via .env:

    OPENAI_API_KEY=...                    (required)
    OPENAI_MODEL=gemini-2.5-flash         (or gpt-5.5)
    OPENAI_BASE_URL=...                   (Gemini endpoint, or blank for OpenAI)

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
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE = "https://api.openai.com/v1"

# Each provider is an OpenAI-compatible endpoint -> (base_url, api_key_env). The base URL is ALWAYS
# explicit: a model picked from the catalog must hit ITS provider's endpoint, never fall back to the
# OPENAI_BASE_URL env var (which configures the default chat model, e.g. the free Gemini endpoint) —
# otherwise selecting GPT-5.5 would send its requests to the Gemini base and fail to connect.
PROVIDERS: Dict[str, tuple] = {
    "gemini":  (GEMINI_BASE, "GEMINI_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openai":  (OPENAI_BASE, "OPENAI_CLOUD_KEY"),
}

# Models offered in the picker: (model_id, provider, vendor, display_name, is_free).
CATALOG = [
    ("gemini-2.5-flash",     "gemini",  "Gemini",  "2.5 Flash", True),
    ("mistral-large-latest", "mistral", "Mistral", "Large",     True),
    ("codestral-latest",     "mistral", "Mistral", "Codestral", True),
    ("gpt-5.5",              "openai",  "OpenAI",  "GPT-5.5",   False),
]
_MODEL_PROVIDER = {mid: prov for mid, prov, *_ in CATALOG}

_AFFORD_RE = re.compile(r"can only afford (\d+)")


def route_model(model: str):
    """(base_url, api_key) for a model id — resolves the right OpenAI-compatible
    endpoint + key by the model's provider in CATALOG (with a sensible fallback)."""
    m = (model or "").strip()
    prov = _MODEL_PROVIDER.get(m)
    if prov is None:                       # unlisted model: best-effort guess
        prov = "gemini" if m.lower().startswith("gemini") else "openai"
    base, key_env = PROVIDERS[prov]
    key = os.getenv(key_env, "")
    if prov == "gemini" and not key:
        key = os.getenv("GOOGLE_API_KEY", "")
    return base, key


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
        # (Gemini, Mistral, OpenAI o-series, local Ollama, …) just works.
        return [
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens},
        ]

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3,
                    yield_reasoning=False, timeout=None):
        """Yield answer content as strings. When `yield_reasoning=True`, also yields
        the model's hidden reasoning/'thinking' as {"reasoning": "..."} dicts (for
        reasoning models that expose it) so the UI can show it. `timeout` (seconds) caps the request so
        a stalled response can't hang the caller indefinitely; None = the client default."""
        import openai

        client = (openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
                  if self.base_url else openai.OpenAI(api_key=self.api_key, timeout=timeout))
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

    # Per-request model/endpoint/key from the request context (set at request entry from the user's
    # selection), so concurrent requests on different models stay isolated; env is the off-request
    # default (CLI / tests / a fresh process).
    from backend.common import request_context as _rc
    return OpenAIProvider(
        model=_rc.request_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        api_key=_rc.request_str("OPENAI_API_KEY", "") or "",
        base_url=_rc.request_str("OPENAI_BASE_URL", None) or None,
    )


def resolve_model_settings(model: Optional[str]) -> Dict[str, str]:
    """The request-context settings for a chosen chat model: its name + (for a catalog model) the
    endpoint/key inferred from the name. Empty when no model is given (the env default then applies).
    Bound to the per-request context at request entry — never written to os.environ."""
    m = (model or "").strip()
    if not m:
        return {}
    base, key = route_model(m)
    out: Dict[str, str] = {"OPENAI_MODEL": m}
    if base:
        out["OPENAI_BASE_URL"] = base
    if key:
        out["OPENAI_API_KEY"] = key
    return out


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
