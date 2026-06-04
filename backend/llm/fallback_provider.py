"""
backend/llm_providers.py  --  Multi-provider LLM abstraction (Batch 12C)

Goal: same interface, different backends. The rest of the system calls
provider.generate(...) without caring whether it's OpenAI or Ollama.

Supports:
  - OpenAI (paid, default)
  - Ollama (free, fallback)

Auto-fallback: if OpenAI fails (network/auth/rate-limit), automatically
retries with Ollama and tells the caller it fell back.

Cost tracking: every OpenAI call records tokens + cost via cost_tracker.

Security:
  - API key read from os.environ['OPENAI_API_KEY']
  - Never logged, never returned in error messages
  - Missing key -> falls back to Ollama with clear warning
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------

@dataclass
class LLMResult:
    """What every provider returns."""
    text: str
    provider: str           # "openai" or "ollama"
    model: str              # e.g. "gpt-4o-mini" or "llama3.2:3b"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    fell_back: bool = False
    fallback_reason: Optional[str] = None
    error: Optional[str] = None   # if non-None, generation failed entirely


# ----------------------------------------------------------------------
# OpenAI pricing (USD per 1M tokens, as of 2025-2026)
# https://openai.com/api/pricing/
# ----------------------------------------------------------------------

OPENAI_PRICING = {
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},   # cheapest
    "gpt-4o":         {"input": 2.50,  "output": 10.00},  # main quality option
    "gpt-4-turbo":    {"input": 10.00, "output": 30.00},  # older, expensive
    "gpt-3.5-turbo":  {"input": 0.50,  "output": 1.50},   # legacy
}

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# Models we expose to the user (filtered subset of full OpenAI catalog)
OPENAI_AVAILABLE_MODELS = [
    "gpt-4o-mini",    # default
    "gpt-4o",         # bump up for important queries
    "gpt-4-turbo",    # fallback if 4o issues
    "gpt-3.5-turbo",  # cheap legacy
]


# ----------------------------------------------------------------------
# Provider interface
# ----------------------------------------------------------------------

class LLMProvider(ABC):
    """All providers implement this interface."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> LLMResult:
        """Generate a response.

        messages: list of {role: "user"|"assistant", content: str}
        system: optional system prompt
        model: model name (provider-specific)
        Returns LLMResult.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Quick health check. Should be cheap (no API call required)."""
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return list of model names this provider exposes."""
        ...


# ----------------------------------------------------------------------
# Ollama provider (free, local)
# ----------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Local Ollama provider. Free, no API key needed."""

    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")
        self._available_cache = None
        self._available_cache_time = 0.0

    def is_available(self) -> bool:
        # Cache result for 30 seconds to avoid spamming Ollama with health checks
        now = time.time()
        if self._available_cache is not None and (now - self._available_cache_time) < 30:
            return self._available_cache

        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=2.0)
            self._available_cache = (r.status_code == 200)
        except Exception:
            self._available_cache = False
        self._available_cache_time = now
        return self._available_cache

    def list_models(self) -> List[str]:
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        # Fallback common ones
        return ["qwen2.5:7b-instruct", "llama3.2:3b"]

    def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> LLMResult:
        import requests

        if model is None:
            model = "qwen2.5:7b-instruct"

        # Ollama uses its own /api/chat format
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(messages)

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "").strip()

            return LLMResult(
                text=text,
                provider="ollama",
                model=model,
                tokens_in=data.get("prompt_eval_count", 0),
                tokens_out=data.get("eval_count", 0),
                cost_usd=0.0,
            )
        except requests.exceptions.Timeout:
            return LLMResult(text="", provider="ollama", model=model,
                             error="Ollama timed out")
        except requests.exceptions.ConnectionError:
            return LLMResult(text="", provider="ollama", model=model,
                             error="Ollama not running (start it with: ollama serve)")
        except Exception as exc:
            return LLMResult(text="", provider="ollama", model=model,
                             error=f"Ollama error: {type(exc).__name__}")


# ----------------------------------------------------------------------
# OpenAI provider (paid, primary)
# ----------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI provider. Reads OPENAI_API_KEY from environment."""

    name = "openai"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._client = None
        self._import_error: Optional[str] = None

    def _get_client(self):
        """Lazy import + create client. Returns None if SDK missing or no key."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            self._import_error = "OPENAI_API_KEY not set in environment"
            return None
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            return self._client
        except ImportError:
            self._import_error = (
                "openai SDK not installed. "
                "Run: pip install openai>=1.0"
            )
            return None
        except Exception as exc:
            self._import_error = f"Could not init OpenAI client: {type(exc).__name__}"
            return None

    def is_available(self) -> bool:
        """Returns True if we have a key AND the SDK is installed.
        Does NOT make a network call (cheap check)."""
        if not self.api_key:
            return False
        return self._get_client() is not None

    def list_models(self) -> List[str]:
        return list(OPENAI_AVAILABLE_MODELS)

    def _calc_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Compute USD cost from token counts."""
        prices = OPENAI_PRICING.get(model)
        if not prices:
            return 0.0
        cost_in = (tokens_in / 1_000_000.0) * prices["input"]
        cost_out = (tokens_out / 1_000_000.0) * prices["output"]
        return cost_in + cost_out

    def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> LLMResult:
        client = self._get_client()
        if client is None:
            return LLMResult(
                text="", provider="openai", model=model or OPENAI_DEFAULT_MODEL,
                error=self._import_error or "OpenAI not configured",
            )

        if model is None:
            model = OPENAI_DEFAULT_MODEL

        # Build messages array with system prompt prepended
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            text = response.choices[0].message.content or ""
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            cost = self._calc_cost(model, tokens_in, tokens_out)

            # Record cost (lazy import to avoid circular dep at module load)
            try:
                from backend.llm.cost_tracker import record_call
                record_call(provider="openai", model=model,
                            tokens_in=tokens_in, tokens_out=tokens_out,
                            cost_usd=cost)
            except Exception:
                # Never let cost tracker break the call
                pass

            return LLMResult(
                text=text.strip(),
                provider="openai",
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )
        except Exception as exc:
            # Classify the error WITHOUT leaking key in message
            exc_name = type(exc).__name__
            msg = str(exc)
            # Strip any accidental key leak
            if self.api_key and self.api_key in msg:
                msg = msg.replace(self.api_key, "[REDACTED]")
            return LLMResult(
                text="", provider="openai", model=model,
                error=f"OpenAI {exc_name}: {msg[:200]}",
            )


# ----------------------------------------------------------------------
# Factory + auto-fallback wrapper
# ----------------------------------------------------------------------

# Cached singleton providers (created once per process)
_ollama_singleton: Optional[OllamaProvider] = None
_openai_singleton: Optional[OpenAIProvider] = None


def get_provider(name: str) -> LLMProvider:
    """Return a provider instance by name."""
    global _ollama_singleton, _openai_singleton
    name = (name or "").lower()
    if name == "openai":
        if _openai_singleton is None:
            _openai_singleton = OpenAIProvider()
        return _openai_singleton
    elif name == "ollama":
        if _ollama_singleton is None:
            _ollama_singleton = OllamaProvider()
        return _ollama_singleton
    else:
        raise ValueError(f"Unknown provider: {name}")


def generate_with_fallback(
    primary: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    model: Optional[str] = None,
    fallback: str = "ollama",
    fallback_model: Optional[str] = None,
    **kwargs,
) -> LLMResult:
    """Try primary provider; if it errors, retry with fallback.

    The result's .fell_back flag tells the caller whether a fallback
    happened so they can show a warning in the UI.
    """
    primary_provider = get_provider(primary)
    result = primary_provider.generate(messages, system=system, model=model, **kwargs)

    if result.error is None and result.text.strip():
        # Primary succeeded
        return result

    # Primary failed -- try fallback
    if fallback == primary:
        # No point falling back to same provider
        return result

    fallback_provider = get_provider(fallback)
    if not fallback_provider.is_available():
        # Fallback also not available -- return original error
        return result

    fallback_result = fallback_provider.generate(
        messages, system=system, model=fallback_model, **kwargs
    )
    fallback_result.fell_back = True
    fallback_result.fallback_reason = (
        f"{primary} failed: {result.error or 'empty response'}"
    )
    return fallback_result


# ----------------------------------------------------------------------
# Convenience: test the connection with minimal cost
# ----------------------------------------------------------------------

def test_connection(provider_name: str, model: Optional[str] = None) -> LLMResult:
    """Send a 1-token test query to verify the provider works.
    For OpenAI this costs roughly $0.00001 -- effectively free.
    For Ollama it's free."""
    provider = get_provider(provider_name)
    return provider.generate(
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        max_tokens=5,
        temperature=0.0,
    )
