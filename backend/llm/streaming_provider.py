"""
Streaming LLM backend: OpenAI.

The chat UI calls this; nothing else cares about the details. Configure via .env:

    OPENAI_API_KEY=sk-...                 (required)
    OPENAI_MODEL=gpt-4o                   (default; e.g. gpt-4o-mini, gpt-4.1, gpt-5.5)
    OPENAI_BASE_URL=https://api.openai.com/v1   (optional; Azure / OpenAI-compatible proxy)

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
        # Newer OpenAI models (GPT-5 family, o-series) require `max_completion_tokens`
        # and only allow the default temperature; older models (gpt-4o/4.1/3.5) use
        # `max_tokens` + a custom temperature. Try the right shape first, then fall
        # back so any current or future model just works.
        newer = self._model.startswith(("gpt-5", "o1", "o3", "o4"))
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
        msgs: List[Dict[str, str]] = []
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


def get_provider() -> LLMProvider:
    """Construct the OpenAI provider from .env.

    Always returns a provider object; callers must check `.is_available` to catch
    a missing API key or a missing dependency.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

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
