"""
Streaming LLM backend — OpenAI.

The chat UI calls this; nothing else cares about the details. Configure via .env:

    OPENAI_API_KEY=sk-...                 (required)
    OPENAI_MODEL=gpt-4o                    (default; e.g. gpt-4o-mini, gpt-4.1)
    OPENAI_BASE_URL=https://api.openai.com/v1   (optional; only for Azure/proxies)

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
from typing import Iterator, List, Dict, Optional

DEFAULT_MODEL = "gpt-4o"


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

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions client with token streaming.

    base_url is normally OpenAI's default; override it only for Azure OpenAI or an
    OpenAI-compatible proxy.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self._model = model or DEFAULT_MODEL
        self.api_key = api_key
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

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3):
        import openai

        client = (openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                  if self.base_url else openai.OpenAI(api_key=self.api_key))
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


def get_provider() -> LLMProvider:
    """Construct the OpenAI provider from .env. Always returns a usable object —
    the caller must check `.is_available` (e.g. to detect a missing API key)."""
    from dotenv import load_dotenv
    load_dotenv(override=False)

    return OpenAIProvider(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
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
