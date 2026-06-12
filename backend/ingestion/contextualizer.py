"""
Anthropic-style Contextual Retrieval.

For each chunk we ask the configured LLM (Gemini, via get_provider()) for one short sentence that
situates the chunk within its document. That sentence is stored separately (the `context_text`
column) and prepended to the chunk ONLY for indexing — the original chunk text is unchanged and is
what users see in citations. The contextual prefix measurably improves vector + BM25 recall.

Production-safe:
  - Off-switch: CONTEXTUAL_CHUNKS=false → callers get empty contexts and fall back to plain chunks.
  - Fail-safe: any LLM error / unavailable provider → "" (plain chunk), never raises.
  - Cheap re-runs: generated contexts are cached on disk keyed by (document, chunk).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = ROOT / "data" / "extracted" / "contextual_cache.json"

_PROMPT = (
    "<document>\n{doc}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context (one sentence, roughly 50-100 tokens) to situate this chunk "
    "within the overall document, to improve search retrieval of the chunk. "
    "Answer ONLY with the succinct context and nothing else."
)
_SYSTEM = "You write one concise context sentence to help document search. Output the sentence only."


def contextual_enabled() -> bool:
    return os.getenv("CONTEXTUAL_CHUNKS", "true").strip().lower() == "true"


def _doc_chars() -> int:
    return int(os.getenv("CONTEXTUAL_DOC_CHARS", "12000"))


def _max_tokens() -> int:
    # Generous: reasoning models (Gemini 2.5) spend tokens "thinking", which would otherwise
    # starve the short answer. The situating sentence itself is only ~50-100 tokens.
    return int(os.getenv("CONTEXTUAL_MAX_TOKENS", "1024"))


def _retries() -> int:
    return max(1, int(os.getenv("CONTEXTUAL_RETRIES", "3")))


def _cache_key(doc_text: str, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update((doc_text or "")[: _doc_chars()].encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update((chunk_text or "").encode("utf-8", "ignore"))
    return h.hexdigest()


def _load_cache() -> Dict[str, str]:
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict[str, str]) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _provider_chain():
    """Chat providers to try, in order: the configured model (e.g. Gemini), then the fallback
    models (e.g. Mistral Large) — so contextualization keeps working if the first model is
    rate-limited/quota-exhausted. Only available providers (key present) are included; deduped."""
    from backend.llm.streaming_provider import get_provider
    out, seen = [], set()
    names = [None] + [m.strip() for m in
                      os.getenv("CONTEXTUAL_FALLBACK_MODELS", "mistral-large-latest").split(",")]
    for name in names:
        if name is not None and not name:
            continue
        try:
            p = get_provider(name) if name else get_provider()
        except Exception:
            continue
        mid = getattr(p, "model", name)
        if mid in seen or not getattr(p, "is_available", False):
            continue
        seen.add(mid)
        out.append(p)
    return out


def _complete(provider, prompt: str) -> str:
    """One situating-sentence call against a single provider. Returns "" on error/empty/unavailable
    (never raises). Skips reasoning dicts; keeps answer text only."""
    try:
        if not getattr(provider, "is_available", False):
            return ""
        parts: List[str] = []
        for piece in provider.stream_chat(
            [{"role": "user", "content": prompt}],
            system=_SYSTEM, max_tokens=_max_tokens(), temperature=0.0,
        ):
            if isinstance(piece, str):
                parts.append(piece)
        return " ".join("".join(parts).split()).strip()
    except Exception:
        return ""


def _run_chain(chain: list, prompt: str) -> str:
    """Try each provider in order; on success promote the winner so later chunks skip dead models.
    Retries the whole chain with backoff for transient rate-limit / empty replies."""
    if not chain:
        return ""
    for attempt in range(_retries()):
        for i, prov in enumerate(list(chain)):
            out = _complete(prov, prompt)
            if out:
                if i > 0:
                    chain.insert(0, chain.pop(i))   # prefer the working model next time
                return out
        if attempt < _retries() - 1:
            time.sleep(min(2 ** attempt, 20))
    return ""


def generate_context(doc_text: str, chunk_text: str, provider=None) -> str:
    """One situating sentence for `chunk_text`. Returns "" on any failure (caller falls back to the
    plain chunk). With no `provider`, tries the configured model then the fallback model(s).
    `provider` is injectable for tests (single provider, single attempt)."""
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return ""
    prompt = _PROMPT.format(doc=(doc_text or "")[: _doc_chars()], chunk=chunk_text[:4000])
    if provider is not None:
        return _complete(provider, prompt)
    try:
        chain = _provider_chain()
    except Exception:
        return ""
    return _run_chain(chain, prompt)


def contextualize_chunks(doc_text: str, chunks: List[dict], provider=None) -> List[str]:
    """Return one context string per chunk (same order). Builds the provider chain once, uses the
    on-disk cache, and only calls the LLM for cache misses. Returns ['']*len when disabled. Never
    raises."""
    n = len(chunks)
    if not contextual_enabled() or n == 0:
        return [""] * n
    chain = [provider] if provider is not None else _provider_chain()
    cache = _load_cache()
    out: List[str] = []
    dirty = False
    for ch in chunks:
        ctext = (ch.get("text") or "").strip()
        if not ctext:
            out.append("")
            continue
        key = _cache_key(doc_text, ctext)
        if key in cache:
            out.append(cache[key])
            continue
        prompt = _PROMPT.format(doc=(doc_text or "")[: _doc_chars()], chunk=ctext[:4000])
        ctx = _run_chain(chain, prompt)
        cache[key] = ctx
        dirty = True
        out.append(ctx)
    if dirty:
        _save_cache(cache)
    return out
