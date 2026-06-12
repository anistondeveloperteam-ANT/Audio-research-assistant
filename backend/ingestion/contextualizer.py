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


def generate_context(doc_text: str, chunk_text: str, provider=None) -> str:
    """One situating sentence for `chunk_text`. Returns "" on any failure (caller falls back to the
    plain chunk). Retries with backoff on transient rate‑limit/quota/5xx errors. `provider` is
    injectable for tests; otherwise the configured chat model is used."""
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return ""
    try:
        if provider is None:
            from backend.llm.streaming_provider import get_provider
            provider = get_provider()
    except Exception:
        return ""
    if not getattr(provider, "is_available", False):
        return ""
    prompt = _PROMPT.format(doc=(doc_text or "")[: _doc_chars()], chunk=chunk_text[:4000])
    retries = _retries()
    for attempt in range(retries):
        try:
            parts: List[str] = []
            for piece in provider.stream_chat(
                [{"role": "user", "content": prompt}],
                system=_SYSTEM, max_tokens=_max_tokens(), temperature=0.0,
            ):
                if isinstance(piece, str):      # skip any reasoning dicts; keep answer text only
                    parts.append(piece)
            out = " ".join("".join(parts).split()).strip()
            if out:
                return out
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(k in msg for k in
                            ("429", "rate", "quota", "timeout", "unavailable", "503", "500"))
            if not transient:
                return ""                       # hard error (bad key/request) -> don't hammer
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 20))   # backoff before retrying a rate-limit / empty reply
    return ""


def contextualize_chunks(doc_text: str, chunks: List[dict], provider=None) -> List[str]:
    """Return one context string per chunk (same order). Uses the on-disk cache and only calls the
    LLM for cache misses. Returns ['']*len(chunks) when disabled. Never raises."""
    n = len(chunks)
    if not contextual_enabled() or n == 0:
        return [""] * n
    cache = _load_cache()
    out: List[str] = []
    dirty = False
    for ch in chunks:
        ctext = (ch.get("text") or "").strip()
        key = _cache_key(doc_text, ctext)
        if key in cache:
            out.append(cache[key])
            continue
        ctx = generate_context(doc_text, ctext, provider=provider)
        cache[key] = ctx
        dirty = True
        out.append(ctx)
    if dirty:
        _save_cache(cache)
    return out
