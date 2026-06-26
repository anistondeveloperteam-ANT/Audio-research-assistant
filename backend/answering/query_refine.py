"""
query_refine.py  --  Silently fix a user's spelling/grammar BEFORE search.

A typo poisons the whole pipeline: the embedding is off, BM25 misses the
misspelled term, and external search queries the typo verbatim. `query_sanity`
only *rejects* gibberish; it never *corrects* a real-but-misspelled query.

This module corrects the query in the backend, then search proceeds normally
(the user sees nothing — the correction is silent). Design goals:

  * Context-aware  -- an LLM fixes typos/grammar while PRESERVING technical
                      terms, acronyms, proper nouns, code, numbers, and meaning
                      (a generic spell dictionary would corrupt "kubernetes",
                      "MVDR", "FastAPI" -> wrong common words).
  * Low latency    -- a cheap gate skips the LLM for already-clean queries;
                      results are cached so repeats are free.
  * Never breaks   -- a tight timeout + a catch-all fall back to the ORIGINAL
                      query on any provider error/timeout, so a request is
                      never slowed down or broken by this step.

No new dependency: reuses the LLM provider the app already loads.

Public API:
    refine_query("i want to exploer delhi") -> "i want to explore delhi"
    clear_cache()                            -> reset the in-process LRU (tests)

Toggle / tune via .env (read live, never frozen into constants):
    QUERY_REFINE=true|false           (default true)
    QUERY_REFINE_TIMEOUT=3.0          (seconds; fall back to original after this)
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import List, Optional

from backend.answering.query_sanity import _LEGIT_WORDS, _is_legit_word

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------

_MIN_CHARS = 4              # nothing shorter is worth a correction pass
_MIN_TOKEN_LEN = 3         # tokens shorter than this don't gate the LLM
_MAX_TOKENS = 96           # the corrected query is short; cap the LLM output
_HARD_CHAR_CAP = 400       # stop reading the stream past this (runaway answer)
_DEFAULT_TIMEOUT = 3.0     # seconds before we give up and use the original
_CACHE_MAX = 512           # bounded in-process LRU

_SYSTEM_PROMPT = (
    "You correct a user's search query before it is sent to a search engine. "
    "Fix ONLY spelling, typos, and basic grammar. Keep all technical terms, "
    "acronyms, product names, proper nouns, numbers, and the original meaning "
    "and language unchanged. Do NOT answer the query, do NOT add or remove "
    "information, do NOT explain. Reply with ONLY the corrected query as a "
    "single line."
)

_LABEL_RE = re.compile(r"^(?:corrected(?:\s+query)?|query|answer)\s*:\s*", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[^A-Za-z']+")

# ----------------------------------------------------------------------
# In-process LRU cache (same query -> same correction, for free)
# ----------------------------------------------------------------------

_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Empty the correction cache (used by tests)."""
    with _cache_lock:
        _cache.clear()


def _cache_get(key: str) -> Optional[str]:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _cache_put(key: str, value: str) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


# ----------------------------------------------------------------------
# Config (read live so .env / test monkeypatching takes effect)
# ----------------------------------------------------------------------

def _enabled() -> bool:
    return os.getenv("QUERY_REFINE", "true").strip().lower() not in ("0", "false", "no", "off")


def _timeout() -> float:
    try:
        return max(0.3, float(os.getenv("QUERY_REFINE_TIMEOUT", str(_DEFAULT_TIMEOUT))))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


# ----------------------------------------------------------------------
# Gate: is this query already clean? (cheap, dependency-free)
# ----------------------------------------------------------------------

def _is_known_token(raw: str) -> bool:
    """A token is 'known' (not a typo signal) if it's a recognized word, an
    acronym, a code/identifier, or carries a digit. Over-counting as known only
    risks skipping a correction; it never corrupts text."""
    if any(c.isdigit() for c in raw):
        return True
    if len(raw) >= 2 and raw.isupper():            # acronym: MVDR, FFT, RAG
        return True
    if any(c.isupper() for c in raw[1:]):          # camelCase / FastAPI / arXiv
        return True
    return _is_legit_word(raw) or raw.lower() in _LEGIT_WORDS


def _should_refine(q: str) -> bool:
    """True if any meaningful token looks unrecognized (a possible typo)."""
    meaningful = [t for t in _TOKEN_RE.split(q) if len(t) >= _MIN_TOKEN_LEN]
    if not meaningful:
        return False
    return any(not _is_known_token(t) for t in meaningful)


# ----------------------------------------------------------------------
# LLM correction + output sanitization
# ----------------------------------------------------------------------

def _sanitize(raw: str, original: str) -> Optional[str]:
    """Pull a single clean corrected line out of the model's reply, or None if
    it doesn't look like a plain query (rambled / answered / empty)."""
    if not raw:
        return None
    # First non-empty line only — the corrected query is a single line.
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    if not line:
        return None
    line = _LABEL_RE.sub("", line).strip()
    if (line.startswith(('"', "'", "`")) and line.endswith(('"', "'", "`")) and len(line) >= 2):
        line = line[1:-1].strip()
    if not line:
        return None
    # A real correction stays close in length; a long reply means the model
    # answered/explained instead of correcting -> reject, keep the original.
    if len(line) > max(80, len(original) * 3):
        return None
    return line


def _llm_correct(q: str) -> Optional[str]:
    """One short, timeout-bounded LLM call. Returns the corrected query, or
    None on unavailability / timeout / error (caller falls back to original)."""
    from backend.llm.streaming_provider import get_provider

    provider = get_provider()
    if not provider.is_available:
        return None

    def _run() -> str:
        parts: List[str] = []
        total = 0
        for tok in provider.stream_chat(
            [{"role": "user", "content": q}],
            system=_SYSTEM_PROMPT,
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        ):
            if not isinstance(tok, str):           # ignore reasoning dicts, etc.
                continue
            parts.append(tok)
            total += len(tok)
            if total > _HARD_CHAR_CAP:
                break
        return "".join(parts)

    from backend.common.request_context import ContextThreadPoolExecutor
    with ContextThreadPoolExecutor(max_workers=1) as ex:   # worker inherits the request's model
        fut = ex.submit(_run)
        try:
            raw = fut.result(timeout=_timeout())
        except Exception:                           # noqa: BLE001 - timeout or provider error
            return None
    return _sanitize(raw, q)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def refine_query(query: Optional[str]) -> str:
    """Return a spelling/grammar-corrected version of `query` for search.

    Silent and safe: skips already-clean queries (no LLM call), caches results,
    and falls back to the original on disabled/short input or any error/timeout.
    Never raises.
    """
    original = query or ""
    try:
        if not _enabled():
            return original
        q = original.strip()
        if len(q) < _MIN_CHARS or not _should_refine(q):
            return original
        key = " ".join(q.lower().split())
        cached = _cache_get(key)
        if cached is not None:
            return cached
        corrected = (_llm_correct(q) or "").strip() or q
        _cache_put(key, corrected)
        return corrected
    except Exception:                               # noqa: BLE001 - never break a request
        return original
