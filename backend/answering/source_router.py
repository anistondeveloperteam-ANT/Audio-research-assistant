"""
source_router.py — decide WHICH source a question needs BEFORE answering, so the indexed corpus is
never the default for everything.

Routes:
  "reasoning" — self-contained / general knowledge / definition / basic fact / calculation / logic.
                Answer from the model's own knowledge; do NOT retrieve or cite the corpus.
  "web"       — current / time-sensitive ("latest", "now", "state of the art today"). Search the web and
                answer from current sources; never serve stale corpus/training content as current.
  "corpus"    — genuinely about the indexed documents' subject matter / specific literature / findings.
                Retrieve and cite (the existing CRAG path).

Deterministic fast-paths (passed in by the caller) settle the obvious cases — a calculation -> reasoning,
a recency cue -> web; the LLM classifier decides the rest. Gated (SOURCE_ROUTER, default on), cached, and
FAIL-OPEN to "corpus" so any disabled/empty/error case keeps the existing retrieve-first behaviour. No
new dependency.
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import List, Optional

REASONING = "reasoning"
WEB = "web"
CORPUS = "corpus"
_ROUTES = (REASONING, WEB, CORPUS)

_MAX_TOKENS = 24
_HARD_CHAR_CAP = 120
_CACHE_MAX = 512

_SOURCE_SYSTEM = (
    "You decide WHICH source is needed to correctly answer a user's question in a research assistant. "
    "Choose by what answering it REQUIRES, not by how familiar the topic is:\n"
    '  "reasoning" — the question is self-contained / general knowledge / a definition / a basic fact / '
    "a calculation / a logic or standard textbook-concept question that you can answer CORRECTLY from "
    "your own knowledge and reasoning, with NO external source (no documents or web needed).\n"
    '  "web" — answering REQUIRES current / up-to-date / recent information: "latest", "current", '
    '"newest", "now", "this year", a recent date, "state of the art today", prices, ongoing events. '
    "Static training knowledge would be stale, so the web must be searched.\n"
    '  "documents" — answering genuinely DEPENDS on specific research literature / findings / methods / '
    "datasets, or specific external facts you should not guess; it should be grounded in and cited from "
    "indexed source documents.\n"
    "Tie-breakers: if torn between reasoning and documents, prefer 'reasoning' UNLESS the answer truly "
    "depends on specific documents. If recency is plausibly required, prefer 'web' over stale content. "
    "Reply with ONLY one word: reasoning, web, or documents."
)


def source_router_enabled() -> bool:
    return os.getenv("SOURCE_ROUTER", "true").strip().lower() not in ("0", "false", "no", "off")


def _timeout() -> float:
    try:
        return max(0.3, float(os.getenv("SOURCE_ROUTER_TIMEOUT", "3.0")))
    except (TypeError, ValueError):
        return 3.0


# In-process LRU cache (same pattern as task_classifier).
_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
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


def _parse(raw: str) -> Optional[str]:
    """Map the model's one-word reply to a route, tolerant of quotes/JSON/punctuation. 'documents'/
    'corpus' -> corpus, 'current' -> web. Returns None when nothing recognisable is present."""
    if not raw:
        return None
    m = re.search(r"\b(reasoning|web|documents|corpus|current)\b", raw.strip().lower())
    if not m:
        return None
    w = m.group(1)
    if w in ("documents", "corpus"):
        return CORPUS
    if w == "current":
        return WEB
    return w                                  # reasoning | web


def classify_source(provider, question: str) -> Optional[str]:
    """One bounded LLM call -> 'reasoning' | 'web' | 'corpus', or None on unavailability / timeout /
    parse failure (the caller falls open to 'corpus'). Never raises."""
    if provider is None or not getattr(provider, "is_available", False):
        return None

    def _run() -> str:
        parts: List[str] = []
        for tok in provider.stream_chat(
            [{"role": "user", "content": question}],
            system=_SOURCE_SYSTEM, max_tokens=_MAX_TOKENS, temperature=0.0,
        ):
            if not isinstance(tok, str):
                continue
            parts.append(tok)
            if sum(len(p) for p in parts) > _HARD_CHAR_CAP:
                break
        return "".join(parts)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            raw = fut.result(timeout=_timeout())
        except Exception:                     # noqa: BLE001 - timeout / provider error -> fall open
            return None
    return _parse(raw)


def decide_source(provider, question: Optional[str], *, freshness: bool, calc: bool) -> str:
    """Decide the source a question needs: 'reasoning' | 'web' | 'corpus'.

    Deterministic fast-paths first (a self-contained calculation -> reasoning; a recency cue -> web),
    then the LLM classifier. FAIL-OPEN to 'corpus' (the existing retrieve-first behaviour) when the
    router is disabled, the provider is unavailable, or anything fails — so it never regresses."""
    q = (question or "").strip()
    if not q:
        return CORPUS
    if calc:
        return REASONING
    if freshness:
        return WEB
    if not source_router_enabled():
        return CORPUS
    key = " ".join(q.lower().split())
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        verdict = classify_source(provider, q)
    except Exception:                         # noqa: BLE001 - never break routing
        verdict = None
    result = verdict if verdict in _ROUTES else CORPUS
    if verdict is not None:                   # don't cache transient failures
        _cache_put(key, result)
    return result
