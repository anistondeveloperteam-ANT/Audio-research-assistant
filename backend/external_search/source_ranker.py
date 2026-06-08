"""
De-duplicate and re-rank external sources against the original user query.

Re-ranking reuses the project's cross-encoder reranker when it's already loaded
(same model the local pipeline uses); if it isn't available it falls back to a
cheap lexical overlap score. De-dup is by content hash (url / path / text).
"""
from __future__ import annotations

import datetime
import re
from typing import List

from backend.external_search.base import ExternalSource, env_flag, logger

_TOKEN = re.compile(r"[a-z0-9]+")
_RECENCY_WORDS = re.compile(
    r"\b(latest|newest|recent|recently|today|now|current|currently|state[- ]of[- ]the[- ]art|"
    r"sota|up[- ]?to[- ]?date|this year|new)\b", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _wants_latest(query: str) -> bool:
    q = query or ""
    return bool(_RECENCY_WORDS.search(q) or _YEAR.search(q))


def _recency_score(published) -> float:
    """0..1 by publication year: ~1 this year, decaying over ~6 years."""
    if not published:
        return 0.0
    m = re.match(r"(\d{4})", str(published))
    if not m:
        return 0.0
    try:
        this_year = datetime.date.today().year
    except Exception:
        this_year = 2026
    age = max(0, this_year - int(m.group(1)))
    return max(0.0, 1.0 - age / 6.0)

# Use the local cross-encoder reranker for external sources only when explicitly
# enabled (it loads torch + the model). Off by default so a web-only production
# deploy stays light and uses the fast lexical scorer.
USE_CROSS_ENCODER = env_flag("EXTERNAL_RERANK_CROSS_ENCODER", default=env_flag("ENABLE_LOCAL_RAG"))


def deduplicate(sources: List[ExternalSource]) -> List[ExternalSource]:
    """Collapse duplicates by content hash, keeping the higher-scoring copy."""
    best: dict[str, ExternalSource] = {}
    for s in sources:
        h = s.content_hash()
        if h not in best or s.score > best[h].score:
            best[h] = s
    return list(best.values())


def _lexical_score(query: str, text: str) -> float:
    q = set(_TOKEN.findall((query or "").lower()))
    t = set(_TOKEN.findall((text or "").lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def rerank_sources(query: str, sources: List[ExternalSource], top_k: int = 6) -> List[ExternalSource]:
    """De-dup, score each source against the query, and return the top_k."""
    sources = deduplicate(sources)
    if not sources:
        return []

    scored_by_model = False
    if USE_CROSS_ENCODER:
        try:
            from backend.retrieval.hybrid_retrieve import get_reranker
            reranker = get_reranker()
            pairs = [(query, (s.text or s.snippet or s.title or "")[:1200]) for s in sources]
            preds = reranker.predict(pairs)
            for s, p in zip(sources, preds):
                s.score = float(p)
            scored_by_model = True
        except Exception as exc:
            logger.info("external rerank fell back to lexical (%s)", type(exc).__name__)

    if not scored_by_model:
        for s in sources:
            s.score = _lexical_score(query, s.text or s.snippet or s.title)

    # Blend in recency: normalize relevance to 0..1, then add a freshness bonus so
    # newly published papers/repos rank up — strongly when the user asks for "latest".
    weight = 0.5 if _wants_latest(query) else 0.12
    vals = [float(s.score) for s in sources]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    for s in sources:
        rel = (float(s.score) - lo) / rng
        s.score = rel + weight * _recency_score(s.published)

    sources.sort(key=lambda s: s.score, reverse=True)
    return sources[:top_k]
