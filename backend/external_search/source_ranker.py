"""
De-duplicate and re-rank external sources against the original user query.

Re-ranking reuses the project's cross-encoder reranker when it's already loaded
(same model the local pipeline uses); if it isn't available it falls back to a
cheap lexical overlap score. De-dup is by content hash (url / path / text).
"""
from __future__ import annotations

import re
from typing import List

from backend.external_search.base import ExternalSource, logger

_TOKEN = re.compile(r"[a-z0-9]+")


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

    sources.sort(key=lambda s: s.score, reverse=True)
    return sources[:top_k]
