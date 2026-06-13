"""
Reference-code retrieval (any domain). For a code task, fetch 1-2 stars-first GitHub reference
implementations of the NAMED algorithm/topic and present them to the agent clearly marked
REFERENCE (with stars/license) — to ADAPT, not copy. No hardcoded repos; the topic is extracted
from the query, so it works for RTF-MVDR, Black-Scholes, Dijkstra, anything.
"""
from __future__ import annotations

import re
from typing import List

# Request boilerplate to strip so the GitHub query is the algorithm/topic itself.
_STOP = {
    "give", "me", "the", "a", "an", "please", "code", "python", "script", "program",
    "function", "implementation", "implement", "write", "generate", "show", "build",
    "create", "make", "simulate", "simulation", "for", "to", "that", "which", "of",
    "in", "with", "using", "and", "or", "compute", "calculate", "provide", "need", "want",
}


def topic_of(query: str) -> str:
    """The algorithm/topic phrase to search GitHub for (request boilerplate stripped)."""
    words = re.findall(r"[A-Za-z0-9+#.\-]+", query or "")
    kept = [w for w in words if w.lower() not in _STOP]
    return " ".join(kept).strip() or (query or "").strip()


def fetch_reference_code(query: str, max_repos: int = 2, max_chars: int = 4000) -> str:
    """A REFERENCE block of 1-2 stars-first GitHub implementations of the query's topic, each
    marked 'adapt, do not copy' with its stars/license. Returns "" on any failure (never fatal)."""
    topic = topic_of(query)
    if not topic:
        return ""
    try:
        from backend.external_search.github_search import github_search
        sources = github_search(topic, max_repos=max_repos) or []
    except Exception:
        return ""
    blocks: List[str] = []
    for s in sources[:max_repos]:
        text = (getattr(s, "text", "") or getattr(s, "snippet", "") or "").strip()
        if not text:
            continue
        title = getattr(s, "title", "") or getattr(s, "url", "") or "repository"
        lic = getattr(s, "license", None) or "license unknown"
        blocks.append(
            f"REFERENCE — {title} ({lic}); ADAPT the approach, do NOT copy verbatim:\n"
            f"{text[:max_chars]}"
        )
    return "\n\n".join(blocks)
