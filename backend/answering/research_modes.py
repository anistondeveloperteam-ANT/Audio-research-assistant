"""
Run profiles: FAST (default) vs DEEP.

FAST is local-first and quick — no multi-query planning, light external search, a single
verification round, no auto-review. DEEP does the full sweep — sub-queries, broader web/arXiv
reading, multiple verification rounds, and auto-review — for expensive research questions.

`resolve_research_mode(mode)` returns the profile as an ENV-KEYED settings map (e.g. VECTOR_TOP_K,
EXTERNAL_TOP_K, AGENTIC_MAX_VERIFY_ROUNDS). The caller binds it to the per-request context
(`backend.common.request_context`); the retrieval + chat pipelines read those knobs via the context's
typed getters, so each concurrent request reads ONLY its own profile. Nothing is written to os.environ,
so two simultaneous requests with different modes can never clobber each other.
"""
from __future__ import annotations

from typing import Any, Dict

# Retrieval knobs are shared across modes (the retrieval engine is already tuned).
_RETRIEVAL_BASE: Dict[str, Any] = {
    "max_query_routes": 3,
    "total_source_limit": 12,
    "per_topic_source_limit": 2,
    "max_sources_per_paper": 2,
    "vector_top_k": 24,
    "bm25_top_k": 24,
    "rerank_top_n": 24,
    "context_budget_chars": 24000,
    "question_memory_threshold": 0.72,
    "use_question_memory": True,
}

# FAST = the recommended local-first, quick defaults.
FAST_SETTINGS: Dict[str, Any] = {
    **_RETRIEVAL_BASE, "mode": "fast",
    "deep_search_subqueries": 0,
    "external_top_k": 8,
    "web_max_results": 4,
    "arxiv_read_pdf_count": 0,
    "agentic_max_verify_rounds": 1,
    "deep_max_loops": 1,
    "agentic_min_verify_score": 80,
    "auto_review": False,
    "evidence_budget_chars": 14000,
    "answer_max_tokens": 3000,
    "external_gather_timeout": 12,
}

# DEEP = the full research sweep (the project's prior rich defaults).
DEEP_SETTINGS: Dict[str, Any] = {
    **_RETRIEVAL_BASE, "mode": "deep",
    "deep_search_subqueries": 3,
    "external_top_k": 20,
    "web_max_results": 8,
    "arxiv_read_pdf_count": 3,
    "agentic_max_verify_rounds": 3,
    # Match max_verify_rounds so DEEP keeps its full thoroughness: the loop early-stops when a
    # verdict has no concrete, fixable gap (waste removal), but a query that genuinely needs 3
    # gap-filling rounds still gets all 3. Lower this only to trade quality for speed deliberately.
    "deep_max_loops": 3,
    "agentic_min_verify_score": 80,
    "auto_review": True,
    "evidence_budget_chars": 28000,
    "answer_max_tokens": 8000,
    "external_gather_timeout": 30,
}

MODE_SETTINGS: Dict[str, Dict[str, Any]] = {"fast": FAST_SETTINGS, "deep": DEEP_SETTINGS}
DEFAULT_RETRIEVAL_SETTINGS = FAST_SETTINGS   # back-compat alias


def normalize_mode(mode: str | None = None) -> str:
    """Map any input to 'fast' (default) or 'deep'. 'Deep'/'Deep Research' -> deep; else fast."""
    m = (mode or "").strip().lower()
    if m in ("deep", "deep research", "research", "thorough"):
        return "deep"
    return "fast"


def get_mode_settings(mode: str | None = None) -> Dict[str, Any]:
    return dict(MODE_SETTINGS[normalize_mode(mode)])


# Settings key -> the ENV-style name the pipeline's request-context getters read.
_ENV_KEYS: Dict[str, str] = {
    "RESEARCH_MODE": "mode",
    "MAX_QUERY_ROUTES": "max_query_routes",
    "TOTAL_SOURCE_LIMIT": "total_source_limit",
    "PER_TOPIC_SOURCE_LIMIT": "per_topic_source_limit",
    "MAX_SOURCES_PER_PAPER": "max_sources_per_paper",
    "VECTOR_TOP_K": "vector_top_k",
    "BM25_TOP_K": "bm25_top_k",
    "RERANK_TOP_N": "rerank_top_n",
    "SOURCE_CONTEXT_BUDGET_CHARS": "context_budget_chars",
    "QUESTION_MEMORY_THRESHOLD": "question_memory_threshold",
    "USE_QUESTION_MEMORY": "use_question_memory",
    "DEEP_SEARCH_SUBQUERIES": "deep_search_subqueries",
    "EXTERNAL_TOP_K": "external_top_k",
    "WEB_MAX_RESULTS": "web_max_results",
    "ARXIV_READ_PDF_COUNT": "arxiv_read_pdf_count",
    "AGENTIC_MAX_VERIFY_ROUNDS": "agentic_max_verify_rounds",
    "DEEP_MAX_LOOPS": "deep_max_loops",
    "AGENTIC_MIN_VERIFY_SCORE": "agentic_min_verify_score",
    "AUTO_REVIEW": "auto_review",
    "EVIDENCE_BUDGET_CHARS": "evidence_budget_chars",
    "ANSWER_MAX_TOKENS": "answer_max_tokens",
    "EXTERNAL_GATHER_TIMEOUT": "external_gather_timeout",
}


def resolve_research_mode(mode: str | None = None) -> Dict[str, Any]:
    """The Fast/Deep run profile as an ENV-KEYED settings map (typed values) to BIND to the per-request
    context — never written to os.environ, so concurrent Fast/Deep requests can't clobber one another.
    During a request the bound profile is the authority for these knobs (matching the prior behaviour
    where the mode was applied per request); OFF-request, the getters fall back to env then default."""
    s = get_mode_settings(mode)
    return {env: s[key] for env, key in _ENV_KEYS.items()}


if __name__ == "__main__":
    import sys
    for k, v in resolve_research_mode(sys.argv[1] if len(sys.argv) > 1 else None).items():
        print(f"{k}: {v}")
