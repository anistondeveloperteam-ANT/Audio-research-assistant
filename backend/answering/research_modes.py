"""
Run profiles: FAST (default) vs DEEP.

FAST is local-first and quick — no multi-query planning, light external search, a single
verification round, no auto-review. DEEP does the full sweep — sub-queries, broader web/arXiv
reading, multiple verification rounds, and auto-review — for expensive research questions.

`apply_research_mode(mode)` writes the profile to the process environment (NOT the .env file).
The retrieval pipeline (hybrid_retrieve.py) and the chat pipeline (chat_logic.py / orchestrator.py
/ agentic_answer.py) read those vars LIVE, so a mode applies per request. Note: env is process-
global, so concurrent requests with different modes can interleave — fine for local single-user use.
"""
from __future__ import annotations

import os
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


def apply_research_mode(mode: str | None = None) -> Dict[str, Any]:
    """Write the chosen profile to the process environment so retrieval + chat read it live."""
    s = get_mode_settings(mode)
    os.environ["RESEARCH_MODE"] = s["mode"]
    # Retrieval engine knobs (read by hybrid_retrieve via _mode_int).
    os.environ["MAX_QUERY_ROUTES"] = str(s["max_query_routes"])
    os.environ["TOTAL_SOURCE_LIMIT"] = str(s["total_source_limit"])
    os.environ["PER_TOPIC_SOURCE_LIMIT"] = str(s["per_topic_source_limit"])
    os.environ["MAX_SOURCES_PER_PAPER"] = str(s["max_sources_per_paper"])
    os.environ["VECTOR_TOP_K"] = str(s["vector_top_k"])
    os.environ["BM25_TOP_K"] = str(s["bm25_top_k"])
    os.environ["RERANK_TOP_N"] = str(s["rerank_top_n"])
    os.environ["SOURCE_CONTEXT_BUDGET_CHARS"] = str(s["context_budget_chars"])
    os.environ["QUESTION_MEMORY_THRESHOLD"] = str(s["question_memory_threshold"])
    os.environ["USE_QUESTION_MEMORY"] = "true" if s["use_question_memory"] else "false"
    # Chat-pipeline cost knobs (read live by chat_logic / orchestrator / agentic_answer).
    os.environ["DEEP_SEARCH_SUBQUERIES"] = str(s["deep_search_subqueries"])
    os.environ["EXTERNAL_TOP_K"] = str(s["external_top_k"])
    os.environ["WEB_MAX_RESULTS"] = str(s["web_max_results"])
    os.environ["ARXIV_READ_PDF_COUNT"] = str(s["arxiv_read_pdf_count"])
    os.environ["AGENTIC_MAX_VERIFY_ROUNDS"] = str(s["agentic_max_verify_rounds"])
    os.environ["AGENTIC_MIN_VERIFY_SCORE"] = str(s["agentic_min_verify_score"])
    os.environ["AUTO_REVIEW"] = "true" if s["auto_review"] else "false"
    os.environ["EVIDENCE_BUDGET_CHARS"] = str(s["evidence_budget_chars"])
    os.environ["ANSWER_MAX_TOKENS"] = str(s["answer_max_tokens"])
    os.environ["EXTERNAL_GATHER_TIMEOUT"] = str(s["external_gather_timeout"])
    return s


if __name__ == "__main__":
    import sys
    for k, v in apply_research_mode(sys.argv[1] if len(sys.argv) > 1 else None).items():
        print(f"{k}: {v}")
