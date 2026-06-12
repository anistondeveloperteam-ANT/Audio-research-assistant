"""
Fast (default) vs Deep run profiles. Fast is local-first + cheap; Deep does the full sweep.
The verification accuracy bar (AGENTIC_MIN_VERIFY_SCORE) is the SAME in both — speed never
lowers the threshold.
"""
import os

import pytest

from backend.answering.research_modes import apply_research_mode, normalize_mode

_KEYS = ["RESEARCH_MODE", "DEEP_SEARCH_SUBQUERIES", "EXTERNAL_TOP_K", "ANSWER_MAX_TOKENS",
         "AGENTIC_MAX_VERIFY_ROUNDS", "AGENTIC_MIN_VERIFY_SCORE", "AUTO_REVIEW",
         "ARXIV_READ_PDF_COUNT", "WEB_MAX_RESULTS", "EVIDENCE_BUDGET_CHARS",
         "EXTERNAL_GATHER_TIMEOUT", "VECTOR_TOP_K", "BM25_TOP_K"]


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in _KEYS}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_normalize_mode_defaults_to_fast():
    assert normalize_mode("deep") == "deep"
    assert normalize_mode("Deep Research") == "deep"
    assert normalize_mode("fast") == "fast"
    assert normalize_mode("Default") == "fast"
    assert normalize_mode("") == "fast"
    assert normalize_mode(None) == "fast"


def test_fast_profile_is_cheap_and_local_first():
    s = apply_research_mode("fast")
    assert s["deep_search_subqueries"] == 0 and s["arxiv_read_pdf_count"] == 0
    assert s["agentic_max_verify_rounds"] == 1 and s["auto_review"] is False
    assert os.getenv("DEEP_SEARCH_SUBQUERIES") == "0"
    assert os.getenv("AUTO_REVIEW") == "false"
    assert os.getenv("ANSWER_MAX_TOKENS") == "3000"
    assert os.getenv("EXTERNAL_GATHER_TIMEOUT") == "12"


def test_deep_profile_is_rich():
    s = apply_research_mode("deep")
    assert s["deep_search_subqueries"] == 3 and s["arxiv_read_pdf_count"] == 3
    assert s["agentic_max_verify_rounds"] == 3 and s["auto_review"] is True
    assert os.getenv("AGENTIC_MAX_VERIFY_ROUNDS") == "3"


def test_accuracy_threshold_unchanged_across_modes():
    assert apply_research_mode("fast")["agentic_min_verify_score"] == 80
    assert apply_research_mode("deep")["agentic_min_verify_score"] == 80


def test_chat_logic_live_getters_follow_the_mode():
    import webapp.chat_logic as CL
    apply_research_mode("fast")
    assert CL._deep_subqueries() == 0 and CL._answer_max_tokens() == 3000 and CL._external_top_k() == 8
    apply_research_mode("deep")
    assert CL._deep_subqueries() == 3 and CL._answer_max_tokens() == 8000 and CL._external_top_k() == 20
