"""
Fast (default) vs Deep run profiles, now bound PER REQUEST (no process-global env mutation). Fast is
local-first + cheap; Deep does the full sweep. The verification accuracy bar (AGENTIC_MIN_VERIFY_SCORE)
is the SAME in both — speed never lowers the threshold. `resolve_research_mode` returns an ENV-KEYED
settings map to bind to the request context; it writes NOTHING to os.environ.
"""
import os

import pytest

from backend.answering.research_modes import (get_mode_settings, normalize_mode,
                                              resolve_research_mode)
from backend.common import request_context as rc

# Mode knobs that must stay UNSET as env so the profile values come through (resolve honours an explicit
# env override, which would otherwise mask the profile).
_KEYS = ["RESEARCH_MODE", "DEEP_SEARCH_SUBQUERIES", "EXTERNAL_TOP_K", "ANSWER_MAX_TOKENS",
         "AGENTIC_MAX_VERIFY_ROUNDS", "AGENTIC_MIN_VERIFY_SCORE", "AUTO_REVIEW", "ARXIV_READ_PDF_COUNT",
         "WEB_MAX_RESULTS", "EVIDENCE_BUDGET_CHARS", "EXTERNAL_GATHER_TIMEOUT", "VECTOR_TOP_K",
         "BM25_TOP_K"]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)       # no static override -> pure profile values
    rc.clear_request_settings()
    yield
    rc.clear_request_settings()


def test_normalize_mode_defaults_to_fast():
    assert normalize_mode("deep") == "deep"
    assert normalize_mode("Deep Research") == "deep"
    assert normalize_mode("fast") == "fast"
    assert normalize_mode("Default") == "fast"
    assert normalize_mode("") == "fast"
    assert normalize_mode(None) == "fast"


def test_fast_profile_is_cheap_and_local_first():
    s = get_mode_settings("fast")
    assert s["deep_search_subqueries"] == 0 and s["arxiv_read_pdf_count"] == 0
    assert s["agentic_max_verify_rounds"] == 1 and s["auto_review"] is False
    env = resolve_research_mode("fast")            # ENV-KEYED map; nothing written to os.environ
    assert env["DEEP_SEARCH_SUBQUERIES"] == 0 and env["AUTO_REVIEW"] is False
    assert env["ANSWER_MAX_TOKENS"] == 3000 and env["EXTERNAL_GATHER_TIMEOUT"] == 12


def test_deep_profile_is_rich():
    s = get_mode_settings("deep")
    assert s["deep_search_subqueries"] == 3 and s["arxiv_read_pdf_count"] == 3
    assert s["agentic_max_verify_rounds"] == 3 and s["auto_review"] is True
    assert resolve_research_mode("deep")["AGENTIC_MAX_VERIFY_ROUNDS"] == 3


def test_accuracy_threshold_unchanged_across_modes():
    assert get_mode_settings("fast")["agentic_min_verify_score"] == 80
    assert get_mode_settings("deep")["agentic_min_verify_score"] == 80


def test_resolve_never_writes_os_environ():
    before = dict(os.environ)
    resolve_research_mode("deep")
    resolve_research_mode("fast")
    assert dict(os.environ) == before              # pure resolver — no process-global mutation


def test_bound_profile_is_the_authority_over_a_stale_env(monkeypatch):
    # Even with a stale env value present, the BOUND request profile wins during the request (matching the
    # prior apply-per-request behaviour). Off-request the getter would fall back to env/default.
    monkeypatch.setenv("EXTERNAL_TOP_K", "999")
    import webapp.chat_logic as CL
    rc.set_request_settings(resolve_research_mode("fast"))
    assert CL._external_top_k() == 8               # the request's profile, not the stale env 999


def test_chat_logic_live_getters_follow_the_bound_mode():
    import webapp.chat_logic as CL
    rc.set_request_settings(resolve_research_mode("fast"))
    assert CL._deep_subqueries() == 0 and CL._answer_max_tokens() == 3000 and CL._external_top_k() == 8
    rc.set_request_settings(resolve_research_mode("deep"))
    assert CL._deep_subqueries() == 3 and CL._answer_max_tokens() == 8000 and CL._external_top_k() == 20
