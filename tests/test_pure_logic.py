"""Unit tests for the pure backend logic (no DB, no models, no network)."""
import subprocess
import sys
from pathlib import Path

import pytest

from backend.common.device import resolve_device
from backend.answering.query_sanity import check_query_sanity
from backend.answering.research_modes import normalize_mode, get_mode_settings
from backend.retrieval.hyde_generator import hyde_expand
from backend.retrieval.retrieval_fusion import (
    field_weighted_bm25,
    reciprocal_rank_fusion,
    mmr_diversify,
)


# ----------------------------------------------------------------------
# Device selection (GPU / CPU)
# ----------------------------------------------------------------------
def test_resolve_device_explicit_cpu(monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    assert resolve_device("EMBEDDING_DEVICE") == "cpu"


def test_resolve_device_role_override(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    assert resolve_device("EMBEDDING_DEVICE") == "cpu"   # role wins over DEVICE


def test_resolve_device_explicit_value_passthrough(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda:0")
    monkeypatch.delenv("RERANKER_DEVICE", raising=False)
    assert resolve_device("RERANKER_DEVICE") == "cuda:0"


def test_resolve_device_auto(monkeypatch):
    monkeypatch.setenv("DEVICE", "auto")
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    assert resolve_device("EMBEDDING_DEVICE") in {"cuda", "cpu"}


# ----------------------------------------------------------------------
# Query sanity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "a", "??"])
def test_query_sanity_rejects_bad(bad):
    assert check_query_sanity(bad).ok is False


def test_query_sanity_accepts_real_question():
    res = check_query_sanity("How does MVDR beamforming reduce noise in microphone arrays?")
    assert res.ok is True


# ----------------------------------------------------------------------
# Retrieval mode — single optimized "Default" (no Fast/Balanced/Deep)
# ----------------------------------------------------------------------
def test_normalize_mode_fast_or_deep():
    # Default (and any non-deep input) -> fast; deep variants -> deep.
    assert normalize_mode(None) == "fast"
    assert normalize_mode("Fast") == "fast"
    assert normalize_mode("Balanced") == "fast"
    assert normalize_mode("not-a-mode") == "fast"
    assert normalize_mode("Deep") == "deep"
    assert normalize_mode("Deep Research") == "deep"


def test_get_mode_settings_distinguishes_fast_and_deep():
    fast = get_mode_settings("fast")
    deep = get_mode_settings("deep")
    assert isinstance(fast, dict) and "max_query_routes" in fast
    assert fast["mode"] == "fast" and deep["mode"] == "deep"
    # fast is cheaper than deep, but the accuracy bar is identical
    assert fast["deep_search_subqueries"] < deep["deep_search_subqueries"]
    assert fast["agentic_min_verify_score"] == deep["agentic_min_verify_score"] == 80


def test_run_help_still_works():
    """`python run.py --help` must exit cleanly (launcher unbroken)."""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "run.py", "--help"],
                       cwd=str(root), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()


# ----------------------------------------------------------------------
# HyDE expansion
# ----------------------------------------------------------------------
def test_hyde_expand_returns_nonempty_text():
    out = hyde_expand("How does MVDR beamforming reduce noise?")
    assert isinstance(out, str) and out.strip()


# ----------------------------------------------------------------------
# Field-weighted BM25
# ----------------------------------------------------------------------
def _chunk(title="", concepts="", section="", text=""):
    return {"title": title, "concepts": concepts, "section": section, "text": text}


def test_bm25_scores_match_positive_and_miss_zero():
    df, N, avgdl = {"mvdr": 1}, 10, 5.0
    hit = field_weighted_bm25(["mvdr"], _chunk(text="mvdr beamforming reduces noise"), df, N, avgdl)
    miss = field_weighted_bm25(["mvdr"], _chunk(text="unrelated content here"), df, N, avgdl)
    assert hit > 0
    assert miss == 0


def test_bm25_title_weighted_higher_than_body():
    df, N, avgdl = {"mvdr": 1}, 10, 1.0
    in_title = field_weighted_bm25(["mvdr"], _chunk(title="mvdr"), df, N, avgdl)
    in_body = field_weighted_bm25(["mvdr"], _chunk(text="mvdr"), df, N, avgdl)
    assert in_title >= in_body


# ----------------------------------------------------------------------
# Reciprocal Rank Fusion
# ----------------------------------------------------------------------
def test_rrf_merges_and_ranks():
    rankings = [
        [{"id": "a"}, {"id": "b"}],
        [{"id": "a"}, {"id": "c"}],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)
    ids = [x["id"] for x in fused]
    assert set(ids) == {"a", "b", "c"}     # all candidates present
    assert fused[0]["id"] == "a"           # appears top of both -> ranked first
    assert all("rrf_score" in x for x in fused)


# ----------------------------------------------------------------------
# MMR diversification
# ----------------------------------------------------------------------
def test_mmr_empty_returns_empty():
    assert mmr_diversify([], top_k=5) == []


def test_mmr_respects_top_k():
    cands = [
        {"title": f"P{i}", "text": f"distinct topic number {i}", "rerank_score": 1.0 - i * 0.1}
        for i in range(6)
    ]
    out = mmr_diversify(cands, top_k=3, max_per_paper=3)
    assert len(out) == 3


def test_mmr_respects_per_paper_cap_when_alternatives_exist():
    # 3 chunks each from two papers; cap=2; top_k=4 -> 2 from each, cap honored.
    cands = (
        [{"title": "P1", "text": f"alpha {i} beta {i}", "rerank_score": 0.95 - i * 0.01} for i in range(3)]
        + [{"title": "P2", "text": f"gamma {i} delta {i}", "rerank_score": 0.90 - i * 0.01} for i in range(3)]
    )
    out = mmr_diversify(cands, top_k=4, max_per_paper=2)
    titles = [x["title"] for x in out]
    assert len(out) == 4
    assert titles.count("P1") <= 2
    assert titles.count("P2") <= 2


def test_mmr_tops_off_past_cap_when_no_alternatives():
    # Only one paper available: the cap is relaxed (top-off) to still fill top_k.
    cands = [
        {"title": "SamePaper", "text": f"chunk {i} unique words {i}", "rerank_score": 1.0 - i * 0.1}
        for i in range(5)
    ]
    out = mmr_diversify(cands, top_k=5, max_per_paper=2)
    assert len(out) == 5
    assert all(x["title"] == "SamePaper" for x in out)
