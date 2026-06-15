"""
evidence_grader.py — Corrective RAG (CRAG) evidence grading.

After local PDF retrieval, grade how well the retrieved chunks cover the question, then act on the
grade. This borrows the LangGraph CRAG / Self-RAG design (grade-then-correct) and implements it in
our own code — no LangGraph dependency.

    STRONG  — enough high-relevance chunks  -> answer from the PDFs.
    PARTIAL — some relevant but thin        -> keep PDF evidence AND search externally to fill gaps.
    NONE    — nothing relevant              -> go fully external.

Grading reuses the reranker scores already computed during retrieval, so it is a fast, deterministic
check with no extra model/LLM call. Thresholds are read live from .env so they can be tuned without
a redeploy and without touching code.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

STRONG = "STRONG"
PARTIAL = "PARTIAL"
NONE = "NONE"


# ----------------------------------------------------------------------
# Config (live reads)
# ----------------------------------------------------------------------
def crag_enabled() -> bool:
    return os.getenv("CRAG_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int, lo: int = 1) -> int:
    try:
        return max(lo, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def crag_strong_min() -> float:
    """A chunk this relevant (reranker score) is strong evidence."""
    return _float_env("CRAG_STRONG_MIN", 0.55)


def crag_partial_min() -> float:
    """Below this, a chunk is not relevant enough to count (matches SOURCE_MIN_SCORE)."""
    return _float_env("CRAG_PARTIAL_MIN", 0.30)


def crag_strong_count() -> int:
    """How many high-relevance chunks make the grade STRONG."""
    return _int_env("CRAG_STRONG_COUNT", 2)


def crag_paper_min_chunks() -> int:
    """Below this many relevant chunks, a paper is too thin to be a complete code spec on its own
    -> supplement with GitHub reference implementations."""
    return _int_env("CRAG_PAPER_MIN_CHUNKS", 2)


# ----------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------
def _item_score(item: Dict[str, Any]) -> float:
    """The chunk's relevance score (cross-encoder rerank score), tolerant of either field name."""
    for key in ("rerank_score", "score"):
        v = item.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def grade_evidence(items: List[Dict[str, Any]]) -> str:
    """Grade local PDF evidence from the reranker scores already on each item.

    STRONG  : >= CRAG_STRONG_COUNT chunks at/above CRAG_STRONG_MIN.
    PARTIAL : at least one chunk at/above CRAG_PARTIAL_MIN (but not STRONG).
    NONE    : nothing at/above CRAG_PARTIAL_MIN (or no items)."""
    if not items:
        return NONE
    strong_min = crag_strong_min()
    partial_min = crag_partial_min()
    if sum(1 for it in items if _item_score(it) >= strong_min) >= crag_strong_count():
        return STRONG
    if any(_item_score(it) >= partial_min for it in items):
        return PARTIAL
    return NONE


def relevant_items(items: List[Dict[str, Any]],
                   min_score: Optional[float] = None) -> List[Dict[str, Any]]:
    """The genuinely-relevant chunks (>= the partial threshold), best-first."""
    floor = crag_partial_min() if min_score is None else min_score
    rel = [it for it in items if _item_score(it) >= floor]
    rel.sort(key=_item_score, reverse=True)
    return rel


def paper_is_thin(items: List[Dict[str, Any]]) -> bool:
    """True when the relevant PDF evidence is too sparse to be a complete code spec on its own."""
    return len(relevant_items(items)) < crag_paper_min_chunks()


# ----------------------------------------------------------------------
# Code-from-paper: turn the relevant chunks into an algorithm spec
# ----------------------------------------------------------------------
def _title_of(item: Dict[str, Any]) -> str:
    return (item.get("title") or "Untitled").strip()


def extract_algorithm_spec(items: List[Dict[str, Any]], question: str = "",
                           max_chars: int = 3000) -> Tuple[str, str]:
    """Build a code spec from the most relevant PDF chunks (the algorithm description) plus a
    citation naming the source paper(s). The whole returned spec (header included) is kept within
    `max_chars`: chunk text is packed up to the budget and the boundary chunk is truncated to fit.
    Returns ("", "") when there is no relevant evidence (or none of it carries usable text), so an
    empty spec is never paired with a non-empty citation."""
    rel = relevant_items(items)
    if not rel:
        return "", ""

    titles: List[str] = []
    for it in rel:
        t = _title_of(it)
        if t and t not in titles:
            titles.append(t)
    citation = "; ".join(titles[:3])

    header = ("Algorithm description extracted from the user's research library"
              + (f" (source: {citation})" if citation else "")
              + ". Implement the algorithm EXACTLY as the paper describes it:\n\n")
    budget = max(0, max_chars - len(header))          # reserve room so header+blocks <= max_chars

    parts: List[str] = []
    for it in rel:
        text = (it.get("text") or it.get("chunk_text") or "").strip()
        if not text:
            continue
        section = (it.get("section") or it.get("section_name") or "").strip()
        head = f"[from {_title_of(it)}" + (f" — {section}" if section else "") + "]"
        block = f"{head}\n{text}"
        remaining = budget - len("\n\n".join(parts)) - (2 if parts else 0)
        if remaining <= 0:
            break
        if len(block) > remaining:                    # truncate the boundary chunk to fit the budget
            parts.append(block[: max(0, remaining - 1)].rstrip() + "…")
            break
        parts.append(block)

    if not parts:                                     # relevant by score, but no usable text
        return "", ""
    return header + "\n\n".join(parts), citation
