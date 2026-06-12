"""
retrieval_fusion.py  --  Batch 2 (High-Impact Retrieval Layer)

Three retrieval-quality primitives used by hybrid_retrieve v2:

  H1  Field-weighted BM25 (BM25F-style)
      Title / concepts / section matter more than body text.

  H4  Reciprocal Rank Fusion (RRF)
      Rank-based fusion of vector + BM25. Robust to score-scale
      mismatch between different rankers.

  H2  MMR diversity (Maximal Marginal Relevance)
      Balances relevance against content overlap so the final
      evidence set isn't five near-duplicate chunks. Also enforces
      a per-paper cap as a hard constraint.

This file is purely additive. It does not replace anything else.
hybrid_retrieve.py calls into these helpers.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Tokenization (kept consistent with hybrid_retrieve.tokenize)
# ----------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9][a-z0-9\-\_]+", text)


# ======================================================================
# H1 -- Field-weighted BM25 (BM25F-style)
# ======================================================================

# Field weights for research papers.
# Title is short and high-signal: matches there matter most.
# 'concepts' is the extracted concept tag list per chunk: also high-signal.
# Section name is a moderate signal (e.g. "Method" beats "References").
# Body 'text' is the baseline.
FIELD_WEIGHTS: Dict[str, float] = {
    "title":    3.0,
    "concepts": 2.5,
    "section":  2.0,
    "text":     1.0,
    "context":  1.0,   # Contextual Retrieval: the LLM situating sentence (body-level signal)
}


def field_weighted_bm25(
    query_tokens: List[str],
    chunk: Dict[str, Any],
    df: Dict[str, int],
    N: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    BM25F-style scoring. Same corpus-wide IDF for all fields, but TF
    and document length are computed per field, scaled by field weights.

    Why it helps: a chunk whose TITLE contains 'MVDR' should beat a
    chunk that mentions 'MVDR' once 1500 words into the body.
    """
    score = 0.0

    for field, weight in FIELD_WEIGHTS.items():
        field_text = chunk.get(field) or ""
        if not field_text:
            continue

        field_tokens = tokenize(field_text)
        if not field_tokens:
            continue

        field_dl = len(field_tokens)
        field_freq = Counter(field_tokens)

        field_score = 0.0
        for term in query_tokens:
            if term not in field_freq:
                continue
            doc_freq = df.get(term, 0)
            idf = math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
            tf = field_freq[term]
            denom = tf + k1 * (1 - b + b * field_dl / max(avgdl, 1))
            field_score += idf * ((tf * (k1 + 1)) / denom)

        score += weight * field_score

    return score


# ======================================================================
# H4 -- Reciprocal Rank Fusion
# ======================================================================

def reciprocal_rank_fusion(
    rankings: List[List[Dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of multiple ranked candidate lists.

    rankings: list of ranked lists. Each list is candidates from one
              ranker, in best-first order. Each candidate is a dict.

    Returns one fused list sorted by RRF score descending. Each output
    item carries the original fields from whichever input had the
    most-complete record, plus 'rrf_score' and 'hybrid_score'.

    RRF is robust because it uses RANKS, not raw scores. Vector cosine
    and BM25 live in completely different ranges; min-max linear fusion
    is fragile. RRF does not care about score scales.

    The constant k=60 is the standard from Cormack et al. 2009.
    Higher k -> flatter fusion. Lower k -> sharper.
    """
    scores: Dict[Any, float] = defaultdict(float)
    items_by_id: Dict[Any, Dict[str, Any]] = {}
    sources_by_id: Dict[Any, List[str]] = defaultdict(list)

    for ranker_idx, ranking in enumerate(rankings):
        if not ranking:
            continue
        for rank, item in enumerate(ranking, 1):
            item_id = item.get(id_key)
            if item_id is None:
                # Stable fallback identity if 'id' is missing
                item_id = (
                    item.get("chunk_id")
                    or (
                        str(item.get("title") or "")
                        + "::"
                        + str(item.get("section") or "")
                        + "::"
                        + str(item.get("page_start") or "")
                    )
                )
                if not item_id:
                    continue

            scores[item_id] += 1.0 / (k + rank)

            # Prefer the richer record (more populated fields)
            existing = items_by_id.get(item_id)
            if existing is None or len(item) > len(existing):
                items_by_id[item_id] = dict(item)

            src = item.get("source") or f"ranker_{ranker_idx}"
            if src not in sources_by_id[item_id]:
                sources_by_id[item_id].append(src)

    fused: List[Dict[str, Any]] = []
    for item_id, score in scores.items():
        out = items_by_id[item_id]
        out["rrf_score"] = float(score)
        # Backward-compatible name used elsewhere in the codebase
        out["hybrid_score"] = float(score)
        out["retrieval_sources"] = sources_by_id[item_id]
        fused.append(out)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# ======================================================================
# H2 -- MMR diversity
# ======================================================================

def _jaccard(text_a: str, text_b: str, cap: int = 300) -> float:
    """
    Word-level Jaccard similarity, capped at first N tokens for speed.
    Cheap and good enough as a similarity proxy inside MMR.
    """
    a = set(tokenize(text_a or "")[:cap])
    b = set(tokenize(text_b or "")[:cap])
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _paper_identity(item: Dict[str, Any]) -> str:
    return str(
        item.get("title")
        or item.get("paper")
        or item.get("paper_title")
        or "unknown"
    ).strip().lower()


def mmr_diversify(
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
    max_per_paper: int = 3,
    lambda_param: float = 0.7,
    relevance_key: str = "rerank_score",
) -> List[Dict[str, Any]]:
    """
    Greedy Maximal Marginal Relevance.

    Picks items that are BOTH relevant (high relevance_key score) AND
    content-different from items already picked (low Jaccard similarity
    to anything in selected).

    Per-paper cap is enforced inside the loop so one paper cannot fill
    the slate even if its chunks all score very high.

    lambda_param = 0.7 means 70% relevance weight, 30% diversity weight.
    Audio DSP questions are usually technical; relevance dominates.
    """
    if not candidates:
        return []

    selected: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = list(candidates)
    paper_counts: Dict[str, int] = {}

    while len(selected) < top_k and remaining:
        best_score = float("-inf")
        best_item: Optional[Dict[str, Any]] = None
        best_index = -1

        for i, cand in enumerate(remaining):
            paper = _paper_identity(cand)
            if paper_counts.get(paper, 0) >= max_per_paper:
                continue

            relevance = float(cand.get(relevance_key, 0.0))

            if not selected:
                mmr_score = relevance
            else:
                cand_text = cand.get("text") or ""
                max_sim = 0.0
                for s in selected:
                    sim = _jaccard(cand_text, s.get("text") or "")
                    if sim > max_sim:
                        max_sim = sim
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = cand
                best_index = i

        if best_item is None:
            # All remaining are blocked by per-paper cap. Stop strict
            # selection and fall through to top-off below.
            break

        best_item["mmr_score"] = float(best_score)
        selected.append(best_item)
        paper_counts[_paper_identity(best_item)] = (
            paper_counts.get(_paper_identity(best_item), 0) + 1
        )
        remaining.pop(best_index)

    # Top-off: if per-paper caps prevented filling top_k, take the best
    # remaining items regardless of paper.
    if len(selected) < top_k and remaining:
        selected_ids = {id(s) for s in selected}
        for cand in remaining:
            if id(cand) in selected_ids:
                continue
            selected.append(cand)
            if len(selected) >= top_k:
                break

    return selected[:top_k]
