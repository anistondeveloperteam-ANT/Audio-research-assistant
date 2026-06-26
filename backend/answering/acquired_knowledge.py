"""
acquired_knowledge.py — the "UPDATE my RAG" layer (Phase 2: a GROWN CORPUS from verified findings).

When the assistant produces a VERIFIED, good answer grounded in EXTERNAL findings (web / paper / patent /
repo / online-pdf) that it actually CITED, those passages are embedded and stored as acquired knowledge.
On a future question they are RECALLED into retrieval alongside the local corpus — so the agent answers
from a corpus that GREW out of its own verified research, without re-fetching. The corpus gets better day
by day.

Latency-free by construction: CAPTURE runs in the BACKGROUND (the embedding call is the one slow step),
so it adds nothing to the answer; RECALL reuses the query embedding the pipeline already computed, so it
makes no extra network call. Recency-weighted (latest wins; stale fades), deduped by content hash,
bounded, gated by CORPUS_GROWTH (default on), and fail-open everywhere — it can never break an answer.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from backend.answering import background as _background
from backend.answering import tuning as _tuning

logger = logging.getLogger(__name__)

# External source types worth ingesting. 'local_pdf' is already in the corpus; everything else is a
# finding we fetched and can keep.
_INGESTIBLE = {"web", "research_paper", "patent", "github_repo", "github_code", "online_pdf"}


def acquired_enabled() -> bool:
    return os.getenv("CORPUS_GROWTH", "true").strip().lower() not in ("0", "false", "no", "off")


def _int_env(name: str, default: int, lo: int) -> int:
    try:
        return max(lo, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, lo: float) -> float:
    try:
        return max(lo, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _min_relevance() -> float:
    return _tuning.tuned("CORPUS_MIN_RELEVANCE", _float_env("CORPUS_MIN_RELEVANCE", 0.5, 0.0))


def _top_k() -> int:
    return int(_tuning.tuned("CORPUS_TOP_K", _int_env("CORPUS_TOP_K", 3, 0)))


def _half_life_days() -> float:
    return _tuning.tuned("CORPUS_HALF_LIFE_DAYS", _float_env("CORPUS_HALF_LIFE_DAYS", 120.0, 1.0))


def _max_text() -> int:
    return _int_env("CORPUS_MAX_TEXT_CHARS", 4000, 200)


def _min_text() -> int:
    return _int_env("CORPUS_MIN_TEXT_CHARS", 80, 0)


def content_hash(url: str, text: str) -> str:
    """Stable dedup key for a finding: its URL + a prefix of its text (so the same passage from the same
    page is stored once). Mirrors external_search.base.ExternalSource.content_hash."""
    h = hashlib.sha256()
    h.update(((url or "").strip().lower().rstrip("/") + "|" + (text or "")[:400]).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _embed_passages(texts: List[str]):
    """(vectors, meta) for passages, embedded as DOCUMENTS with the SAME meta string the query embedder
    (_query_embedding in chat_logic) uses — so at recall time query_meta == the stored embedding_meta and
    semantic matching fires. ([], None) on any failure, so a stored passage simply falls back to lexical
    recall."""
    if not texts:
        return [], None
    try:
        from backend.common.embeddings import embed_documents, provider as _emb_provider
        vecs = embed_documents(texts)
        if not vecs or len(vecs) != len(texts) or not vecs[0]:
            return [], None
        meta = f"{_emb_provider()}:{os.getenv('EMBEDDING_MODEL', '')}:{len(vecs[0])}"
        return vecs, meta
    except Exception:                                  # noqa: BLE001 - embedding is optional / may fail
        logger.warning("passage embedding failed; storing finding(s) without a vector", exc_info=True)
        return [], None


def _select_cited_findings(items: Sequence[Dict[str, Any]],
                           cited_sources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The external findings the answer actually CITED, deduped by content hash, with FULL text pulled
    from `items` (the cited-source dicts carry text capped at 600). Skips local-corpus hits and findings
    too short to be worth keeping."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    n_items = len(items or [])
    min_chars = _min_text()
    max_chars = _max_text()
    for s in (cited_sources or []):
        st = (s.get("source_type") or "").strip()
        if st not in _INGESTIBLE:
            continue
        url = (s.get("url") or "").strip()
        n = s.get("n")
        full = items[n - 1] if (isinstance(n, int) and 1 <= n <= n_items) else s
        text = (full.get("text") or s.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        ch = content_hash(url, text)
        if ch in seen:
            continue
        seen.add(ch)
        out.append({
            "content_hash": ch,
            "source_type": st,
            "title": (s.get("title") or "").strip(),
            "url": url,
            "snippet": ((s.get("snippet") or text)[:600]).strip(),
            "text": text[:max_chars],
            "provider": s.get("provider"),
            "published": s.get("published"),
        })
    return out


def _do_capture(mem, *, user_id: str, question: str, findings: List[Dict[str, Any]],
                logic_version: int) -> None:
    """Runs on a BACKGROUND thread: embed only the findings we haven't stored before, then upsert all of
    them (re-capture strengthens an existing finding without re-embedding). Never raises (background
    guard wraps it), but we still keep store calls individually fail-soft."""
    if not findings:
        return
    hashes = [f["content_hash"] for f in findings]
    try:
        existing = mem.existing_source_hashes(user_id=user_id, hashes=hashes)
    except Exception:                                  # noqa: BLE001
        existing = set()
    new_findings = [f for f in findings if f["content_hash"] not in existing]
    vecs, meta = _embed_passages([f["text"] for f in new_findings])
    vec_by_hash = {new_findings[i]["content_hash"]: vecs[i]
                   for i in range(min(len(vecs), len(new_findings)))} if vecs else {}
    stored = 0
    for f in findings:
        emb = vec_by_hash.get(f["content_hash"])
        try:
            mem.record_learned_source(
                user_id=user_id, content_hash=f["content_hash"], text=f["text"],
                source_type=f["source_type"], title=f["title"], url=f["url"], snippet=f["snippet"],
                provider=f["provider"], published=f["published"], question=question,
                embedding=emb, embedding_meta=(meta if emb else None),
                confidence=1.0, logic_version=logic_version)
            stored += 1
        except Exception:                              # noqa: BLE001
            logger.warning("storing acquired finding failed", exc_info=True)
    if stored:
        try:
            mem.prune_learned_sources(user_id=user_id)   # bound the table ONCE per batch, not per row
        except Exception:                              # noqa: BLE001
            pass
        logger.info("grew corpus with %d verified finding(s)", stored)


def capture_findings(mem, *, user_id: str, question: str, items: Sequence[Dict[str, Any]],
                     cited_sources: Sequence[Dict[str, Any]], verified: bool = True,
                     logic_version: int = 0) -> None:
    """Schedule capture of the CITED external findings from a VERIFIED answer into the grown corpus — in
    the BACKGROUND, so it adds NO latency to this answer. No-op when disabled, not verified, or nothing
    citable was cited. Fail-open."""
    if not acquired_enabled() or not verified:
        return
    try:
        findings = _select_cited_findings(items, cited_sources)
        if not findings:
            return
        _background.run(_do_capture, mem, user_id=user_id, question=question,
                        findings=findings, logic_version=logic_version)
    except Exception:                                  # noqa: BLE001 - learning never breaks a turn
        logger.warning("scheduling corpus capture failed", exc_info=True)


def _to_evidence_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """Shape an acquired passage as an evidence item the answer pipeline can merge, cite, and display —
    same shape as a local/external item, with a rerank_score on the 0..1 scale select/grade expect."""
    rel = round(float(row.get("relevance") or 0.0), 3)
    return {
        "source_type": row.get("source_type") or "web",
        "title": row.get("title") or "Untitled",
        "section": "",
        "page_start": None, "page_end": None,
        "url": row.get("url") or "",
        "file_path": None, "line_start": None, "line_end": None, "page": None,
        "provider": row.get("provider") or "learned",
        "license": None,
        "text": (row.get("text") or "").strip(),
        "score": rel,
        "rerank_score": rel,
        "retrieval_sources": ["learned"],
        "graph_reason": "",
        "concepts": "",
        "published": row.get("published"),
    }


def recall_items(mem, *, user_id: str, question: str,
                 query_embedding: Optional[List[float]] = None,
                 query_meta: Optional[str] = None) -> List[Dict[str, Any]]:
    """Acquired passages relevant to THIS question, as evidence items (most relevant first). Reuses the
    already-computed query embedding — no extra network call. Fail-open: [] on disabled / any error."""
    if not acquired_enabled():
        return []
    try:
        rows = mem.recall_learned_sources(
            user_id=user_id, question=question, query_embedding=query_embedding,
            query_meta=query_meta, min_relevance=_min_relevance(), top_k=_top_k(),
            half_life_days=_half_life_days())
    except Exception:                                  # noqa: BLE001 - recall must never break retrieval
        return []
    return [_to_evidence_item(r) for r in rows]
