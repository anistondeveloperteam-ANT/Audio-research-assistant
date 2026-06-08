"""
Read a PDF from a public URL.

Safely downloads the bytes (SSRF-guarded, size-capped, cached), confirms it is a
real PDF, then extracts per-page text with PyMuPDF so answers can cite the PDF
URL + page number. Returns page-grouped `ExternalSource` chunks; the orchestrator
re-ranks them against the query and keeps the most relevant.
"""
from __future__ import annotations

import os
from typing import Dict, List

from backend.external_search.base import ExternalSource, cache_get, cache_set, logger, safe_get

MAX_PAGES = int(os.getenv("ONLINE_PDF_MAX_PAGES", "30"))
MAX_CHUNKS = int(os.getenv("ONLINE_PDF_MAX_CHUNKS", "12"))
CHUNK_CHARS = int(os.getenv("ONLINE_PDF_CHUNK_CHARS", "1600"))


def _extract_pages(pdf_bytes: bytes) -> List[Dict]:
    """Per-page text + a best-effort document title via PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        try:
            fitz.TOOLS.mupdf_display_errors(False)  # silence noisy MuPDF stderr spam
        except Exception:
            pass
    except Exception:
        logger.info("PyMuPDF not available; cannot read online PDF")
        return []
    pages: List[Dict] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            title = (doc.metadata or {}).get("title") or ""
            for i, page in enumerate(doc):
                if i >= MAX_PAGES:
                    break
                text = (page.get_text() or "").strip()
                if text:
                    pages.append({"page": i + 1, "title": title, "text": text})
    except Exception as exc:
        logger.info("PDF parse failed: %s", type(exc).__name__)
        return []
    return pages


def _chunk_pages(pages: List[Dict], url: str) -> List[Dict]:
    """Group page text into ~CHUNK_CHARS chunks, each tagged with its page."""
    title = (pages[0].get("title") if pages else "") or url.rsplit("/", 1)[-1] or "Online PDF"
    chunks: List[Dict] = []
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), CHUNK_CHARS):
            chunks.append({"page": p["page"], "title": title, "text": text[start:start + CHUNK_CHARS]})
            if len(chunks) >= MAX_CHUNKS:
                return chunks
    return chunks


def read_online_pdf(url: str) -> List[ExternalSource]:
    """Download + parse a PDF URL into cited, page-numbered evidence chunks.
    Returns [] on any failure (bad URL, not a PDF, parse error) — never raises."""
    cache_key = f"online_pdf::{url}"
    cached_chunks = cache_get(cache_key)
    if cached_chunks is None:
        body = safe_get(url, expect="bytes")
        if not body or not body[:5].startswith(b"%PDF"):
            return []
        pages = _extract_pages(body)
        cached_chunks = _chunk_pages(pages, url)
        cache_set(cache_key, cached_chunks)

    sources: List[ExternalSource] = []
    for ch in cached_chunks:
        sources.append(ExternalSource(
            source_type="online_pdf",
            title=ch.get("title") or "Online PDF",
            url=url,
            page=ch.get("page"),
            text=(ch.get("text") or "")[:8000],
            snippet=(ch.get("text") or "")[:600],
            provider="online_pdf",
        ))
    return sources


def looks_like_pdf_url(url: str) -> bool:
    u = (url or "").lower().split("?")[0]
    return u.endswith(".pdf") or "/pdf/" in u or "arxiv.org/pdf" in u
