"""
Research-paper and patent channels.

- arXiv: free Atom API (no key) -> cited research papers (title, abstract, PDF URL).
- Patents: routed through the configured web-search provider with a patent focus
  (Google Patents), since there is no free first-party patent API. Returns
  `patent`-typed sources; needs a web search key.

Both are best-effort and never raise.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List

from backend.external_search.base import ExternalSource, cached, logger, safe_get

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_MAX = int(os.getenv("ARXIV_MAX_RESULTS", "4"))
PATENT_MAX = int(os.getenv("PATENT_MAX_RESULTS", "3"))
_ATOM = "{http://www.w3.org/2005/Atom}"


def _parse_arxiv(xml_text: str) -> List[dict]:
    out: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip()
        summary = (entry.findtext(f"{_ATOM}summary") or "").strip()
        published = (entry.findtext(f"{_ATOM}published") or "").strip()[:10]
        abs_url = (entry.findtext(f"{_ATOM}id") or "").strip()
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href") or ""
        if title:
            out.append({"title": title, "summary": summary, "published": published,
                        "url": pdf_url or abs_url})
    return out


def arxiv_search(query: str, max_results: int = ARXIV_MAX) -> List[ExternalSource]:
    """Search arXiv for relevant research papers (free, no API key)."""
    def _produce():
        xml_text = safe_get(ARXIV_API, params={"search_query": f"all:{query}",
                                               "start": 0, "max_results": max_results},
                            expect="text")
        return _parse_arxiv(xml_text) if xml_text else None

    entries = cached(f"arxiv::{query}::{max_results}", _produce) or []
    sources: List[ExternalSource] = []
    for e in entries[:max_results]:
        sources.append(ExternalSource(
            source_type="research_paper",
            title=e["title"],
            url=e["url"],
            text=(e.get("summary") or "")[:4000],
            snippet=(e.get("summary") or "")[:600],
            provider="arxiv",
            published=e.get("published") or None,
        ))
    return sources


def patent_search(query: str, max_results: int = PATENT_MAX) -> List[ExternalSource]:
    """Patent results via the configured web provider (Google Patents focus)."""
    from backend.external_search.web_search import web_search  # avoid import cycle
    try:
        hits = web_search(f"{query} patent site:patents.google.com", max_results=max_results)
    except Exception as exc:
        logger.info("patent search failed: %s", type(exc).__name__)
        return []
    out: List[ExternalSource] = []
    for h in hits[:max_results]:
        h.source_type = "patent"
        h.provider = f"{h.provider or 'web'}/patents"
        out.append(h)
    return out
