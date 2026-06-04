"""
web_search.py  --  Batch 10 (Phase 2)

External paper search via arXiv and Semantic Scholar. No API key,
no account. Both are free public APIs.

Public surface:

    results = search_web(query, max_results=8,
                         sources=("arxiv", "semantic_scholar"))
    # -> list of dicts, deduped by title, ranked by source order

Each result is a uniform dict with the keys:
    title           str
    authors         list[str]
    abstract        str
    year            int | None
    venue           str | None     (journal, conference, or 'arXiv')
    url             str
    pdf_url         str | None
    source          str            ('arxiv' or 'semantic_scholar')
    citation_count  int | None
    paper_id        str            (arxiv id or s2 paperId)

Why no API key:
  - arXiv:        https://export.arxiv.org/api  (no auth, polite rate limit)
  - Semantic Scholar: https://api.semanticscholar.org  (no auth tier, capped)

Polite usage:
  - arXiv asks for >=3s between calls. We enforce a 3s gap PER PROCESS.
  - Semantic Scholar free tier: ~1 req/sec. We enforce 1s.
  - Both calls share a 10s connect timeout, 25s read timeout.
  - All exceptions are caught; on error the empty list is returned for
    that source and the other source's results still come back.

Design choices:
  - No network in __init__. The module imports cheaply.
  - All search functions are pure: input -> requests -> normalized dicts.
  - Dedup by lower-cased title prefix (first 120 chars). Bias toward
    arXiv for ties since arXiv often has the freshest preprint.
  - This module knows nothing about Streamlit, the LLM, or the chat UI.
    The chat UI calls search_web() and renders the results itself.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence


_ARXIV_BASE = "https://export.arxiv.org/api/query"
_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

# Globally serialized rate gates. Per-process; sufficient for one Streamlit app.
_arxiv_lock = threading.Lock()
_s2_lock = threading.Lock()
_arxiv_last_call = [0.0]   # using list as mutable holder
_s2_last_call = [0.0]

_ARXIV_MIN_GAP = 3.0   # seconds
_S2_MIN_GAP = 1.0


def _wait_for_gate(last_call_holder, lock, min_gap):
    with lock:
        elapsed = time.time() - last_call_holder[0]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        last_call_holder[0] = time.time()


# ----------------------------------------------------------------------
# arXiv
# ----------------------------------------------------------------------

_ARXIV_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _arxiv_year(published_text: str) -> Optional[int]:
    if not published_text:
        return None
    m = re.match(r"^(\d{4})", published_text)
    return int(m.group(1)) if m else None


def _parse_arxiv_response(xml_text: str) -> List[Dict[str, Any]]:
    """Parse arXiv Atom XML into our uniform dicts. Tolerant of missing
    fields; skips entries that don't have at least a title + id."""
    out: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    for entry in root.findall("atom:entry", _ARXIV_ATOM_NS):
        title_el = entry.find("atom:title", _ARXIV_ATOM_NS)
        id_el = entry.find("atom:id", _ARXIV_ATOM_NS)
        if title_el is None or id_el is None or not (title_el.text and id_el.text):
            continue
        title = re.sub(r"\s+", " ", title_el.text).strip()
        url = id_el.text.strip()

        # Authors
        authors = []
        for a in entry.findall("atom:author/atom:name", _ARXIV_ATOM_NS):
            if a.text:
                authors.append(a.text.strip())

        # Abstract
        summary_el = entry.find("atom:summary", _ARXIV_ATOM_NS)
        abstract = re.sub(r"\s+", " ", summary_el.text).strip() if (summary_el is not None and summary_el.text) else ""

        # Published year
        published_el = entry.find("atom:published", _ARXIV_ATOM_NS)
        year = _arxiv_year(published_el.text if (published_el is not None and published_el.text) else "")

        # PDF link
        pdf_url = None
        for link in entry.findall("atom:link", _ARXIV_ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        # arXiv id (the trailing "1706.03762" portion)
        paper_id = url.rsplit("/", 1)[-1]

        out.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "venue": "arXiv",
            "url": url,
            "pdf_url": pdf_url,
            "source": "arxiv",
            "citation_count": None,
            "paper_id": paper_id,
        })
    return out


def search_arxiv(query: str, max_results: int = 8,
                 timeout: float = 25.0) -> List[Dict[str, Any]]:
    """Search arXiv. Returns up to max_results normalized dicts."""
    if not query or not query.strip():
        return []
    try:
        import requests
    except ImportError:
        return []

    params = {
        "search_query": f"all:{query.strip()}",
        "start": "0",
        "max_results": str(max(1, min(max_results, 50))),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = _ARXIV_BASE + "?" + urllib.parse.urlencode(params)

    _wait_for_gate(_arxiv_last_call, _arxiv_lock, _ARXIV_MIN_GAP)
    try:
        r = requests.get(
            url,
            timeout=(10.0, timeout),
            headers={"User-Agent": "AudioResearchAssistant/1.0 (research)"},
        )
        if r.status_code != 200:
            return []
        return _parse_arxiv_response(r.text)
    except Exception:
        return []


# ----------------------------------------------------------------------
# Semantic Scholar
# ----------------------------------------------------------------------

_S2_FIELDS = (
    "title,abstract,year,venue,authors,url,citationCount,externalIds"
)


def _parse_s2_response(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return out
    data = payload.get("data") or []
    for p in data:
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or "").strip()
        if not title:
            continue
        authors_list = p.get("authors") or []
        authors = [a.get("name", "") for a in authors_list if isinstance(a, dict) and a.get("name")]
        abstract = (p.get("abstract") or "").strip()
        year = p.get("year") if isinstance(p.get("year"), int) else None
        venue = (p.get("venue") or "").strip() or None
        url = p.get("url") or ""
        citation_count = p.get("citationCount") if isinstance(p.get("citationCount"), int) else None
        ext = p.get("externalIds") or {}
        # Prefer arXiv if present, fall back to DOI, fall back to s2 paperId
        paper_id = ext.get("ArXiv") or ext.get("DOI") or p.get("paperId") or ""
        pdf_url = None  # S2's open-access PDF requires extra field; skip for simplicity

        out.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "venue": venue,
            "url": url,
            "pdf_url": pdf_url,
            "source": "semantic_scholar",
            "citation_count": citation_count,
            "paper_id": str(paper_id),
        })
    return out


def search_semantic_scholar(query: str, max_results: int = 8,
                            timeout: float = 25.0) -> List[Dict[str, Any]]:
    """Search Semantic Scholar. Returns up to max_results normalized dicts."""
    if not query or not query.strip():
        return []
    try:
        import requests
    except ImportError:
        return []

    params = {
        "query": query.strip(),
        "limit": str(max(1, min(max_results, 50))),
        "fields": _S2_FIELDS,
    }
    url = _S2_SEARCH + "?" + urllib.parse.urlencode(params)

    _wait_for_gate(_s2_last_call, _s2_lock, _S2_MIN_GAP)
    try:
        r = requests.get(
            url,
            timeout=(10.0, timeout),
            headers={"User-Agent": "AudioResearchAssistant/1.0 (research)"},
        )
        if r.status_code != 200:
            return []
        return _parse_s2_response(r.json())
    except Exception:
        return []


# ----------------------------------------------------------------------
# Unified search
# ----------------------------------------------------------------------

def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()[:120]


def dedupe_by_title(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable dedup: first occurrence wins, ties broken by input order."""
    seen = set()
    out = []
    for r in results:
        key = _norm_title(r.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def search_web(
    query: str,
    max_results: int = 8,
    sources: Sequence[str] = ("arxiv", "semantic_scholar"),
) -> List[Dict[str, Any]]:
    """Search the configured sources, dedupe, return up to max_results.

    On error from any source, that source contributes [] and the
    other source's results still come back. Total returned is capped
    at max_results across both sources combined.
    """
    if not query or not query.strip():
        return []
    per_source = max(2, max_results)  # over-fetch so dedupe can prune
    combined: List[Dict[str, Any]] = []

    for src in sources:
        if src == "arxiv":
            combined.extend(search_arxiv(query, max_results=per_source))
        elif src == "semantic_scholar":
            combined.extend(search_semantic_scholar(query, max_results=per_source))
        # silently skip unknown source names

    deduped = dedupe_by_title(combined)
    return deduped[:max_results]


def format_results_for_llm(results: Sequence[Dict[str, Any]],
                           max_chars_per_abstract: int = 700) -> str:
    """Render web results as a compact, numbered evidence block. Returns
    empty string if results is empty -- caller decides what to do."""
    if not results:
        return ""
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Untitled"
        authors = r.get("authors") or []
        if authors:
            authors_str = ", ".join(authors[:4])
            if len(authors) > 4:
                authors_str += " et al."
        else:
            authors_str = "(authors unknown)"
        year = r.get("year")
        venue = r.get("venue") or r.get("source")
        header = f"[W{i}] {title}"
        meta = f"   {authors_str}, {venue or 'unknown venue'}"
        if year:
            meta += f", {year}"
        cc = r.get("citation_count")
        if isinstance(cc, int) and cc > 0:
            meta += f"  ({cc} citations)"
        url = r.get("url")
        if url:
            meta += f"\n   {url}"

        abstract = (r.get("abstract") or "").strip()
        if not abstract:
            abstract = "(no abstract available)"
        if len(abstract) > max_chars_per_abstract:
            abstract = abstract[:max_chars_per_abstract].rsplit(" ", 1)[0] + "..."

        lines.append(header)
        lines.append(meta)
        lines.append(f"   Abstract: {abstract}")
        lines.append("")  # blank line between entries
    return "\n".join(lines).rstrip()
