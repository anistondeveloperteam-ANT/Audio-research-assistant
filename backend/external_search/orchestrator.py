"""
External-evidence orchestrator.

Runs the optional external channels (web pages, GitHub repos/code, online PDFs),
each isolated so one failure never blocks the others, then de-duplicates and
re-ranks everything against the original query. Returns a small, cited set of
`ExternalSource` plus a list of non-fatal warnings.

Disabled by default. Turn on with ENABLE_WEB_SEARCH=true *and* a configured web
search provider key. Local PDF RAG is unaffected either way.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from backend.external_search.base import ExternalSource, env_flag, logger
from backend.external_search.github_search import github_search
from backend.external_search.pdf_reader import looks_like_pdf_url, read_online_pdf
from backend.external_search.scholar_search import arxiv_search, patent_search
from backend.external_search.source_ranker import rerank_sources
from backend.external_search.web_search import fetch_page_text, get_web_provider, web_search

MAX_PDFS = int(os.getenv("EXTERNAL_MAX_PDFS", "2"))


def is_web_search_enabled() -> bool:
    """Master switch for the automatic external-search fallback (web / arXiv /
    patents / GitHub). On by default. The web + patent channels additionally need
    a provider key; arXiv and GitHub work for free, so the fallback is still
    useful without any key."""
    return env_flag("ENABLE_WEB_SEARCH", default=True)


def _web_channel(query: str, max_results: int, warnings: List[str]) -> Tuple[List[ExternalSource], List[str]]:
    """Web results (HTML enriched with page text); returns (sources, pdf_urls)."""
    try:
        results = web_search(query, max_results=max_results)
    except Exception as exc:
        logger.info("web search failed: %s", type(exc).__name__)
        warnings.append("Web search failed; continuing with local sources.")
        return [], []
    sources: List[ExternalSource] = []
    pdf_urls: List[str] = []
    for s in results:
        if looks_like_pdf_url(s.url):
            pdf_urls.append(s.url)
            continue
        if not (s.text or "").strip():
            try:
                page = fetch_page_text(s.url)
                if page:
                    s.text = page
            except Exception:
                pass
        sources.append(s)
    return sources, pdf_urls


def gather_external_evidence(query: str, max_results: int = 8) -> Tuple[List[ExternalSource], List[str]]:
    """Collect + rank external evidence across all channels — web pages, research
    papers (arXiv), patents, GitHub repos/code, and online PDFs. Never raises; on
    any channel failure it records a warning and returns whatever else succeeded.
    Web + patent channels need a provider key; arXiv + GitHub are free."""
    warnings: List[str] = []
    collected: List[ExternalSource] = []
    have_web = get_web_provider() is not None

    # Web pages (+ any online PDFs they surface) — needs a web provider key.
    if have_web:
        web_sources, pdf_urls = _web_channel(query, max_results, warnings)
        collected.extend(web_sources)
        for url in pdf_urls[:MAX_PDFS]:
            try:
                collected.extend(read_online_pdf(url))
            except Exception:
                warnings.append("An online PDF could not be read.")

    # Research papers (arXiv) — free, no key.
    try:
        collected.extend(arxiv_search(query))
    except Exception as exc:
        logger.info("arxiv search failed: %s", type(exc).__name__)
        warnings.append("Research-paper (arXiv) search failed.")

    # Patents — via the web provider (Google Patents focus).
    if have_web:
        try:
            collected.extend(patent_search(query))
        except Exception:
            warnings.append("Patent search failed.")

    # GitHub repos/code — free (a GITHUB_TOKEN raises limits + enables code search).
    try:
        collected.extend(github_search(query))
    except Exception as exc:
        logger.info("github search failed: %s", type(exc).__name__)
        warnings.append("GitHub search failed; continuing without it.")

    if not have_web:
        warnings.append("No web search key set — used free sources (arXiv, GitHub). "
                        "Add TAVILY_API_KEY for web pages & patents.")

    if not collected:
        return [], warnings

    try:
        ranked = rerank_sources(query, collected, top_k=max_results)
    except Exception as exc:
        logger.info("external rerank failed: %s", type(exc).__name__)
        ranked = collected[:max_results]
    return ranked, warnings
