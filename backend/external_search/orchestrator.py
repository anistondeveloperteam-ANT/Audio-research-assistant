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

import concurrent.futures
import os
import time as _time
from typing import List, Tuple

from backend.external_search.base import ExternalSource, clean_query, env_flag, logger
from backend.external_search.github_search import github_search
from backend.external_search.pdf_reader import looks_like_pdf_url, read_online_pdf
from backend.external_search.scholar_search import (
    arxiv_search, patent_search, semantic_scholar_search, wikipedia_search,
)
from backend.external_search.source_ranker import rerank_sources
from backend.external_search.web_search import fetch_page_text, get_web_provider, web_search

MAX_PDFS = int(os.getenv("EXTERNAL_MAX_PDFS", "3"))           # online PDFs from web results
WEB_MAX = int(os.getenv("WEB_MAX_RESULTS", "8"))              # web pages per query
ARXIV_READ_PDF_COUNT = int(os.getenv("ARXIV_READ_PDF_COUNT", "3"))  # read this many papers in full
EXTERNAL_GATHER_TIMEOUT = float(os.getenv("EXTERNAL_GATHER_TIMEOUT", "30"))  # shared cap across channels


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


def gather_external_evidence(query: str, max_results: int = 20) -> Tuple[List[ExternalSource], List[str]]:
    """Collect + rank external evidence across all channels — web pages, research
    papers (arXiv), patents, GitHub repos/code, and online PDFs. Never raises; on
    any channel failure it records a warning and returns whatever else succeeded.
    Web + patent channels need a provider key; arXiv + GitHub are free."""
    warnings: List[str] = []
    collected: List[ExternalSource] = []
    have_web = get_web_provider() is not None
    # Keyword query for the search APIs; the full question is kept for re-ranking.
    sq = clean_query(query)

    # Each channel is independent and returns (sources, warnings). They run concurrently
    # (all network-bound) so wall-clock is the slowest channel, not the sum of them.
    def _ch_web() -> Tuple[List[ExternalSource], List[str]]:
        w: List[str] = []
        srcs, pdf_urls = _web_channel(sq, WEB_MAX, w)
        for url in pdf_urls[:MAX_PDFS]:
            try:
                srcs.extend(read_online_pdf(url))
            except Exception:
                w.append("An online PDF could not be read.")
        return srcs, w

    def _ch_arxiv() -> Tuple[List[ExternalSource], List[str]]:
        w: List[str] = []
        out: List[ExternalSource] = []
        try:
            papers = arxiv_search(sq)
            out.extend(papers)
            for p in papers[:ARXIV_READ_PDF_COUNT]:
                if p.url and looks_like_pdf_url(p.url):
                    try:
                        out.extend(read_online_pdf(p.url))
                    except Exception:
                        pass
        except Exception as exc:
            logger.info("arxiv search failed: %s", type(exc).__name__)
            w.append("Research-paper (arXiv) search failed.")
        return out, w

    def _ch_semantic() -> Tuple[List[ExternalSource], List[str]]:
        try:
            return semantic_scholar_search(sq), []
        except Exception as exc:
            logger.info("semantic scholar failed: %s", type(exc).__name__)
            return [], []

    def _ch_wiki() -> Tuple[List[ExternalSource], List[str]]:
        try:
            return wikipedia_search(sq), []
        except Exception as exc:
            logger.info("wikipedia failed: %s", type(exc).__name__)
            return [], []

    def _ch_patents() -> Tuple[List[ExternalSource], List[str]]:
        try:
            return patent_search(sq), []
        except Exception:
            return [], ["Patent search failed."]

    def _ch_github() -> Tuple[List[ExternalSource], List[str]]:
        try:
            return github_search(sq), []
        except Exception as exc:
            logger.info("github search failed: %s", type(exc).__name__)
            return [], ["GitHub search failed; continuing without it."]

    channels: List[Tuple[str, object]] = [
        ("arxiv", _ch_arxiv), ("semantic_scholar", _ch_semantic),
        ("wikipedia", _ch_wiki), ("github", _ch_github),
    ]
    if have_web:                       # web + patents need a provider key
        channels = [("web", _ch_web), ("patents", _ch_patents)] + channels

    def _timed(label: str, fn) -> Tuple[str, List[ExternalSource], List[str], float]:
        start = _time.time()
        try:
            srcs, w = fn()
        except Exception:
            srcs, w = [], []
        return label, srcs, w, _time.time() - start

    t0 = _time.time()
    done = 0
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, len(channels)))
    try:
        futures = [ex.submit(_timed, label, fn) for label, fn in channels]
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=EXTERNAL_GATHER_TIMEOUT):
                label, srcs, w, secs = fut.result()
                collected.extend(srcs)
                warnings.extend(w)
                done += 1
                logger.info("external channel %-16s %2d sources in %4.1fs", label, len(srcs), secs)
        except concurrent.futures.TimeoutError:
            warnings.append("Some external channels timed out; using partial results.")
    finally:
        ex.shutdown(wait=False)        # don't block the response on stragglers
    logger.info("external gather: %d/%d channels, %d sources, %.1fs total",
                done, len(channels), len(collected), _time.time() - t0)

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
