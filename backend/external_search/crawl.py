"""
Crawl4AI page-text extraction (optional, fallback-safe).

Renders a page in a headless browser (JS on), then emits BM25-filtered **markdown** keyed to
the search query — better signal than raw HTML stripping. Used effectively, not wholesale:
we touch only the high-level `AsyncWebCrawler` + a `BM25ContentFilter` markdown generator; no
LLM/litellm features are imported.

A single browser is launched once on a dedicated background event loop and reused across
pages (relaunching Chromium per URL would be slow). Every call is guarded: if crawl4ai is
missing, the browser can't start, or anything errors, `crawl_markdown` returns None and the
caller falls back to the BeautifulSoup path. Disable entirely with EXTERNAL_USE_CRAWL4AI=false.
"""
from __future__ import annotations

import asyncio
import atexit
import importlib.util
import os
import threading
from typing import Optional

from backend.external_search.base import logger

_loop: Optional[asyncio.AbstractEventLoop] = None
_crawler = None
_unavailable = False                  # set once if the browser can't start (e.g. not installed)
_start_lock = threading.Lock()
_run_lock = threading.Lock()          # serialize crawls through the shared browser


def use_crawl4ai() -> bool:
    return os.getenv("EXTERNAL_USE_CRAWL4AI", "true").strip().lower() == "true"


def available() -> bool:
    return importlib.util.find_spec("crawl4ai") is not None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, name="crawl4ai-loop", daemon=True).start()
        atexit.register(_shutdown)
    return _loop


async def _get_crawler():
    global _crawler
    if _crawler is None:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        _crawler = AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False))
        await _crawler.start()
    return _crawler


async def _crawl(url: str, query: str, timeout_s: float) -> Optional[str]:
    from crawl4ai import CrawlerRunConfig
    from crawl4ai.content_filter_strategy import BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    md_gen = DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(user_query=query)) if query.strip() \
        else DefaultMarkdownGenerator()
    cfg = CrawlerRunConfig(page_timeout=int(timeout_s * 1000), markdown_generator=md_gen)
    crawler = await _get_crawler()
    result = await crawler.arun(url, config=cfg)
    if not getattr(result, "success", False):
        return None
    md_obj = getattr(result, "markdown", None)
    if md_obj is None:
        return None
    # With a content_filter, fit_markdown is the BM25-relevant subset; else raw_markdown.
    return (getattr(md_obj, "fit_markdown", None)
            or getattr(md_obj, "raw_markdown", None)
            or (md_obj if isinstance(md_obj, str) else None))


def crawl_markdown(url: str, query: str = "", *, timeout: float, max_chars: int) -> Optional[str]:
    """BM25-filtered markdown for `url` (JS rendered), capped to max_chars. None on any
    failure — caller must fall back. Honors the same timeout as the HTTP path."""
    global _unavailable
    if _unavailable or not use_crawl4ai() or not available():
        return None
    with _run_lock:
        try:
            with _start_lock:
                loop = _ensure_loop()
            fut = asyncio.run_coroutine_threadsafe(_crawl(url, query, timeout), loop)
            md = fut.result(timeout=timeout + 10)
        except Exception as exc:
            logger.info("crawl4ai failed for %s (%s); falling back", url, type(exc).__name__)
            if _crawler is None:   # browser never started -> stop retrying it this process
                _unavailable = True
                logger.info("crawl4ai disabled for this process (browser unavailable); "
                            "run crawl4ai-setup to enable. Using BeautifulSoup.")
            return None
    md = (md or "").strip()
    return md[:max_chars] or None


def _shutdown() -> None:
    try:
        if _crawler is not None and _loop is not None:
            asyncio.run_coroutine_threadsafe(_crawler.close(), _loop).result(timeout=10)
    except Exception:
        pass
    try:
        if _loop is not None:
            _loop.call_soon_threadsafe(_loop.stop)
    except Exception:
        pass
