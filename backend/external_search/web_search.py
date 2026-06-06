"""
Web search providers + page reading.

A small, swappable provider layer. Pick one with `WEB_SEARCH_PROVIDER`
(tavily | brave | serpapi) and set the matching API key. Each provider returns
a list of `ExternalSource` (source_type="web") with title/url/snippet; the
orchestrator then fetches the top pages and extracts readable text.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from backend.external_search.base import ExternalSource, cached, logger, safe_get


# ----------------------------------------------------------------------
# HTML -> readable text
# ----------------------------------------------------------------------
_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header", "aside",
                     "form", "noscript", "svg", "button", "iframe"]


def extract_readable_text(html: str, max_chars: int = 8000) -> str:
    """Strip boilerplate (nav/footer/scripts) and return the main readable text."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # No parser available -> crude fallback (strip tags).
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:max_chars]

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 1]
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return out.strip()[:max_chars]


def fetch_page_text(url: str, max_chars: int = 8000) -> Optional[str]:
    """Safely download a page (SSRF-guarded, cached) and extract readable text."""
    def _produce():
        html = safe_get(url, expect="text")
        return extract_readable_text(html, max_chars=max_chars) if html else None
    return cached(f"page::{url}", _produce)


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------
class WebSearchProvider(ABC):
    name = "abstract"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def search(self, query: str, max_results: int = 6) -> List[ExternalSource]: ...


class TavilyProvider(WebSearchProvider):
    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"

    def is_available(self) -> bool:
        return bool(os.getenv("TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 6) -> List[ExternalSource]:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            return []
        try:
            import requests
            resp = requests.post(self.ENDPOINT, timeout=15, json={
                "api_key": key, "query": query, "max_results": max_results,
                "search_depth": "advanced", "include_raw_content": True,
            })
            if resp.status_code >= 400:
                return []
            data = resp.json()
        except Exception as exc:
            logger.info("tavily search failed: %s", type(exc).__name__)
            return []
        out: List[ExternalSource] = []
        for r in (data.get("results") or [])[:max_results]:
            out.append(ExternalSource(
                source_type="web",
                title=r.get("title") or r.get("url") or "Untitled",
                url=r.get("url") or "",
                snippet=(r.get("content") or "")[:600],
                text=(r.get("raw_content") or r.get("content") or "")[:8000],
                provider="tavily",
                published=r.get("published_date"),
                score=float(r.get("score") or 0.0),
            ))
        return out


class BraveProvider(WebSearchProvider):
    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def is_available(self) -> bool:
        return bool(os.getenv("BRAVE_SEARCH_API_KEY"))

    def search(self, query: str, max_results: int = 6) -> List[ExternalSource]:
        key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not key:
            return []
        data = safe_get(
            self.ENDPOINT,
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            params={"q": query, "count": max_results},
            expect="json",
        )
        if not data:
            return []
        out: List[ExternalSource] = []
        for r in (((data.get("web") or {}).get("results")) or [])[:max_results]:
            out.append(ExternalSource(
                source_type="web",
                title=r.get("title") or r.get("url") or "Untitled",
                url=r.get("url") or "",
                snippet=(r.get("description") or "")[:600],
                provider="brave",
                published=r.get("age"),
            ))
        return out


class SerpApiProvider(WebSearchProvider):
    name = "serpapi"
    ENDPOINT = "https://serpapi.com/search"

    def is_available(self) -> bool:
        return bool(os.getenv("SERPAPI_API_KEY"))

    def search(self, query: str, max_results: int = 6) -> List[ExternalSource]:
        key = os.getenv("SERPAPI_API_KEY")
        if not key:
            return []
        data = safe_get(self.ENDPOINT, params={"q": query, "api_key": key,
                                                "num": max_results, "engine": "google"},
                        expect="json")
        if not data:
            return []
        out: List[ExternalSource] = []
        for r in (data.get("organic_results") or [])[:max_results]:
            out.append(ExternalSource(
                source_type="web",
                title=r.get("title") or r.get("link") or "Untitled",
                url=r.get("link") or "",
                snippet=(r.get("snippet") or "")[:600],
                provider="serpapi",
                published=r.get("date"),
            ))
        return out


_PROVIDERS = {p.name: p for p in (TavilyProvider(), BraveProvider(), SerpApiProvider())}


def get_web_provider() -> Optional[WebSearchProvider]:
    """Resolve the configured provider (WEB_SEARCH_PROVIDER), else the first one
    whose API key is set. Returns None if no provider is configured."""
    name = (os.getenv("WEB_SEARCH_PROVIDER") or "").strip().lower()
    if name in _PROVIDERS and _PROVIDERS[name].is_available():
        return _PROVIDERS[name]
    for prov in _PROVIDERS.values():
        if prov.is_available():
            return prov
    return None


def web_search(query: str, max_results: int = 6) -> List[ExternalSource]:
    provider = get_web_provider()
    if provider is None:
        return []
    return provider.search(query, max_results=max_results)
