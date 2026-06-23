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
from urllib.parse import parse_qs, unquote, urlparse

from backend.external_search.base import ExternalSource, cached, safe_get

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


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
    """Safely download a page (SSRF-guarded via safe_get, cached) and extract readable text
    with requests + BeautifulSoup."""
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
        # Routed through safe_get (not a raw requests.post) so a 429 is retried with
        # backoff like every other channel, instead of silently dropping the web results.
        data = safe_get(self.ENDPOINT, expect="json", timeout=15, json_body={
            "api_key": key, "query": query, "max_results": max_results,
            "search_depth": "advanced", "include_raw_content": True,
        })
        if not isinstance(data, dict):
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


def _ddg_unwrap(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded>."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    p = urlparse(href)
    if "duckduckgo.com" in (p.netloc or "") and p.path.startswith("/l/"):
        qs = parse_qs(p.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


class DuckDuckGoProvider(WebSearchProvider):
    """Free general web search — no API key. Best-effort HTML parsing of the
    DuckDuckGo lite/html endpoint; degrades to [] if the page can't be parsed."""
    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    def is_available(self) -> bool:
        return True

    def search(self, query: str, max_results: int = 8) -> List[ExternalSource]:
        # POST (GET is bot-blocked by DuckDuckGo).
        html = safe_get(self.ENDPOINT, data={"q": query, "kl": "us-en"},
                        headers={"User-Agent": _BROWSER_UA,
                                 "Referer": "https://duckduckgo.com/"}, expect="text")
        if not html:
            return []
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return []
        soup = BeautifulSoup(html, "html.parser")
        out: List[ExternalSource] = []
        for res in soup.select(".result, .web-result")[:max_results * 2]:
            a = res.select_one("a.result__a")
            if not a:
                continue
            url = _ddg_unwrap(a.get("href", ""))
            if not url.startswith("http"):
                continue
            snip = res.select_one(".result__snippet")
            out.append(ExternalSource(
                source_type="web",
                title=a.get_text(" ", strip=True) or url,
                url=url,
                snippet=(snip.get_text(" ", strip=True) if snip else "")[:600],
                provider="duckduckgo",
            ))
            if len(out) >= max_results:
                break
        return out


_KEYED = {p.name: p for p in (TavilyProvider(), BraveProvider(), SerpApiProvider())}
_FREE_WEB = DuckDuckGoProvider()


def get_web_provider() -> Optional[WebSearchProvider]:
    """Resolve the web provider: the one named in WEB_SEARCH_PROVIDER if its key is
    set, else the first keyed provider available, else the FREE DuckDuckGo provider
    (no key). So general web search always works."""
    name = (os.getenv("WEB_SEARCH_PROVIDER") or "").strip().lower()
    if name == "duckduckgo":
        return _FREE_WEB
    if name in _KEYED and _KEYED[name].is_available():
        return _KEYED[name]
    for prov in _KEYED.values():
        if prov.is_available():
            return prov
    return _FREE_WEB


def web_search(query: str, max_results: int = 6) -> List[ExternalSource]:
    provider = get_web_provider()
    if provider is None:
        return []
    return provider.search(query, max_results=max_results)
