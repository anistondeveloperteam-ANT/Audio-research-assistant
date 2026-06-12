"""
Crawl4AI page extraction must be fallback-safe and keep is_safe_url behavior. These run
fully offline (crawl4ai + the network are mocked — no browser, no requests).
"""
import backend.external_search.web_search as ws


def _no_fetch(*a, **k):
    raise AssertionError("must not fetch this URL")


def _run_producer(monkeypatch):
    # bypass the TTL cache so the producer actually runs in-test
    monkeypatch.setattr(ws, "cached", lambda key, producer, **k: producer())


def test_falls_back_to_beautifulsoup_when_crawl4ai_returns_none(monkeypatch):
    _run_producer(monkeypatch)
    monkeypatch.setattr(ws, "crawl_markdown", lambda url, query="", **k: None)   # crawl4ai off/fails
    monkeypatch.setattr(ws, "is_safe_url", lambda url: (True, ""))
    monkeypatch.setattr(ws, "safe_get",
                        lambda url, expect="text": "<html><body><article>Hello <b>world</b></article></body></html>")
    out = ws.fetch_page_text("https://example.com/x", query="hello")
    assert out and "Hello" in out and "world" in out          # BeautifulSoup path used


def test_uses_crawl4ai_markdown_and_threads_the_query(monkeypatch):
    _run_producer(monkeypatch)
    seen = {}

    def fake_crawl(url, query="", **k):
        seen["url"], seen["query"] = url, query
        return "# Title\n\nrelevant markdown"

    monkeypatch.setattr(ws, "crawl_markdown", fake_crawl)
    monkeypatch.setattr(ws, "is_safe_url", lambda url: (True, ""))
    monkeypatch.setattr(ws, "safe_get", _no_fetch)            # crawl4ai succeeded -> no bs4 fetch
    out = ws.fetch_page_text("https://example.com/page", query="hello query")
    assert out == "# Title\n\nrelevant markdown"
    assert seen == {"url": "https://example.com/page", "query": "hello query"}   # BM25 query threaded


def test_unsafe_url_is_never_fetched(monkeypatch):
    _run_producer(monkeypatch)
    monkeypatch.setattr(ws, "is_safe_url", lambda url: (False, "blocked"))
    crawled = {"v": False}
    monkeypatch.setattr(ws, "crawl_markdown",
                        lambda *a, **k: crawled.__setitem__("v", True) or "md")
    monkeypatch.setattr(ws, "safe_get", _no_fetch)
    out = ws.fetch_page_text("http://169.254.169.254/latest/meta-data", query="x")
    assert out is None
    assert crawled["v"] is False                              # is_safe_url gate ran first


def test_crawl_markdown_is_noop_when_disabled(monkeypatch):
    import backend.external_search.crawl as crawl
    monkeypatch.setenv("EXTERNAL_USE_CRAWL4AI", "false")
    assert crawl.crawl_markdown("https://example.com", "q", timeout=5, max_chars=100) is None
