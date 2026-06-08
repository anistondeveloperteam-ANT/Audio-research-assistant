"""
Unit tests for the external-search package. All network calls are mocked — these
run with no internet, no API keys, and no models.
"""
import base64
import sys
import types

import pytest

from backend.external_search.base import ExternalSource, is_safe_url


@pytest.fixture(autouse=True)
def _ssrf_guard_on(monkeypatch):
    """Ensure the SSRF guard is active for tests regardless of the local .env."""
    monkeypatch.delenv("EXTERNAL_ALLOW_UNSAFE_URLS", raising=False)
from backend.external_search.source_ranker import deduplicate, rerank_sources
from backend.external_search.web_search import (
    BraveProvider, TavilyProvider, extract_readable_text, get_web_provider,
)
import backend.external_search.github_search as gh
import backend.external_search.pdf_reader as pr


# ----------------------------------------------------------------------
# URL safety (SSRF guard) — IP literals / schemes need no DNS
# ----------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "", None, "ftp://host/x", "file:///etc/passwd", "http://localhost/x",
    "http://127.0.0.1/x", "https://127.0.0.1", "http://10.0.0.1/", "http://192.168.1.5/",
    "http://172.16.0.9/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
])
def test_is_safe_url_blocks_unsafe(url):
    ok, _ = is_safe_url(url)
    assert ok is False


def test_is_safe_url_allows_public(monkeypatch):
    monkeypatch.setattr("backend.external_search.base.socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    ok, reason = is_safe_url("https://example.com/page")
    assert ok is True, reason


def test_is_safe_url_blocks_dns_rebind_to_private(monkeypatch):
    # Host name resolves to a private IP -> must be blocked (SSRF / DNS rebinding).
    monkeypatch.setattr("backend.external_search.base.socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 80))])
    ok, _ = is_safe_url("https://evil.example.com/")
    assert ok is False


# ----------------------------------------------------------------------
# HTML extraction
# ----------------------------------------------------------------------
def test_extract_readable_text_strips_boilerplate():
    html = ("<html><body><nav>menu nav</nav>"
            "<article><h1>Heading</h1><p>Real meaningful content here.</p></article>"
            "<footer>footer junk</footer><script>var x=1;</script></body></html>")
    out = extract_readable_text(html)
    assert "Real meaningful content" in out
    assert "menu nav" not in out
    assert "footer junk" not in out
    assert "var x" not in out


def test_extract_readable_text_empty():
    assert extract_readable_text("") == ""


# ----------------------------------------------------------------------
# Provider interface
# ----------------------------------------------------------------------
def test_get_web_provider_falls_back_to_free_duckduckgo(monkeypatch):
    from backend.external_search.web_search import DuckDuckGoProvider
    for k in ("WEB_SEARCH_PROVIDER", "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    prov = get_web_provider()
    assert isinstance(prov, DuckDuckGoProvider)   # free web search, no key required


def test_tavily_parses_mocked_response(monkeypatch):
    import requests
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class FakeResp:
        status_code = 200
        def json(self):
            return {"results": [{"title": "T", "url": "https://ex.com/a",
                                 "content": "snippet", "raw_content": "full body text",
                                 "score": 0.9, "published_date": "2026-01-01"}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp())
    results = TavilyProvider().search("query", max_results=3)
    assert len(results) == 1
    s = results[0]
    assert s.source_type == "web" and s.url == "https://ex.com/a"
    assert s.text == "full body text" and s.provider == "tavily"


def test_brave_provider_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert BraveProvider().is_available() is False
    assert BraveProvider().search("q") == []


# ----------------------------------------------------------------------
# GitHub result parsing (mock the API layer)
# ----------------------------------------------------------------------
def test_github_search_parses_repos(monkeypatch):
    readme = base64.b64encode(b"# Title\nAlgorithm explanation here.\nLine 3").decode()

    def fake_api_get(url, params=None):
        if "/search/repositories" in url:
            return {"items": [{"full_name": "foo/bar",
                               "html_url": "https://github.com/foo/bar",
                               "license": {"spdx_id": "MIT"}}]}
        if url.endswith("/readme"):
            return {"path": "README.md",
                    "html_url": "https://github.com/foo/bar/blob/main/README.md",
                    "encoding": "base64", "content": readme}
        if url.endswith("/license"):
            return {"license": {"spdx_id": "MIT"}}
        return None

    monkeypatch.setattr(gh, "cached", lambda key, producer, ttl=None: producer())
    monkeypatch.setattr(gh, "_api_get", fake_api_get)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)  # skip code search

    out = gh.github_search("noise reduction")
    assert len(out) == 1
    s = out[0]
    assert s.source_type == "github_repo"
    assert s.file_path == "README.md"
    assert "Algorithm explanation" in s.text
    assert s.license == "MIT"
    assert s.url.startswith("https://github.com/foo/bar")


# ----------------------------------------------------------------------
# Online PDF fetch failure handling
# ----------------------------------------------------------------------
def test_read_online_pdf_non_pdf_returns_empty(monkeypatch):
    monkeypatch.setattr(pr, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(pr, "safe_get", lambda *a, **k: b"<html>not a pdf</html>")
    assert pr.read_online_pdf("https://ex.com/fake.pdf") == []


def test_read_online_pdf_network_none_returns_empty(monkeypatch):
    monkeypatch.setattr(pr, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(pr, "safe_get", lambda *a, **k: None)
    assert pr.read_online_pdf("https://ex.com/x.pdf") == []


def test_looks_like_pdf_url():
    assert pr.looks_like_pdf_url("https://arxiv.org/pdf/2106.00001")
    assert pr.looks_like_pdf_url("https://x.com/paper.PDF")
    assert not pr.looks_like_pdf_url("https://x.com/page.html")


# ----------------------------------------------------------------------
# Source de-duplication + ranking
# ----------------------------------------------------------------------
def test_deduplicate_collapses_same_content_keeps_best():
    a = ExternalSource(source_type="web", title="A", url="https://e.com/x", text="identical text", score=0.5)
    b = ExternalSource(source_type="web", title="A2", url="https://e.com/x/", text="identical text", score=0.9)
    out = deduplicate([a, b])
    assert len(out) == 1
    assert out[0].score == 0.9   # trailing-slash URL + same text -> dup; keep higher score


def _inject_fake_reranker(monkeypatch, predictor):
    fake = types.ModuleType("backend.retrieval.hybrid_retrieve")
    fake.get_reranker = lambda: types.SimpleNamespace(predict=predictor)
    monkeypatch.setitem(sys.modules, "backend.retrieval.hybrid_retrieve", fake)


def test_rerank_uses_model_scores(monkeypatch):
    monkeypatch.setattr("backend.external_search.source_ranker.USE_CROSS_ENCODER", True)
    _inject_fake_reranker(monkeypatch, lambda pairs: [0.1, 0.9])
    srcs = [ExternalSource(source_type="web", title="a", url="https://e.com/1", text="t1"),
            ExternalSource(source_type="web", title="b", url="https://e.com/2", text="t2")]
    out = rerank_sources("q", srcs, top_k=2)
    assert out[0].url == "https://e.com/2"   # higher model score first


def test_rerank_lexical_fallback(monkeypatch):
    def boom():
        raise RuntimeError("no model")
    fake = types.ModuleType("backend.retrieval.hybrid_retrieve")
    fake.get_reranker = boom
    monkeypatch.setitem(sys.modules, "backend.retrieval.hybrid_retrieve", fake)
    srcs = [ExternalSource(source_type="web", title="a", url="https://e.com/1",
                           text="beamforming mvdr noise reduction speech"),
            ExternalSource(source_type="web", title="b", url="https://e.com/2",
                           text="chocolate cake recipe baking")]
    out = rerank_sources("mvdr beamforming noise", srcs, top_k=2)
    assert out[0].url == "https://e.com/1"   # lexical overlap ranks the relevant one first


# ----------------------------------------------------------------------
# External source formatting / validation
# ----------------------------------------------------------------------
def test_external_source_validates_type():
    with pytest.raises(ValueError):
        ExternalSource(source_type="not-a-type", title="x")


def test_external_source_to_public_and_citation():
    s = ExternalSource(source_type="github_code", title="r/f", url="https://github.com/r/f",
                       file_path="src/a.py", line_start=10, line_end=40, license="Apache-2.0",
                       text="some code", score=0.812)
    pub = s.to_public()
    assert pub["source_type"] == "github_code"
    assert pub["url"] == "https://github.com/r/f"
    assert pub["score"] == 0.812 and "license" in pub
    assert "src/a.py:10-40" in s.citation()


def test_arxiv_search_parses(monkeypatch):
    import backend.external_search.scholar_search as sch
    atom = ('<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            '<title>Deep Noise Suppression</title>'
            '<summary>An abstract about denoising.</summary>'
            '<published>2026-01-02T00:00:00Z</published>'
            '<id>http://arxiv.org/abs/2601.00001v1</id>'
            '<link title="pdf" href="http://arxiv.org/pdf/2601.00001v1" type="application/pdf"/>'
            '</entry></feed>')
    monkeypatch.setattr(sch, "cached", lambda key, producer, ttl=None: producer())
    monkeypatch.setattr(sch, "safe_get", lambda *a, **k: atom)
    out = sch.arxiv_search("denoising")
    assert len(out) == 1
    assert out[0].source_type == "research_paper"
    assert out[0].url.endswith("2601.00001v1")
    assert "abstract" in out[0].text.lower()


def test_duckduckgo_parses(monkeypatch):
    from backend.external_search.web_search import DuckDuckGoProvider
    html = ('<div class="result">'
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">'
            'Example Title</a>'
            '<a class="result__snippet">a useful snippet</a></div>')
    monkeypatch.setattr("backend.external_search.web_search.safe_get", lambda *a, **k: html)
    out = DuckDuckGoProvider().search("query", max_results=5)
    assert len(out) == 1
    assert out[0].url == "https://example.com/page"
    assert out[0].source_type == "web" and "Example Title" in out[0].title


def test_semantic_scholar_parses(monkeypatch):
    import backend.external_search.scholar_search as sch
    data = {"data": [{"title": "Paper X", "abstract": "abstract text", "year": 2026,
                      "openAccessPdf": {"url": "https://x.com/p.pdf"},
                      "url": "https://semanticscholar.org/p"}]}
    monkeypatch.setattr(sch, "cached", lambda key, producer, ttl=None: producer())
    monkeypatch.setattr(sch, "safe_get", lambda *a, **k: data)
    out = sch.semantic_scholar_search("query")
    assert len(out) == 1 and out[0].source_type == "research_paper"
    assert out[0].url == "https://x.com/p.pdf" and out[0].published == "2026"


def test_wikipedia_parses(monkeypatch):
    import backend.external_search.scholar_search as sch
    data = {"query": {"search": [{"title": "Noise reduction",
                                  "snippet": "<b>noise</b> reduction is a method"}]}}
    monkeypatch.setattr(sch, "cached", lambda key, producer, ttl=None: producer())
    monkeypatch.setattr(sch, "safe_get", lambda *a, **k: data)
    out = sch.wikipedia_search("noise")
    assert len(out) == 1 and out[0].source_type == "web"
    assert out[0].url.endswith("Noise_reduction") and "<b>" not in out[0].text


def test_patent_search_tags_patent(monkeypatch):
    import backend.external_search.scholar_search as sch
    def fake_web_search(q, max_results=3):
        return [ExternalSource(source_type="web", title="US123",
                               url="https://patents.google.com/patent/US123",
                               snippet="a claim", provider="tavily")]
    monkeypatch.setattr("backend.external_search.web_search.web_search", fake_web_search)
    out = sch.patent_search("noise cancellation")
    assert out and out[0].source_type == "patent"
    assert "patents.google.com" in out[0].url


def test_gather_uses_free_sources_without_web_provider(monkeypatch):
    import backend.external_search.orchestrator as orch
    monkeypatch.setattr(orch, "get_web_provider", lambda: None)   # force no web provider
    monkeypatch.setattr(orch, "arxiv_search",
                        lambda q: [ExternalSource(source_type="research_paper", title="P",
                                                  url="http://arxiv.org/abs/1", text="relevant abc")])
    # Mock the other (network) channels so the test stays offline.
    monkeypatch.setattr(orch, "semantic_scholar_search", lambda q: [])
    monkeypatch.setattr(orch, "wikipedia_search", lambda q: [])
    monkeypatch.setattr(orch, "github_search", lambda q: [])
    srcs, warns = orch.gather_external_evidence("query abc", max_results=5)
    assert any(s.source_type == "research_paper" for s in srcs)


def test_format_evidence_tags_local_and_external():
    from webapp.chat_logic import format_evidence
    ev = format_evidence([
        {"source_type": "web", "title": "W", "url": "https://e.com", "text": "web text"},
        {"source_type": "local_pdf", "title": "P", "section": "Method",
         "page_start": 1, "page_end": 2, "text": "paper text"},
    ])
    assert "[1] (web)" in ev
    assert "[2] (paper)" in ev
    assert "https://e.com" in ev
