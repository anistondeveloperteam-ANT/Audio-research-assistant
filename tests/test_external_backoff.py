"""429-safe retry-with-backoff for external search. The shared HTTP fetcher (base.safe_get) retries
on rate-limit / transient 5xx with exponential backoff, and EVERY channel (incl. the Tavily web
provider, now routed through safe_get) inherits it — so bounded concurrency can never exhaust an API
and silently drop sources. Fully offline: requests + sleep are mocked, no network.
"""
import backend.external_search.base as base
from backend.external_search.web_search import TavilyProvider


# --------------------------------------------------------------------------------------------
# A fake `requests` response good enough for safe_get's streaming read.
# --------------------------------------------------------------------------------------------
class _Resp:
    url = "https://api.example.com/x"
    encoding = "utf-8"

    def __init__(self, status, body=b'{"ok": true}'):
        self.status_code = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        yield self._body


def _patch_requests(monkeypatch, statuses):
    calls = []
    slept = []

    def fake_request(method, url, **k):
        calls.append((method, url))
        return _Resp(statuses.pop(0))

    monkeypatch.setattr(base.requests, "request", fake_request)
    monkeypatch.setattr(base.time, "sleep", lambda s: slept.append(s))
    return calls, slept


# --------------------------------------------------------------------------------------------
# safe_get retry-with-backoff
# --------------------------------------------------------------------------------------------
def test_safe_get_retries_on_429_then_succeeds(monkeypatch):
    calls, slept = _patch_requests(monkeypatch, [429, 429, 200])
    out = base.safe_get("https://api.example.com/x", expect="json", retries=2)
    assert out == {"ok": True}                       # recovered after the rate limit cleared
    assert len(calls) == 3                            # two retries, then success
    assert slept == [1, 2]                            # exponential backoff: 2**0, 2**1


def test_safe_get_gives_up_after_max_retries(monkeypatch):
    calls, slept = _patch_requests(monkeypatch, [429, 429, 429])
    out = base.safe_get("https://api.example.com/x", expect="json", retries=2)
    assert out is None                               # exhausted the bounded retries -> None (no crash)
    assert len(calls) == 3 and slept == [1, 2]       # backed off between attempts, capped


def test_safe_get_non_retryable_status_fails_fast(monkeypatch):
    calls, slept = _patch_requests(monkeypatch, [404])
    out = base.safe_get("https://api.example.com/x", expect="json", retries=2)
    assert out is None
    assert len(calls) == 1 and slept == []           # a 404 is not retried — no wasted backoff


# --------------------------------------------------------------------------------------------
# The Tavily web provider is now routed through safe_get (so it gets the backoff too).
# --------------------------------------------------------------------------------------------
def test_tavily_routes_through_safe_get_with_backoff(monkeypatch):
    seen = {}

    def fake_safe_get(url, *, expect=None, timeout=None, json_body=None, **k):
        seen.update(url=url, expect=expect, json_body=json_body)
        return {"results": [{"title": "T", "url": "http://x/t", "content": "c", "raw_content": "r"}]}

    monkeypatch.setattr("backend.external_search.web_search.safe_get", fake_safe_get)
    monkeypatch.setenv("TAVILY_API_KEY", "k")

    out = TavilyProvider().search("mvdr beamforming", max_results=5)

    assert seen["url"] == TavilyProvider.ENDPOINT and seen["expect"] == "json"
    assert seen["json_body"]["query"] == "mvdr beamforming"   # the request really went via safe_get
    assert len(out) == 1 and out[0].title == "T"


def test_tavily_empty_when_safe_get_fails(monkeypatch):
    # safe_get returns None after exhausting retries -> the provider degrades to no results, no crash.
    monkeypatch.setattr("backend.external_search.web_search.safe_get", lambda *a, **k: None)
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert TavilyProvider().search("q") == []
