"""
B3: external channels are fetched concurrently, so a slow channel never serializes the
others and one hanging channel can't block the whole gather. Channels are mocked with
sleeps (no network), and we assert on wall-clock.
"""
import time

import backend.external_search.orchestrator as orch


def test_channels_run_concurrently(monkeypatch):
    delay = 0.3

    def slow(_q, *a, **k):
        time.sleep(delay)
        return []

    monkeypatch.setattr(orch, "get_web_provider", lambda: None)   # 4 free channels
    monkeypatch.setattr(orch, "arxiv_search", slow)
    monkeypatch.setattr(orch, "semantic_scholar_search", slow)
    monkeypatch.setattr(orch, "wikipedia_search", slow)
    monkeypatch.setattr(orch, "github_search", slow)
    monkeypatch.setattr(orch, "rerank_sources", lambda q, items, top_k=20: items)

    start = time.time()
    orch.gather_external_evidence("anything")
    elapsed = time.time() - start
    # 4 channels x 0.3s = 1.2s if sequential; concurrent should finish well under that
    assert elapsed < delay * 2, f"channels did not run concurrently ({elapsed:.2f}s)"


def test_one_slow_channel_does_not_block_the_gather(monkeypatch):
    monkeypatch.setattr(orch, "get_web_provider", lambda: None)
    monkeypatch.setenv("EXTERNAL_GATHER_TIMEOUT", "0.3")   # read live via _gather_timeout()

    def fast(_q, *a, **k):
        return []

    def very_slow(_q, *a, **k):
        time.sleep(1.0)
        return []

    monkeypatch.setattr(orch, "arxiv_search", fast)
    monkeypatch.setattr(orch, "semantic_scholar_search", fast)
    monkeypatch.setattr(orch, "wikipedia_search", fast)
    monkeypatch.setattr(orch, "github_search", very_slow)
    monkeypatch.setattr(orch, "rerank_sources", lambda q, items, top_k=20: items)

    start = time.time()
    _, warnings = orch.gather_external_evidence("anything")
    elapsed = time.time() - start
    assert elapsed < 0.9, f"a slow channel blocked the gather ({elapsed:.2f}s)"
    assert any("timed out" in w for w in warnings)
