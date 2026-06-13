"""Contextual Retrieval helper: prompt shape, fail-safe fallback, and on-disk caching.
The LLM is always mocked, so these run fully offline."""
import backend.ingestion.contextualizer as ctx


class _FakeProvider:
    def __init__(self, reply="This chunk explains MVDR within the Methods section.", available=True):
        self.is_available = available
        self.reply = reply
        self.calls = 0

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3, **kw):
        self.calls += 1
        content = messages[0]["content"]
        assert "<document>" in content and "<chunk>" in content   # Anthropic-style prompt
        yield self.reply


def test_generate_context_returns_one_sentence():
    p = _FakeProvider()
    out = ctx.generate_context("Full document about beamforming.", "MVDR minimizes noise.", provider=p)
    assert out == "This chunk explains MVDR within the Methods section."
    assert p.calls == 1


def test_generate_context_falls_back_to_empty_on_error():
    class Boom:
        is_available = True

        def stream_chat(self, *a, **k):
            raise RuntimeError("LLM down")

    assert ctx.generate_context("doc", "chunk", provider=Boom()) == ""


def test_generate_context_empty_when_provider_unavailable():
    assert ctx.generate_context("doc", "chunk", provider=_FakeProvider(available=False)) == ""


def test_generate_context_empty_for_blank_chunk():
    p = _FakeProvider()
    assert ctx.generate_context("doc", "   ", provider=p) == "" and p.calls == 0


def test_contextualize_chunks_disabled_returns_blanks(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "false")
    chunks = [{"text": "a"}, {"text": "b"}]
    assert ctx.contextualize_chunks("doc", chunks, provider=_FakeProvider()) == ["", ""]


def test_contextualize_chunks_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true")
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    chunks = [{"text": "MVDR minimizes noise."}]

    p1 = _FakeProvider()
    first = ctx.contextualize_chunks("doc about beamforming", chunks, provider=p1)
    assert first == ["This chunk explains MVDR within the Methods section."]
    assert p1.calls == 1

    # Same (doc, chunk) again -> served from the on-disk cache, no new LLM call.
    p2 = _FakeProvider()
    second = ctx.contextualize_chunks("doc about beamforming", chunks, provider=p2)
    assert second == first
    assert p2.calls == 0


def test_contextualize_chunks_many_misses_one_call_each_in_order(monkeypatch, tmp_path):
    """Every cache-miss chunk gets exactly one LLM call and results stay in chunk order (indexed
    writes). Uses the serial path (CONTEXTUAL_CONCURRENCY=1) so the offline suite spawns no threads;
    the parallel path is the same logic dispatched through a thread pool."""
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true")
    monkeypatch.setenv("CONTEXTUAL_CONCURRENCY", "1")
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    chunks = [{"text": f"distinct chunk number {i}"} for i in range(6)]
    p = _FakeProvider(reply="ctx")
    out = ctx.contextualize_chunks("doc", chunks, provider=p)
    assert out == ["ctx"] * 6
    assert p.calls == 6
