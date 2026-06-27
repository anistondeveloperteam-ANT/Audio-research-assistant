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
    monkeypatch.setenv("CONTEXTUAL_BATCH_SIZE", "1")          # per-chunk path (batching tested separately)
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    chunks = [{"text": f"distinct chunk number {i}"} for i in range(6)]
    p = _FakeProvider(reply="ctx")
    out = ctx.contextualize_chunks("doc", chunks, provider=p)
    assert out == ["ctx"] * 6
    assert p.calls == 6


# ---- the rate-limit fix: concurrency-safe primary + cross-vendor fallback + persistent promotion ----
class _NamedProvider:
    def __init__(self, model, reply, available=True):
        self.model = model
        self.reply = reply
        self.is_available = available
        self.calls = 0

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3, **kw):
        self.calls += 1
        if self.reply:
            yield self.reply


def test_provider_chain_prefers_concurrency_safe_primary_then_cross_vendor(monkeypatch):
    """The contextual chain must lead with the concurrency-safe primary (Gemini) and still include a
    cross-vendor fallback (Mistral), regardless of which model is the chat default — so one provider's
    429 can't blank the whole paper."""
    import backend.llm.streaming_provider as sp
    made = {}
    monkeypatch.setattr(sp, "get_provider",
                        lambda name=None: made.setdefault(name or "mistral-large-latest",
                                                           _NamedProvider(name or "mistral-large-latest", "x")))
    monkeypatch.delenv("CONTEXTUAL_MODEL", raising=False)
    monkeypatch.delenv("CONTEXTUAL_FALLBACK_MODELS", raising=False)
    models = [p.model for p in ctx._provider_chain()]
    assert models[0] == "gemini-2.5-flash"             # concurrency-safe primary leads
    assert "mistral-large-latest" in models            # cross-vendor fallback present
    assert len(models) == len(set(models))             # deduped


def test_warmup_promotes_working_provider_so_dead_one_is_not_retried_per_chunk(monkeypatch, tmp_path):
    """If the first provider is down/rate-limited (returns empty), the sequential warm-up must promote
    the WORKING provider in the shared chain, so the parallel workers don't re-pay the dead provider's
    failure on every chunk. Serial path for determinism."""
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true")
    monkeypatch.setenv("CONTEXTUAL_CONCURRENCY", "1")
    monkeypatch.setenv("CONTEXTUAL_BATCH_SIZE", "1")          # per-chunk path (batching tested separately)
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    dead = _NamedProvider("dead", "")                  # rate-limited/down -> empty
    good = _NamedProvider("good", "ctx")
    monkeypatch.setattr(ctx, "_provider_chain", lambda: [dead, good])
    chunks = [{"text": f"chunk {i}"} for i in range(5)]
    out = ctx.contextualize_chunks("doc", chunks)      # no provider= -> uses the (mocked) chain
    assert out == ["ctx"] * 5                           # every chunk got a real context
    assert dead.calls == 1                              # tried once in warm-up, then skipped (promotion persisted)
    assert good.calls == 5


# ---- Step 4: batched contextualization (many chunks per LLM call) ----
import json as _json                                    # noqa: E402
import re as _re                                         # noqa: E402


class _BatchProvider:
    """Returns a JSON array of contexts for a batch prompt (one per numbered [CHUNK n] marker); a single
    sentence for a single-chunk prompt. Records calls + batch sizes."""
    def __init__(self, available=True):
        self.is_available = available
        self.calls = 0
        self.batch_sizes = []

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3, **kw):
        self.calls += 1
        n = len(_re.findall(r"\[CHUNK \d+\]", messages[0]["content"]))   # only real numbered markers
        if n >= 2:
            self.batch_sizes.append(n)
            yield _json.dumps([f"context for chunk {i + 1}" for i in range(n)])
        else:
            yield "single context sentence"


class _PlainProvider:
    """Always returns a plain (non-JSON) sentence — exercises the per-chunk fallback for batches."""
    def __init__(self):
        self.is_available = True
        self.calls = 0

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3, **kw):
        self.calls += 1
        yield "plain sentence"


def test_parse_batch_response_variants():
    assert ctx._parse_batch_response(_json.dumps(["a", "b", "c"]), 3) == ["a", "b", "c"]
    assert ctx._parse_batch_response('```json\n["a", "b"]\n```', 2) == ["a", "b"]      # fenced
    assert ctx._parse_batch_response('Sure: ["a", "b"] done', 2) == ["a", "b"]        # prose-wrapped
    assert ctx._parse_batch_response(_json.dumps(["a", "b"]), 3) is None              # wrong count
    assert ctx._parse_batch_response("not json at all", 2) is None
    assert ctx._parse_batch_response(_json.dumps({"x": 1}), 1) is None                # not an array
    assert ctx._parse_batch_response(_json.dumps([1, 2]), 2) is None                  # non-string items
    assert ctx._parse_batch_response("", 2) is None


def test_contextualize_chunks_batches_into_one_call(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true")
    monkeypatch.setenv("CONTEXTUAL_CONCURRENCY", "1")
    monkeypatch.setenv("CONTEXTUAL_BATCH_SIZE", "8")
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    chunks = [{"text": f"distinct chunk {i}"} for i in range(6)]
    p = _BatchProvider()
    out = ctx.contextualize_chunks("doc", chunks, provider=p)
    assert p.calls == 1                                  # all 6 chunks situated in ONE batched call
    assert p.batch_sizes == [6]
    assert out == [f"context for chunk {i + 1}" for i in range(6)]   # parsed back in chunk order


def test_contextualize_chunks_two_batches(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true")
    monkeypatch.setenv("CONTEXTUAL_CONCURRENCY", "1")
    monkeypatch.setenv("CONTEXTUAL_BATCH_SIZE", "4")
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    chunks = [{"text": f"chunk {i}"} for i in range(10)]
    p = _BatchProvider()
    out = ctx.contextualize_chunks("doc", chunks, provider=p)
    assert p.calls == 3                                  # 10 chunks / batch 4 -> 3 calls (4,4,2)
    assert sorted(p.batch_sizes) == [2, 4, 4]
    assert all(c.startswith("context for chunk") for c in out) and len(out) == 10


def test_contextualize_chunks_falls_back_per_chunk_on_unparseable_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true")
    monkeypatch.setenv("CONTEXTUAL_CONCURRENCY", "1")
    monkeypatch.setenv("CONTEXTUAL_BATCH_SIZE", "8")
    monkeypatch.setattr(ctx, "CACHE_FILE", tmp_path / "cache.json")
    chunks = [{"text": f"chunk {i}"} for i in range(4)]
    p = _PlainProvider()                                 # non-JSON reply -> batch parse fails
    out = ctx.contextualize_chunks("doc", chunks, provider=p)
    assert out == ["plain sentence"] * 4                 # every chunk still filled via per-chunk fallback
    assert p.calls == 1 + 4                              # 1 failed batch attempt + 4 per-chunk fallbacks
