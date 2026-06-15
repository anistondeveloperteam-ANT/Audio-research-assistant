"""Embedding layer: Gemini calls are BATCHED (one request per batch, not one per chunk — the fix
for the 429 + slowness), order is preserved, transient errors back off, and an exhausted quota
raises a clear, actionable error. The genai client is faked — no network."""
import math

import pytest

import backend.common.embeddings as emb


class _FakeEmb:
    def __init__(self, values):
        self.values = values


class _FakeResp:
    def __init__(self, embs):
        self.embeddings = embs


class _FakeModels:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []                      # one entry per embed_content request

    def embed_content(self, model, contents, config):
        self.calls.append(contents)
        return self._behavior(contents)


class _FakeClient:
    def __init__(self, behavior):
        self.models = _FakeModels(behavior)


def _ok(contents):
    """A 2D vector per input; component[1] (after L2) decreases with length -> an order signal."""
    items = contents if isinstance(contents, (list, tuple)) else [contents]
    return _FakeResp([_FakeEmb([float(len(str(c))), 1.0]) for c in items])


@pytest.fixture(autouse=True)
def _google_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(emb.time, "sleep", lambda *a, **k: None)   # no real backoff waits


def _install(monkeypatch, behavior):
    client = _FakeClient(behavior)
    monkeypatch.setattr(emb, "_google_client", lambda: client)
    return client


def test_documents_are_embedded_in_ONE_batched_request(monkeypatch):
    client = _install(monkeypatch, _ok)
    vecs = emb.embed_documents(["1", "2", "3", "4", "5"])
    assert len(vecs) == 5
    assert len(client.models.calls) == 1                  # ONE request — not five
    assert client.models.calls[0] == ["1", "2", "3", "4", "5"]   # sent as a batch list
    assert all(abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9 for v in vecs)  # L2-normalized


def test_query_is_a_single_request(monkeypatch):
    client = _install(monkeypatch, _ok)
    v = emb.embed_query("how does MVDR work")
    assert isinstance(v, list) and len(client.models.calls) == 1


def test_order_preserved_across_multiple_batches(monkeypatch):
    monkeypatch.setattr(emb, "EMBED_BATCH_SIZE", 2)
    client = _install(monkeypatch, _ok)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]           # lengths 1..5
    vecs = emb.embed_documents(texts)
    assert len(vecs) == 5
    assert len(client.models.calls) == 3                  # ceil(5 / 2) batches
    comp1 = [v[1] for v in vecs]
    assert comp1 == sorted(comp1, reverse=True)           # 1/sqrt(len^2+1) ↓ with len -> in order


def test_transient_error_backs_off_then_succeeds(monkeypatch):
    state = {"n": 0}

    def behavior(contents):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("429 rate limit, please retry shortly")
        return _ok(contents)

    _install(monkeypatch, behavior)
    vecs = emb.embed_documents(["a", "b"])
    assert len(vecs) == 2 and state["n"] == 2             # retried once, then succeeded


def test_exhausted_quota_raises_clear_actionable_error(monkeypatch):
    def behavior(contents):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota")

    _install(monkeypatch, behavior)
    with pytest.raises(emb.EmbeddingQuotaError) as ei:
        emb.embed_documents(["a"])
    msg = str(ei.value)
    assert "quota" in msg.lower() and "EMBEDDING_PROVIDER=local" in msg   # tells the user the fix


def test_batch_rejection_falls_back_to_per_text(monkeypatch):
    # Defensive: if an older API rejects a list payload, we must still embed (per-text), not fail.
    def behavior(contents):
        if isinstance(contents, (list, tuple)):
            raise ValueError("invalid argument: contents must be a single value")
        return _ok(contents)

    client = _install(monkeypatch, behavior)
    vecs = emb.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3
    assert len(client.models.calls) == 4                  # 1 failed batch + 3 per-text
