"""Concurrency tests for the Gemini embedding path (backend/common/embeddings.py).

Fully offline: the google client is replaced with a fake that records how many requests are in
flight, so we can prove the output is still correct 768-d vectors AND that concurrency is bounded
by EMBED_CONCURRENCY. No network, no real API key."""
import math
import threading
import time

import pytest

from backend.common import embeddings as eg


def _fake_raw(text, dim):
    """Deterministic raw vector for a text — distinct per text so order can be verified."""
    seed = float(sum(ord(c) for c in text) + 1)
    return [seed + j for j in range(dim)]


class _Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0
        self.calls = 0

    def enter(self):
        with self.lock:
            self.inflight += 1
            self.calls += 1
            self.max_inflight = max(self.max_inflight, self.inflight)

    def leave(self):
        with self.lock:
            self.inflight -= 1


class _Emb:
    def __init__(self, values):
        self.values = values


class _Resp:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _Models:
    def __init__(self, tracker, fail_batches=False, sleep=0.02):
        self.tracker = tracker
        self.fail_batches = fail_batches
        self.sleep = sleep

    def embed_content(self, model, contents, config):
        self.tracker.enter()
        try:
            time.sleep(self.sleep)                          # hold the slot so concurrency is visible
            items = contents if isinstance(contents, list) else [contents]
            if self.fail_batches and len(items) > 1:
                raise RuntimeError("simulated: provider rejects multi-text batches")
            dim = getattr(config, "output_dimensionality", 768) or 768
            return _Resp([_Emb(_fake_raw(t, dim)) for t in items])
        finally:
            self.tracker.leave()


class _Client:
    def __init__(self, models):
        self.models = models


def _install(monkeypatch, tracker, *, fail_batches=False, sleep=0.02):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setattr(eg, "_google_client",
                        lambda: _Client(_Models(tracker, fail_batches, sleep)))


def _expected(texts, dim=768):
    return [eg._l2(_fake_raw(t, dim)) for t in texts]


def test_returns_correct_dim_normalized_and_in_order(monkeypatch):
    tr = _Tracker()
    _install(monkeypatch, tr)
    monkeypatch.setattr(eg, "EMBED_CONCURRENCY", 8)
    monkeypatch.setattr(eg, "EMBED_BATCH_SIZE", 4)          # 10 texts -> 3 batches, run concurrently
    texts = [f"chunk number {i}" for i in range(10)]
    out = eg.embed_documents(texts)
    exp = _expected(texts)
    assert len(out) == 10
    for i in range(10):
        assert len(out[i]) == 768                          # same 768-d output
        assert out[i] == pytest.approx(exp[i])             # correct AND in input order
        assert abs(math.sqrt(sum(x * x for x in out[i])) - 1.0) < 1e-6   # L2-normalized


def test_concurrency_is_bounded(monkeypatch):
    tr = _Tracker()
    _install(monkeypatch, tr, sleep=0.03)
    monkeypatch.setattr(eg, "EMBED_CONCURRENCY", 4)
    monkeypatch.setattr(eg, "EMBED_BATCH_SIZE", 1)          # each text -> its own request
    texts = [f"t{i}" for i in range(20)]
    out = eg.embed_documents(texts)
    assert len(out) == 20
    assert tr.calls == 20
    assert tr.max_inflight <= 4                             # never exceeds EMBED_CONCURRENCY
    assert tr.max_inflight >= 2                             # genuinely parallel, not sequential


def test_per_text_fallback_when_batch_rejected(monkeypatch):
    tr = _Tracker()
    _install(monkeypatch, tr, fail_batches=True)           # any multi-text batch raises
    monkeypatch.setattr(eg, "EMBED_CONCURRENCY", 8)
    monkeypatch.setattr(eg, "EMBED_BATCH_SIZE", 100)       # 5 texts -> 1 batch (rejected) -> per-text
    texts = [f"paper {i}" for i in range(5)]
    out = eg.embed_documents(texts)
    exp = _expected(texts)
    assert len(out) == 5
    for i in range(5):
        assert out[i] == pytest.approx(exp[i])             # fallback still correct + ordered
    assert tr.calls == 1 + 5                               # one failed batch try + five per-text


def test_concurrency_one_is_sequential(monkeypatch):
    tr = _Tracker()
    _install(monkeypatch, tr)
    monkeypatch.setattr(eg, "EMBED_CONCURRENCY", 1)
    monkeypatch.setattr(eg, "EMBED_BATCH_SIZE", 1)
    out = eg.embed_documents([f"t{i}" for i in range(6)])
    assert len(out) == 6
    assert tr.max_inflight == 1                            # strictly sequential


def test_embed_query_is_single_request(monkeypatch):
    tr = _Tracker()
    _install(monkeypatch, tr)
    monkeypatch.setattr(eg, "EMBED_CONCURRENCY", 8)
    v = eg.embed_query("how does mvdr beamforming work")
    assert len(v) == 768
    assert tr.calls == 1                                   # single text -> one request, no pool
