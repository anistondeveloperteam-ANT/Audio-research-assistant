"""
BGE local embeddings — switch to BAAI/bge-large-en-v1.5 (1024-d), fp16 on CUDA with CPU fallback, and
the asymmetric query instruction. Everything is MOCKED (a fake SentenceTransformer + a fake DB cursor),
so the offline suite never downloads the 1.3 GB model or touches Oracle; an opt-in test exercises the
real model.

Proves: (a) embeds to 1024-d L2-normalized vectors; (b) the retrieval instruction is prepended to
QUERIES only (not documents), and is empty for bge-m3; (e) fp16 on CUDA + CPU fallback on OOM (never
crashes); (f) graceful fallback if the model can't load on CUDA; (c)/(d) the re-ingestion reset drops +
recreates the VECTOR column at the new dimension and clears old embeddings.
"""
import math
import sys
import types

import pytest

import backend.common.embeddings as emb


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(emb, "_st_model", None, raising=False)
    monkeypatch.setattr(emb, "_st_device", None, raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    yield


def _install_fake_st(monkeypatch, *, dim=1024, device="cpu", load_fail_on=None, oom_first=False):
    """Inject a fake `sentence_transformers` + force the embedding device, recording what's encoded."""
    rec = {"inputs": [], "device": None, "halved": False, "oom_done": False}

    class _Model:
        def __init__(self, name, device="cpu"):
            self.device = device
            rec["device"] = device

        def half(self):
            rec["halved"] = True
            return self

        def to(self, d):
            self.device = d
            rec["device"] = d
            return self

        def float(self):
            return self

        def encode(self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True, **k):
            rec["inputs"].append(list(texts))
            if oom_first and not rec["oom_done"] and str(self.device).startswith("cuda"):
                rec["oom_done"] = True
                raise RuntimeError("CUDA out of memory")
            import numpy as np
            out = []
            for _t in texts:
                v = np.linspace(0.1, 1.0, dim).astype("float32")
                if normalize_embeddings:
                    v = v / (float(np.linalg.norm(v)) or 1.0)
                out.append(v)
            return np.array(out)

    def _ctor(name, device="cpu"):
        if load_fail_on is not None and str(device).startswith(load_fail_on):
            raise RuntimeError(f"cannot load on {device}")
        return _Model(name, device)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _ctor
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    import backend.common.device as dev
    monkeypatch.setattr(dev, "resolve_device", lambda role=None: device)
    return rec


# ---- (a) 1024-d, L2-normalized ----
def test_embeds_to_1024_dim_normalized(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    _install_fake_st(monkeypatch, dim=1024)
    vecs = emb.embed_documents(["a document about beamforming"])
    assert len(vecs) == 1 and len(vecs[0]) == 1024
    assert abs(math.sqrt(sum(x * x for x in vecs[0])) - 1.0) < 1e-5


# ---- (b) query instruction on QUERIES only; documents stay raw; bge-m3 instruction-free ----
def test_query_instruction_applied_to_queries_not_documents(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    rec = _install_fake_st(monkeypatch)
    emb.embed_query("what is MVDR beamforming")
    emb.embed_documents(["MVDR is a minimum-variance distortionless-response beamformer."])
    q_in, d_in = rec["inputs"][0][0], rec["inputs"][1][0]
    assert q_in.startswith("Represent this sentence for searching relevant passages:")
    assert "what is MVDR beamforming" in q_in
    assert d_in == "MVDR is a minimum-variance distortionless-response beamformer."   # no instruction


def test_bge_m3_query_is_instruction_free(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    rec = _install_fake_st(monkeypatch)
    emb.embed_query("hola mundo")
    assert rec["inputs"][0][0] == "hola mundo"
    assert emb.query_instruction() == ""


def test_query_instruction_default_and_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    assert emb.query_instruction() == "Represent this sentence for searching relevant passages:"
    monkeypatch.setenv("BGE_QUERY_INSTRUCTION", "Query:")
    assert emb.query_instruction() == "Query:"


# ---- (e) fp16 on CUDA + CPU fallback on OOM (never crashes) ----
def test_fp16_applied_on_cuda(monkeypatch):
    rec = _install_fake_st(monkeypatch, device="cuda")
    emb.embed_documents(["x"])
    assert rec["halved"] is True                       # model.half() called -> fp16


def test_cuda_oom_falls_back_to_cpu(monkeypatch):
    rec = _install_fake_st(monkeypatch, device="cuda", oom_first=True)
    vecs = emb.embed_documents(["x"])                   # OOMs on CUDA -> retries on CPU
    assert len(vecs) == 1 and len(vecs[0]) == 1024
    assert rec["device"] == "cpu" and emb._st_device == "cpu"


def test_fp16_disabled_skips_half(monkeypatch):
    monkeypatch.setenv("EMBED_FP16", "false")
    rec = _install_fake_st(monkeypatch, device="cuda")
    emb.embed_documents(["x"])
    assert rec["halved"] is False


# ---- (f) graceful load fallback ----
def test_cuda_load_failure_falls_back_to_cpu(monkeypatch):
    _install_fake_st(monkeypatch, device="cuda", load_fail_on="cuda")
    emb.embed_documents(["x"])
    assert emb._st_device == "cpu"                      # loaded on CPU after CUDA load failed


def test_total_load_failure_raises(monkeypatch):
    _install_fake_st(monkeypatch, device="cuda", load_fail_on="")   # fails on cuda AND cpu
    with pytest.raises(Exception):
        emb.embed_documents(["x"])


def test_provider_label_reflects_local_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    assert emb.provider_label() == "local · BAAI/bge-large-en-v1.5 (1024d)"


# ---- concurrent encodes serialize (chat query + background ingest share one GPU model) ----
def test_concurrent_local_embeds_are_serialized(monkeypatch):
    """_encode_lock must serialize model.encode so a chat query and a background ingest embedding
    (two server threads, one shared model) never run on it at once."""
    import threading
    import time
    import numpy as np

    active = {"n": 0, "max": 0}
    guard = threading.Lock()

    class _Model:
        def encode(self, texts, **k):
            with guard:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.02)                               # widen the overlap window
            with guard:
                active["n"] -= 1
            return np.zeros((len(texts), 8), dtype="float32")

    monkeypatch.setattr(emb, "_st_device", "cpu", raising=False)
    monkeypatch.setattr(emb, "_local_model", lambda: _Model())
    threads = [threading.Thread(target=lambda: emb.embed_documents(["x"])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["max"] == 1                              # the lock kept encodes one-at-a-time


# ---- (c)/(d) re-ingestion reset: drop + recreate VECTOR column at the new dim, clear old embeddings ----
def test_reset_recreates_vector_column_and_clears_embeddings():
    from backend.database import reset_embeddings as r
    calls = []

    class _Cur:
        _fetch = None

        def execute(self, sql, *a):
            calls.append(sql)
            low = sql.lower()
            if "user_tab_columns" in low:
                self._fetch = (1,)                     # column already exists -> must be dropped
            elif "count(*) from chunks" in low:
                self._fetch = (42,)

        def fetchone(self):
            return self._fetch

    class _Conn:
        def commit(self):
            pass

    n = r.reset(_Cur(), _Conn(), 1024)
    joined = " ".join(calls).lower()
    assert n == 42
    assert "drop column embedding_vec" in joined                       # old 768-d vectors purged
    assert "add embedding_vec vector(1024, float32)" in joined         # recreated at the new dim
    assert "update chunks set embedding = null" in joined              # whole corpus re-embeds next


# ---- opt-in: actually load the real bge-large model (downloads ~1.3 GB) ----
@pytest.mark.skipif(
    __import__("os").getenv("EMBEDDING_INTEGRATION_TEST", "").strip().lower() not in ("1", "true", "yes"),
    reason="opt-in: set EMBEDDING_INTEGRATION_TEST=true to download + run the real bge-large model")
def test_real_bge_large_embeds_1024(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    emb._st_model = None
    emb._st_device = None
    v = emb.embed_query("speech enhancement with deep learning")
    assert len(v) == 1024 and abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-3
