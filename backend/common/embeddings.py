"""
Local text-embedding layer (sentence-transformers).

The embedding backend is a LOCAL model on your GPU/CPU — free, no API key, no network, no quota:

  - default  BAAI/bge-large-en-v1.5  (1024-d) — top-tier retrieval, fits a 6 GB GPU in fp16.
  - EMBEDDING_MODEL=BAAI/bge-m3      (1024-d, multilingual, instruction-free).

It returns L2-normalized vectors of length `embedding_dim()`, so the rest of the pipeline (Oracle
VECTOR column, cosine search) is unchanged. bge models are encoded ASYMMETRICALLY: a retrieval
instruction is prepended to QUERIES only (not documents) for best recall. fp16 on CUDA with an automatic
CPU fallback on out-of-memory, and a CPU fallback if the model can't load on the GPU.

(Google/Gemini embeddings were removed — `provider()` is always "local"; a stale EMBEDDING_PROVIDER /
EMBEDDING_MODEL / EMBEDDING_DIM in .env is ignored.)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List

logger = logging.getLogger(__name__)

# Local (sentence-transformers) defaults. bge-large-en-v1.5 is 1024-d, free, fits 6 GB in fp16.
_DEFAULT_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"
# bge English v1.5 models want this instruction on the QUERY side only; bge-m3 is instruction-free.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
# Output dimensions of the common local models, used for a cheap label/column hint without loading them.
_KNOWN_DIMS = {"bge-large": 1024, "bge-m3": 1024, "bge-base": 768, "bge-small": 384}


def provider() -> str:
    """The embedding backend — ALWAYS local (Google/Gemini embeddings were removed). Kept so callers that
    branch on provider() keep working; a stale EMBEDDING_PROVIDER in .env is ignored."""
    return "local"


def _local_model_name() -> str:
    """The sentence-transformers model name. A stale Gemini/Google name left in .env (e.g.
    'gemini-embedding-2') is ignored so the loader always receives a real local model."""
    name = (os.getenv("EMBEDDING_MODEL") or "").strip()
    low = name.lower()
    if not name or "gemini" in low or low.startswith(("models/", "text-embedding", "embedding-")):
        return _DEFAULT_LOCAL_MODEL
    return name


def _dim_hint() -> int:
    """The model's output dimension WITHOUT loading it — from the known-model table, else EMBEDDING_DIM,
    else 1024. (For the authoritative dimension, use embedding_dim(), which asks the loaded model.)"""
    low = _local_model_name().lower()
    for key, dim in _KNOWN_DIMS.items():
        if key in low:
            return dim
    try:
        return int(os.getenv("EMBEDDING_DIM", "1024"))
    except (TypeError, ValueError):
        return 1024


def provider_label() -> str:
    return f"local · {_local_model_name()} ({_dim_hint()}d)"


def query_instruction() -> str:
    """The bge retrieval query instruction, applied to QUERIES only. Empty for bge-m3 (instruction-free)
    and whenever BGE_QUERY_INSTRUCTION is set blank — so the asymmetric prefix is correct per model."""
    if "bge-m3" in _local_model_name().lower():
        return ""
    return os.getenv("BGE_QUERY_INSTRUCTION", _BGE_QUERY_INSTRUCTION)


# ----------------------------------------------------------------------
# Local sentence-transformers model (loaded once)
# ----------------------------------------------------------------------
_st_model = None
_st_device = None
# Serializes model load + encode: a chat query (embed_query) and a background ingest (embed_documents)
# run in different server threads but share this ONE GPU model — concurrent .encode() could race it.
_encode_lock = threading.Lock()


def _embed_fp16() -> bool:
    return os.getenv("EMBED_FP16", "true").strip().lower() not in ("0", "false", "no", "off")


def _local_model():
    """Load the embedding model ONCE: fp16 on CUDA (≈2× faster, half the VRAM, so bge-large fits the 6 GB
    3050 alongside the reranker). Fail-open on load — a CUDA load failure retries on CPU (logged) rather
    than crashing."""
    global _st_model, _st_device
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        from backend.common.device import resolve_device
        name = _local_model_name()
        device = resolve_device("EMBEDDING_DEVICE")
        try:
            model = SentenceTransformer(name, device=device)
        except Exception as exc:                       # noqa: BLE001 - load fail-open: fall back to CPU
            if str(device).startswith("cuda"):
                logger.warning("embedding model %s failed to load on %s (%s); falling back to CPU",
                               name, device, exc)
                device = "cpu"
                model = SentenceTransformer(name, device="cpu")
            else:
                raise
        precision = "fp32"
        if _embed_fp16() and str(device).startswith("cuda"):
            try:
                model.half()
                precision = "fp16"
            except Exception as exc:                   # noqa: BLE001 - fp16 optional
                logger.info("embedding fp16 unavailable (%s); using fp32", exc)
        _st_model, _st_device = model, device
        logger.info("embedding model: %s on %s (%s)", name, device, precision)
    return _st_model


def embedding_dim() -> int:
    """The AUTHORITATIVE output dimension of the embedding model (e.g. 1024 for bge-large) — used to size
    the Oracle VECTOR column, so a stale EMBEDDING_DIM in .env can never create a wrong-sized column.
    Asks the loaded model; falls back to the cheap hint if the model can't report it."""
    try:
        d = _local_model().get_sentence_embedding_dimension()
        if d:
            return int(d)
    except Exception:                                  # noqa: BLE001
        pass
    return _dim_hint()


def _is_cuda_oom(exc: Exception) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:                                  # noqa: BLE001 - torch optional in this check
        pass
    return "out of memory" in str(exc).lower()


def _local_embed(texts: List[str]) -> List[List[float]]:
    """Encode in bounded batches (EMBED_ST_BATCH, default 16) so VRAM stays in budget, returning
    L2-normalized vectors. On a CUDA OOM the model is moved to CPU permanently and the batch is retried
    there (logged) — it NEVER OOM-crashes. Held under `_encode_lock` so a chat query and a background
    ingest embedding (two threads, one shared GPU model) serialize instead of racing the model."""
    global _st_model, _st_device
    with _encode_lock:
        model = _local_model()
        bs = max(1, int(os.getenv("EMBED_ST_BATCH", "16")))
        try:
            vecs = model.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
        except Exception as exc:                       # noqa: BLE001 - OOM -> CPU; anything else re-raises
            if not (_is_cuda_oom(exc) and str(_st_device).startswith("cuda")):
                raise
            logger.warning("embedding CUDA OOM; moving the model to CPU and retrying (batch=%d)", bs)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                          # noqa: BLE001
                pass
            model = model.to("cpu").float()            # CPU does fp32; never OOM again
            _st_model, _st_device = model, "cpu"
            vecs = model.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
    return [[float(x) for x in v.tolist()] for v in vecs]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed passages/chunks for indexing (documents are embedded RAW — no query instruction)."""
    if not texts:
        return []
    return _local_embed(texts)


def embed_query(text: str) -> List[float]:
    """Embed a single search query. The bge retrieval INSTRUCTION is prepended here (to the query ONLY —
    documents in embed_documents stay raw), which is the asymmetric encoding bge needs for best recall."""
    instr = query_instruction()
    q = f"{instr} {(text or '').strip()}".strip() if instr else (text or "")
    return _local_embed([q])[0]
