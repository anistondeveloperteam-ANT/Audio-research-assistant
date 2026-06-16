"""
Unified text-embedding layer.

Pick the backend with EMBEDDING_PROVIDER in .env:
  - "google"  -> Gemini Embedding API (gemini-embedding-2), free tier via a
                 GEMINI_API_KEY from https://aistudio.google.com/apikey
  - "local"   -> sentence-transformers model on the local GPU/CPU (BAAI/bge-*)

Both expose the same interface and return L2-normalized vectors of length
EMBEDDING_DIM, so the rest of the pipeline (Oracle VECTOR column, cosine
search) is unchanged.
"""
from __future__ import annotations

import concurrent.futures
import math
import os
import time
from typing import List, Optional

# Gemini's embed_content accepts a LIST of contents and returns one embedding each, so we send a
# whole BATCH per request — far fewer round-trips (faster) and far less rate-limit pressure than
# one request per chunk (which is what tripped the free-tier 429). Tune with EMBED_BATCH_SIZE.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
# How many embedding requests to send CONCURRENTLY (batches + any per-text fallback). The provider,
# model, and 768-d output are unchanged — only the request pattern goes from sequential to parallel.
# Set 1 to force the old sequential behavior. Each request keeps its own 429/backoff (see _embed_call).
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))


class EmbeddingQuotaError(RuntimeError):
    """The embedding provider's quota / rate limit is exhausted (HTTP 429)."""


def provider() -> str:
    return (os.getenv("EMBEDDING_PROVIDER", "local") or "local").strip().lower()


def provider_label() -> str:
    if provider() == "google":
        return f"google · {os.getenv('EMBEDDING_MODEL', 'gemini-embedding-2')} ({os.getenv('EMBEDDING_DIM', '768')}d)"
    return f"local · {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')}"


def _l2(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


# ----------------------------------------------------------------------
# Retrieval text formatting (Gemini gemini-embedding-2)
# Giving the embedder a little structure — the task for queries, and the
# title/section/concepts for documents — improves query↔document matching.
# These are applied only for the Google provider so local models keep raw text.
# ----------------------------------------------------------------------
def format_retrieval_query(query: str) -> str:
    return f"task: question answering | query: {(query or '').strip()}"


def format_retrieval_document(title=None, section=None, concepts=None, text: str = "") -> str:
    """Build a metadata-enriched document string for embedding a chunk."""
    if isinstance(concepts, (list, tuple)):
        concepts = ", ".join(str(c).strip() for c in concepts if str(c).strip())
    parts = [f"title: {str(title).strip() if title else 'none'}"]
    if section and str(section).strip():
        parts.append(f"section: {str(section).strip()}")
    if concepts and str(concepts).strip():
        parts.append(f"concepts: {str(concepts).strip()}")
    parts.append(f"text: {(text or '').strip()}")
    return " | ".join(parts)


# ----------------------------------------------------------------------
# Google Gemini embeddings
# ----------------------------------------------------------------------
_genai_client = None


def _google_client():
    global _genai_client
    if _genai_client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env."
            )
        _genai_client = genai.Client(api_key=key)
    return _genai_client


_TRANSIENT = ("rate", "quota", "429", "resource_exhausted", "deadline",
              "unavailable", "503", "500", "internal")


def _is_transient(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in _TRANSIENT)


def _is_quota(msg: str) -> bool:
    m = (msg or "").lower()
    return "quota" in m or "resource_exhausted" in m or "429" in m


def _embed_call(client, model, cfg, contents) -> List[List[float]]:
    """One embed_content request — `contents` may be a single string OR a batch list. Returns one
    L2-normalized vector per input, in order. Exponential backoff on transient errors; a clear
    EmbeddingQuotaError when the quota is exhausted (instead of a raw 429 traceback)."""
    delay = 1.0
    for attempt in range(6):
        try:
            resp = client.models.embed_content(model=model, contents=contents, config=cfg)
            return [_l2(list(e.values)) for e in resp.embeddings]
        except Exception as exc:                       # noqa: BLE001 - classify by message
            msg = str(exc)
            if _is_transient(msg) and attempt < 5:
                time.sleep(min(delay, 30))
                delay *= 2
                continue
            if _is_quota(msg):
                raise EmbeddingQuotaError(
                    "Gemini embedding quota exhausted (HTTP 429). Either wait for the quota to "
                    "reset, or set EMBEDDING_PROVIDER=local in .env to embed on your GPU/CPU with "
                    "no quota, then re-run `python pipeline.py` to (re)build the index."
                ) from exc
            raise


def _try_batch(client, model, cfg, batch: List[str]) -> Optional[List[List[float]]]:
    """Embed `batch` as ONE request. Returns the vectors, or None to signal the caller should fall
    back to per-text (the batch was rejected or returned a partial result). Quota errors propagate."""
    try:
        vecs = _embed_call(client, model, cfg, batch)
        if len(vecs) == len(batch):
            return vecs
    except EmbeddingQuotaError:
        raise
    except Exception:                                          # noqa: BLE001 - batch rejected
        pass
    return None


def _embed_per_text(client, model, cfg, batch: List[str], ex) -> List[List[float]]:
    """One request per text, run concurrently on `ex` (or sequentially if ex is None). Order kept."""
    one = lambda t: _embed_call(client, model, cfg, t)[0]      # noqa: E731 - tiny inline mapper
    if ex is None:
        return [one(t) for t in batch]
    return list(ex.map(one, batch))


def _google_embed(texts: List[str], task_type: str) -> List[List[float]]:
    from google.genai import types
    client = _google_client()
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    dim = int(os.getenv("EMBEDDING_DIM", "768"))
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)

    bs = max(1, EMBED_BATCH_SIZE)
    batches = [texts[i:i + bs] for i in range(0, len(texts), bs)]
    conc = max(1, EMBED_CONCURRENCY)

    # Sequential when concurrency is disabled or there's nothing to parallelize (e.g. one query).
    if conc == 1 or len(texts) <= 1:
        out: List[List[float]] = []
        for batch in batches:
            vecs = _try_batch(client, model, cfg, batch)
            out.extend(vecs if vecs is not None else _embed_per_text(client, model, cfg, batch, None))
        return out

    # Parallel: at most `conc` requests in flight. Phase 1 tries every batch as one request
    # concurrently; Phase 2 re-embeds any rejected batch's texts concurrently (the real slow path
    # when the provider won't accept multi-text batches). Results stay in input order.
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        attempts = list(ex.map(lambda b: _try_batch(client, model, cfg, b), batches))
        out = []
        for batch, vecs in zip(batches, attempts):
            out.extend(vecs if vecs is not None else _embed_per_text(client, model, cfg, batch, ex))
    return out


# ----------------------------------------------------------------------
# Local sentence-transformers embeddings
# ----------------------------------------------------------------------
_st_model = None


def _local_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        from backend.common.device import resolve_device
        name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        _st_model = SentenceTransformer(name, device=resolve_device("EMBEDDING_DEVICE"))
    return _st_model


def _local_embed(texts: List[str]) -> List[List[float]]:
    model = _local_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [[float(x) for x in v.tolist()] for v in vecs]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed passages/chunks for indexing."""
    if not texts:
        return []
    if provider() == "google":
        return _google_embed(texts, "RETRIEVAL_DOCUMENT")
    return _local_embed(texts)


def embed_query(text: str) -> List[float]:
    """Embed a single search query."""
    if provider() == "google":
        return _google_embed([format_retrieval_query(text)], "RETRIEVAL_QUERY")[0]
    return _local_embed([text])[0]
