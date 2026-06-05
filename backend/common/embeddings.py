"""
Unified text-embedding layer.

Pick the backend with EMBEDDING_PROVIDER in .env:
  - "google"  -> Gemini Embedding API (gemini-embedding-001), free tier via a
                 GEMINI_API_KEY from https://aistudio.google.com/apikey
  - "local"   -> sentence-transformers model on the local GPU/CPU (BAAI/bge-*)

Both expose the same interface and return L2-normalized vectors of length
EMBEDDING_DIM, so the rest of the pipeline (Oracle VECTOR column, cosine
search) is unchanged.
"""
from __future__ import annotations

import math
import os
import time
from typing import List

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "4"))
EMBED_SLEEP = float(os.getenv("EMBED_SLEEP", "0.0"))  # optional pause between calls


def provider() -> str:
    return (os.getenv("EMBEDDING_PROVIDER", "local") or "local").strip().lower()


def provider_label() -> str:
    if provider() == "google":
        return f"google · {os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')} ({os.getenv('EMBEDDING_DIM', '768')}d)"
    return f"local · {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')}"


def _l2(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


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


def _google_embed(texts: List[str], task_type: str) -> List[List[float]]:
    from google.genai import types
    client = _google_client()
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    dim = int(os.getenv("EMBEDDING_DIM", "768"))
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)

    def call(batch: List[str]) -> List[List[float]]:
        for attempt in range(5):
            try:
                resp = client.models.embed_content(model=model, contents=batch, config=cfg)
                return [_l2(list(e.values)) for e in resp.embeddings]
            except Exception as exc:  # rate limit / transient -> backoff
                msg = str(exc).lower()
                transient = any(k in msg for k in ("rate", "quota", "429", "resource_exhausted", "deadline", "unavailable", "503"))
                if transient and attempt < 4:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise

    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        try:
            out.extend(call(batch))
        except Exception:
            # If a multi-item batch is rejected, fall back to one at a time.
            if len(batch) > 1:
                for t in batch:
                    out.extend(call([t]))
            else:
                raise
        if EMBED_SLEEP:
            time.sleep(EMBED_SLEEP)
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
        return _google_embed([text], "RETRIEVAL_QUERY")[0]
    return _local_embed([text])[0]
