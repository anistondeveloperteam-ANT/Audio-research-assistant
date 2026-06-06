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

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

# The Gemini embedding API returns ONE embedding per request, so we send one
# text per call and run the calls concurrently for speed.
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "6"))


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


def _google_embed(texts: List[str], task_type: str) -> List[List[float]]:
    from google.genai import types
    client = _google_client()
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    dim = int(os.getenv("EMBEDDING_DIM", "768"))
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)

    def one(text: str) -> List[float]:
        """Embed a single text (the API returns exactly one embedding per call)."""
        for attempt in range(6):
            try:
                resp = client.models.embed_content(model=model, contents=text, config=cfg)
                return _l2(list(resp.embeddings[0].values))
            except Exception as exc:  # rate-limit / transient -> exponential backoff
                msg = str(exc).lower()
                transient = any(k in msg for k in (
                    "rate", "quota", "429", "resource_exhausted",
                    "deadline", "unavailable", "503", "500", "internal",
                ))
                if transient and attempt < 5:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise

    if len(texts) <= 1:
        return [one(t) for t in texts]

    # Fire requests concurrently; ThreadPoolExecutor.map preserves order.
    workers = max(1, min(EMBED_CONCURRENCY, len(texts)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, texts))


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
