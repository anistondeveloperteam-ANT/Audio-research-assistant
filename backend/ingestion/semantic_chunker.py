"""Optional semantic chunking (SEMANTIC_CHUNKER=true).

Splits prose into TOPIC-coherent segments using Chonkie's SemanticChunker with a Model2Vec static
embedder (CPU-only, ~30 MB, no GPU, no torch inference) — so a chunk starts and ends where an idea
does, instead of at a fixed character budget. Default OFF; when off (or if Chonkie/Model2Vec is
unavailable) callers fall back to the legacy sentence-packing splitter. The heavy model is built ONCE
and cached. This module never imports chonkie at import time, so it's free to import unconditionally."""
from __future__ import annotations

import os
import threading
from typing import List, Optional

_LOCK = threading.Lock()
_CHUNKER = None          # cached SemanticChunker (or None if unavailable)
_TRIED = False           # have we attempted to build it yet?

_DEFAULT_MODEL = "minishlab/potion-base-8M"      # tiny static CPU embedder


def semantic_chunker_enabled() -> bool:
    """Whether topic-coherent semantic chunking is on. OFF by default — the legacy splitter is the
    safe, dependency-free path. Set SEMANTIC_CHUNKER=true to enable."""
    return (os.getenv("SEMANTIC_CHUNKER", "false") or "false").strip().lower() in ("1", "true", "yes", "on")


def _model_name() -> str:
    return (os.getenv("SEMANTIC_CHUNKER_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _threshold() -> float:
    """Similarity threshold (0-1): higher = split more eagerly at topic shifts."""
    try:
        return min(1.0, max(0.0, float(os.getenv("SEMANTIC_CHUNKER_THRESHOLD", "0.75"))))
    except (TypeError, ValueError):
        return 0.75


def _chunk_size() -> int:
    """Max tokens per chunk. 512 aligns with bge-large's embedding window, so a chunk is never bigger
    than what the embedder can actually encode."""
    try:
        return max(64, int(os.getenv("SEMANTIC_CHUNK_SIZE", "512")))
    except (TypeError, ValueError):
        return 512


def _get_chunker():
    """Build the SemanticChunker once (the Model2Vec load is the cost) and cache it. Returns None if
    chonkie / the model can't be loaded, so the caller degrades to the legacy splitter."""
    global _CHUNKER, _TRIED
    with _LOCK:
        if _TRIED:
            return _CHUNKER
        _TRIED = True
        try:
            from chonkie import SemanticChunker
            _CHUNKER = SemanticChunker(embedding_model=_model_name(),
                                       threshold=_threshold(), chunk_size=_chunk_size())
        except Exception:
            _CHUNKER = None          # missing dep / offline / bad model -> legacy fallback
        return _CHUNKER


def semantic_split(text: str) -> Optional[List[str]]:
    """Split `text` into topic-coherent segments. Returns a list of strings, or **None** when the
    feature is off / unavailable / produced nothing — the signal for the caller to use the legacy
    splitter. Fail-soft: any error returns None rather than raising into the ingest pipeline."""
    if not text or not text.strip() or not semantic_chunker_enabled():
        return None
    chunker = _get_chunker()
    if chunker is None:
        return None
    try:
        chunks = chunker(text)
    except Exception:
        return None
    segments = [c.text.strip() for c in chunks if getattr(c, "text", "").strip()]
    return segments or None


def _reset_for_tests() -> None:
    """Drop the cached chunker so a test can re-exercise the lazy build."""
    global _CHUNKER, _TRIED
    with _LOCK:
        _CHUNKER = None
        _TRIED = False
