"""
Central compute-device selection for the embedding + reranker models.

Lets you spread model load across GPU and CPU via .env, which matters on
small GPUs (e.g. a 6 GB laptop card; the chat LLM runs in the cloud via OpenAI):

    DEVICE            global default: auto | cuda | cpu        (default: auto)
    EMBEDDING_DEVICE  override for the embedding model         (falls back to DEVICE)
    RERANKER_DEVICE   override for the cross-encoder reranker  (falls back to DEVICE)

"auto" means: use the GPU if CUDA is available, otherwise CPU.

Example .env to use BOTH GPU and CPU (embeddings on GPU, heavy reranker on CPU):
    DEVICE=auto
    EMBEDDING_DEVICE=cuda
    RERANKER_DEVICE=cpu
"""
from __future__ import annotations

import os


def _auto() -> str:
    """Return 'cuda' if a CUDA GPU is usable, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_device(role_env: str | None = None) -> str:
    """
    Resolve the device for a model.

    role_env: name of a role-specific env var (e.g. "EMBEDDING_DEVICE").
              If unset, falls back to the global DEVICE env var, then "auto".
    Returns one of: "cuda", "cpu" (or any explicit value the user set, e.g. "cuda:0").
    """
    value = ""
    if role_env:
        value = (os.getenv(role_env) or "").strip()
    if not value:
        value = (os.getenv("DEVICE") or "auto").strip()

    if value.lower() == "auto":
        return _auto()
    return value
