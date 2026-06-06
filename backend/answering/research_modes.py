"""
Single retrieval configuration.

The old Fast / Balanced / Deep "research modes" have been removed — the app now
always runs ONE optimized retrieval configuration tuned for high accuracy with
good speed. The functions below are kept (and accept a `mode` argument) only for
backward compatibility; the passed mode is ignored.

The retrieval pipeline (backend/retrieval/hybrid_retrieve.py) reads the env vars
that `apply_research_mode()` sets — VECTOR_TOP_K, BM25_TOP_K, RERANK_TOP_N,
MAX_SOURCES_PER_PAPER, etc.
"""
from __future__ import annotations

import os
from typing import Any, Dict

# One optimized configuration. Started from the evaluator's best baseline (Fast)
# and tuned slightly wider for accuracy without adding meaningful latency.
DEFAULT_RETRIEVAL_SETTINGS: Dict[str, Any] = {
    "mode": "Default",
    "description": "Single optimized retrieval mode (high accuracy, good speed).",
    "max_query_routes": 3,
    "total_source_limit": 12,
    "per_topic_source_limit": 2,
    "max_sources_per_paper": 2,
    "vector_top_k": 24,
    "bm25_top_k": 24,
    "rerank_top_n": 24,
    "context_budget_chars": 24000,
    "question_memory_threshold": 0.72,
    "use_question_memory": True,
}

# Backward-compatibility alias: anything that imported MODE_SETTINGS still works,
# but there is only one entry now.
MODE_SETTINGS: Dict[str, Dict[str, Any]] = {"Default": DEFAULT_RETRIEVAL_SETTINGS}


def normalize_mode(mode: str | None = None) -> str:
    """There is only one retrieval mode now — always returns 'Default'.
    Any input (Fast/Balanced/Deep/invalid/None) maps to 'Default'."""
    return "Default"


def get_mode_settings(mode: str | None = None) -> Dict[str, Any]:
    """Return the single default retrieval settings. `mode` is ignored."""
    return dict(DEFAULT_RETRIEVAL_SETTINGS)


def apply_research_mode(mode: str | None = None) -> Dict[str, Any]:
    """Apply the single default retrieval config to the process environment so the
    retrieval pipeline picks it up. `mode` is accepted but ignored (back-compat)."""
    settings = get_mode_settings()

    os.environ["RESEARCH_MODE"] = "Default"
    os.environ["MAX_QUERY_ROUTES"] = str(settings["max_query_routes"])
    os.environ["TOTAL_SOURCE_LIMIT"] = str(settings["total_source_limit"])
    os.environ["PER_TOPIC_SOURCE_LIMIT"] = str(settings["per_topic_source_limit"])
    os.environ["MAX_SOURCES_PER_PAPER"] = str(settings["max_sources_per_paper"])
    os.environ["VECTOR_TOP_K"] = str(settings["vector_top_k"])
    os.environ["BM25_TOP_K"] = str(settings["bm25_top_k"])
    os.environ["RERANK_TOP_N"] = str(settings["rerank_top_n"])
    os.environ["SOURCE_CONTEXT_BUDGET_CHARS"] = str(settings["context_budget_chars"])
    os.environ["QUESTION_MEMORY_THRESHOLD"] = str(settings["question_memory_threshold"])
    os.environ["USE_QUESTION_MEMORY"] = "true" if settings["use_question_memory"] else "false"

    return settings


if __name__ == "__main__":
    for k, v in apply_research_mode().items():
        print(f"{k}: {v}")
