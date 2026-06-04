
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class RetrieverMode:
    name: str
    query_routes: int
    vector_top_k: int
    bm25_top_k: int
    rerank_top_n: int
    total_source_limit: int
    per_topic_source_limit: int
    max_sources_per_paper: int
    context_budget_chars: int
    use_question_memory: bool
    question_memory_threshold: float


MODES: Dict[str, RetrieverMode] = {
    "fast": RetrieverMode(
        name="Fast",
        query_routes=3,
        vector_top_k=18,
        bm25_top_k=18,
        rerank_top_n=18,
        total_source_limit=8,
        per_topic_source_limit=2,
        max_sources_per_paper=2,
        context_budget_chars=18_000,
        use_question_memory=True,
        question_memory_threshold=0.68,
    ),
    "balanced": RetrieverMode(
        name="Balanced",
        query_routes=4,
        vector_top_k=35,
        bm25_top_k=35,
        rerank_top_n=35,
        total_source_limit=14,
        per_topic_source_limit=3,
        max_sources_per_paper=3,
        context_budget_chars=30_000,
        use_question_memory=True,
        question_memory_threshold=0.72,
    ),
    "deep": RetrieverMode(
        name="Deep",
        query_routes=6,
        vector_top_k=60,
        bm25_top_k=60,
        rerank_top_n=60,
        total_source_limit=22,
        per_topic_source_limit=5,
        max_sources_per_paper=4,
        context_budget_chars=52_000,
        use_question_memory=False,
        question_memory_threshold=0.86,
    ),
}


def normalize_mode(mode: str | None = None) -> str:
    raw = (mode or os.getenv("RESEARCH_MODE") or "Balanced").strip().lower()
    if raw not in MODES:
        return "balanced"
    return raw


def get_mode(mode: str | None = None) -> RetrieverMode:
    return MODES[normalize_mode(mode)]


def apply_mode_to_env(mode: str | None = None) -> RetrieverMode:
    cfg = get_mode(mode)

    os.environ["RESEARCH_MODE"] = cfg.name
    os.environ["MAX_QUERY_ROUTES"] = str(cfg.query_routes)
    os.environ["VECTOR_TOP_K"] = str(cfg.vector_top_k)
    os.environ["BM25_TOP_K"] = str(cfg.bm25_top_k)
    os.environ["RERANK_TOP_N"] = str(cfg.rerank_top_n)
    os.environ["TOTAL_SOURCE_LIMIT"] = str(cfg.total_source_limit)
    os.environ["PER_TOPIC_SOURCE_LIMIT"] = str(cfg.per_topic_source_limit)
    os.environ["MAX_SOURCES_PER_PAPER"] = str(cfg.max_sources_per_paper)
    os.environ["SOURCE_CONTEXT_BUDGET_CHARS"] = str(cfg.context_budget_chars)
    os.environ["USE_QUESTION_MEMORY"] = "true" if cfg.use_question_memory else "false"
    os.environ["QUESTION_MEMORY_THRESHOLD"] = str(cfg.question_memory_threshold)

    return cfg


def mode_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, str(default)))
    except Exception:
        return default


def mode_float(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, str(default)))
    except Exception:
        return default


def vector_top_k(default: int = 35) -> int:
    return mode_int("VECTOR_TOP_K", default)


def bm25_top_k(default: int = 35) -> int:
    return mode_int("BM25_TOP_K", default)


def rerank_top_n(default: int = 35) -> int:
    return mode_int("RERANK_TOP_N", default)


def max_query_routes(default: int = 4) -> int:
    return mode_int("MAX_QUERY_ROUTES", default)


def total_source_limit(default: int = 14) -> int:
    return mode_int("TOTAL_SOURCE_LIMIT", default)


def per_topic_source_limit(default: int = 3) -> int:
    return mode_int("PER_TOPIC_SOURCE_LIMIT", default)


def max_sources_per_paper(default: int = 3) -> int:
    return mode_int("MAX_SOURCES_PER_PAPER", default)


def context_budget_chars(default: int = 30000) -> int:
    return mode_int("SOURCE_CONTEXT_BUDGET_CHARS", default)


def use_question_memory(default: bool = True) -> bool:
    raw = os.getenv("USE_QUESTION_MEMORY")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def question_memory_threshold(default: float = 0.72) -> float:
    return mode_float("QUESTION_MEMORY_THRESHOLD", default)


def limit_sources_by_mode(sources: Sequence[Any], default: int = 14) -> List[Any]:
    return list(sources)[:total_source_limit(default)]


def limit_routes_by_mode(routes: Sequence[Any], default: int = 4) -> List[Any]:
    return list(routes)[:max_query_routes(default)]


def trim_text_to_mode_budget(text: str, default: int = 30000) -> str:
    budget = context_budget_chars(default)
    if not text or len(text) <= budget:
        return text
    return text[:budget].rsplit("\n", 1)[0] + "\n\n[trimmed by selected research mode budget]"


if __name__ == "__main__":
    for mode_name in ["Fast", "Balanced", "Deep"]:
        cfg = apply_mode_to_env(mode_name)
        print("=" * 80)
        print(cfg.name)
        print("MAX_QUERY_ROUTES =", os.environ["MAX_QUERY_ROUTES"])
        print("VECTOR_TOP_K =", os.environ["VECTOR_TOP_K"])
        print("BM25_TOP_K =", os.environ["BM25_TOP_K"])
        print("RERANK_TOP_N =", os.environ["RERANK_TOP_N"])
        print("TOTAL_SOURCE_LIMIT =", os.environ["TOTAL_SOURCE_LIMIT"])
        print("SOURCE_CONTEXT_BUDGET_CHARS =", os.environ["SOURCE_CONTEXT_BUDGET_CHARS"])
        print("USE_QUESTION_MEMORY =", os.environ["USE_QUESTION_MEMORY"])
