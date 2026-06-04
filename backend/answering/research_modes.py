
from __future__ import annotations

import os
from typing import Dict, Any


MODE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "Fast": {
        "description": "Fastest mode. Best for quick questions and repeated/similar questions.",
        "max_query_routes": 3,
        "total_source_limit": 8,
        "per_topic_source_limit": 2,
        "max_sources_per_paper": 2,
        "vector_top_k": 18,
        "bm25_top_k": 18,
        "rerank_top_n": 18,
        "context_budget_chars": 18000,
        "question_memory_threshold": 0.68,
        "use_question_memory": True,
    },
    "Balanced": {
        "description": "Recommended mode. Strong quality with good speed for most questions.",
        "max_query_routes": 4,
        "total_source_limit": 14,
        "per_topic_source_limit": 3,
        "max_sources_per_paper": 3,
        "vector_top_k": 35,
        "bm25_top_k": 35,
        "rerank_top_n": 35,
        "context_budget_chars": 30000,
        "question_memory_threshold": 0.72,
        "use_question_memory": True,
    },
    "Deep": {
        "description": "Most powerful local mode. Uses more evidence and refreshes retrieval.",
        "max_query_routes": 6,
        "total_source_limit": 22,
        "per_topic_source_limit": 5,
        "max_sources_per_paper": 4,
        "vector_top_k": 60,
        "bm25_top_k": 60,
        "rerank_top_n": 60,
        "context_budget_chars": 52000,
        "question_memory_threshold": 0.86,
        "use_question_memory": False,
    },
}


def normalize_mode(mode: str | None) -> str:
    mode = (mode or "Balanced").strip().title()
    if mode not in MODE_SETTINGS:
        return "Balanced"
    return mode


def get_mode_settings(mode: str | None) -> Dict[str, Any]:
    mode = normalize_mode(mode)
    settings = dict(MODE_SETTINGS[mode])
    settings["mode"] = mode
    return settings


def apply_research_mode(mode: str | None) -> Dict[str, Any]:
    settings = get_mode_settings(mode)
    mode_name = settings["mode"]

    os.environ["RESEARCH_MODE"] = mode_name
    os.environ["PARSER_MODE"] = "auto"
    os.environ["ENABLE_DOCLING"] = "true"
    os.environ["ENABLE_MARKER"] = "false"
    os.environ["ENABLE_OCR"] = "true"
    os.environ["ANSWER_PROVIDER"] = os.getenv("ANSWER_PROVIDER", "manual") or "manual"

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
    for name in ["Fast", "Balanced", "Deep"]:
        s = apply_research_mode(name)
        print("=" * 80)
        print(name)
        for k, v in s.items():
            print(f"{k}: {v}")
