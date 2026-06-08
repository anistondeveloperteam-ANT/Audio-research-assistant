"""Configuration for the optional Memgraph GraphRAG layer."""
from __future__ import annotations

import dataclasses
import os


def env_flag(name: str, default: bool = False) -> bool:
    return (os.getenv(name, "true" if default else "false") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


@dataclasses.dataclass(frozen=True)
class GraphRagConfig:
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = ""
    password: str = ""
    database: str | None = None
    max_hops: int = 2
    max_results: int = 12
    timeout_seconds: float = 3.0
    build_batch_size: int = 200

    @classmethod
    def from_env(cls) -> "GraphRagConfig":
        database = (os.getenv("MEMGRAPH_DATABASE", "") or "").strip() or None
        return cls(
            enabled=env_flag("ENABLE_GRAPH_RAG", default=False),
            uri=(os.getenv("MEMGRAPH_URI", "bolt://localhost:7687") or "").strip(),
            user=(os.getenv("MEMGRAPH_USER", "") or "").strip(),
            password=os.getenv("MEMGRAPH_PASSWORD", "") or "",
            database=database,
            max_hops=env_int("GRAPH_RAG_MAX_HOPS", 2, minimum=1),
            max_results=env_int("GRAPH_RAG_MAX_RESULTS", 12, minimum=1),
            timeout_seconds=env_float("GRAPH_RAG_TIMEOUT_SECONDS", 3.0, minimum=0.5),
            build_batch_size=env_int("GRAPH_RAG_BUILD_BATCH_SIZE", 200, minimum=1),
        )


def graph_rag_enabled() -> bool:
    """Read live env state so tests and long-running app processes can flip it."""
    return GraphRagConfig.from_env().enabled
