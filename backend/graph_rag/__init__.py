"""Optional Memgraph-backed GraphRAG helpers.

The graph layer is deliberately optional: imports should stay cheap, and the
main app must work when Memgraph or the neo4j driver are not installed.
"""
from __future__ import annotations

from backend.graph_rag.config import GraphRagConfig, graph_rag_enabled
from backend.graph_rag.retrieve_graph import graph_retrieve

__all__ = ["GraphRagConfig", "graph_rag_enabled", "graph_retrieve"]
