"""Small Memgraph client wrapper using the Neo4j Bolt driver.

Memgraph speaks the Bolt protocol, so the official ``neo4j`` Python driver is a
practical client. The import is lazy so the rest of the app works without the
optional dependency until GraphRAG is enabled or the graph build CLI is run.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from backend.graph_rag.config import GraphRagConfig

logger = logging.getLogger("graph_rag")


class GraphRagUnavailable(RuntimeError):
    """Raised when Memgraph or its Python driver is not available."""


class MemgraphClient:
    def __init__(self, config: GraphRagConfig | None = None):
        self.config = config or GraphRagConfig.from_env()
        self._driver = None

    def __enter__(self) -> "MemgraphClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            raise GraphRagUnavailable(
                "The optional 'neo4j' package is required for Memgraph GraphRAG. "
                "Install requirements.txt, then retry."
            ) from exc

        auth = None
        if self.config.user or self.config.password:
            auth = (self.config.user, self.config.password)
        try:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=auth,
                connection_timeout=self.config.timeout_seconds,
            )
            self._driver.verify_connectivity()
        except Exception as exc:
            self._driver = None
            raise GraphRagUnavailable(f"Memgraph is unavailable at {self.config.uri}: {exc}") from exc

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def execute(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.connect()
        assert self._driver is not None
        session_kwargs: dict[str, Any] = {}
        if self.config.database:
            session_kwargs["database"] = self.config.database
        with self._driver.session(**session_kwargs) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def execute_many(self, statements: Iterable[tuple[str, dict[str, Any]]]) -> None:
        self.connect()
        assert self._driver is not None
        session_kwargs: dict[str, Any] = {}
        if self.config.database:
            session_kwargs["database"] = self.config.database
        with self._driver.session(**session_kwargs) as session:
            for cypher, params in statements:
                session.run(cypher, **params).consume()
