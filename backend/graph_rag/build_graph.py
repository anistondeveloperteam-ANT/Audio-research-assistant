"""Build the optional Memgraph knowledge graph from Oracle paper chunks.

Run:
    python -m backend.graph_rag.build_graph

The graph is derived from existing structured data only: papers, chunks,
sections, chunk types, and detected audio concepts. Oracle remains the source of
truth for full text and citations.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import oracledb
from dotenv import load_dotenv

from backend.graph_rag.config import GraphRagConfig
from backend.graph_rag.memgraph_client import GraphRagUnavailable, MemgraphClient
from backend.graph_rag.schema import (
    COUNT_CYPHER,
    INDEX_STATEMENTS,
    UPSERT_CHUNKS_CYPHER,
    UPSERT_RELATED_CONCEPTS_CYPHER,
    concept_pairs,
    split_concepts,
    text_preview,
)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def read_lob(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "read"):
            return value.read()
    except Exception:
        return str(value)
    return value


def connect_oracle():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def load_oracle_rows() -> list[dict[str, Any]]:
    conn = connect_oracle()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            p.id,
            p.title,
            p.year,
            p.file_name,
            p.file_hash,
            c.id,
            c.section_name,
            c.chunk_type,
            c.page_start,
            c.page_end,
            c.audio_concepts,
            c.chunk_text
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        ORDER BY p.id, c.chunk_index, c.id
        """
    )
    rows = []
    for row in cur.fetchall():
        (
            paper_id,
            title,
            year,
            file_name,
            file_hash,
            chunk_id,
            section,
            chunk_type,
            page_start,
            page_end,
            concepts,
            text,
        ) = row
        title = read_lob(title) or "Untitled"
        section = read_lob(section) or "Unknown"
        chunk_type = read_lob(chunk_type) or "text"
        concepts = split_concepts(read_lob(concepts))
        text = read_lob(text) or ""
        rows.append(
            {
                "paper_id": int(paper_id),
                "title": str(title),
                "year": int(year) if year is not None else None,
                "file_name": str(file_name or ""),
                "file_hash": str(file_hash or ""),
                "chunk_id": int(chunk_id),
                "section": str(section),
                "chunk_type": str(chunk_type),
                "page_start": int(page_start) if page_start is not None else None,
                "page_end": int(page_end) if page_end is not None else None,
                "concepts": concepts,
                "text_preview": text_preview(text),
            }
        )
    cur.close()
    conn.close()
    return rows


def chunks(items: list[dict[str, Any]], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def ensure_indexes(client: MemgraphClient) -> None:
    for stmt in INDEX_STATEMENTS:
        try:
            client.execute(stmt)
        except Exception:
            # Memgraph raises if an index already exists; index creation is an
            # optimization, so graph build should continue.
            pass


def build_graph() -> dict[str, int]:
    cfg = GraphRagConfig.from_env()
    rows = load_oracle_rows()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for pair in concept_pairs(row["concepts"]):
            pair_counts[(pair["source"], pair["target"])] += 1
    pairs = [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in pair_counts.items()
    ]

    with MemgraphClient(cfg) as client:
        ensure_indexes(client)
        for batch in chunks(rows, cfg.build_batch_size):
            client.execute(UPSERT_CHUNKS_CYPHER, rows=batch)
        for batch in chunks(pairs, cfg.build_batch_size):
            client.execute(UPSERT_RELATED_CONCEPTS_CYPHER, pairs=batch)
        counts = {}
        for name, cypher in COUNT_CYPHER.items():
            records = client.execute(cypher)
            counts[name] = int(records[0]["count"]) if records else 0
        return counts


def main() -> int:
    try:
        counts = build_graph()
    except GraphRagUnavailable as exc:
        print(f"GraphRAG unavailable: {exc}")
        return 1
    except Exception as exc:
        print(f"Graph build failed: {exc}")
        return 1

    print("Memgraph GraphRAG build complete")
    print("-" * 40)
    for key in ("papers", "chunks", "concepts", "relationships"):
        print(f"{key:14s}: {counts.get(key, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
