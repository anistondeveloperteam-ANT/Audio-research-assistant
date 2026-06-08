"""Cypher schema helpers for the Memgraph GraphRAG layer."""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Iterable


_SPACE = re.compile(r"\s+")


INDEX_STATEMENTS = [
    "CREATE INDEX ON :Paper(id)",
    "CREATE INDEX ON :Paper(file_hash)",
    "CREATE INDEX ON :Chunk(oracle_chunk_id)",
    "CREATE INDEX ON :Concept(name)",
    "CREATE INDEX ON :Section(name)",
    "CREATE INDEX ON :ChunkType(name)",
]


UPSERT_CHUNKS_CYPHER = """
UNWIND $rows AS row
MERGE (p:Paper {id: row.paper_id})
SET p.title = row.title,
    p.year = row.year,
    p.file_name = row.file_name,
    p.file_hash = row.file_hash
MERGE (c:Chunk {oracle_chunk_id: row.chunk_id})
SET c.section = row.section,
    c.chunk_type = row.chunk_type,
    c.page_start = row.page_start,
    c.page_end = row.page_end,
    c.text_preview = row.text_preview
MERGE (p)-[:HAS_CHUNK]->(c)
MERGE (s:Section {name: row.section})
MERGE (c)-[:IN_SECTION]->(s)
MERGE (t:ChunkType {name: row.chunk_type})
MERGE (c)-[:HAS_TYPE]->(t)
FOREACH (concept_name IN row.concepts |
    MERGE (concept:Concept {name: concept_name})
    MERGE (c)-[:MENTIONS]->(concept)
    MERGE (p)-[:MENTIONS]->(concept)
)
"""


UPSERT_RELATED_CONCEPTS_CYPHER = """
UNWIND $pairs AS pair
MERGE (a:Concept {name: pair.source})
MERGE (b:Concept {name: pair.target})
MERGE (a)-[r:RELATED_TO]->(b)
SET r.weight = coalesce(r.weight, 0) + pair.weight,
    r.reason = "chunk co-occurrence"
"""


COUNT_CYPHER = {
    "papers": "MATCH (n:Paper) RETURN count(n) AS count",
    "chunks": "MATCH (n:Chunk) RETURN count(n) AS count",
    "concepts": "MATCH (n:Concept) RETURN count(n) AS count",
    "relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
}


def normalize_name(value: Any) -> str:
    text = str(value or "").strip()
    text = _SPACE.sub(" ", text)
    return text


def split_concepts(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items: Iterable[Any] = raw
    else:
        items = str(raw).replace(";", ",").split(",")
    out = []
    seen = set()
    for item in items:
        name = normalize_name(item)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def text_preview(value: Any, limit: int = 500) -> str:
    text = normalize_name(value)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


def concept_pairs(concepts: list[str]) -> list[dict[str, Any]]:
    """Stable concept co-occurrence pairs for safe RELATED_TO edges."""
    names = sorted({normalize_name(c) for c in concepts if normalize_name(c)}, key=str.lower)
    pairs = []
    for a, b in combinations(names, 2):
        if a.lower() == b.lower():
            continue
        source, target = sorted([a, b], key=str.lower)
        pairs.append({"source": source, "target": target, "weight": 1})
    return pairs
