"""Graph expansion retrieval against Memgraph.

Graph retrieval is not a replacement for vector/BM25 retrieval. It starts from
Oracle chunk IDs found by the existing hybrid retriever and finds related Oracle
chunks through graph relationships.
"""
from __future__ import annotations

from typing import Any, Iterable

from backend.graph_rag.config import GraphRagConfig
from backend.graph_rag.memgraph_client import GraphRagUnavailable, MemgraphClient, logger


SHARED_CONCEPT_CYPHER = """
MATCH (seed:Chunk)-[:MENTIONS]->(concept:Concept)<-[:MENTIONS]-(cand:Chunk)
WHERE seed.oracle_chunk_id IN $seed_ids
  AND NOT cand.oracle_chunk_id IN $seed_ids
RETURN cand.oracle_chunk_id AS chunk_id,
       3.0 * count(DISTINCT concept) AS graph_score,
       ["shared concept"] AS reasons,
       collect(DISTINCT concept.name)[0..4] AS labels
LIMIT $limit
"""

SAME_SECTION_CYPHER = """
MATCH (seed:Chunk)-[:IN_SECTION]->(section:Section)<-[:IN_SECTION]-(cand:Chunk)
WHERE seed.oracle_chunk_id IN $seed_ids
  AND NOT cand.oracle_chunk_id IN $seed_ids
RETURN cand.oracle_chunk_id AS chunk_id,
       1.0 * count(DISTINCT section) AS graph_score,
       ["same section"] AS reasons,
       collect(DISTINCT section.name)[0..2] AS labels
LIMIT $limit
"""

SAME_PAPER_CYPHER = """
MATCH (paper:Paper)-[:HAS_CHUNK]->(seed:Chunk)
MATCH (paper)-[:HAS_CHUNK]->(cand:Chunk)
WHERE seed.oracle_chunk_id IN $seed_ids
  AND NOT cand.oracle_chunk_id IN $seed_ids
RETURN cand.oracle_chunk_id AS chunk_id,
       0.75 * count(DISTINCT paper) AS graph_score,
       ["same paper"] AS reasons,
       collect(DISTINCT paper.title)[0..1] AS labels
LIMIT $limit
"""

RELATED_CONCEPT_CYPHER = """
MATCH (seed:Chunk)-[:MENTIONS]->(:Concept)-[:RELATED_TO*1..2]-(related:Concept)<-[:MENTIONS]-(cand:Chunk)
WHERE seed.oracle_chunk_id IN $seed_ids
  AND NOT cand.oracle_chunk_id IN $seed_ids
RETURN cand.oracle_chunk_id AS chunk_id,
       1.5 * count(DISTINCT related) AS graph_score,
       ["related concept"] AS reasons,
       collect(DISTINCT related.name)[0..4] AS labels
LIMIT $limit
"""


def _clean_seed_ids(seed_chunk_ids: Iterable[Any] | None) -> list[int]:
    out = []
    seen = set()
    for value in seed_chunk_ids or []:
        try:
            chunk_id = int(value)
        except Exception:
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(chunk_id)
    return out


def _flatten_label_groups(groups: Any) -> list[str]:
    labels = []
    seen = set()
    for group in groups or []:
        if isinstance(group, str):
            group_items = [group]
        else:
            group_items = group or []
        for item in group_items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(text)
    return labels


def _merge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            chunk_id = int(record.get("chunk_id"))
        except Exception:
            continue
        item = merged.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "graph_score": 0.0,
                "graph_reasons": [],
                "graph_labels": [],
                "source": "memgraph_graph",
            },
        )
        item["graph_score"] += float(record.get("graph_score") or 0.0)
        for reason in record.get("reasons") or []:
            text = str(reason or "").strip()
            if text and text not in item["graph_reasons"]:
                item["graph_reasons"].append(text)
        for label in _flatten_label_groups([record.get("labels") or []]):
            if label not in item["graph_labels"]:
                item["graph_labels"].append(label)
    return sorted(merged.values(), key=lambda x: x["graph_score"], reverse=True)


def graph_retrieve(
    query: str,
    seed_chunk_ids: list[int] | None = None,
    top_k: int | None = None,
    config: GraphRagConfig | None = None,
) -> list[dict[str, Any]]:
    """Return graph-expanded Oracle chunk IDs with explainable scores.

    ``query`` is accepted for API symmetry and future query-aware graph search;
    v1 uses only seed chunks to avoid hallucinated graph-only evidence.
    """
    cfg = config or GraphRagConfig.from_env()
    if not cfg.enabled:
        return []

    seed_ids = _clean_seed_ids(seed_chunk_ids)
    if not seed_ids:
        return []

    limit = int(top_k or cfg.max_results)
    try:
        related_hops = max(1, min(int(cfg.max_hops), 4))
        related_query = RELATED_CONCEPT_CYPHER.replace(
            "[:RELATED_TO*1..2]",
            f"[:RELATED_TO*1..{related_hops}]",
        )
        with MemgraphClient(cfg) as client:
            records = []
            for cypher in (
                SHARED_CONCEPT_CYPHER,
                SAME_SECTION_CYPHER,
                SAME_PAPER_CYPHER,
                related_query,
            ):
                records.extend(client.execute(cypher, seed_ids=seed_ids, limit=limit))
    except GraphRagUnavailable as exc:
        logger.info("GraphRAG unavailable: %s", exc)
        return []
    except Exception as exc:
        logger.info("GraphRAG retrieval failed: %s", type(exc).__name__)
        return []

    return _merge_records(records)[:limit]
