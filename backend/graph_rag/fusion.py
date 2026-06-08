"""Fusion helpers for adding graph-expanded chunks to hybrid retrieval."""
from __future__ import annotations

from typing import Any


def _source_list(item: dict[str, Any]) -> list[str]:
    sources = item.get("retrieval_sources")
    if isinstance(sources, list):
        return [str(s) for s in sources]
    source = item.get("source")
    return [str(source)] if source else []


def _merge_graph_metadata(item: dict[str, Any], hit: dict[str, Any]) -> None:
    sources = _source_list(item)
    if "memgraph_graph" not in sources:
        sources.append("memgraph_graph")
    item["retrieval_sources"] = sources
    item["graph_score"] = float(hit.get("graph_score") or item.get("graph_score") or 0.0)
    reasons = hit.get("graph_reasons") or []
    labels = hit.get("graph_labels") or []
    if reasons:
        item["graph_reasons"] = list(dict.fromkeys(str(r) for r in reasons if str(r).strip()))
        item["graph_reason"] = ", ".join(item["graph_reasons"])
    if labels:
        item["graph_labels"] = list(dict.fromkeys(str(l) for l in labels if str(l).strip()))


def merge_graph_candidates(
    candidates: list[dict[str, Any]],
    graph_hits: list[dict[str, Any]],
    chunks_by_id: dict[int, dict[str, Any]],
    max_new: int | None = None,
) -> list[dict[str, Any]]:
    """Merge graph hits into a candidate list by Oracle chunk ID.

    Existing candidates get graph metadata added. New graph-only candidates are
    converted back to regular chunk dicts so the cross-encoder can rerank them
    and the final answer can cite the original Oracle paper chunk.
    """
    merged = [dict(c) for c in candidates]
    by_id: dict[int, dict[str, Any]] = {}
    for item in merged:
        try:
            by_id[int(item.get("id"))] = item
        except Exception:
            continue

    added = 0
    for hit in graph_hits:
        try:
            chunk_id = int(hit.get("chunk_id"))
        except Exception:
            continue
        if chunk_id in by_id:
            _merge_graph_metadata(by_id[chunk_id], hit)
            continue
        if max_new is not None and added >= max_new:
            break
        source_chunk = chunks_by_id.get(chunk_id)
        if not source_chunk:
            continue
        item = dict(source_chunk)
        item["source"] = "memgraph_graph"
        item["hybrid_score"] = float(hit.get("graph_score") or 0.0)
        item["rrf_score"] = item["hybrid_score"]
        _merge_graph_metadata(item, hit)
        merged.append(item)
        by_id[chunk_id] = item
        added += 1
    return merged
