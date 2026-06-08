"""Offline tests for the optional Memgraph GraphRAG layer."""
from __future__ import annotations

from backend.graph_rag.config import GraphRagConfig
from backend.graph_rag.fusion import merge_graph_candidates
from backend.graph_rag.schema import concept_pairs, split_concepts
import backend.graph_rag.retrieve_graph as rg


def test_graph_retrieve_disabled_returns_empty(monkeypatch):
    monkeypatch.delenv("ENABLE_GRAPH_RAG", raising=False)
    assert rg.graph_retrieve("query", seed_chunk_ids=[1, 2]) == []


def test_graph_retrieve_no_seeds_returns_empty():
    cfg = GraphRagConfig(enabled=True)
    assert rg.graph_retrieve("query", seed_chunk_ids=[], config=cfg) == []


def test_graph_retrieve_maps_memgraph_records(monkeypatch):
    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, cypher, **params):
            assert params["seed_ids"] == [10]
            assert params["limit"] == 2
            self.calls.append(cypher)
            if "MENTIONS" in cypher and "shared concept" in cypher:
                return [
                    {
                        "chunk_id": 42,
                        "graph_score": 4.5,
                        "reasons": ["shared concept", "same section"],
                        "labels": ["MVDR", "PESQ", "Method"],
                    }
                ]
            assert "RELATED_TO*1..2" in cypher or "IN_SECTION" in cypher or "HAS_CHUNK" in cypher
            return []

    monkeypatch.setattr(rg, "MemgraphClient", FakeClient)
    cfg = GraphRagConfig(enabled=True, max_hops=2)
    out = rg.graph_retrieve("mvdr", seed_chunk_ids=[10], top_k=2, config=cfg)
    assert out == [
        {
            "chunk_id": 42,
            "graph_score": 4.5,
            "graph_reasons": ["shared concept", "same section"],
            "graph_labels": ["MVDR", "PESQ", "Method"],
            "source": "memgraph_graph",
        }
    ]


def test_split_concepts_deduplicates_and_trims():
    assert split_concepts(" MVDR, PESQ; mvdr, STOI ") == ["MVDR", "PESQ", "STOI"]


def test_concept_pairs_are_stable():
    pairs = concept_pairs(["PESQ", "MVDR", "STOI"])
    assert {"source": "MVDR", "target": "PESQ", "weight": 1} in pairs
    assert {"source": "PESQ", "target": "STOI", "weight": 1} in pairs


def test_merge_graph_candidates_adds_metadata_and_new_chunks():
    candidates = [{"id": 1, "title": "A", "source": "bm25", "text": "seed"}]
    chunks_by_id = {
        2: {
            "id": 2,
            "title": "B",
            "section": "Method",
            "text": "related chunk",
            "concepts": "MVDR",
        }
    }
    graph_hits = [
        {"chunk_id": 1, "graph_score": 3.0, "graph_reasons": ["same paper"]},
        {"chunk_id": 2, "graph_score": 5.0, "graph_reasons": ["shared concept"]},
    ]

    out = merge_graph_candidates(candidates, graph_hits, chunks_by_id)
    by_id = {item["id"]: item for item in out}

    assert len(out) == 2
    assert "memgraph_graph" in by_id[1]["retrieval_sources"]
    assert by_id[1]["graph_reason"] == "same paper"
    assert by_id[2]["source"] == "memgraph_graph"
    assert by_id[2]["graph_reason"] == "shared concept"
