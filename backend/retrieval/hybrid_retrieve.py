"""
hybrid_retrieve.py  --  Batch 3 (Smart Query Layer)

Pipeline (builds on Batches 1+2):

  1. Vector search (original question)         -> ranking A
  2. Vector search (HyDE-expanded passage)     -> ranking B   <-- NEW (H3)
  3. Field-weighted BM25 (original question)   -> ranking C
  4. RRF 3-way fusion                          -> fused pool
  5. Rerank against ORIGINAL question
  6. Chunk-type aware boost
  7. MMR diversification (with per-paper cap)

Toggle:
  ENABLE_HYDE=true (default)   set to "false" to fall back to Batch 2
                               (single vector search + BM25 + RRF)

Mode env vars honoured (from Batch 1):
  VECTOR_TOP_K, BM25_TOP_K, RERANK_TOP_N, MAX_SOURCES_PER_PAPER

Batch 2 tunables still active:
  RRF_K        (default 60)
  MMR_LAMBDA   (default 0.7)

Backward compatible:
  hybrid_retrieve(query, top_k=N) signature unchanged.
"""

import os
import re
import warnings
import logging
from collections import defaultdict

from dotenv import load_dotenv
import oracledb
from sentence_transformers import CrossEncoder

from backend.retrieval.vector_retriever import vector_search
from backend.retrieval.retrieval_fusion import (
    field_weighted_bm25,
    reciprocal_rank_fusion,
    mmr_diversify,
)
from backend.retrieval.hyde_generator import hyde_expand
from backend.graph_rag.config import graph_rag_enabled
from backend.graph_rag.fusion import merge_graph_candidates
from backend.graph_rag.retrieve_graph import graph_retrieve

# ---------------------------------------------------------------------
# Quiet backend output
# ---------------------------------------------------------------------
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
RRF_K = int(os.getenv("RRF_K", "60"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"

_reranker = None
_chunks_cache = None
_bm25_cache = None


def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)


def read_lob(value):
    if value is None:
        return ""
    try:
        if hasattr(value, "read"):
            return value.read()
    except Exception:
        return str(value)
    return value


def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def get_reranker():
    global _reranker
    if _reranker is None:
        from backend.common.device import resolve_device
        device = resolve_device("RERANKER_DEVICE")
        debug_print("Loading reranker:", RERANKER_MODEL, "on", device)
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
    return _reranker


def tokenize(text: str):
    text = (text or "").lower()
    return re.findall(r"[a-z0-9][a-z0-9\-\_]+", text)


def load_chunks():
    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache

    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            c.id,
            p.title,
            c.section_name,
            c.chunk_text,
            c.chunk_type,
            c.page_start,
            c.page_end,
            c.audio_concepts
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        ORDER BY c.id
    """)

    chunks = []
    for row in cur.fetchall():
        (chunk_id, title, section, text, chunk_type,
         page_start, page_end, concepts) = row

        title = read_lob(title)
        section = read_lob(section)
        text = read_lob(text)
        chunk_type = read_lob(chunk_type)
        concepts = read_lob(concepts)

        chunks.append({
            "id": int(chunk_id),
            "title": str(title or ""),
            "section": str(section or ""),
            "text": str(text or ""),
            "chunk_type": str(chunk_type or ""),
            "page_start": int(page_start) if page_start is not None else None,
            "page_end": int(page_end) if page_end is not None else None,
            "concepts": str(concepts or ""),
            "source": "bm25",
        })

    cur.close()
    conn.close()
    _chunks_cache = chunks
    return chunks


def build_bm25_index():
    """Build corpus-wide BM25 statistics (df, N, avgdl)."""
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    chunks = load_chunks()
    doc_lengths = []
    df = defaultdict(int)

    for chunk in chunks:
        full_text = " ".join([
            chunk.get("title") or "",
            chunk.get("section") or "",
            chunk.get("concepts") or "",
            chunk.get("text") or "",
        ])
        toks = tokenize(full_text)
        doc_lengths.append(len(toks))
        for tok in set(toks):
            df[tok] += 1

    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    _bm25_cache = {
        "chunks": chunks,
        "df": df,
        "avgdl": avgdl,
        "N": len(chunks),
    }
    return _bm25_cache


def keyword_search(query: str, top_k: int = 10):
    """Field-weighted BM25 (BM25F-style) search."""
    index = build_bm25_index()
    chunks = index["chunks"]
    df = index["df"]
    N = index["N"]
    avgdl = index["avgdl"]

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for chunk in chunks:
        score = field_weighted_bm25(query_tokens, chunk, df, N, avgdl)
        if score <= 0:
            continue
        item = dict(chunk)
        item["keyword_score"] = float(score)
        item["source"] = "bm25"
        scored.append(item)

    scored.sort(key=lambda x: x["keyword_score"], reverse=True)
    return scored[:top_k]


def rerank(query: str, candidates, top_k: int = 10):
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = []
    for item in candidates:
        text = " ".join([
            item.get("title") or "",
            item.get("section") or "",
            item.get("concepts") or "",
            item.get("text") or "",
        ])
        pairs.append((query, text[:3000]))

    scores = reranker.predict(pairs)
    reranked = []
    for item, score in zip(candidates, scores):
        new_item = dict(item)
        new_item["rerank_score"] = float(score)
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def apply_chunk_type_boost(query: str, results):
    """Boost equation / algorithm / metrics chunks for matching questions."""
    q = (query or "").lower()

    wants_equation = any(k in q for k in [
        "equation", "formula", "derive", "math", "proof", "expression",
    ])
    wants_algorithm = any(k in q for k in [
        "algorithm", "pseudocode", "step", "procedure", "implement",
        "how to", "method", "pipeline",
    ])
    wants_metrics = any(k in q for k in [
        "metric", "pesq", "stoi", "sdr", "snr", "score", "benchmark",
        "performance", "evaluation", "result", "compare",
    ])

    for r in results:
        chunk_type = (r.get("chunk_type") or "").lower()
        base = float(r.get("rerank_score", 0.0))
        boost = 0.0

        if wants_equation and "equation" in chunk_type:
            boost += 0.08
        if wants_algorithm and "algorithm" in chunk_type:
            boost += 0.10
        if wants_metrics and ("table" in chunk_type or "metrics" in chunk_type):
            boost += 0.08

        r["chunk_type_boost"] = boost
        r["rerank_score"] = base + boost

    results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return results


def _result_identity(item):
    if not isinstance(item, dict):
        return id(item)
    return (
        item.get("id")
        or item.get("chunk_id")
        or item.get("source_id")
        or (
            str(item.get("title") or item.get("paper") or "unknown")
            + "::"
            + str(item.get("page_start") or "")
            + "::"
            + str(item.get("section") or item.get("section_name") or "")
        )
    )


def _paper_key(item):
    if not isinstance(item, dict):
        return "unknown"
    return str(
        item.get("title")
        or item.get("paper")
        or item.get("paper_title")
        or "unknown"
    ).strip().lower()


def diversify_results(results, top_k=10, max_per_paper=3):
    """Backward-compat per-paper-count diversifier. Main pipeline uses MMR."""
    if not isinstance(results, list):
        return results

    selected = []
    paper_counts = {}
    selected_ids = set()

    for item in results:
        key = _paper_key(item)
        identity = _result_identity(item)
        if identity in selected_ids:
            continue
        if paper_counts.get(key, 0) >= max_per_paper:
            continue
        selected.append(item)
        selected_ids.add(identity)
        paper_counts[key] = paper_counts.get(key, 0) + 1
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in results:
            identity = _result_identity(item)
            if identity in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(identity)
            if len(selected) >= top_k:
                break

    return selected


def _mode_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, str(default)))
    except Exception:
        return default


def _hybrid_retrieve_core(query: str, top_k: int = 10):
    vec_k = _mode_int("VECTOR_TOP_K", max(top_k * 3, 20))
    bm_k = _mode_int("BM25_TOP_K", max(top_k * 3, 20))
    rerank_n = _mode_int("RERANK_TOP_N", max(top_k * 2, 20))
    max_per_paper = _mode_int("MAX_SOURCES_PER_PAPER", 3)

    debug_print(
        f"Mode binding: VECTOR_TOP_K={vec_k} BM25_TOP_K={bm_k} "
        f"RERANK_TOP_N={rerank_n} MAX_PER_PAPER={max_per_paper} "
        f"ENABLE_HYDE={ENABLE_HYDE}"
    )

    # Ranking A -- vector search with original question
    debug_print("Vector search (original question)...")
    vector_orig = vector_search(query, top_k=vec_k)
    for r in vector_orig:
        r["source"] = f"{r.get('source') or 'vector'}_orig"

    rankings = [vector_orig]

    # Ranking B -- vector search with HyDE-expanded passage (NEW Batch 3)
    if ENABLE_HYDE:
        try:
            hyde_text = hyde_expand(query)
            if hyde_text and hyde_text.strip() != query.strip():
                debug_print("Vector search (HyDE expansion)...")
                vector_hyde = vector_search(hyde_text, top_k=vec_k)
                for r in vector_hyde:
                    r["source"] = f"{r.get('source') or 'vector'}_hyde"
                rankings.append(vector_hyde)
        except Exception as exc:
            debug_print(f"HyDE expansion failed (non-fatal): {exc}")

    # Ranking C -- field-weighted BM25 with original question
    debug_print("Field-weighted BM25 search...")
    keyword_results = keyword_search(query, top_k=bm_k)
    for r in keyword_results:
        r["source"] = "bm25"
    rankings.append(keyword_results)

    # RRF fusion across all rankings
    debug_print(f"RRF fusion of {len(rankings)} rankings (k={RRF_K})...")
    fused = reciprocal_rank_fusion(rankings, k=RRF_K, id_key="id")

    candidates = fused[:max(rerank_n, 30)]
    debug_print("Candidates before rerank:", len(candidates))

    if graph_rag_enabled():
        try:
            seed_ids = [int(c["id"]) for c in candidates if c.get("id") is not None]
            graph_hits = graph_retrieve(query, seed_chunk_ids=seed_ids)
            if graph_hits:
                chunks_by_id = {int(c["id"]): c for c in load_chunks()}
                candidates = merge_graph_candidates(
                    candidates,
                    graph_hits,
                    chunks_by_id,
                    max_new=len(graph_hits),
                )
                debug_print("Candidates after Memgraph expansion:", len(candidates))
        except Exception as exc:
            debug_print(f"Memgraph GraphRAG expansion failed (non-fatal): {exc}")

    # Reranker uses the ORIGINAL question (not HyDE expansion) because
    # cross-encoders are trained on natural-language query / doc pairs.
    debug_print("Reranking against original question...")
    reranked = rerank(query, candidates, top_k=rerank_n)

    reranked = apply_chunk_type_boost(query, reranked)

    debug_print(f"MMR (lambda={MMR_LAMBDA}, max_per_paper={max_per_paper})...")
    results = mmr_diversify(
        reranked,
        top_k=top_k,
        max_per_paper=max_per_paper,
        lambda_param=MMR_LAMBDA,
        relevance_key="rerank_score",
    )
    return results


def hybrid_retrieve(query, top_k=10, *args, **kwargs):
    """Public retrieval function. Backward-compatible signature."""
    try:
        raw = _hybrid_retrieve_core(query, top_k=top_k, *args, **kwargs)
    except TypeError:
        raw = _hybrid_retrieve_core(query, *args, **kwargs)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["results", "sources", "source_cards"]:
            if isinstance(raw.get(key), list):
                raw[key] = raw[key][:top_k]
        return raw

    return raw


if __name__ == "__main__":
    q = input("Ask retrieval query: ").strip()
    results = hybrid_retrieve(q, top_k=8)

    for i, r in enumerate(results, 1):
        print("\n" + "=" * 100)
        print("Rank:", i)
        print("Paper:", r.get("title"))
        print("Section:", r.get("section"))
        print("Pages:", r.get("page_start"), "-", r.get("page_end"))
        print("Type:", r.get("chunk_type"))
        print("Concepts:", r.get("concepts"))
        print("Sources:", ", ".join(r.get("retrieval_sources", [])))
        print("RRF:", round(r.get("rrf_score", 0), 4))
        print("Rerank:", round(r.get("rerank_score", 0), 4))
        if r.get("chunk_type_boost"):
            print("Chunk type boost:", round(r["chunk_type_boost"], 4))
        if r.get("mmr_score") is not None:
            print("MMR:", round(r.get("mmr_score", 0), 4))
        print((r.get("text") or "")[:700])
