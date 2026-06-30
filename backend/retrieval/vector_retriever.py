import os
import json
import threading
import warnings
import logging

from dotenv import load_dotenv
import oracledb

# ---------------------------------------------------------------------
# Quiet Hugging Face / Transformers warnings for user-friendly output
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
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def read_lob(value):
    """
    Oracle CLOB/BLOB values must be read before the connection closes.
    This prevents DPY-1001 not connected errors.
    """
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


_embed_lock = threading.Lock()


def embed_query(query: str):
    """Embed the query via the configured provider (Google Gemini or local). Serialized so
    concurrent retrieval stages (parallel vector + HyDE) are safe with a non-thread-safe local
    embedding model. Uncontended (negligible) on single-threaded paths like ingestion."""
    from backend.common.embeddings import embed_query as _embed
    with _embed_lock:
        return _embed(query)


def _oracle_vector_search(query_vector, top_k: int = 10):
    """
    Oracle native VECTOR semantic search.

    Uses:
    - chunks.embedding_vec native VECTOR column
    - VECTOR_DISTANCE(..., COSINE)
    """

    qvec_json = json.dumps(query_vector)

    conn = connect()
    cur = conn.cursor()

    sql = f"""
        SELECT
            c.id,
            p.title,
            c.section_name,
            c.chunk_text,
            c.chunk_type,
            c.page_start,
            c.page_end,
            c.concept_tags,
            VECTOR_DISTANCE(c.embedding_vec, TO_VECTOR(:qvec), COSINE) AS distance
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE c.embedding_vec IS NOT NULL
        ORDER BY distance
        FETCH FIRST {int(top_k)} ROWS ONLY
    """

    cur.execute(sql, {"qvec": qvec_json})

    results = []

    for row in cur.fetchall():
        (
            chunk_id,
            title,
            section,
            text,
            chunk_type,
            page_start,
            page_end,
            concepts,
            distance,
        ) = row

        # CRITICAL FIX:
        # Convert all Oracle LOB values before closing cursor/connection.
        title = read_lob(title)
        section = read_lob(section)
        text = read_lob(text)
        chunk_type = read_lob(chunk_type)
        concepts = read_lob(concepts)

        distance = float(distance)
        score = 1.0 - distance

        results.append({
            "id": int(chunk_id),
            "title": str(title or ""),
            "section": str(section or ""),
            "text": str(text or ""),
            "chunk_type": str(chunk_type or ""),
            "page_start": int(page_start) if page_start is not None else None,
            "page_end": int(page_end) if page_end is not None else None,
            "concepts": str(concepts or ""),
            "vector_score": score,
            "distance": distance,
            "source": "oracle_vector",
        })

    cur.close()
    conn.close()

    return results


def vector_search(query: str, top_k: int = 10):
    """Semantic vector search with an optional turbovec accelerator.

    The public result schema stays unchanged. When `VECTOR_BACKEND=turbovec`
    or `TURBOVEC_ENABLED=true`, turbovec is tried first. Oracle remains the
    fallback and source of truth unless `TURBOVEC_STRICT=true`.
    """
    qvec = embed_query(query)

    try:
        from backend.retrieval.turbovec_index import (
            search_by_vector as turbovec_search_by_vector,
            strict_enabled as turbovec_strict_enabled,
            turbovec_enabled,
        )
        if turbovec_enabled():
            try:
                results = turbovec_search_by_vector(qvec, top_k=top_k)
                if results:
                    return results
                if turbovec_strict_enabled():
                    return results
            except Exception as exc:
                if turbovec_strict_enabled():
                    raise
                if DEBUG_MODE:
                    print(f"turbovec unavailable; falling back to Oracle VECTOR: {exc}")
    except Exception as exc:
        if DEBUG_MODE:
            print(f"turbovec backend not loaded; using Oracle VECTOR: {exc}")

    return _oracle_vector_search(qvec, top_k=top_k)


if __name__ == "__main__":
    q = input("Ask vector query: ").strip()

    results = vector_search(q, top_k=5)

    print("\nTop Oracle VECTOR results:")

    for i, r in enumerate(results, 1):
        print("\n" + "=" * 80)
        print(i, r["title"])
        print("Section:", r["section"])
        print("Pages:", r["page_start"], "-", r["page_end"])
        print("Score:", round(r["vector_score"], 4))
        print("Concepts:", r["concepts"])
        print((r["text"] or "")[:700])
