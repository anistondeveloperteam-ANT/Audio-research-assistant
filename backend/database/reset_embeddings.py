"""
Reset the corpus embeddings for a CLEAN re-ingestion — used when the embedding model/dimension changes
(e.g. switching to BAAI/bge-large-en-v1.5, 768 -> 1024).

It (1) drops + re-adds the native `chunks.embedding_vec` column at the CURRENT EMBEDDING_DIM, fully
removing every old-dimension vector (so 768-d and 1024-d can never be mixed), and (2) NULLs every chunk's
CLOB `embedding`, so the following `embed_chunks` stage re-embeds the WHOLE corpus with the new model.
Destructive on purpose; run via `python pipeline.py --reembed`.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def reset(cur, conn, dim: int) -> int:
    """Drop + recreate `chunks.embedding_vec` at `dim`, and NULL all `chunks.embedding`. Pure DB logic
    (takes an open cursor/connection) so it is unit-testable with a mock cursor. Returns the chunk count.
    """
    cur.execute(
        "SELECT COUNT(*) FROM user_tab_columns "
        "WHERE table_name = 'CHUNKS' AND column_name = 'EMBEDDING_VEC'"
    )
    if cur.fetchone()[0] > 0:
        print("Dropping old embedding_vec column (purging all old-dimension vectors)...")
        cur.execute("ALTER TABLE chunks DROP COLUMN embedding_vec")
        conn.commit()
    print(f"Adding embedding_vec VECTOR({dim}, FLOAT32)...")
    cur.execute(f"ALTER TABLE chunks ADD embedding_vec VECTOR({dim}, FLOAT32)")
    conn.commit()
    print("Clearing CLOB embeddings so the whole corpus re-embeds with the new model...")
    cur.execute("UPDATE chunks SET embedding = NULL")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM chunks")
    n = int(cur.fetchone()[0])
    print(f"Reset complete: {n} chunk(s) cleared; embedding_vec recreated at {dim}-dim. Re-embed next.")
    return n


def main() -> None:
    import oracledb
    from backend.common.embeddings import embedding_dim
    conn = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )
    cur = conn.cursor()
    try:
        reset(cur, conn, embedding_dim())              # the model's true dimension, not a stale env value
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
