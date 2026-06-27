"""
Copy each chunk's JSON embedding (CLOB `embedding`) into Oracle's native `embedding_vec` VECTOR
column, which powers `VECTOR_DISTANCE(... COSINE)` search.

The VECTOR column is sized to the CURRENT embedding dimension, detected from the stored vectors —
so it is correct regardless of a stale EMBEDDING_DIM in .env. If the column already exists at a
DIFFERENT dimension (e.g. a leftover VECTOR(768) after switching to a 1024-d model), it is dropped
and recreated at the right dimension and the vectors are re-migrated from the CLOBs — otherwise every
insert fails with ORA-51803 ("Vector dimension count must match the column definition").
"""
import os
import json
import array

from dotenv import load_dotenv
import oracledb

load_dotenv()

# Fallback when nothing is embedded yet (the live value is detected from stored data at run time).
# bge-large-en-v1.5 = 1024, bge-m3 = 1024, bge-base-en-v1.5 = 768, bge-small-en-v1.5 = 384.
EXPECTED_DIM_DEFAULT = int(os.getenv("EMBEDDING_DIM", "1024"))


def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def detect_embedding_dim(cur, default: int) -> int:
    """The ACTUAL embedding dimension = the length of a stored CLOB vector, so the VECTOR column +
    validation always match the current model regardless of a stale EMBEDDING_DIM in .env. Falls
    back to `default` when nothing is embedded yet."""
    try:
        cur.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL FETCH FIRST 1 ROWS ONLY")
        row = cur.fetchone()
        if row and row[0] is not None:
            emb = row[0].read() if hasattr(row[0], "read") else row[0]
            if isinstance(emb, bytes):
                emb = emb.decode("utf-8")
            dim = len(json.loads(emb))
            print(f"Detected embedding dimension from stored vectors: {dim}")
            return dim
    except Exception as exc:                           # noqa: BLE001 - fall back to the default below
        print("Could not detect dimension from data; using", default, "-", str(exc)[:100])
        return default
    print(f"No embedded chunks yet; using dimension {default}.")
    return default


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM user_tab_columns WHERE table_name = :t AND column_name = :c",
        {"t": table.upper(), "c": column.upper()})
    return cur.fetchone()[0] > 0


def column_vector_dim(cur, table: str, column: str):
    """The declared dimension of a VECTOR column — Oracle stores it in `user_tab_columns.char_length`
    (verified: a VECTOR(768) column reports char_length=768). Returns the int dimension, or None if
    the column is absent or its dimension can't be read (then we leave it alone)."""
    cur.execute(
        "SELECT data_type, char_length FROM user_tab_columns "
        "WHERE table_name = :t AND column_name = :c",
        {"t": table.upper(), "c": column.upper()})
    row = cur.fetchone()
    if not row:
        return None
    data_type, char_len = row[0], row[1]
    if data_type and str(data_type).upper() == "VECTOR" and char_len:
        return int(char_len)
    return None


def ensure_embedding_vec_column(cur, conn, dim: int) -> str:
    """Ensure `chunks.embedding_vec` is a VECTOR(dim). Create it if missing; if it exists at a
    DIFFERENT dimension (after an embedding-model switch — e.g. a stale VECTOR(768) vs new 1024-d
    vectors), DROP + recreate it at `dim` so the vectors store instead of failing ORA-51803. The
    source-of-truth CLOB `embedding` is untouched, so all rows are then re-migrated. Returns
    'created' | 'recreated' | 'ok'."""
    if not column_exists(cur, "chunks", "embedding_vec"):
        print(f"Adding EMBEDDING_VEC native VECTOR column ({dim}-dim)...")
        cur.execute(f"ALTER TABLE chunks ADD embedding_vec VECTOR({dim}, FLOAT32)")
        conn.commit()
        return "created"
    existing = column_vector_dim(cur, "chunks", "embedding_vec")
    if existing is not None and existing != dim:
        print(f"EMBEDDING_VEC is VECTOR({existing}) but embeddings are {dim}-dim — recreating the "
              f"column at {dim} (vectors are re-migrated from the CLOBs)...")
        cur.execute("ALTER TABLE chunks DROP COLUMN embedding_vec")
        conn.commit()
        cur.execute(f"ALTER TABLE chunks ADD embedding_vec VECTOR({dim}, FLOAT32)")
        conn.commit()
        return "recreated"
    print("EMBEDDING_VEC already exists at the right dimension.")
    return "ok"


def index_exists(cur, index_name: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM user_indexes WHERE index_name = :n", {"n": index_name.upper()})
    return cur.fetchone()[0] > 0


def migrate_clob_to_vector(cur, conn, expected_dim: int):
    """Copy CLOB `embedding` JSON into the native `embedding_vec` for rows not yet migrated. Rows
    whose stored dimension != `expected_dim` are skipped (so a mixed-dimension index is impossible).
    Returns (updated, skipped)."""
    cur.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL AND embedding_vec IS NULL")
    rows = cur.fetchall()
    print("Rows to migrate:", len(rows))
    updated = 0
    skipped = 0
    for chunk_id, emb in rows:
        try:
            if hasattr(emb, "read"):
                emb = emb.read()
            if isinstance(emb, bytes):
                emb = emb.decode("utf-8")
            values = json.loads(emb)
            if len(values) != expected_dim:
                print(f"Skipping chunk {chunk_id}: dim={len(values)} expected={expected_dim}")
                skipped += 1
                continue
            vec = array.array("f", [float(x) for x in values])
            cur.execute("UPDATE chunks SET embedding_vec = :vec WHERE id = :id",
                        {"vec": vec, "id": chunk_id})
            updated += 1
            if updated % 100 == 0:
                conn.commit()
                print("Migrated:", updated)
        except Exception as e:                         # noqa: BLE001 - skip a bad row, keep the rest
            print(f"Skipping chunk {chunk_id}: {e}")
            skipped += 1
    conn.commit()
    return updated, skipped


def maybe_create_index(cur, conn) -> None:
    # A vector index is OPTIONAL. Exact COSINE search needs no index and is fast for small/medium
    # libraries (and avoids ORA-51962, the HNSW in-memory pool running out). Opt in for very large
    # libraries with CREATE_VECTOR_INDEX=true; we use an on-disk IVF index (no in-memory pool).
    if os.getenv("CREATE_VECTOR_INDEX", "false").lower() != "true":
        print("Using exact vector search (no index needed at this scale).")
        return
    if index_exists(cur, "idx_chunks_embedding_vec"):
        print("Vector index already exists.")
        return
    try:
        cur.execute("""
            CREATE VECTOR INDEX idx_chunks_embedding_vec
            ON chunks (embedding_vec)
            ORGANIZATION NEIGHBOR PARTITIONS
            DISTANCE COSINE
            WITH TARGET ACCURACY 90
        """)
        conn.commit()
        print("Vector index (IVF) created.")
    except Exception as e:                             # noqa: BLE001
        print("Vector index not created; exact search will be used.", str(e)[:140])


def main() -> None:
    conn = connect()
    cur = conn.cursor()
    try:
        print("Checking CHUNKS table...")
        expected_dim = detect_embedding_dim(cur, EXPECTED_DIM_DEFAULT)
        ensure_embedding_vec_column(cur, conn, expected_dim)
        print("Migrating old CLOB embeddings into native VECTOR column...")
        updated, skipped = migrate_clob_to_vector(cur, conn, expected_dim)
        print("Vector migration complete.")
        print("Updated:", updated)
        print("Skipped:", skipped)
        maybe_create_index(cur, conn)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
