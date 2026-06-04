import os
import json
import array

from dotenv import load_dotenv
import oracledb

load_dotenv()

# Must match the embedding model's output dimension (EMBEDDING_DIM in .env).
# bge-base-en-v1.5 = 768, bge-small-en-v1.5 = 384.
EXPECTED_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

conn = oracledb.connect(
    user=os.getenv("ORACLE_USER"),
    password=os.getenv("ORACLE_PASSWORD"),
    dsn=os.getenv("ORACLE_DSN"),
)

cur = conn.cursor()


def column_exists(table, column):
    cur.execute("""
        SELECT COUNT(*)
        FROM user_tab_columns
        WHERE table_name = :table_name
          AND column_name = :column_name
    """, {
        "table_name": table.upper(),
        "column_name": column.upper(),
    })
    return cur.fetchone()[0] > 0


def index_exists(index_name):
    cur.execute("""
        SELECT COUNT(*)
        FROM user_indexes
        WHERE index_name = :index_name
    """, {
        "index_name": index_name.upper(),
    })
    return cur.fetchone()[0] > 0


print("Checking CHUNKS table...")

if not column_exists("chunks", "embedding_vec"):
    print("Adding EMBEDDING_VEC native VECTOR column...")
    cur.execute(f"ALTER TABLE chunks ADD embedding_vec VECTOR({EXPECTED_DIM}, FLOAT32)")
    conn.commit()
else:
    print("EMBEDDING_VEC already exists.")


print("Migrating old CLOB embeddings into native VECTOR column...")

cur.execute("""
    SELECT id, embedding
    FROM chunks
    WHERE embedding IS NOT NULL
      AND embedding_vec IS NULL
""")

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

        if len(values) != EXPECTED_DIM:
            print(f"Skipping chunk {chunk_id}: dim={len(values)} expected={EXPECTED_DIM}")
            skipped += 1
            continue

        vec = array.array("f", [float(x) for x in values])

        cur.execute(
            "UPDATE chunks SET embedding_vec = :vec WHERE id = :id",
            {
                "vec": vec,
                "id": chunk_id,
            },
        )

        updated += 1

        if updated % 100 == 0:
            conn.commit()
            print("Migrated:", updated)

    except Exception as e:
        print(f"Skipping chunk {chunk_id}: {e}")
        skipped += 1

conn.commit()

print("Vector migration complete.")
print("Updated:", updated)
print("Skipped:", skipped)


print("Creating HNSW vector index...")

if not index_exists("idx_chunks_embedding_vec_hnsw"):
    try:
        cur.execute("""
            CREATE VECTOR INDEX idx_chunks_embedding_vec_hnsw
            ON chunks (embedding_vec)
            ORGANIZATION INMEMORY NEIGHBOR GRAPH
            DISTANCE COSINE
            WITH TARGET ACCURACY 90
        """)
        conn.commit()
        print("HNSW vector index created.")
    except Exception as e:
        print("Could not create HNSW vector index.")
        print("Reason:", e)
        print("Vector search can still work without the index, but slower.")
else:
    print("HNSW vector index already exists.")

cur.close()
conn.close()