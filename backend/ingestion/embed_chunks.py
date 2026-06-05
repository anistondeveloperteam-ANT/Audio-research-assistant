import os
import warnings
import logging

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import json
import time

import oracledb
from dotenv import load_dotenv
from tqdm import tqdm

from backend.common.embeddings import embed_documents, provider_label

load_dotenv()

# How many chunks to pull from the DB per loop (the embedding provider may
# sub-batch further internally).
BATCH_SIZE = int(os.getenv("EMBED_DB_BATCH", "16"))


def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def main():
    print("Embedding provider:", provider_label())

    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, chunk_text
        FROM chunks
        WHERE embedding IS NULL
        ORDER BY id
    """)
    rows = cur.fetchall()
    print(f"Chunks needing embeddings: {len(rows)}")

    if not rows:
        print("All chunks already have embeddings.")
        cur.close()
        conn.close()
        return

    start = time.time()

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Embedding chunks"):
        batch = rows[i:i + BATCH_SIZE]
        ids = [row[0] for row in batch]

        texts = []
        for row in batch:
            value = row[1]
            texts.append(value.read() if hasattr(value, "read") else str(value))

        embeddings = embed_documents(texts)

        for chunk_id, emb in zip(ids, embeddings):
            emb_json = json.dumps([float(x) for x in emb])
            cur.execute(
                "UPDATE chunks SET embedding = :embedding WHERE id = :chunk_id",
                {"embedding": emb_json, "chunk_id": chunk_id},
            )
        conn.commit()

    elapsed = time.time() - start

    cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    embedded_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks")
    total_count = cur.fetchone()[0]

    print("\nEmbedding summary:")
    print(f"Embedded chunks: {embedded_count}/{total_count}")
    print(f"Time taken: {elapsed:.2f} seconds")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
