import os
import warnings
import logging

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
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
BATCH_SIZE = 16

def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )

def main():
    from backend.common.device import resolve_device
    print("Embedding model:", MODEL_NAME)
    print("CUDA available:", torch.cuda.is_available())

    device = resolve_device("EMBEDDING_DEVICE")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, chunk_text
        FROM chunks
        WHERE embedding IS NULL
        ORDER BY id
    """)

    rows = cur.fetchall()
    print(f"\nChunks needing embeddings: {len(rows)}")

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
            text = value.read() if hasattr(value, "read") else str(value)
            texts.append(text)

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for chunk_id, emb in zip(ids, embeddings):
            emb_json = json.dumps([float(x) for x in emb])

            cur.execute(
                """
                UPDATE chunks
                SET embedding = :embedding
                WHERE id = :chunk_id
                """,
                {
                    "embedding": emb_json,
                    "chunk_id": chunk_id,
                },
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