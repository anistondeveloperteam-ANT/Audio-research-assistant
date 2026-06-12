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

from backend.common.embeddings import (
    embed_documents, provider_label, provider, format_retrieval_document,
)

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

    # Pull metadata alongside the text so document embeddings can be enriched
    # (title / section / audio concepts) — improves retrieval matching.
    cur.execute("""
        SELECT c.id, c.chunk_text, c.context_text, c.section_name, c.audio_concepts, p.title
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE c.embedding IS NULL
        ORDER BY c.id
    """)
    rows = cur.fetchall()
    print(f"Chunks needing embeddings: {len(rows)}")

    if not rows:
        print("All chunks already have embeddings.")
        cur.close()
        conn.close()
        return

    start = time.time()
    enrich = provider() == "google"   # only Gemini gets the metadata-formatted docs

    def _read(value):
        if value is None:
            return ""
        return value.read() if hasattr(value, "read") else str(value)

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Embedding chunks"):
        batch = rows[i:i + BATCH_SIZE]
        ids = [row[0] for row in batch]

        texts = []
        for row in batch:
            text = _read(row[1])
            context = _read(row[2]).strip()
            # Contextual Retrieval: prepend the situating sentence to what we EMBED (better recall).
            # The stored chunk_text shown to users is unchanged.
            body = (context + "\n" + text) if context else text
            if enrich:
                section = _read(row[3])
                concepts_raw = _read(row[4])
                try:
                    concepts = json.loads(concepts_raw) if concepts_raw else None
                except Exception:
                    concepts = concepts_raw or None
                title = row[5]
                texts.append(format_retrieval_document(
                    title=title, section=section, concepts=concepts, text=body))
            else:
                texts.append(body)

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
