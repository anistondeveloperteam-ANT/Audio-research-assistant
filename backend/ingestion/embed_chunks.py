import os
import warnings
import logging

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# How many chunks to pull + embed per DB-commit batch (the model sub-batches further internally).
BATCH_SIZE = int(os.getenv("EMBED_DB_BATCH", "16"))


def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def _read(value):
    if value is None:
        return ""
    return value.read() if hasattr(value, "read") else str(value)


def _embed_text(title: str, section: str, context: str, chunk: str) -> str:
    """The text we EMBED for a chunk (NOT what users see). A 'contextual chunk header' — the paper
    title + section name — is prepended so the chunk's vector reflects WHERE it sits in the document,
    which sharply improves retrieval recall at ZERO cost (no LLM call, instant). Any LLM situating
    sentence (`context`) is included too when present. The stored chunk_text shown in citations is
    unchanged. This is what makes ingestion fast AND accurate without the per-chunk LLM step."""
    head = " | ".join(p for p in (title.strip(), section.strip())
                      if p and p.strip().lower() not in ("", "unknown"))
    return "\n".join(p for p in (head, context.strip(), chunk) if p)


def embed_pending_chunks(progress=None, should_cancel=None):
    """Embed every chunk whose embedding IS NULL **in the current process**, reusing the warm bge
    model — no subprocess, no second model load. (A fresh model-loading subprocess is what OOM-
    crashes a small GPU/box: VRAM -> 0xC0000005, system commit -> WinError 1455. Reusing the
    already-loaded model avoids both.)

    `progress(done, total)` is called after each committed DB-batch; `should_cancel()` (optional)
    is polled between batches to stop early. Returns {'embedded': n, 'total': m, 'cancelled': bool}.
    Idempotent (only NULL chunks), so it can be called repeatedly to embed papers incrementally as
    each one finishes parsing — which is how the web UI overlaps parsing (CPU) with embedding (GPU).
    """
    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT c.id, c.chunk_text, c.context_text, c.section_name, p.title "
            "FROM chunks c JOIN papers p ON p.id = c.paper_id "
            "WHERE c.embedding IS NULL ORDER BY c.id"
        )
        rows = cur.fetchall()
        total = len(rows)
        if total == 0:
            return {"embedded": 0, "total": 0, "cancelled": False}

        done = 0
        for i in range(0, total, BATCH_SIZE):
            if should_cancel is not None and should_cancel():
                return {"embedded": done, "total": total, "cancelled": True}
            batch = rows[i:i + BATCH_SIZE]
            ids = [r[0] for r in batch]
            # Embed with a contextual chunk header (title + section [+ LLM context]); the stored
            # chunk_text is untouched, so citations still show the original text.
            texts = [_embed_text(_read(r[4]), _read(r[3]), _read(r[2]).strip(), _read(r[1]))
                     for r in batch]
            embeddings = embed_documents(texts)
            for chunk_id, emb in zip(ids, embeddings):
                cur.execute(
                    "UPDATE chunks SET embedding = :embedding WHERE id = :chunk_id",
                    {"embedding": json.dumps([float(x) for x in emb]), "chunk_id": chunk_id},
                )
            conn.commit()                     # per-batch commit -> progress survives an interruption
            done += len(batch)
            if progress is not None:
                progress(done, total)
        return {"embedded": done, "total": total, "cancelled": False}
    finally:
        cur.close()
        conn.close()


def main():
    """CLI / pipeline entry point — embeds all pending chunks in THIS process. Used by
    `pipeline.py`, where no server is running to compete for the GPU, so loading the model here is
    fine. (The web UI does NOT spawn this; it calls embed_pending_chunks in the warm server.)"""
    warnings.filterwarnings("ignore")     # quiet the CLI run only (not when imported by the server)
    print("Embedding provider:", provider_label())
    start = time.time()
    bar = {"p": None}

    def _progress(done, total):
        if bar["p"] is None:
            bar["p"] = tqdm(total=total, desc="Embedding chunks")
        bar["p"].n = done
        bar["p"].refresh()

    result = embed_pending_chunks(progress=_progress)
    if bar["p"] is not None:
        bar["p"].close()

    if result["total"] == 0:
        print("All chunks already have embeddings.")
        return
    print("\nEmbedding summary:")
    print(f"Embedded chunks: {result['embedded']}/{result['total']}")
    print(f"Time taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
