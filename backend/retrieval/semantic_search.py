import os
import json
import re
import numpy as np
import oracledb
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

QUERY_TERMS = [
    "MVDR",
    "LCMV",
    "GSC",
    "DNN",
    "DeepFilterNet",
    "RNNoise",
    "dereverberation",
    "speech enhancement",
    "beamforming",
    "beamformer",
    "DOA",
    "direction of arrival",
    "microphone array",
    "acoustic echo cancellation",
    "AEC",
    "WPE",
]

def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )

def read_clob(value):
    return value.read() if hasattr(value, "read") else value

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)

def keyword_bonus(query, text, concepts):
    query_lower = query.lower()
    text_lower = text.lower()
    concepts_lower = (concepts or "").lower()

    bonus = 0.0
    matched = []

    for term in QUERY_TERMS:
        term_lower = term.lower()

        if term_lower in query_lower:
            if term_lower in text_lower:
                bonus += 0.035
                matched.append(term)

            if term_lower in concepts_lower:
                bonus += 0.025

    return bonus, sorted(set(matched))

def load_chunks():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            c.id,
            p.id AS paper_id,
            p.title,
            c.section_name,
            c.chunk_type,
            c.page_start,
            c.page_end,
            c.audio_concepts,
            c.chunk_text,
            c.embedding
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE c.embedding IS NOT NULL
    """)

    chunks = []

    for row in cur.fetchall():
        text = read_clob(row[8])
        embedding_json = read_clob(row[9])

        try:
            embedding = json.loads(embedding_json)
        except Exception:
            continue

        chunks.append({
            "chunk_id": row[0],
            "paper_id": row[1],
            "title": row[2],
            "section": row[3],
            "chunk_type": row[4],
            "page_start": row[5],
            "page_end": row[6],
            "concepts": read_clob(row[7]) if row[7] else "",
            "text": text,
            "embedding": embedding,
        })

    cur.close()
    conn.close()

    return chunks

def diversify_results(results, top_k=12, max_per_paper=2):
    final = []
    paper_counts = {}

    for result in results:
        paper_id = result["paper_id"]
        count = paper_counts.get(paper_id, 0)

        if count >= max_per_paper:
            continue

        final.append(result)
        paper_counts[paper_id] = count + 1

        if len(final) >= top_k:
            break

    return final

def semantic_search(query, top_k=12, max_per_paper=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("\nQuestion:", query)
    print("Creating query embedding...")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    print("Loading chunks from Oracle...")
    chunks = load_chunks()
    print("Chunks loaded:", len(chunks))

    results = []

    for chunk in chunks:
        semantic_score = cosine_similarity(query_embedding, chunk["embedding"])
        bonus, matched_terms = keyword_bonus(query, chunk["text"], chunk["concepts"])

        final_score = semantic_score + bonus

        item = dict(chunk)
        item["semantic_score"] = semantic_score
        item["keyword_bonus"] = bonus
        item["final_score"] = final_score
        item["matched_terms"] = matched_terms

        results.append(item)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    diversified = diversify_results(
        results,
        top_k=top_k,
        max_per_paper=max_per_paper,
    )

    return diversified

if __name__ == "__main__":
    query = "Compare MVDR, LCMV, GSC, DNN speech enhancement, and dereverberation methods"
    results = semantic_search(query, top_k=12, max_per_paper=2)

    print("\nTop diversified semantic search results:\n")

    for i, r in enumerate(results, 1):
        print("=" * 100)
        print(f"Rank: {i}")
        print(f"Final score: {r['final_score']:.4f}")
        print(f"Semantic score: {r['semantic_score']:.4f}")
        print(f"Keyword bonus: {r['keyword_bonus']:.4f}")
        print(f"Matched terms: {', '.join(r['matched_terms']) if r['matched_terms'] else 'None'}")
        print(f"Paper: {r['title']}")
        print(f"Section: {r['section']}")
        print(f"Type: {r['chunk_type']}")
        print(f"Pages: {r['page_start']}-{r['page_end']}")
        print(f"Concepts: {r['concepts']}")
        print("\nPreview:")
        print(r["text"][:900].replace("\n", " "))
        print()