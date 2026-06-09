# Turbovec Accelerator

This project can optionally use `turbovec` as a compressed local dense-vector
search accelerator for indexed PDFs.

Oracle remains the source of truth. It still stores papers, chunk text,
metadata, page numbers, embeddings, and citations. turbovec stores only a
compressed vector index keyed by Oracle `chunks.id`.

## When To Use It

Use turbovec when the local PDF corpus grows large enough that Oracle exact
vector search becomes the slow part of retrieval, or when you want a smaller
local vector cache.

Keep Oracle as the default when the corpus is small or when you want exact
vector search with the least moving parts.

## Configuration

```env
ENABLE_LOCAL_RAG=true

VECTOR_BACKEND=turbovec
TURBOVEC_ENABLED=true
TURBOVEC_INDEX_PATH=data/vector_cache/chunks.tvim
TURBOVEC_BIT_WIDTH=4
TURBOVEC_OVERFETCH=3
TURBOVEC_AUTOBUILD=true
TURBOVEC_STRICT=false
```

Recommended starting point:

- `TURBOVEC_BIT_WIDTH=4` for better recall.
- `TURBOVEC_OVERFETCH=3` so the cross-encoder reranker still gets a wide
  candidate pool.
- `TURBOVEC_STRICT=false` so Oracle is used as fallback if the cache is missing,
  stale, or the package is not installed.

## Build And Inspect

```bash
python -m backend.retrieval.turbovec_index build
python -m backend.retrieval.turbovec_index status
python -m backend.retrieval.turbovec_index clear
```

When `TURBOVEC_ENABLED=true` or `VECTOR_BACKEND=turbovec`, `python pipeline.py`
also builds the cache after Oracle vector migration.

## Query Flow

```text
question
  -> embedding provider
  -> turbovec IdMapIndex search
  -> fetch matching chunk rows from Oracle by chunks.id
  -> existing RRF + BM25 + reranker + MMR pipeline
  -> cited answer
```

If turbovec cannot serve a query, `backend/retrieval/vector_retriever.py` falls
back to Oracle exact vector search unless `TURBOVEC_STRICT=true`.

## Cache Validity

The cache has two files:

```text
data/vector_cache/chunks.tvim
data/vector_cache/chunks.tvim.manifest.json
```

The manifest records:

- vector count
- embedding dimension
- embedding provider/model
- bit width
- Oracle chunk id signature

If the Oracle chunk set changes, the manifest becomes stale and the cache is
rebuilt or bypassed.

## Accuracy Check

Before making turbovec the default for a deployment, compare against Oracle:

```bash
VECTOR_BACKEND=oracle python -m backend.evaluation.evaluate_retrieval --top-k 10
VECTOR_BACKEND=turbovec TURBOVEC_ENABLED=true python -m backend.evaluation.evaluate_retrieval --top-k 10
```

Accept the switch only if recall, MRR, and nDCG stay close while latency improves.

## Why The Full Repo Is Not Vendored

The useful part for this project is the PyPI package API:

- `IdMapIndex`
- `add_with_ids`
- `search(..., allowlist=...)`
- `write` / `load`
- `prepare`

The framework integrations in the turbovec repo are for LangChain, LlamaIndex,
Haystack, and Agno. This project has its own Oracle-backed retrieval pipeline,
so vendoring those wrappers would add maintenance cost without improving the
model.
