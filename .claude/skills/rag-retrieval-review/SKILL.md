---
name: rag-retrieval-review
description: "Review a change to the retrieval / RAG pipeline — chunking, embeddings, hybrid search, reranking, evidence formatting, citation grounding, and freshness — for this research assistant."
origin: ECC (adapted)
---

# RAG / Retrieval Review Skill

A focused checklist for reviewing changes to retrieval and answer-grounding in this
project. The pipeline is: parse → chunk → embed → store (Oracle/turbovec) → hybrid
retrieve (vector + BM25 + RRF) → rerank → MMR → format evidence → grounded, cited answer.

## When to Use

Invoke this skill when a change touches:
- `backend/retrieval/` (hybrid retrieve, RRF, reranker, MMR, HyDE)
- `backend/ingestion/` or chunking/embedding code
- `backend/external_search/` (web, arXiv, Semantic Scholar, patents, GitHub)
- `webapp/chat_logic.py` evidence building (`format_evidence`, `select_sources`, `_deep_queries`)
- anything that decides **what evidence reaches the model** or **how answers cite it**

## Review Phases

### 1. Retrieval quality
- Hybrid search keeps vector **and** keyword paths; RRF fuses ranks (scale-independent).
- Reranker scores `(query, passage)` pairs; `select_sources` keeps relevant ones between
  `SOURCE_MIN`/`SOURCE_MAX` rather than a blind fixed top-k.
- Query handling is robust: long natural-language questions are cleaned/expanded (HyDE) so
  recall does not collapse to zero.
- Deduplication is stable (`_item_key` / `_extend_unique`) and does not drop distinct sources
  that merely share a domain.

### 2. Grounding & citations (most important)
- The answer is built **only** from the numbered evidence; `[n]` markers map 1:1 to the
  evidence actually sent to the model (watch for off-by-one when evidence is trimmed but the
  source list is numbered separately).
- No invented titles, URLs, numbers, or page/line references.
- "I couldn't find it" is preserved when evidence is missing — never silently guessed.

### 3. Evidence budget & cost
- Evidence fed to the LLM is **bounded** (`EVIDENCE_MAX_ITEMS`, `EVIDENCE_BUDGET_CHARS`,
  per-source `max_chars`) so deep/multi-angle search cannot blow the context or the bill.
- The auto-shrink path still produces an answer on small token budgets (low-credit accounts).
- More sources are gathered for coverage than are pushed into the prompt — the best subset wins.

### 4. Freshness
- External results can surface recent material: arXiv/S2 fetch newest + most-relevant, GitHub
  sorts by update, and the reranker boosts recency when the query says "latest/recent/<year>".
- Caches have a sane TTL so new content appears without manual clearing.

### 5. Safety & robustness
- Every fetched URL passes the SSRF guard; sizes/timeouts are capped.
- Optional systems degrade gracefully: local RAG off → web still answers; Oracle/Memgraph down
  → no crash; web search fails → local evidence still works.

## Output

Report findings as `severity · file:line · issue · fix`, then a verdict
(**approve / approve-with-fixes / block**). Prefer measuring a retrieval change with
`backend/evaluation/evaluate_retrieval.py` (precision@1, MRR) over eyeballing it.
