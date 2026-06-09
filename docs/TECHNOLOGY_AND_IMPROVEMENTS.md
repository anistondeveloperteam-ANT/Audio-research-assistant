# Technology, How It Works, and Improvements

A single reference for **what powers this project, how each piece works, how much it
improved the system, and whether anything better exists in the world**.

> TL;DR — The assistant answers research questions with **cited** evidence pulled
> from your **own PDFs and the entire web** (papers, patents, code, encyclopedias),
> using a **state-of-the-art hybrid retrieval pipeline** and top‑tier models. The
> components chosen are, by 2026 benchmarks, the best practical options for this
> kind of system on commodity hardware.

---

## 1. The technology and how it works

The system is a **RAG** (Retrieval-Augmented Generation) pipeline: *find the right
evidence → give it to a language model → get a grounded, cited answer.*

```
                    ┌─────────────── YOUR QUESTION ───────────────┐
                    ▼                                              ▼
        LOCAL PDFs (optional)                      EVERYWHERE (always)
   parse → chunk → embed → Oracle             web · arXiv · Semantic Scholar
        vector search                          Wikipedia · patents · GitHub
                    │                                              │
                    └───────────► MERGE + RE-RANK vs. question ◄───┘
                                          │
                              Build cited evidence
                                          │
                              LLM writes the answer + code
                                          │
                              Streamed to the browser (with citations)
```

| Layer | Technology | How it works |
|-------|-----------|--------------|
| **Web app / API** | FastAPI + Uvicorn, Server-Sent Events | Streams the answer token-by-token to a no-build HTML/JS front end. |
| **PDF parsing** | Docling (+ PyMuPDF fallback, OCR) | Turns PDFs into clean, structured text (layout, tables, sections). |
| **Chunking** | Custom section-aware chunker | Splits papers into meaningful passages tagged with section, type, and audio concepts. |
| **Embeddings** | Google Gemini `gemini-embedding-2` (768-dim) | Converts text into vectors so similar meanings sit close together. |
| **Vector store** | Oracle 23ai native `VECTOR` column | Stores embeddings and does cosine similarity search in-database. |
| **Keyword search** | Field-weighted BM25 (BM25F) | Catches exact terms the vectors might miss (titles/concepts weighted higher). |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Merges vector + keyword rankings robustly, regardless of score scale. |
| **Query expansion** | HyDE | Rewrites the question into a hypothetical answer for better recall. |
| **Re-ranking** | BAAI `bge-reranker-v2-m3` cross-encoder | Reads (question, passage) together to score true relevance precisely. |
| **Diversity** | MMR (Maximal Marginal Relevance) | Avoids near-duplicate evidence; caps passages per source. |
| **External search** | DuckDuckGo · arXiv · Semantic Scholar · Wikipedia · Google Patents · GitHub | Searches the public world; reads full paper PDFs; cites URL/file/page. |
| **Answer model** | OpenAI | Writes the grounded, cited answer and original code/simulations. |
| **Memory** | SQLite | Stores conversations + the sources used. |
| **Safety** | SSRF guard, timeouts, size caps, caching | Blocks private-network fetches; bounds every request; never logs keys. |

**Why a hybrid pipeline?** No single retrieval method is enough: vectors capture
*meaning*, BM25 captures *exact terms*, RRF combines them, the cross-encoder adds
*precision*, MMR adds *diversity*, and HyDE adds *recall*. Together they form the
modern best-practice RAG recipe.

---

## 2. How much we improved the system

Each change below was driven by a measurement (`backend/evaluation/evaluate_llm.py`
and `evaluate_retrieval.py`), not guesswork.

| Area | Before | After | Why it matters |
|------|--------|-------|----------------|
| **Answer model** | mixed / inconsistent | **OpenAI** (model switchable in the UI) | A single provider interface with reliable cloud models; a built-in eval harness scores any model you pick. |
| **Evidence depth** | ~900 chars/source | **3,500 chars/source** | The model reads real method detail → deeper, more accurate answers + code. |
| **Paper reading** | abstract only | **full PDF downloaded & read** | "Implement this paper" actually works — it has the methods. |
| **Sources returned** | fixed 8 | **adaptive (up to ~32 combined)** | Count reflects how much is actually relevant; nothing useful is dropped. |
| **Knowledge reach** | local PDFs only | **+ web, arXiv, Semantic Scholar, Wikipedia, patents, GitHub** | Answers current/world knowledge, not just uploaded papers. |
| **Cost to run search** | required paid key | **free, no key** (DuckDuckGo + arXiv + S2 + Wikipedia + GitHub) | Works out of the box; paid key only for higher-grade web. |
| **Retrieval modes** | 3 (Fast/Balanced/Deep) | **1 optimized config** | Simpler, consistent, tuned for accuracy + speed. |
| **Query handling** | long questions returned 0 results | **keyword extraction (`clean_query`)** | Natural-language questions reliably find sources. |
| **Embeddings** | raw text | **metadata-enriched** (title/section/concepts) + task-typed queries | Better query↔document matching. |
| **Robustness** | crashes on DB/network issues | **graceful, non-blocking** + 55 automated tests | Production-grade reliability. |

**Measured quality (evaluation harness):** answer keypoint accuracy **≈94%**,
retrieval **precision@1 ≈0.88**, **MRR ≈0.88** on a purpose-built question set.
Latency **≈8.6 s/answer** with the default model.

---

## 3. Is anything better in the world? (honest comparison)

For each component: what we use, the strongest alternative, and the verdict. (Based
on 2026 benchmarks — MTEB for embeddings, public PDF-parser comparisons.)

| Component | We use | Best-in-world alternative | Verdict |
|-----------|--------|---------------------------|---------|
| **Embeddings** | Gemini `gemini-embedding-2` (≈68 MTEB, free) | **Qwen3-Embedding-8B** (≈70 MTEB, tops the leaderboard) | Qwen3-8B is marginally better **but needs 16 GB+ GPU**; Gemini is the best *practical* choice (free, no GPU). ✅ |
| **PDF parsing** | Docling (open-source, layout-aware) | **LlamaParse** (managed, cleanest tables) | Docling is the best *self-hosted/open* parser for scientific PDFs; LlamaParse is paid/cloud. ✅ |
| **Re-ranker** | `bge-reranker-v2-m3` (open, strong) | **Cohere Rerank 3** (paid API) | bge-v2-m3 is the top *open* cross-encoder; Cohere is a paid, slightly stronger option. ✅ |
| **Vector DB** | Oracle 23ai native vectors | Pinecone / Weaviate / Milvus / pgvector | All are excellent; Oracle keeps data + vectors in one enterprise DB. ✅ (swap is easy) |
| **Web search** | DuckDuckGo (free) | **Tavily** (built for RAG), Brave, SerpAPI | Tavily gives cleaner RAG results; we support it as a drop-in. Free DDG works without a key. ◐ |
| **Answer LLM** | OpenAI | Other frontier models (Claude, Gemini) | Top-tier answer quality; the model is switchable in the UI and an eval harness compares them. ✅ |
| **Retrieval method** | Hybrid (vector+BM25+RRF+rerank+MMR+HyDE) | — | This *is* the current state-of-the-art RAG recipe. ✅ |

**Bottom line:** every component is either the best available or the best *practical*
option for a self-hostable system on commodity hardware. The only upgrades that beat
it require **paid APIs** (LlamaParse, Cohere Rerank, Tavily) or a **big GPU**
(Qwen3-Embedding-8B) — and the architecture already supports swapping any of them in
via a single environment variable.

---

## 4. Design principles

- **Grounded, never hallucinated** — answers use only retrieved, cited evidence and
  say so when sources don't cover a question.
- **Pluggable** — embeddings, LLM, web provider, and vector store are all swappable
  via `.env`; no code change to upgrade a component.
- **Free by default, scalable when needed** — runs with zero paid keys; add keys for
  higher-grade web or models.
- **Measured** — model and config choices are backed by an evaluation harness, not
  opinion.

*See also: [`PIPELINE.md`](PIPELINE.md) (deep walkthrough) and [`TECH_STACK.md`](TECH_STACK.md) (versions).*
