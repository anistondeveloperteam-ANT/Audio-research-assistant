# Research Assistant — Technology & Tools

**A Retrieval-Augmented Generation (RAG) assistant for research papers.**
It ingests PDFs, builds a hybrid (vector + keyword) search index in an Oracle vector
database, and answers questions with cited, grounded answers through a fast web app.

*Document generated: 5 June 2026 · reflects the current codebase.*

---

## 1. At a glance

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **Web server / API** | FastAPI + Uvicorn (streaming via Server‑Sent Events) |
| **Front end** | HTML + CSS + vanilla JavaScript (no build step) |
| **Database** | Oracle Database Free 23ai — native vector search (in Docker) |
| **Optional vector accelerator** | turbovec compressed local dense-vector cache |
| **PDF parsing** | Docling (primary) + PyMuPDF (fallback) + OCR (optional) |
| **Embeddings** | Google Gemini `gemini-embedding-2` (768‑dim) |
| **Re‑ranker** | BAAI `bge-reranker-v2-m3` (cross‑encoder) |
| **Chat models** | OpenAI - selectable in the UI |
| **Conversation memory** | SQLite |
| **Compute** | GPU (CUDA) and/or CPU, configurable |

---

## 2. How the pieces fit (pipeline)

```
PDFs ──► Parse (Docling) ──► Chunk + tag ──► Embed (Gemini) ──► Oracle VECTOR(768)
                                                                      │
Question ──► HyDE + Vector + BM25 ──► RRF fusion ──► Cross‑encoder rerank ──► MMR
                                                                      │
                                          Build cited prompt ──► LLM ──► Streamed answer (web UI)
```

- **Two commands:** `python pipeline.py` builds/refreshes the index; `python run.py`
  launches the web app at `http://localhost:8600`.

---

## 3. Core platform

| Tool | Version | Role |
|------|---------|------|
| **Python** | 3.11 | Implementation language |
| **FastAPI** | 0.136.1 | Web framework — REST API + streaming chat endpoint |
| **Uvicorn** | 0.46.0 | ASGI server that runs the app |
| **python‑dotenv** | 1.2.2 | Loads configuration from `.env` |
| **Server‑Sent Events (SSE)** | — | Streams the answer to the browser token by token |

---

## 4. AI / ML models

### Embeddings (turn text into meaning vectors)
| Tool | Version | Role |
|------|---------|------|
| **Google Gemini Embedding** (`gemini-embedding-2`) | via `google-genai` 1.75.0 | Default embedding model — 768‑dim, free tier, top‑tier retrieval accuracy. Runs as an API call, so it frees the local GPU. |
| **sentence‑transformers** | 5.5.0 | Optional local embedding backend (e.g. `BAAI/bge-base-en-v1.5`) |

### Re‑ranking (precise final ordering of retrieved passages)
| Tool | Role |
|------|------|
| **BAAI `bge-reranker-v2-m3`** | Cross‑encoder that reads the question + passage together to score relevance |

### Chat / answer models
| Provider | How | Models |
|----------|-----|--------|
| **OpenAI** | `OPENAI_API_KEY` (cloud) | `gpt-4o` (default), `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` — pick in the UI |

### ML runtime
| Tool | Version | Role |
|------|---------|------|
| **PyTorch** | 2.7.1 (CUDA 12.6 build) | Tensor / GPU backend for the local models |
| **transformers** | 4.57.6 | Model runtime under sentence‑transformers |
| **OpenAI SDK** | 1.109.1 | Client for OpenAI chat APIs (streaming) |

---

## 5. Document processing (PDF → clean text)

| Tool | Version | Role |
|------|---------|------|
| **Docling** (IBM) | 2.93.0 | **Primary parser** — AI layout analysis, reading order, tables, equations, section structure. Best quality for scientific papers. Runs on GPU when available. |
| **PyMuPDF** (`fitz`) | 1.27.2.3 | Fast fallback parser if Docling errors on a file |
| **pypdf** | 6.11.0 | Lightweight PDF utilities |
| **PaddleOCR / Tesseract** | optional | OCR — only for scanned / image‑only PDFs (`ENABLE_OCR`) |

After parsing, text is split into **section‑aware chunks** tagged with section name,
chunk type (prose / caption / algorithm), equation/table flags, and audio concepts
(MVDR, PESQ, STOI, U‑Net, spectral subtraction, deep filtering, etc.).

---

## 6. Retrieval engine (finding the right evidence)

A **hybrid, multi‑signal retriever** combines several techniques:

| Technique | What it does |
|-----------|--------------|
| **Vector / semantic search** | Oracle `VECTOR_DISTANCE … COSINE` — meaning‑based matching |
| **turbovec** *(optional)* | Compressed dense-vector candidate search, hydrated back from Oracle |
| **HyDE** (Hypothetical Document Embeddings) | Rewrites the question into a hypothetical answer passage for better recall |
| **BM25F** (field‑weighted keyword search) | Exact‑term matching; title/concepts/section weighted higher |
| **RRF** (Reciprocal Rank Fusion) | Merges the vector + keyword rankings robustly |
| **Cross‑encoder re‑ranking** | `bge-reranker-v2-m3` re‑scores the top candidates precisely |
| **MMR** (Maximal Marginal Relevance) | Diversifies results + caps passages per paper |

Supporting numeric libraries: **NumPy 2.3.5**, **SciPy 1.17.1**, **pandas 3.0.3**.

---

## 7. Database & storage

| Tool | Version | Role |
|------|---------|------|
| **Oracle Database Free 23ai** | latest (Docker) | Relational store **+ native vector search** (`embedding_vec VECTOR(768, FLOAT32)`, exact COSINE; optional HNSW/IVF index) |
| **turbovec** | 0.7.0 | Optional compressed local vector-search cache; Oracle remains source of truth |
| **python‑oracledb** | 4.0.0 | Oracle driver |
| **SQLite** | built‑in | Conversation memory — sessions, turns (with saved sources), facts (`data/memory.db`) |

---

## 8. Front end (the web UI)

| Tool | Role |
|------|------|
| **HTML / CSS / vanilla JavaScript** | Single‑page chat UI — **no build step, no framework** |
| **marked.js** | Renders the model's markdown answers |
| **Inter** (web font) | Typography |
| **Server‑Sent Events** | Live token‑by‑token streaming, citation chips, source drawer, live timer + speed badge |

Features: streaming cited answers, source‑passage drawer, multi‑session sidebar,
PDF upload + live indexing, model switcher, dark/light theme, and per‑question
copy / edit‑and‑resend / delete.

---

## 9. Compute (GPU + CPU)

Device placement is configured in `.env` (`DEVICE`, `EMBEDDING_DEVICE`,
`RERANKER_DEVICE` = `auto` | `cuda` | `cpu`). Because embeddings use the Gemini API,
the local GPU is free for the **re‑ranker** and **Docling** parsing — efficient even
on a small (6 GB) laptop GPU.

---

## 10. Developer & evaluation tools

| Tool | Version | Role |
|------|---------|------|
| **pytest** | 9.0.3 | Unit test suite |
| **pyflakes** | 3.4.0 | Lint — unused imports / undefined names |
| **autoflake** | 2.3.3 | Removes unused imports |
| **vulture** | 2.16 | Finds dead code |
| **evaluate_retrieval** | — | Scores retrieval quality against a question set |
| **evaluate_llm** | — | Scores/compares LLM answer accuracy (keyword coverage, citations, optional LLM‑judge) |
| **tqdm** | 4.67.3 | Progress bars for long jobs |
| **Docker Desktop** | — | Runs the Oracle 23ai database container |
| **Git** | — | Version control |
| **VS Code** | — | Editor + debugger configuration included |

---

## 11. External services / APIs

| Service | Used for | Cost |
|---------|----------|------|
| **Google AI Studio (Gemini API)** | Text embeddings | Free tier |
| **OpenAI** | Cloud chat models (`gpt-4o` family) | Pay‑as‑you‑go |

---

## 12. Why these choices

- **Docling + Gemini embeddings** are independently rated **best‑in‑class for 2026**
  for self‑hosted scientific‑paper RAG on modest hardware (the only stronger embedding
  model needs 16 GB+ of GPU memory).
- **OpenAI** provides cloud chat models through a simple API key, and the
  active provider/model is switchable from the UI.
- **No‑build front end** (plain HTML/CSS/JS) keeps the UI fast, dependency‑free, and
  easy for anyone to read and modify.
- **Hybrid retrieval** (semantic + keyword + re‑rank + diversify) is more accurate than
  any single method, and the cited‑answer design keeps the assistant grounded and
  honest — it answers only from the uploaded papers.

---

*Research Assistant — Python 3.11 · FastAPI · Oracle 23ai · Docling · Gemini · OpenAI.*
