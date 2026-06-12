# Architecture, Technology & Full Pipeline

The complete, deep reference for **what this project uses and how every piece fits together** —
the tech stack, every module, and each pipeline (chat, retrieval, ingestion, the coding agent,
external search, evaluation, observability, GPU, and the optional distributed layer).

> Plain‑English overview: see the [README](../README.md). This doc is the engineering map.

---

## 1. The system in one picture

```mermaid
flowchart TB
    subgraph UI["Frontend — vanilla HTML/CSS/JS (no build)"]
      B[index.html · app.js · styles.css]
    end
    subgraph API["webapp/ — FastAPI + Uvicorn"]
      S[server.py<br/>routes · auth · SSE/NDJSON]
      CL[chat_logic.py<br/>chat orchestration]
    end
    subgraph CORE["backend/ — the engine"]
      RET[retrieval/<br/>hybrid search + GPU rerank]
      EXT[external_search/<br/>web · papers · GitHub]
      LLM[llm/<br/>OpenAI-compatible router]
      ANS[answering/<br/>verify · review · cite · modes]
      AG[agent/<br/>code agent + LangGraph]
      ING[ingestion/<br/>PDF → chunks → vectors]
      MEM[memory/<br/>sessions · cache]
      OBS[observability/<br/>Langfuse tracing]
    end
    subgraph DATA["Data & models"]
      ORA[(Oracle 23ai<br/>vectors)]
      SQL[(SQLite<br/>cache · chats · auth · checkpoints)]
      TV[(turbovec<br/>compressed vectors)]
      MG[(Memgraph<br/>GraphRAG · optional)]
      GPU[/NVIDIA GPU<br/>reranker + embeddings/]
    end
    B <-->|HTTP + NDJSON stream| S --> CL
    CL --> RET & EXT & LLM & ANS & MEM & OBS
    AG --> LLM
    RET --> ORA & TV & MG & GPU
    ING --> ORA & TV
    MEM --> SQL
    EXT -->|cloud APIs| Web((Internet))
    LLM -->|cloud / local| Models((LLM provider))
```

**Design principles**
- **Reuse, don't rewrite.** Every optional/advanced layer wraps proven functions; the core paths are untouched when flags are off.
- **Optional + fallback‑safe.** Oracle off → web‑only; Memgraph down → no graph hits; Redis down → in‑process; GPU absent → CPU; Crawl4AI absent → BeautifulSoup. Nothing hard‑fails.
- **Flags default OFF** for all heavy/experimental infra (observability, queue, multi‑agent graph).

---

## 2. Technology stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| **Web framework** | FastAPI | 0.136.1 | HTTP API + routing + dependency‑injected auth |
| | Uvicorn | 0.46.0 | ASGI server (`python run.py` → `:8600`) |
| | Starlette `SessionMiddleware` + itsdangerous | 2.2.0 | signed session cookie |
| | python‑multipart | 0.0.28 | PDF upload form parsing |
| **Frontend** | vanilla HTML / CSS / JS | — | **no build step, no framework**; NDJSON stream via `fetch` + `ReadableStream` |
| | marked.js · highlight.js · KaTeX | CDN | markdown · code highlight · math |
| **Numerics** | NumPy / SciPy / pandas | 2.3.5 / 1.17.1 / 3.0.3 | vectors, scoring, data |
| **Embeddings / rerank** | sentence‑transformers | 5.5.0 | local embedder + cross‑encoder reranker |
| | transformers | 4.57.6 | model backbone |
| | **PyTorch (CUDA)** | 2.7.1+cu126 | **GPU inference** (reranker + embeddings, fp16) |
| **Vector DB** | Oracle 23ai (`oracledb`) | 4.0.0 | `VECTOR` columns + `VECTOR_DISTANCE` search |
| | **turbovec** | 0.7.0 | optional compressed (4‑bit) local vector index |
| **Graph (optional)** | Memgraph via `neo4j` Bolt | 6.2.0 | GraphRAG concept/section expansion |
| **PDF parsing** | docling · PyMuPDF · pypdf | 2.93 / 1.27 / 6.11 | layout/table‑aware parse + fast fallback |
| **LLM client** | openai SDK | 1.109.1 | OpenAI‑compatible streaming (Gemini/Mistral/GPT/Ollama) |
| **Embeddings (cloud)** | google‑genai | 1.75.0 | optional Gemini embeddings |
| **Agent graphs** | LangGraph | 1.2.4 | research + multi‑agent code `StateGraph` |
| | langgraph‑checkpoint‑sqlite | 3.1.0 | persistent graph checkpoints (resume) |
| **Sandbox** | Docker | host | network‑isolated code execution |
| **HTTP / scraping** | requests · beautifulsoup4 | 2.34 / 4.14 | external fetch + readable‑text extraction |
| | **Crawl4AI** (optional) | 0.8.9 | JS‑rendered, BM25‑filtered markdown |
| **Memory** | SQLite (stdlib) | — | answer cache · chats · auth · checkpoints (WAL) |
| **Observability** (optional) | Langfuse (OpenTelemetry) | 4.7.1 | per‑stage traces; `protobuf` pinned 5.29.5 |
| **Eval** (dev/test) | DeepEval · Ragas‑style metrics | 4.0.6 | faithfulness / relevancy / citation validity |
| **Queue** (optional) | Celery + Redis | 5.6.3 / 8.0.0 | distributed agent execution |
| **Tests / lint** | pytest · pyflakes · vulture | 9.0 / 3.4 / 2.16 | 194 tests, offline/mocked |

---

## 3. Directory map

```
backend/
├── llm/            streaming_provider.py    → OpenAI-compatible router (PROVIDERS + CATALOG)
├── retrieval/      hybrid_retrieve.py       → the retrieval pipeline (orchestrator)
│                   vector_retriever.py      → Oracle VECTOR_DISTANCE + local/Gemini embed (GPU)
│                   retrieval_fusion.py      → field-weighted BM25 · RRF · MMR
│                   hyde_generator.py        → HyDE query expansion (template-based, no LLM)
│                   turbovec_index.py        → compressed 4-bit vector backend
├── external_search/ orchestrator.py         → parallel channels + shared timeout + rerank
│                   web_search.py            → providers (DuckDuckGo/Tavily/Brave/SerpAPI) + page text
│                   crawl.py                 → Crawl4AI adapter (optional, fallback-safe)
│                   scholar_search.py        → arXiv · Semantic Scholar · Wikipedia · patents
│                   github_search.py         → repos + code (stars-first)
│                   pdf_reader.py            → read online PDFs
│                   source_ranker.py         → cross-encoder rerank of external sources
│                   base.py                  → safe_get · is_safe_url · cached · timeouts/caps
├── answering/      agentic_answer.py        → draft → verify → refine loop + sandbox sim
│                   reviewer.py              → automatic peer review + relevance gate
│                   citations.py             → [n] validation/repair  (NEW)
│                   research_modes.py        → fast/deep run profiles   (NEW)
│                   query_sanity.py          → reject junk queries
├── agent/          loop.py                  → code agent: THINK → EXECUTE → REFLECT
│                   code_runner.py           → Docker sandbox runner (network-off, capped)
│                   hooks.py                 → pre-run audit/allow-block gate
│                   memory.py                → two-tier constant-size agent memory
│                   graph.py                 → LangGraph multi-agent pipeline  (NEW, optional)
│                   celery_app.py            → distributed queue task          (NEW, optional)
│                   task_channel.py          → Redis/in-proc event bus         (NEW, optional)
│                   research_agent.py        → deep-research loop
│                   langgraph_research.py    → research as a LangGraph StateGraph
├── ingestion/      ingest_papers.py         → PDF → parse → chunk → DB
│                   pdf_parser.py            → docling/PyMuPDF parse + tables/equations
│                   document_chunker.py      → section/sentence/figure/algorithm chunking
│                   embed_chunks.py          → embed chunks (GPU/Gemini)
│                   incremental_index.py     → only changed PDFs
│                   ocr_fallback.py          → OCR for image-only PDFs
├── database/       create_schema.py · vector_migration.py · db_status.py · reset_*.py
├── graph_rag/      build_graph.py · retrieve_graph.py · fusion.py · memgraph_client.py
├── memory/         store.py                 → SQLite: sessions/turns/facts + answer_cache (WAL)
├── common/         embeddings.py            → provider dispatch (local bge / Gemini)
│                   device.py                → resolve_device (cuda/cpu/auto)
├── observability/  tracing.py               → Langfuse adapter (no-op when off)  (NEW)
├── evaluation/     evaluate_retrieval.py    → recall/latency report (json+md)
│                   evaluate_llm.py          → coverage · citation validity · judge
│                   corpus_report.py         → coverage/gaps report               (NEW)
├── auth/           users.py · google_oauth.py · mailer.py
└── config.py

webapp/
├── server.py       FastAPI app · routes · auth gate · SSE/NDJSON · GPU warmup (lifespan)
├── chat_logic.py   chat orchestration (the 10-stage pipeline)
├── settings.py     model catalog + selection
├── auth.py         session/login glue
├── ingest.py       PDF upload + live ingest progress
└── static/         index.html · app.js · styles.css   (the whole UI)
run.py              launcher (--share / --lan)
pipeline.py         build/refresh index · --status · --corpus-report · --inspect-chunks
```

---

## 4. The chat pipeline — `webapp/chat_logic.py :: stream_chat_events`

One request streams NDJSON events; each labeled stage is also a Langfuse span. **The run mode
(fast/deep) is applied first**, setting live env knobs for this request.

```mermaid
flowchart LR
    Q[question] --> SAN[query_sanity] --> MODE[apply_research_mode<br/>fast / deep]
    MODE --> CC{cache_check}
    CC -- hit --> OUT([stream cached answer])
    CC -- miss --> PLAN[plan angles<br/>_deep_queries]
    PLAN --> PAR
    subgraph PAR["per query — CONCURRENT"]
      LR[local_rag<br/>hybrid_retrieve]
      ES[external_search<br/>gather_external_evidence]
    end
    PAR --> SS[source_selection<br/>_public_sources] --> PB[prompt_build<br/>format_evidence]
    PB --> LS[llm_stream<br/>provider.stream_chat]
    LS --> CS[code_simulation<br/>run_best_python_block]
    CS --> AV{agentic_verify<br/>score >= MIN?}
    AV -- no, rounds left --> PB
    AV -- yes --> AR[auto_review<br/>+ relevance gate]
    AR --> CIT[citation guard<br/>strip out-of-range n]
    CIT --> MS[memory_save<br/>turn + answer cache] --> OUT
```

| Stage (span) | Does | Key code |
|---|---|---|
| `cache_check` | reuse a saved answer (lexical ≥0.97 or semantic ≥0.88) | `MemoryStore.find_cached_answer` |
| `local_rag` ‖ | hybrid retrieval over your PDFs | `hybrid_retrieve` |
| `external_search` ‖ | web/papers/GitHub (capped, partial‑safe) | `gather_external_evidence` |
| `source_selection` | number sources `[1..n]` | `_public_sources` |
| `prompt_build` | numbered evidence block, budget‑bounded | `format_evidence` |
| `llm_stream` | grounded answer (streamed) | `provider.stream_chat` |
| `code_simulation` | run any Python in the answer in Docker | `run_best_python_block` |
| `agentic_verify` | score answer vs evidence; loop if < `AGENTIC_MIN_VERIFY_SCORE` | `verify_answer` |
| `auto_review` | peer review + relevance gate (deep only) | `reviewer.review` |
| **citation guard** | strip `[n]` outside `[1..n_sources]` (saved + display) | `citations.repair_citations` |
| `memory_save` | persist turn + cache the clean body | `append_turn` / `cache_answer` |

**Fast vs Deep** (`research_modes.py`) sets these *live* per request — accuracy bar (`AGENTIC_MIN_VERIFY_SCORE=80`) is identical in both:

| knob | fast (default) | deep |
|---|---|---|
| sub‑queries · ext_top_k · web · arXiv reads | 0 · 8 · 4 · 0 | 3 · 20 · 8 · 3 |
| verify rounds · auto‑review | 1 · off | 3 · on |
| evidence chars · answer tokens · gather timeout | 14k · 3k · 12s | 28k · 8k · 30s |

---

## 5. The retrieval pipeline — `backend/retrieval/hybrid_retrieve.py`

```mermaid
flowchart LR
    Q[query] --> P
    subgraph P["CONCURRENT (thread pool)"]
      VO[vector: original<br/>Oracle VECTOR_DISTANCE]
      VH[vector: HyDE-expanded]
      BM[field-weighted BM25]
    end
    P --> RRF[Reciprocal Rank Fusion<br/>k=60] --> G{Memgraph<br/>expansion?}
    G --> RR[cross-encoder rerank<br/>bge-reranker-v2-m3 · GPU fp16] --> CB[chunk-type boost] --> MMR[MMR diversify<br/>λ=0.7 · per-paper cap] --> R[top-k]
```

- **3 independent rankings run concurrently** (thread‑safe embed lock), then **RRF** fuses them — fusion order preserved, so results are identical to sequential; only wall‑clock improves.
- **HyDE** (`hyde_generator.py`) expands the query into a hypothetical passage *without an LLM* (template + lexicon), then embeds it — a second recall angle.
- **Cross‑encoder rerank** (`BAAI/bge-reranker-v2-m3`) on **GPU in fp16** (`RERANKER_FP16`) — ~0.5s vs 5–37s on CPU. Pre‑warmed at startup (`hybrid_retrieve.warmup`).
- **MMR** diversifies + caps sources per paper; **chunk‑type boost** nudges equations/algorithms for matching intents.
- **Vector backend** is Oracle by default; `VECTOR_BACKEND=turbovec` uses a compressed 4‑bit local index (`turbovec_index.py`) with overfetch + exact re‑rank.
- **GraphRAG** (optional, `ENABLE_GRAPH_RAG`): seed the fused IDs into Memgraph and pull concept/section‑adjacent chunks (`graph_rag/`).

---

## 6. The ingestion pipeline — `pipeline.py` → `backend/ingestion/`

```mermaid
flowchart LR
    PDF[data/papers/*.pdf] --> PARSE[pdf_parser<br/>docling → PyMuPDF → OCR fallback]
    PARSE --> CHUNK[document_chunker<br/>section · sentence · figure · algorithm]
    CHUNK --> EMB[embed_chunks<br/>local bge / Gemini · GPU]
    EMB --> ORA[(Oracle: papers + chunks + VECTOR)]
    ORA --> TVB[turbovec build<br/>optional compressed index]
```

- **Chunking** is structure‑aware: canonical sections, sentence packing (`CHUNK_MAX_CHARS`, overlap), separate chunks for **figure captions** and **algorithm blocks**, plus `chunk_type` (`equation`/`algorithm`/`table`/`text`) and concept tags.
- **Incremental** (`--incremental`) re‑processes only changed PDFs.
- Inspect/verify: `pipeline.py --status` · `--corpus-report` · `--inspect-chunks <id>`.
- Upload from the UI (**＋ Add papers**) streams live progress via `webapp/ingest.py`.

---

## 7. External search — `backend/external_search/orchestrator.py`

Seven channels fetched **in parallel** with a **shared timeout**; slow channels return partial results (never block the gather), then everything is de‑duped and cross‑encoder re‑ranked.

| Channel | Source | Key/free |
|---|---|---|
| web | DuckDuckGo (default) / Tavily / Brave / SerpAPI | free / key |
| arXiv · Semantic Scholar · Wikipedia | scholarly APIs | free |
| patents | Google Patents (via web provider) | key |
| GitHub | repos + code (stars‑first) | free (token raises limits) |
| online PDFs | read PDFs surfaced by web/arXiv | free |

- **Page extraction:** Crawl4AI (JS‑rendered, BM25‑filtered **markdown** keyed to the query) when `EXTERNAL_USE_CRAWL4AI=true` and installed; **automatic BeautifulSoup fallback** otherwise.
- **Safety:** `base.py` enforces timeouts (`EXTERNAL_HTTP_TIMEOUT`), a 3 MB cap, retries, and disk caching. `is_safe_url` validates scheme/host (SSRF guard intentionally permissive by owner choice).

---

## 8. The LLM layer — `backend/llm/streaming_provider.py`

A single **OpenAI‑compatible** router. `PROVIDERS` + `CATALOG` map model names → endpoint/key, so the same `provider.stream_chat(...)` works across:

| Model | Cost | Env |
|---|---|---|
| Gemini 2.5 Flash | free | `GEMINI_API_KEY` |
| Mistral Large / Codestral | free | `MISTRAL_API_KEY` |
| GPT‑5.5 | paid | `OPENAI_CLOUD_KEY` |
| any local (Ollama, …) | free | `OPENAI_BASE_URL` |

`AGENT_MODEL` can route the code agent to a dedicated coder model. The chat model is picked live in the sidebar (`webapp/settings.py`).

---

## 9. The coding agent

**Two implementations**, both wrapping the same sandbox + reviewer:

**A. Hand‑rolled loop — `backend/agent/loop.py`** (default for `/api/agent`)
```mermaid
flowchart LR
    T[THINK<br/>_generate_code] --> H{pre_run<br/>policy gate} --> X[EXECUTE<br/>run_python in Docker] --> RF[REFLECT<br/>_reflect: relevant? correct? done?]
    RF -- not done --> T
    RF -- done / max iters --> BEST[best attempt → peer review → final]
```

**B. LangGraph multi‑agent — `backend/agent/graph.py`** (opt‑in `AGENT_GRAPH_ENABLED`)
```
planner → {fetcher_local ‖ fetcher_external} → coder → sandbox_runner
        → verifier (relevance ‖ citation ‖ code-run, concurrent) → grader
grader: score<MIN & round<MAX → back to planner ; else END
```
- Persistent **SqliteSaver** checkpoints (`data/agent_checkpoints.db`, WAL) → resumable; `CHECKPOINT_BACKEND` switch.
- Per‑task **budgets**: `AGENTIC_MAX_VERIFY_ROUNDS`, `MAX_TOKENS_PER_TASK`, per‑node timeout.
- Falls back to loop.py if LangGraph is unavailable.

**The Docker sandbox — `backend/agent/code_runner.py`** (never weakened): `--network none`, capped `--memory`/`--cpus`/`--pids-limit`, hard timeout, non‑root, auto‑removed; a scientific image (numpy/scipy/pandas/sklearn/sympy/…) is built on first run.

**Distributed (optional, `QUEUE_ENABLED`):** `/api/agent` enqueues to a **Celery/Redis** worker and returns a `task_id`; `/api/agent/{id}/stream` drains events from a Redis list (`task_channel.py`). If Redis is down → automatic in‑process streaming. Worker: `celery -A backend.agent.celery_app worker --pool=solo`.

---

## 10. Data layer

| Store | Tech | Holds |
|---|---|---|
| **Oracle 23ai** (`FREEPDB1`) | `oracledb` | `papers`, `chunks` (+ `VECTOR` column) — the searchable corpus |
| **turbovec** | `data/vector_cache/chunks.tvim` | optional compressed (4‑bit) vector index |
| **conversations.db** | SQLite WAL | sessions · turns · facts (chat history) |
| **memory.db** | SQLite WAL | `answer_cache` (lexical + semantic reuse) |
| **auth store** | SQLite | users / sessions (login) |
| **agent_checkpoints.db** | SQLite WAL | LangGraph checkpoints |
| **Memgraph** (optional) | Bolt / `neo4j` | GraphRAG concept graph |
| **data/external_cache/** | disk | external fetch cache (`EXTERNAL_CACHE_TTL`) |

`backend/memory/store.py` is the single SQLite interface; conversations are split from the cache via an ATTACH‑based one‑time migration (never deletes the old file).

---

## 11. Accuracy, trust & caching

- **Citations** (`answering/citations.py`): every `[n]` outside `[1..n_sources]` is stripped from the **saved answer** (server) and the **live display** (`app.js linkifyCitations`); grouped `[1, 9, 3]` → `[1, 3]`.
- **Verification** (`agentic_answer.py`): draft → score vs evidence → refine until `≥ AGENTIC_MIN_VERIFY_SCORE` or round cap; low‑confidence answers get a styled warning (no raw reviewer jargon).
- **Peer review + relevance gate** (`reviewer.py`): off‑topic answers/code are never marked "verified".
- **Answer cache**: exact + **semantic** (embedding) reuse with freshness guards; cached body is footer‑free + citation‑repaired.

---

## 12. GPU acceleration

- `backend/common/device.py :: resolve_device` → `cuda` when available (`DEVICE`/`EMBEDDING_DEVICE`/`RERANKER_DEVICE=auto`).
- **Reranker** (`bge-reranker-v2-m3`) and **local embedder** (`bge-base-en-v1.5`) load on the GPU; reranker runs **fp16** (`RERANKER_FP16`) — ~2× faster, half VRAM (~1.5 GB total, fits a 6 GB card).
- **Startup pre‑warm** (`hybrid_retrieve.warmup`, FastAPI `lifespan`, background thread) pays the ~14 s model load + CUDA init **once**, so the first user query is fast.
- Measured: retrieval **p50 ~10 s → ~3.2 s**, **p95 ~19.5 s → ~4.1 s** (the CPU reranker's 5–37 s variance is gone). CPU fallback is automatic.

---

## 13. Observability & evaluation

- **Langfuse tracing** (`observability/tracing.py`, off by default): one trace per chat + per agent run; spans for every stage with duration, success, counts, scores, verify rounds, and token‑cost estimate. **No‑op + zero overhead** when `LANGFUSE_ENABLED=false`; never blocks the chat. (`docs/OBSERVABILITY.md`.)
- **Evaluation** (`backend/evaluation/`):
  - `evaluate_retrieval` → term‑recall, recall@k, MRR, nDCG, latency (p50/p95) → JSON **+ markdown** report.
  - `evaluate_llm` → answer coverage, **citation validity**, optional LLM‑judge.
  - **DeepEval gates** (`tests/test_llm_quality.py`, opt‑in) → faithfulness / answer‑relevancy / contextual‑relevancy ≥ 0.7.
  - `corpus_report` → papers, chunks, topic coverage, gaps, duplicates, failed ingestions.

---

## 14. Frontend — `webapp/static/`

`index.html` + `app.js` + `styles.css`, **no build step**. Consumes the NDJSON stream (`fetch` + `ReadableStream`), renders markdown (marked.js) + code (highlight.js) + math (KaTeX). Features: multi‑session sidebar with restore, **clickable numbered citation chips** → source drawer, the **Fast/Deep toggle**, live "thinking/coding" agent cards, model picker, PDF upload with live ingest progress, light/dark theme.

---

## 15. Configuration map (`.env.example`, 16 sections)

| § | Area | Notable flags |
|---|---|---|
| 01 | Runtime | `DEBUG_MODE` |
| 02 | Auth | `ENABLE_AUTH` · `SESSION_MAX_AGE` · `CONVERSATIONS_DB_PATH` |
| 03 | DB / local RAG | `ENABLE_LOCAL_RAG` · `ORACLE_DSN` · `VECTOR_BACKEND` · `TURBOVEC_*` |
| 04 | Chat LLM | `OPENAI_MODEL` · `GEMINI/MISTRAL/OPENAI_CLOUD_KEY` |
| 05 | Embeddings / rerank | `EMBEDDING_PROVIDER` · `EMBEDDING_MODEL` · `RERANKER_MODEL` |
| 06 | External search | `ENABLE_WEB_SEARCH` · `WEB_SEARCH_PROVIDER` · keys |
| 07 | Answer quality | `AGENTIC_MAX_VERIFY_ROUNDS` · `AGENTIC_MIN_VERIFY_SCORE` · `AUTO_REVIEW` · cache |
| 08 | Search limits | `EXTERNAL_HTTP_TIMEOUT` · `EXTERNAL_USE_CRAWL4AI` · per‑channel caps |
| 09 | Code agent | `AGENT_MAX_ITERS` · sandbox `AGENT_MEM_LIMIT`/`CPUS`/`RUN_TIMEOUT` |
| 10 | GraphRAG | `ENABLE_GRAPH_RAG` · `MEMGRAPH_*` |
| 11 | Device / GPU | `DEVICE` · `EMBEDDING_DEVICE` · `RERANKER_DEVICE` · `RERANKER_FP16` |
| 12 | Ingestion | `ENABLE_OCR` · `CHUNK_MAX_CHARS` |
| 13 | Retrieval knobs | `RRF_K` · `MMR_LAMBDA` · `ENABLE_HYDE` |
| 14 | Sharing | `CLOUDFLARE_TUNNEL_*` |
| 15 | Observability/eval | `LANGFUSE_ENABLED` · `DEEPEVAL_ENABLED` |
| 16 | Distributed | `AGENT_GRAPH_ENABLED` · `QUEUE_ENABLED` · `REDIS_URL` · `CHECKPOINT_BACKEND` |

> **Run modes override** §07/§08 knobs per request (fast/deep); the values in `.env` are the fallback defaults for CLI/eval entry points.

---

## 16. Feature flags — default state

| Optional system | Flag | Default | Fallback when off/unavailable |
|---|---|---|---|
| Local PDF RAG | `ENABLE_LOCAL_RAG` | off | web‑only |
| turbovec | `VECTOR_BACKEND=turbovec` | Oracle | Oracle vectors |
| GraphRAG | `ENABLE_GRAPH_RAG` | off | no graph hits |
| Crawl4AI | `EXTERNAL_USE_CRAWL4AI` | on* | BeautifulSoup (*if installed) |
| GPU | `DEVICE=auto` | auto | CPU |
| Langfuse | `LANGFUSE_ENABLED` | off | no‑op |
| DeepEval gates | `DEEPEVAL_ENABLED` | off | tests skip |
| Multi‑agent graph | `AGENT_GRAPH_ENABLED` | off | hand‑rolled loop |
| Distributed queue | `QUEUE_ENABLED` | off | in‑process streaming |
| OCR | `ENABLE_OCR` | on | digital‑text parse only |

---

## 17. End‑to‑end request (deep mode, with a code task)

1. `POST /api/chat` `{question, mode:"deep", …}` → auth gate → `stream_chat_events`.
2. `apply_research_mode("deep")` sets live knobs; `query_sanity` checks the question.
3. `cache_check`: semantic/lexical reuse — if hit, stream and stop.
4. `_deep_queries` plans 3 angles; per angle, `local_rag` ‖ `external_search` run concurrently (Oracle+GPU rerank ‖ 7 web channels, capped).
5. `source_selection` numbers sources; `prompt_build` builds the bounded, cited evidence block.
6. `llm_stream` drafts a grounded answer; `code_simulation` runs any Python in Docker.
7. `agentic_verify` scores it vs the evidence; below `MIN` → search more + refine (≤ rounds).
8. `auto_review` + relevance gate; **citation guard** strips invalid `[n]`.
9. `memory_save` persists the turn and caches the clean body.
10. The whole flow is one Langfuse trace (latency, tokens, scores, rounds) when enabled.

---

_Generated as the engineering reference for this repository. Pair with the [README](../README.md)
(plain‑English), [OBSERVABILITY.md](OBSERVABILITY.md), and [INGESTION_CHECKLIST.md](INGESTION_CHECKLIST.md)._
