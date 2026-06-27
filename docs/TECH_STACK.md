# Technology & Tools — Detailed Reference

Every tool and technology this project uses, **as of now** — with the version, what it is, the
exact role it plays here, and where it lives in the code. Pinned versions come from
`requirements.txt` (the versions verified working in the project's `.venv`).

> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) (how the pieces fit into pipelines),
> [README.md](../README.md) (plain‑English overview).

**Conventions:** 🟢 = always used · 🔵 = used in a specific mode/config · ⚪ = optional, off by default.

---

## 0. Platform & language

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **Python** | 3.11+ | The language the whole backend is written in | Everything in `backend/` and `webapp/` |
| 🟢 **Git** | any | Version control | Branching, history, the commit workflow |
| 🟢 **Windows / PowerShell + Bash** | — | The dev OS + shells | Run/build commands; paths use `pathlib` for portability |

---

## 1. Web server & app framework

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **FastAPI** | 0.136.1 | Modern async Python web framework | Defines every HTTP route + the streaming chat API (`webapp/server.py`) |
| 🟢 **Uvicorn** | 0.46.0 | ASGI web server | Actually serves the app — `python run.py` → `http://localhost:8600` |
| 🟢 **Starlette `SessionMiddleware`** | (with FastAPI) | Signed‑cookie sessions | Login state / "stay signed in" |
| 🟢 **itsdangerous** | 2.2.0 | Cryptographic signing | Signs the session cookie so it can't be forged |
| 🟢 **python‑multipart** | 0.0.28 | Multipart form parsing | Handles **PDF file uploads** ("Add papers") |
| 🟢 **python‑dotenv** | 1.2.2 | Loads `.env` into env vars | All configuration is read from `.env` |

**How requests stream:** the chat API returns **NDJSON** (newline‑delimited JSON) over a
`StreamingResponse`, so the browser shows the answer token‑by‑token. The served `index.html` is
**cache‑busted** (`?v=<file‑mtime>` on `app.js`/`styles.css`) so UI updates load without a hard refresh.

---

## 2. Front end (no build step)

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **HTML / CSS / vanilla JavaScript** | — | The entire UI, hand‑written, **no framework, no bundler** | `webapp/static/{index.html, app.js, styles.css}` |
| 🟢 **marked.js** | CDN | Markdown → HTML | Renders the answer markdown |
| 🟢 **highlight.js** | 11.9.0 (CDN) | Syntax highlighting | Colours code blocks in answers |
| 🟢 **KaTeX** | 0.16.11 (CDN) | Math typesetting | Renders LaTeX math (`$…$`, `$$…$$`) |
| 🟢 **Google Fonts (Inter)** | CDN | Typeface | The UI font |

The UI consumes the stream with `fetch` + `ReadableStream`. Features: streaming cited answers,
**colour‑coded citations** (🟢 local paper / 🔴 web → opens in a new tab), a source drawer,
multi‑session sidebar, Fast/Deep toggle, live "thinking/coding" cards, model picker, PDF upload
with progress, light/dark theme.

---

## 3. Numerics

| Tool | Version | Role here |
|---|---|---|
| 🟢 **NumPy** | 2.3.5 | Vector math, similarity, array ops throughout retrieval |
| 🟢 **SciPy** | 1.17.1 | Scientific routines (scoring, distances) |
| 🟢 **pandas** | 3.0.3 | Tabular data handling (evaluation, reports) |

---

## 4. Machine learning — retrieval engine (the "accuracy" stack)

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **PyTorch** | 2.7.1 (+cu126 for GPU) | Deep‑learning tensor framework | Runs the reranker (and the local embedder, if enabled) — **on the GPU in fp16** |
| 🟢 **sentence‑transformers** | 5.5.0 | High‑level wrapper for embedding + cross‑encoder models | Loads & runs the **cross‑encoder reranker** (and the optional local embedder) |
| 🟢 **transformers** | 4.57.6 | Hugging Face model backbone | The model architecture under sentence‑transformers |
| 🟢 **BAAI/bge‑reranker‑v2‑m3** | model | A **cross‑encoder reranker** | Re‑orders search hits by true query↔document relevance — the key accuracy step. Runs on GPU **fp16** (`RERANKER_FP16=true`), pre‑warmed at startup |
| 🟢 **BAAI/bge‑large‑en‑v1.5** | model | The local embedding model (1024‑d) | **The only** embedder — turns chunks + queries into vectors on the GPU (fp16, CPU fallback). Asymmetric: a query instruction is prepended to queries only |
| 🟢 **CUDA / NVIDIA GPU** | (driver) | GPU compute | Auto‑detected (`DEVICE=auto`); reranker runs here. No GPU → automatic CPU fallback |

> **Why fp16 + pre‑warm:** half precision makes the reranker ~2× faster at half the VRAM (fits a
> 6 GB laptop card), and pre‑warming pays the model‑load + CUDA‑init cost once at startup, so the
> first query isn't slow. Measured retrieval p50 dropped from ~10 s to ~3 s.

Code: `backend/retrieval/hybrid_retrieve.py`, `backend/common/device.py`, `backend/common/embeddings.py`.

---

## 5. Search & ranking algorithms (in‑house, no extra deps)

| Technique | What it is | Role here |
|---|---|---|
| 🟢 **Vector / semantic search** | Nearest‑neighbour over embeddings | Find PDF chunks by *meaning* |
| 🟢 **Contextual Retrieval** (Anthropic‑style) | LLM‑written situating sentence per chunk, prepended for indexing only | Better recall for context‑poor chunks; original text kept for citations — `contextualizer.py` (`CONTEXTUAL_CHUNKS`) |
| 🟢 **HyDE** (Hypothetical Document Embeddings) | Search with a generated hypothetical answer | A second recall angle (template‑based, no LLM call) — `hyde_generator.py` |
| 🟢 **BM25** (field‑weighted) | Classic keyword relevance | Keyword matching, fused with vectors — `retrieval_fusion.py` |
| 🟢 **RRF** (Reciprocal Rank Fusion) | Merge several ranked lists robustly | Combine vector + HyDE + BM25 |
| 🟢 **MMR** (Maximal Marginal Relevance) | Relevance + diversity | Drop near‑duplicates, cap chunks per paper |
| 🟢 **Cross‑encoder rerank** | (see §4) | The final ordering by true relevance |

---

## 6. Databases & storage

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **Oracle Database 23ai** | (Docker image) | Relational DB with native **`VECTOR`** type + `VECTOR_DISTANCE` | Stores `PAPERS` + `CHUNKS` (text + embeddings) — your searchable corpus (`FREEPDB1`, port 1521) |
| 🟢 **oracledb** | 4.0.0 | Python driver for Oracle | Connects + runs the vector/SQL queries |
| 🔵 **turbovec** | 0.7.0 | Compressed (4‑bit) local vector index | Fast local vector search when `VECTOR_BACKEND=turbovec` (the default in this setup); overfetch + exact re‑rank. File: `data/vector_cache/chunks.tvim` |
| 🟢 **SQLite** | (Python stdlib) | Embedded file database (WAL mode) | All app state — see the table below |
| 🟢 **Docker** | host | Container runtime | (1) runs the Oracle DB container; (2) the **code sandbox** |

**The SQLite files (`data/`):**

| File | Holds |
|---|---|
| `conversations.db` | chat history (sessions, turns) — reloads on reopen |
| `memory.db` | the answer cache (reused answers) |
| `auth.db` | login accounts |
| `llm_costs.db` | optional cost tracking |
| `logs/agent_audit.jsonl` | a record of each coding‑agent run (text log, not a DB) |

Code: `backend/memory/store.py` (the single SQLite interface), `backend/database/` (Oracle schema/tools).

---

## 7. Embeddings provider

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **google‑genai** | 1.75.0 | Google's Gemini SDK | Chat model client (`GEMINI_API_KEY`). NOT used for embeddings anymore — those are local bge |
| 🟢 **local bge embedder** | `bge-large-en-v1.5` via sentence‑transformers | On‑device embeddings | The only embedder — 1024‑d vectors for indexing + queries on the GPU (needs a matching re‑index when the model changes) |

---

## 8. PDF parsing & ingestion

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **docling** | 2.93.0 | High‑quality document parser (layout, tables, structure) | Primary PDF → structured text |
| 🟢 **PyMuPDF** | 1.27.2.3 | Fast PDF library | Fallback parser when docling is overkill/fails |
| 🟢 **pypdf** | 6.11.0 | Pure‑Python PDF toolkit | Page handling / fallback |
| ⚪ **PaddleOCR / paddlepaddle / pytesseract** | (commented) | OCR engines | Optional — only for scanned/image‑only PDFs (`ENABLE_OCR`); large downloads, off by default |

Chunking is structure‑aware (sections, sentences, figure captions, algorithm blocks).
Code: `backend/ingestion/`. Run with `python pipeline.py` (`--incremental`, `--status`,
`--corpus-report`, `--inspect-chunks`).

---

## 9. The chat LLM (the answer writer)

| Tool | Version | What it is | Role here |
|---|---|---|---|
| 🟢 **openai SDK** | 1.109.1 | OpenAI‑compatible client | One client that talks to **any** OpenAI‑style endpoint |

The same `provider.stream_chat(...)` works across providers — you pick the model in the sidebar:

| Provider / model | Cost | Env key |
|---|---|---|
| **Google Gemini 2.5 Flash** | free | `GEMINI_API_KEY` |
| **Mistral** (Large, Codestral) | free | `MISTRAL_API_KEY` |
| **OpenAI GPT** | paid | `OPENAI_CLOUD_KEY` |
| **Local (Ollama, etc.)** | free | `OPENAI_BASE_URL` |

Code: `backend/llm/streaming_provider.py`.

---

## 10. External search (the web)

| Tool / source | What it is | Role here |
|---|---|---|
| 🟢 **requests** (2.34.1) | HTTP client | Fetches web pages, PDFs, API calls |
| 🟢 **BeautifulSoup4** (4.14.3) | HTML parser | Extracts the readable text from web pages |
| 🟢 **DuckDuckGo** | web search | Default web provider (no key needed) |
| 🔵 **Tavily / Brave / SerpAPI** | web search | Optional alternative providers (need a key) |
| 🟢 **arXiv · Semantic Scholar · Wikipedia** | scholarly APIs | Paper search (free) |
| 🟢 **GitHub API** | code search | Repos + code, most‑starred first (token raises limits) |
| 🟢 **Google Patents** | patent search | Via the web provider |

Channels run **in parallel** with a shared timeout (partial results never block). Code:
`backend/external_search/` (`orchestrator.py`, `web_search.py`, `scholar_search.py`, `github_search.py`, `pdf_reader.py`, `base.py`).

---

## 11. The coding agent & its sandbox

| Tool | What it is | Role here |
|---|---|---|
| 🟢 **Docker** | Container runtime | Runs generated Python in an isolated container: **`--network none`**, capped CPU/RAM/PIDs, hard timeout, non‑root, auto‑removed |
| 🟢 **Scientific image** (numpy/scipy/pandas/scikit‑learn/sympy/…) | Prebuilt sandbox image | The libraries the agent's code can use; built on first run |

The agent loop (THINK → run in sandbox → REFLECT → repeat) lives in `backend/agent/loop.py`;
the sandbox runner is `backend/agent/code_runner.py`; a pre‑run policy gate is `backend/agent/hooks.py`.
Runs are **saved to the chat** so the code + output reload after reopen.

---

## 12. Observability & evaluation (optional, off by default)

| Tool | Version | What it is | Role here |
|---|---|---|---|
| ⚪ **Langfuse** | 4.7.1 | LLM tracing (OpenTelemetry) | Per‑request traces: latency, token cost, retrieval quality, verify rounds. `LANGFUSE_ENABLED=true`; **no‑op + zero overhead when off**. Code: `backend/observability/tracing.py` |
| ⚪ **protobuf** | 5.29.5 (pinned) | Serialization lib | **Pinned** because Langfuse's OpenTelemetry deps otherwise pull protobuf 6.x, which crashes torch model loading on Windows |
| ⚪ **DeepEval** | 4.0.6 | LLM‑quality test framework | Faithfulness / answer‑relevancy / contextual‑relevancy gates (`tests/test_llm_quality.py`); opt‑in via `DEEPEVAL_ENABLED` |
| 🟢 **Custom evals** (in‑house) | — | Retrieval + answer metrics | `evaluate_retrieval.py` (recall@k, MRR, nDCG, latency), `evaluate_llm.py` (coverage, citation validity), `corpus_report.py` (coverage/gaps) |

---

## 13. Dev / test tooling

| Tool | Version | Role here |
|---|---|---|
| 🟢 **pytest** | 9.0.3 | The test runner — **176 tests**, all offline/mocked |
| 🟢 **pyflakes** | 3.4.0 | Lint: unused imports / undefined names |
| 🟢 **vulture** | 2.16 | Finds dead code |
| 🟢 **tqdm** | 4.67.3 | Progress bars (ingestion, batch jobs) |
| 🟢 **Node.js** (`node --check`) | optional | Quick JS syntax check for `app.js` |

---

## 14. Removed / NOT used (so the stack is unambiguous)

These were evaluated or previously present and **deliberately removed** to keep a lean,
conflict‑free, production dependency tree. If you see them referenced in old notes, they're gone:

| Removed | Was for | Replaced by / why removed |
|---|---|---|
| **Crawl4AI** + Playwright | JS‑rendered page scraping | **BeautifulSoup** (its `litellm` fork conflicted with the pinned `openai`, and it needed a browser install) |
| **neo4j / Memgraph** | GraphRAG concept graph | Removed — unused, needed a separate graph server |
| **LangGraph** (+ checkpoint‑sqlite) | multi‑agent graph + a 2nd research engine | Removed — the in‑process `loop.py` agent is the single path |
| **Celery + Redis** | distributed task queue | Removed — over‑engineering for a local single‑user app |

`pip check` is clean (no version conflicts) after these removals.

---

## 15. At a glance — the whole stack in one list

**Backend:** Python 3.11 · FastAPI · Uvicorn · python‑dotenv · itsdangerous · python‑multipart
**Retrieval/ML:** PyTorch (CUDA, fp16) · sentence‑transformers · transformers · bge‑reranker‑v2‑m3 · NumPy/SciPy/pandas · BM25 · HyDE · RRF · MMR
**Vectors/DB:** Oracle 23ai (`oracledb`) · turbovec · SQLite
**Embeddings:** Google Gemini (`google‑genai`) [local bge optional]
**PDF:** docling · PyMuPDF · pypdf [PaddleOCR/pytesseract optional]
**LLM:** openai SDK → Gemini / Mistral / GPT / Ollama
**Web search:** requests · BeautifulSoup · DuckDuckGo [Tavily/Brave/SerpAPI optional] · arXiv · Semantic Scholar · Wikipedia · GitHub · Patents
**Sandbox:** Docker
**Front end:** HTML/CSS/JS (no build) · marked.js · highlight.js · KaTeX
**Observability/eval (optional):** Langfuse · DeepEval · custom evals
**Dev:** pytest · pyflakes · vulture · tqdm

---

_Accurate to the project's current state. Versions are pinned in `requirements.txt`._
