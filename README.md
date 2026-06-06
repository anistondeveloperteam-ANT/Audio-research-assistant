# Audio Research Assistant

A cited, source-grounded research assistant for audio / speech topics, served
through a fast interactive **web app** (FastAPI + a no-build front end).

- **Web search is the primary, production-ready knowledge source** — it searches
  the public web, GitHub repos/code, and online PDFs, reads them safely, and
  answers with citations (URL · file:line · page). **No database required.**
- **Local PDF RAG is optional** (`ENABLE_LOCAL_RAG=true`): a hybrid
  vector + BM25 retriever over your own PDFs in an Oracle vector DB. Off by
  default so the app deploys with no Oracle and no uploaded papers.

> **Quick start (web mode):** set `ENABLE_WEB_SEARCH=true` and a provider key
> (`TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` / `SERPAPI_API_KEY`) in `.env`, plus an
> LLM key (`OPENROUTER_API_KEY`), then `python run.py`. No Oracle/Docker needed.

## Two entry points

| File | What it does |
|------|--------------|
| **`python run.py`** | Launch the web app → http://localhost:8600 |
| **`python pipeline.py`** | Build / refresh the search index from `data/papers/`. `--incremental` for changed PDFs only. |

That's all most people need. Everything else lives under `backend/` and `webapp/`.

## Project structure

```
Audio-research-assistant/
├── run.py              # Launch the web app
├── pipeline.py         # Build / refresh the search index (ingest → embed → vector-migrate)
├── backend/
│   ├── config.py           # Central settings + data paths (reads .env)
│   ├── common/             # device (GPU/CPU), embeddings (Gemini / local)
│   ├── ingestion/          # pdf_parser, ocr_fallback, document_chunker,
│   │                       #   ingest_papers, embed_chunks, incremental_index
│   ├── retrieval/          # hybrid_retrieve, vector_retriever,
│   │                       #   retrieval_fusion, hyde_generator
│   ├── answering/          # research_modes, query_sanity
│   ├── llm/                # streaming_provider (all chat providers)
│   ├── database/           # vector_migration + DB admin scripts
│   ├── memory/             # store (conversation memory), memory_backup (import/export)
│   └── evaluation/         # evaluate_retrieval
├── webapp/             # The web UI — FastAPI server + static front end
│   ├── server.py           #   API routes + streaming chat (SSE)
│   ├── chat_logic.py       #   orchestrates retrieval + LLM + memory
│   ├── ingest.py           #   PDF upload + live ingestion
│   ├── settings.py         #   model switcher
│   └── static/             #   index.html, app.js, styles.css (no build step)
├── scripts/            # CLI maintenance tools (memory import/export, chat cleanup)
├── viewer_tool/        # show_my_data.py — inspect indexed data & memory
├── tests/              # pytest unit tests
├── data/               # Papers, extracted text, SQLite memory DB (gitignored)
├── docs/               # PIPELINE.md guide + reference PDF
├── CHANGELOG.md        # What changed and when (kept up to date)
├── .vscode/            # Editor + debugger configuration
├── requirements.txt    # Pinned Python dependencies
└── .env.example        # Environment configuration template
```

> The `backend` package uses absolute imports (`from backend.retrieval.hybrid_retrieve import …`).
> Run scripts from the project root, e.g. `python -m backend.database.create_schema`.

## Setup

### 1. Prerequisites
- Python 3.11+
- An Oracle database with vector support (e.g. Oracle 23ai / FREEPDB1)
- For answer generation: Ollama (local, free) **or** an OpenRouter key (one key → DeepSeek, Qwen, GPT, Claude, 300+)

### 2. Install dependencies
The repository already ships with a `.venv`. To use it as-is, just select it in
VSCode. To rebuild from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` pins the CPU build of PyTorch. For CUDA, install
> torch/torchaudio from the CUDA index first (see the comment in `requirements.txt`).

### 3. Configure environment
```powershell
copy .env.example .env
```
Then edit `.env` and fill in your Oracle credentials and any LLM API keys.
**`.env` is gitignored — never commit it.**

## Running

### From VSCode
Open the folder, pick the `.venv` interpreter, then use **Run and Debug**:
- **Web UI: run.py (FastAPI, :8600)**

### From the terminal
```powershell
# Check what is already indexed (no rebuild)
python pipeline.py --status

# First time / after adding PDFs — build the index
python pipeline.py

# Launch the web app
python run.py                      # http://localhost:8600
python run.py --port 9000          # optional: choose another local port
```
> The web app does not auto-open a browser - visit http://localhost:8600 yourself.
> The launcher binds to 127.0.0.1, so it is available only on this PC.

## Maintenance tools

| Command | Purpose |
|---------|---------|
| `python viewer_tool/show_my_data.py` | Inspect indexed PDFs, chunks, embeddings, memory |
| `python scripts/export_memory_cli.py` / `python scripts/import_memory_cli.py` | Back up / restore conversation memory |
| `python scripts/clean_bad_conversations.py` | Prune low-quality stored conversations |
| `python pipeline.py` | Full index rebuild |
| `python pipeline.py --incremental` | Index only changed PDFs |
| `python pipeline.py --status` | Show what's indexed (no rebuild) |
| `python -m backend.database.create_schema` | (Re)create the Oracle schema |
| `python -m backend.database.test_oracle` | Verify the Oracle connection |
| `python -m backend.database.db_status` | Show indexed papers / chunk counts |
| `python -m backend.evaluation.evaluate_retrieval` | Score retrieval quality |
| `python -m backend.evaluation.evaluate_llm` | Measure LLM answer accuracy (keypoint coverage + citations); `--models a,b` to compare, `--judge` for an LLM-graded score |

## External search (automatic — no toggle)

External search is **automatic** and searches **everywhere — with no API key**.
When the local papers don't answer a question (or local RAG is off), the assistant
automatically searches:

- **Web pages** — DuckDuckGo (free, default) or Tavily / Brave / SerpAPI
- **Research papers** — **arXiv** + **Semantic Scholar** (free; top papers' full
  PDFs are downloaded and read, not just abstracts)
- **Wikipedia** — background/encyclopedic (free)
- **Patents** — Google Patents (free)
- **GitHub** — repos / READMEs / code (free; a token enables code search)
- **Online PDFs** surfaced by the search

…reads them safely, re-ranks everything against your question, and answers with
citations (URL · file:line · page). No button to flip — it just falls back.

**Keys are optional** — it all works for free. A paid web key gives higher-quality
general-web results:
```
ENABLE_WEB_SEARCH=true               # on by default; set false to disable entirely
WEB_SEARCH_PROVIDER=duckduckgo       # free default; or tavily | brave | serpapi
TAVILY_API_KEY=optional_key          # higher-quality web (or BRAVE_/SERPAPI_)
GITHUB_TOKEN=optional_token          # raises GitHub limits + enables code search
```

**Behavior & citations**
- Source cards are tagged **Paper / Web / Research / Patent / GitHub / PDF** with
  the URL, file path + line range, and page number; answers cite every claim `[n]`.
- Local papers are preferred when they answer the question; external is the fallback.
- For code/algorithm questions the assistant explains the algorithm, cites the
  source, and writes **original** code in this project's style — it does **not**
  copy repository code; any license is shown.
- If a channel fails, the others continue and a non-blocking warning is shown.

**Security & limits**
- Keys are read **server-side only** — never sent to the browser or logged.
- SSRF-guarded: only `http(s)`, and any host resolving to localhost / private /
  link-local / cloud-metadata IPs is blocked; `file://` is rejected.
- Per-request timeouts, retries, and size caps; downloaded content is never executed.
- Fetches are cached under `data/external_cache/` (gitignored) with a TTL.

**Limitations:** respects provider/API terms and rate limits; results depend on the
chosen provider; very large pages/PDFs are truncated; code search needs a `GITHUB_TOKEN`.

## GPU / CPU

The embedding and reranker models pick their device from `.env`:

| Var | Meaning |
|-----|---------|
| `DEVICE` | global default — `auto` (GPU if available, else CPU), `cuda`, or `cpu` |
| `EMBEDDING_DEVICE` | override for the embedding model (falls back to `DEVICE`) |
| `RERANKER_DEVICE` | override for the reranker (falls back to `DEVICE`) |

The default config runs **embeddings on the GPU and the heavier reranker on the CPU**,
which avoids out-of-memory on a small GPU (e.g. a 6 GB laptop card shared with a local
Ollama LLM). If you have VRAM to spare (or use a cloud LLM), set `RERANKER_DEVICE=cuda`.

## Development & tests

The dev/test tools (pytest, pyflakes, vulture) are included at the bottom of
`requirements.txt`, so a normal install already has them. Run the fast unit suite
(no DB / models / network needed):

```powershell
pytest                          # tests/ — retrieval fusion, query sanity,
                                #   device selection, env helpers, etc.
pyflakes backend webapp         # lint: unused imports / undefined names
vulture backend webapp          # find dead code
```

## Notes
- Large/generated artifacts (`data/papers`, `data/extracted`, `*.db`, `.venv`) are
  gitignored. Source PDFs and the index are machine-local.
