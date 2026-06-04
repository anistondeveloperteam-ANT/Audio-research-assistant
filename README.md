# Audio Research Assistant

A Retrieval-Augmented Generation (RAG) assistant for audio / speech-enhancement
research papers. It ingests PDFs, builds a hybrid (vector + BM25) retrieval index
backed by an Oracle vector database, and answers questions with cited sources
through a Streamlit chat interface.

## Two entry points

| File | What it does |
|------|--------------|
| **`python run.py`** | Launch the app (Chat UI on :8502). `--market` for the Market UI on :8501. |
| **`python pipeline.py`** | Build / refresh the search index from `data/papers/`. `--incremental` for changed PDFs only. |

That's all most people need. Everything else lives under `backend/`, organized by job.

## Project structure

```
Audio-research-assistant/
├── run.py              # Launch the web app
├── pipeline.py         # Build / refresh the search index (ingest → embed → vector-migrate)
├── backend/
│   ├── config.py           # Central settings + data paths (reads .env)
│   ├── common/             # logger_config
│   ├── ingestion/          # pdf_parser, ocr_fallback, document_chunker,
│   │                       #   ingest_papers, embed_chunks, incremental_index
│   ├── retrieval/          # hybrid_retrieve, vector_retriever, retrieval_fusion,
│   │                       #   query_planner, hyde_generator, multi_query_retrieve, ...
│   ├── answering/          # answer_orchestrator, manual_answer_engine, evidence_builder,
│   │                       #   prompt_builder, prompt_quality, research_modes, query_sanity, ...
│   ├── llm/                # provider, multi_provider, router, cost_tracker
│   ├── database/           # oracle_db, create_schema, create_user, vector_migration,
│   │                       #   reset_index, reset_embeddings, inspect_schema, db_status
│   ├── memory/             # store (conversation memory), memory_io (import/export)
│   ├── tools/              # web_search, code_executor, sandbox_runner, dsp_toolkit
│   └── evaluation/         # evaluate_retrieval, quality_test
├── frontend/           # Streamlit UIs (chat_ui.py :8502, streamlit_app.py :8501) + helpers
├── scripts/            # CLI maintenance tools (memory import/export, chat cleanup)
├── viewer_tool/        # show_my_data.py — inspect indexed data & memory
├── data/               # Papers, extracted text, SQLite memory/cost DBs (gitignored)
├── docs/               # Reference material (pipeline PDF)
├── .streamlit/         # Streamlit theme/config
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
- Optional: Ollama, an OpenAI key, or an Anthropic key for answer generation

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
- **Streamlit: Chat UI (port 8502)**
- **Streamlit: Market UI (port 8501)**

### From the terminal
```powershell
# First time / after adding PDFs — build the index
python pipeline.py

# Launch the app
python run.py                      # Chat UI    -> http://localhost:8502
python run.py --market             # Market UI   -> http://localhost:8501
```

## Maintenance tools

| Command | Purpose |
|---------|---------|
| `python viewer_tool/show_my_data.py` | Inspect indexed PDFs, chunks, embeddings, memory |
| `python scripts/export_memory_cli.py` / `python scripts/import_memory_cli.py` | Back up / restore conversation memory |
| `python scripts/clean_bad_conversations.py` | Prune low-quality stored conversations |
| `python pipeline.py` | Full index rebuild |
| `python pipeline.py --incremental` | Index only changed PDFs |
| `python -m backend.database.create_schema` | (Re)create the Oracle schema |
| `python -m backend.database.test_oracle` | Verify the Oracle connection |
| `python -m backend.database.db_status` | Show indexed papers / chunk counts |
| `python -m backend.evaluation.evaluate_retrieval` | Score retrieval quality |

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

## Notes
- Large/generated artifacts (`data/papers`, `data/extracted`, `*.db`, `.venv`) are
  gitignored. Source PDFs and the index are machine-local.
