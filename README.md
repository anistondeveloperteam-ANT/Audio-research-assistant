# Audio Research Assistant

A Retrieval-Augmented Generation (RAG) assistant for audio / speech-enhancement
research papers. It ingests PDFs, builds a hybrid (vector + BM25) retrieval index
backed by an Oracle vector database, and answers questions with cited sources
through a Streamlit chat interface.

## Project structure

```
Audio-research-assistant/
├── backend/            # Retrieval engine, ingestion, LLM providers, DB access
├── frontend/           # Streamlit UIs (chat_ui.py, streamlit_app.py) + helpers
├── scripts/            # CLI maintenance tools (memory import/export, cleanup)
├── viewer_tool/        # show_my_data.py — inspect indexed data & memory
├── data/               # Papers, extracted text, SQLite memory/cost DBs (gitignored)
├── docs/               # Reference material (pipeline PDF)
├── .streamlit/         # Streamlit theme/config
├── .vscode/            # Editor + debugger configuration
├── requirements.txt    # Pinned Python dependencies
└── .env.example        # Environment configuration template
```

### Key components

| Area | Modules |
|------|---------|
| Ingestion | `backend/ingest_papers.py`, `incremental_index.py`, `parsers.py`, `advanced_chunker.py`, `ocr_fallback.py` |
| Embedding / index | `backend/embed_chunks.py`, `create_schema.py`, `oracle_vector_migration.py` |
| Retrieval | `backend/hybrid_retrieve.py`, `oracle_vector_retriever.py`, `retrieval_fusion.py`, `query_planner.py`, `hyde_generator.py` |
| Answering | `backend/answer_orchestrator.py`, `manual_answer_engine.py`, `evidence_builder.py`, `llm_providers.py`, `llm_router.py` |
| Memory / cost | `backend/memory.py`, `memory_io.py`, `cost_tracker.py` |
| UI | `frontend/chat_ui.py` (chat, port 8502), `frontend/streamlit_app.py` (market UI, port 8501) |

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
# Chat UI
.\run_chat_ui.bat
# or
.venv\Scripts\python.exe -m streamlit run frontend\chat_ui.py --server.port 8502

# Market UI
.\run_market_ui.bat
```

## Maintenance tools

| Command | Purpose |
|---------|---------|
| `.\show_my_data.bat` | Inspect indexed PDFs, chunks, embeddings, memory |
| `.\export_memory.bat` / `.\import_memory.bat` | Back up / restore conversation memory |
| `.\clean_bad_chats.bat` | Prune low-quality stored conversations |
| `python backend\ingest_papers.py` | Ingest / re-index PDFs in `data/papers/` |
| `python backend\incremental_index.py` | Index only changed PDFs |

## Notes
- Large/generated artifacts (`data/papers`, `data/extracted`, `*.db`, `.venv`) are
  gitignored. Source PDFs and the index are machine-local.
