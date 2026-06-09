# Project Structure

This repo is organized by product area, not by file type. Keep new code close to
the workflow it belongs to, and avoid creating broad folders such as `utils/` or
`misc/` unless there is a very specific shared responsibility.

## Root

```text
Audio-research-assistant/
|-- run.py                  # local FastAPI launcher
|-- pipeline.py             # local PDF indexing pipeline
|-- requirements.txt        # Python dependencies
|-- pytest.ini              # test config
|-- .env.example            # safe config template; real .env stays private
|-- README.md               # user-facing setup and usage
|-- CHANGELOG.md            # release/change history
|-- backend/                # server-side application code
|-- webapp/                 # FastAPI app and browser UI
|-- tests/                  # pytest suite
|-- scripts/                # operator/admin command-line tools
|-- docs/                   # architecture, pipeline, and project guides
|-- data/                   # local runtime data, caches, papers, logs
```

## Backend Packages

```text
backend/
|-- agent/             # code-writing agent loop, sandbox runner, hooks, memory
|-- answering/         # answer drafting, verification, reviewer, query checks
|-- auth/              # user account and password helpers
|-- common/            # shared runtime helpers with clear ownership
|-- database/          # Oracle setup, reset, inspection, migration scripts
|-- evaluation/        # retrieval and LLM evaluation runners
|-- external_search/   # web, papers, patents, GitHub, online PDF search
|-- graph_rag/         # optional Memgraph graph expansion
|-- ingestion/         # PDF parsing, chunking, embedding, indexing
|-- llm/               # chat provider interface: OpenAI/OpenRouter
|-- memory/            # SQLite conversation memory and backups
|-- retrieval/         # local hybrid retrieval, fusion, HyDE, vector search
```

## Naming Rules

- Python packages and modules use `lower_snake_case`.
- Tests use `test_<area>.py` and should mirror the feature they protect.
- Scripts use verb-first names: `show_data.py`, `export_memory_cli.py`.
- Docs use clear topic names: `PIPELINE.md`, `TECH_STACK.md`,
  `KIMI_CODE_ADOPTION.md`.
- Runtime/generated folders stay under `data/` and should be ignored by Git.
- Do not commit `.env`, local database files, caches, or downloaded model/tool
  artifacts.

## Where New Work Goes

| New work | Put it here |
|----------|-------------|
| New LLM provider | `backend/llm/` plus `webapp/settings.py` |
| New web/search source | `backend/external_search/` |
| New retrieval ranking logic | `backend/retrieval/` |
| New answer verification behavior | `backend/answering/` |
| New sandbox/agent behavior | `backend/agent/` |
| New UI behavior | `webapp/static/` and/or `webapp/server.py` |
| New admin command | `scripts/` |
| New test | `tests/test_<area>.py` |
| New architecture note | `docs/` |
