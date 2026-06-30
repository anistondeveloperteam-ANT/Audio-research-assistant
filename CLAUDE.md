# CLAUDE.md — Research Assistant

## Project
Python 3.11 RAG research assistant. FastAPI + Uvicorn web app (`webapp/`), backend logic in `backend/`, tests in `tests/`. Local PDF RAG (Oracle 23ai vectors), external search (web/arXiv/Semantic Scholar/Wikipedia/patents/GitHub), agentic answering with citations, code agent with Docker sandbox.

## Key folders
- `backend/agent/` — code-writing agent + Docker sandbox runner
- `backend/answering/` — answer drafting, verification, reviewer logic
- `backend/retrieval/` — hybrid local retrieval (vector + BM25 + RRF + rerank)
- `backend/external_search/` — web, papers, patents, GitHub search
- `backend/llm/` — LLM provider interface (`streaming_provider.py`)
- `backend/ingestion/` — PDF parsing, chunking, embedding, indexing

## Code quality rules (non-negotiable)
1. Any code you write MUST run without errors before you finish. Run it. Do not present untested code.
2. After every change, run: `python -m pytest -q`
3. After every change, run: `pyflakes backend webapp`
4. If tests or lint fail, FIX the failures and re-run until clean. Do not stop at the first attempt.
5. Write or update a unit test for every new function you add.
6. Prefer small, focused diffs. Do not refactor unrelated code.
7. Match the existing code style of the file you are editing.

## Commands
- Run app: `python run.py` (serves at http://localhost:8600)
- Build PDF index: `python pipeline.py` (incremental: `--incremental`)
- Tests: `python -m pytest -q`
- Lint: `pyflakes backend webapp`

## Hard limits
- NEVER read, print, edit, or commit `.env` or any API keys/secrets.
- NEVER weaken the Docker sandbox limits (network-off, CPU/mem caps, timeout).
- Generated Python for users must only run inside the Docker sandbox, never on the host.
- External search has NO SSRF/private-IP URL filter — this is intentional (the owner wants
  unrestricted fetching so search reaches anywhere). Do not re-add the guard unless asked.
- Do not add new dependencies without asking first.

## When asked to build a feature
1. Plan first: list the files you will touch and why. Wait for my approval on big changes.
2. Implement → run tests → run lint → fix → re-run until everything passes.
3. Show a summary of what changed and the passing test output.
