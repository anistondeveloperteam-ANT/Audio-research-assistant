---
paths:
  - "backend/**/*.py"
  - "webapp/**/*.py"
  - "tests/**/*.py"
  - "run.py"
  - "pipeline.py"
---

# Python Rules

- Use Python 3.11-compatible code.
- Prefer small pure helpers and typed public functions.
- Keep imports lazy when a feature is optional or heavy, especially Oracle,
  Memgraph, Torch, rerankers, and web-search providers.
- Optional systems must fail closed or degrade gracefully:
  - local RAG off -> app still works
  - Memgraph down -> GraphRAG returns no hits
  - web search fails -> local evidence still works
- Do not import `.env` values into module constants if they must reflect live
  settings during tests or long-running app sessions.
- Networked tests must mock network calls.
- Database/model tests must be optional or mocked unless explicitly requested.
- Use `pathlib` for filesystem paths and keep Windows behavior in mind.
