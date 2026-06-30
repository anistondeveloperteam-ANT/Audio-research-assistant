# Operator & Developer Tools

These are **standalone command-line tools** — run by a human, not imported by the app. They are
intentionally not part of the web runtime, so they can be safely excluded from a deployed image
(see `.dockerignore`). Run them from the repo root with the project virtualenv active.

> A static "is it imported anywhere?" scan flags these as unused — they are **not** dead code. They
> are setup, maintenance, and diagnostic utilities you run by hand (and some are needed to bring up a
> fresh database on a new deployment).

---

## Database setup & admin — `backend/database/`

Needed to bootstrap and operate the Oracle 23ai store.

| Tool | What it does |
|---|---|
| `create_user.py` | Creates the Oracle user/schema owner the app connects as. **Run once on a fresh DB.** |
| `create_schema.py` | Creates the `papers` / `chunks` tables and the native `VECTOR` column. **Run once on a fresh DB.** |
| `check_oracle.py` | Quick connectivity probe — confirms the app can reach Oracle with the `.env` credentials. |
| `db_status.py` | Prints row counts and index status (how many papers/chunks/vectors are indexed). |
| `vector_migration.py` | Copies JSON-CLOB embeddings into the native `VECTOR` column (auto-sizes to the model dimension). Also runs inside `pipeline.py`. |
| `reset_index.py` | Clears indexed papers/chunks from Oracle (requires `--yes`). Destructive. |
| `reset_embeddings.py` | Drops the `VECTOR` column and NULLs embeddings so they can be re-embedded (used by `pipeline.py --reembed`). Destructive. |

### Fresh-deploy database recipe

```bash
# 1) Oracle 23ai reachable (Docker: container oracle-ai-db, service FREEPDB1:1521)
# 2) .env has ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN set
python -m backend.database.create_user      # once: create the app's DB user
python -m backend.database.create_schema    # once: create papers/chunks + VECTOR column
python -m backend.database.check_oracle      # verify connectivity
python pipeline.py                           # index your PDFs (parse -> embed -> migrate)
python -m backend.database.db_status         # confirm the corpus is indexed
```

---

## Maintenance & inspection CLIs — `scripts/`

| Tool | What it does |
|---|---|
| `show_accounts.py` | List registered user accounts (when auth is enabled). |
| `show_data.py` | Inspect what's stored — papers, chunks, sample rows. |
| `show_agent_patterns.py` | Show the code agent's learned failure/success patterns. |
| `suggest_parameters.py` | Suggest tuned retrieval/answering parameters from recorded outcomes. |
| `export_memory_cli.py` / `import_memory_cli.py` | Export / import the conversation + learned-knowledge memory (backup or move between machines). |
| `clean_bad_conversations.py` | Prune malformed/empty conversations from the SQLite store. |
