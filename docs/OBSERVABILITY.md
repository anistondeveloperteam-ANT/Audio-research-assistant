# Observability — LLM tracing with Langfuse

The app can emit a [Langfuse](https://langfuse.com) trace for every chat request and every
code‑agent run, so you can see — per pipeline stage — how long it took, whether it
succeeded, and small metadata (source counts, verification scores, cache hits).

**It is off by default and fully optional.** With `LANGFUSE_ENABLED=false` (the default)
the app behaves byte‑identically to having no tracing at all: no client is created, the
`langfuse` package is never imported, nothing is timed, and nothing is sent. Every tracing
call is also wrapped in `try/except`, so a missing package or an unreachable server can
never crash or slow the chat flow.

---

## What gets traced

**Chat request** (`webapp/chat_logic.py`) — one trace, one span per stage:

`cache_check → local_rag → external_search → source_selection → prompt_build →
llm_stream → code_simulation → agentic_verify → auto_review → memory_save`

**Code agent** (`backend/agent/loop.py`) — one trace per run, four spans per iteration:

`generate → prerun_hook → docker_run → reflect`

### Privacy
Spans record only **durations, success/failure, and small metadata** (counts, scores,
booleans, and short summaries truncated to 500 chars). They **never** carry the user's
question, the answer text, source contents, or any API key.

---

## 1. Self‑host Langfuse (Docker Compose)

Langfuse ships an official compose stack. In a separate folder (not this repo):

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d          # starts Langfuse + its Postgres/ClickHouse/Redis/MinIO
```

Wait ~30s, then open **http://localhost:3000** and create an account (the first user is the
owner). For production, review and change the secrets in their `docker-compose.yml`
(`NEXTAUTH_SECRET`, `SALT`, database passwords) per the
[self‑hosting guide](https://langfuse.com/self-hosting).

### Get your API keys
In the Langfuse UI: create (or open) a **Project → Settings → API Keys → Create**. Copy the
**Public Key** (`pk-lf-…`) and **Secret Key** (`sk-lf-…`).

---

## 2. Enable it here

In your **`.env`** (never committed):

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Install the dependency (already pinned in `requirements.txt`) if you haven't:

```bash
.venv\Scripts\python.exe -m pip install langfuse
```

Restart the app (`python run.py`), ask a question, then watch traces appear under your
project in the Langfuse UI. Spans export in the background, so they never block a response;
buffered spans are flushed on shutdown.

### Turn it back off
Set `LANGFUSE_ENABLED=false` (or remove the keys) and restart — the app returns to the
zero‑overhead no‑op path.

---

## Notes
- Tracing requires **all** of `LANGFUSE_ENABLED=true`, `LANGFUSE_HOST`,
  `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`. Miss any and it stays off.
- The whole adapter lives in `backend/observability/tracing.py` behind a tiny interface
  (`start_trace` / `span` / `flush`); the rest of the app does not depend on the Langfuse SDK.
- Langfuse Cloud works too — set `LANGFUSE_HOST=https://cloud.langfuse.com` (or the EU host)
  and use the keys from your cloud project instead of self‑hosting.
