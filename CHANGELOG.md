# Changelog

Running notes of what changed in this project and why — the human-readable
companion to the git history.

> **Keep this updated.** Whenever you make a meaningful change, add a short bullet
> under a dated heading (newest at the top). **Keep it compact:** once a date's
> bullets grow long or a feature is superseded, fold them into a one-line summary.
> This file is the quick "what's the current state and how did we get here" —
> not an exhaustive log (that's `git log`).

---

## 2026-06-05

### Added Qwen 3.5 (35B) to the OpenRouter model list
- Added `qwen/qwen3.5-35b-a3b` to the dropdown. (There is no literal "Qwen 3.5
  32B"; the closest 3.5-family model is this 35B mixture-of-experts.) It's a
  reasoning model, so it "thinks" before answering — fine with the app's 2048
  token budget, just expect a short pause before the answer appears.

### Slimmed to two LLM providers
Kept **Ollama** (local) + **OpenRouter** (one key → DeepSeek, Qwen, GPT, Claude,
300+) and removed the redundant direct providers — **OpenAI, Anthropic, DeepSeek,
Qwen** — since OpenRouter reaches all of those models with a single key. Dropped
the `AnthropicProvider` class and the direct-OpenAI branch from
`streaming_provider.py`, trimmed `webapp/settings.py`, cleaned the removed keys
out of `.env` / `.env.example`, and dropped the now-unused `anthropic` and
`beautifulsoup4` dependencies. Docs updated.

### LLM accuracy measurement
- Added `backend/evaluation/evaluate_llm.py` + `data/llm_eval_questions.json`:
  runs the real retrieve→answer pipeline over a question set and scores each
  answer on **keypoint coverage**, **citation rate**, and an optional
  **LLM-as-judge** correctness score. `--models a,b,c` compares models head-to-head
  and prints a ranked scorecard, so you can see which model answers best.

### Consolidated config / launcher files
- Removed `enable_sharing.bat` — `run.py` now opens the firewall itself (one UAC
  prompt) and prints the manual `netsh` command if that's declined.
- Merged `requirements-dev.txt` into `requirements.txt` (dev/test tools are a
  clearly-marked section at the bottom), so there's a single requirements file.
- Kept `.env` + `.env.example` as-is: `.env.example` is the committed, secret-free
  template; `.env` is your local filled-in copy and stays gitignored. Merging them
  would mean committing real API keys, so they're intentionally separate.

### Cleanup — removed dead code, fixed the docs
Deleted modules that nothing imported (the live web app answers via
`webapp/chat_logic.py` → `hybrid_retrieve` → `streaming_provider`, so an older
"answer via router" path and unused future-phase tools were dead weight):
- `backend/answering/`: `answer_orchestrator.py`, `evidence_builder.py`, `prompt_quality.py`
- `backend/retrieval/query_planner.py`
- `backend/llm/`: `router.py`, `fallback_provider.py`, `cost_tracker.py`
- `backend/database/oracle_db.py` (unused wrapper — everything uses the `oracledb` driver directly)
- `backend/common/logger_config.py` (only the deleted `evidence_builder` used it)
- `backend/tools/` (whole package: `web_search`, `code_executor`, `sandbox_runner`, `dsp_toolkit`)
- `web_ui.bat` (redundant — `python run.py` is the one launcher)

Supporting changes: inlined the OpenAI model list into `webapp/settings.py`;
rewrote `README.md` structure and `docs/PIPELINE.md` to describe the **real**
pipeline (FastAPI web app, current retrieval/answer path, OpenRouter/DeepSeek/Qwen
providers) instead of the old Streamlit/router design. All 20 tests still pass.

### Sharing on the local network
- `python run.py` now **shares on the Wi-Fi/LAN by default** (binds `0.0.0.0`) and
  prints the teammate URL; `--local` restricts to this PC.
- On Windows it **auto-opens the firewall** for the port via one UAC prompt (and
  prints the manual `netsh` command if that's declined).
- `run.py` also **auto-clears a stale server** still holding the port (fixes the
  recurring `[Errno 10048] only one usage of each socket address`).

### Chat: per-question actions
- Hovering your question shows **Copy**, **Edit & resend** (truncates + regenerates,
  ChatGPT-style), and **Delete** (removes the question + its answer).
- Backed by `MemoryStore.delete_turns_from` / `delete_turn_pair` and two routes:
  `DELETE /api/sessions/{id}/turns/{i}` and `POST …/turns/{i}/truncate`.

### LLM providers
- Added **OpenRouter** (`OPENROUTER_API_KEY`) — one key serves DeepSeek, Qwen and
  300+ models via the OpenAI-compatible endpoint, including `:free` slugs. This is
  the recommended cloud path.
- Added **DeepSeek** (`deepseek-v4-pro`) and **Qwen** (`qwen3-32b`) direct
  providers. Cloud models appear in the Model dropdown only when their key is set.

---

## Earlier (condensed)

- **UI:** retired Streamlit; the FastAPI web app (`webapp/`, port 8600) is the only
  UI — streaming chat over SSE, citations + source drawer, sessions, PDF upload +
  live ingest, model/theme switchers, multi-PDF upload, paper management/delete.
- **Embeddings:** switched to Google **Gemini** (`gemini-embedding-2`, 768-dim,
  free tier) with a pluggable `local` (sentence-transformers) fallback; fixed a bug
  where only some chunks were embedded.
- **Parsing:** **Docling** (Python API, GPU) is the primary PDF parser for quality,
  with PyMuPDF + OCR fallbacks.
- **Vector store:** Oracle 23ai native `VECTOR(768)` with exact COSINE search by
  default; approximate HNSW/IVF index is opt-in (removed the ORA-51962 error).
- **Compute:** configurable GPU/CPU split (`DEVICE` / `EMBEDDING_DEVICE` /
  `RERANKER_DEVICE`).
- **Structure:** reorganized into `backend/` subpackages + `webapp/`, with
  `pipeline.py` (ingest) and `run.py` (serve) as the two entry points.
