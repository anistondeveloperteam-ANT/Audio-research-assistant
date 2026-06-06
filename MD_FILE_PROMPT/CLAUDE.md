# CLAUDE.md — Audio Research Assistant

> Project constitution for Claude Code. Read every turn — keep it lean.
> If anything here conflicts with the actual code, trust the code and tell me.

## 1. What this project is

A **Retrieval-Augmented Generation (RAG)** assistant for **audio / speech-enhancement
research papers**. It ingests PDFs, builds a hybrid (vector + keyword) search index in an
Oracle vector database, and answers questions with **cited, grounded answers** streamed to a
web UI. It is evolving toward an **agentic simulate → verify loop** that designs and validates
audio-enhancement algorithms.

**Non-negotiable product contract:** answers come *only* from retrieved evidence, every
technical claim is cited (`[SOURCE n]`), and the system says plainly when the papers don't
cover something. Never invent facts, citations, or numbers.

## 2. Tech stack (authoritative)

- **Language:** Python 3.11. **OS:** Windows 11 — use PowerShell; paths use `%USERPROFILE%`.
- **API/Web:** FastAPI + Uvicorn, streaming via Server-Sent Events (SSE). Frontend is plain
  HTML/CSS/vanilla JS — **no build step, no framework**. Do not introduce one.
- **Database:** Oracle Database Free 23ai (in Docker), native vectors `VECTOR(768, FLOAT32)`,
  cosine distance, optional HNSW/IVF index. Driver: `python-oracledb`.
- **PDF parsing:** Docling (primary) → PyMuPDF (fallback) → PaddleOCR/Tesseract (OCR, opt-in).
- **Embeddings:** Google **Gemini Embedding 2** (`gemini-embedding-2`, **768-dim**) via
  `google-genai` (free tier, API call — frees the local GPU). Optional local backend:
  `sentence-transformers` (BGE).
- **Reranker:** BAAI `bge-reranker-v2-m3` (cross-encoder).
- **Chat models:** OpenRouter (OpenAI-compatible; default `deepseek-v4-flash`) + Ollama
  (local fallback, e.g. `qwen2.5:7b`). Routed in `router.py` via `ANSWER_PROVIDER`.
- **Memory:** SQLite (`data/memory.db`) — sessions, turns (with sources), facts.
- **Numeric/ML:** NumPy, SciPy, pandas; PyTorch 2.7.1 (CUDA 12.6), transformers.
- **GPU:** single laptop GPU (~6–8 GB). Embeddings are API-side, so the GPU is for the
  reranker + Docling. Be memory-aware; don't load large models concurrently.

## 3. Commands

```powershell
docker start oracle-ai-db            # Oracle must be running first
python pipeline.py                   # full index build/refresh from PDFs
python pipeline.py --incremental     # only re-process changed PDFs (content-hash)
python run.py                        # launch web app -> http://localhost:8600
pytest                               # run tests
python -m pyflakes . ; vulture .     # lint / dead-code
```

Quality harnesses: `evaluate_retrieval` (retrieval quality vs a fixed question set) and
`evaluate_llm` (answer accuracy: keyword coverage, citations, optional LLM-judge). Run the
relevant one after any change to chunking, retrieval, or prompting.

## 4. Pipeline (how the pieces fit)

```
PDFs -> Parse (Docling) -> Chunk + tag -> Embed (Gemini 768) -> Oracle VECTOR(768)
Question -> HyDE + Vector + BM25F -> RRF fusion -> cross-encoder rerank -> MMR
        -> build cited prompt -> LLM (router) -> streamed answer (SSE)
```

Approximate key files (verify against the repo before editing):
`pipeline.py`, `run.py`, `pdf_parser.py`, `document_chunker.py`, `embed_chunks.py`,
`vector_migration.py`, `query_planner.py`, `hyde_generator.py`, `retrieval_fusion.py`,
`evidence_builder.py`, `router.py`.

## 5. Data model

`PAPERS → CHUNKS → CONCEPTS`. Each chunk stores: text, section name, chunk type
(prose / caption / algorithm), equation & table flags, audio concepts (MVDR, PESQ, STOI,
U-Net, spectral subtraction, deep filtering…), and a `VECTOR(768)` with an HNSW index.
**Chunking is section-aware** — figure/table captions and algorithm blocks are kept **whole**
(high-signal). Do not switch to blind fixed-size slicing.

## 6. Code conventions

- Type hints on public functions; clear names over cleverness; small, pure functions.
- Prefer the standard library; justify every new dependency (this app is intentionally lean).
- Tests with `pytest`; aim ~80% coverage on changed code. Add a failing test first for bugs.
- No secrets in code — config via `.env` / `python-dotenv` (`DEVICE`, `EMBEDDING_DEVICE`,
  `RERANKER_DEVICE` = `auto|cuda|cpu`; provider keys; model names). Never commit `.env`.
- Conventional Commit messages (`feat:`, `fix:`, `chore:`…). Small, focused PRs.
- Cross-platform file/path handling (`pathlib`) — this runs on Windows.

## 7. Hard rules / gotchas

- **Embedding dimension is 768 and must match the Oracle column.** If you change the embedding
  model, embed documents *and* queries with the **same** model + same instruction prefix, and
  migrate/rebuild the index. Never mix dimensions or models across stored vs query vectors.
- **Don't break the cited-answer contract.** Keep the `[SOURCE n]` evidence format and the
  evidence-only system prompt. If retrieval returns nothing relevant, the answer says so.
- **Oracle runs in Docker** — start the container before `pipeline.py` or `run.py`.
- **Gemini free tier is rate-limited** (~1500 req/day, low RPM). Batch embeddings; handle
  429s with backoff; cache by content hash. Don't assume unlimited calls.
- **Keep MCP servers < 10** if any are enabled — each tool description eats the context window.
- Model-written code only runs in the **locked-down sandbox** (import allowlist, timeout, no
  file/network). Never execute model code outside it.
- Log every paid API call (tokens + USD) to the cost tracker.

## 8. When working on the agentic loop (current direction)

The loop is: **propose pipeline → simulate (pyroomacoustics) → score → reflect → iterate**,
then deploy the chosen algorithm to the audio-processing host (not "into the mic").

- The **referee is the objective harness, not the model**: PESQ, STOI, SI-SDR, DNSMOS, plus
  latency/RTF. Reject non-causal/over-budget candidates regardless of quality score.
- **Branch on constraints:** single- vs multi-channel; streaming/causal vs offline.
- **Tune on a train set of scenes, select/report on a held-out set** (different rooms, SNRs,
  noise types) — guard against overfitting.
- **Route models by role:** cheap/fast (DeepSeek V4 Flash) for glue; stronger reasoner for
  planning/verification; zero LLM calls in simulate/score.
- Working memory: offload large sim outputs to disk, keep metrics/params/paths **verbatim**,
  summarize only reasoning. (See `agent_memory.py`.)

## 9. Do NOT

- Add a frontend build step or framework.
- Re-derive DSP math in ad-hoc code — call the vetted DSP toolkit (MVDR, delay-and-sum, MUSIC,
  SRP-PHAT…).
- Bulk-install third-party config kits globally, stack install methods, or auto-enable bundled
  MCP servers.
- Hardcode secrets, model names, or device choices — read them from `.env`.
- Claim a result is "novel/best" without baselining it against a standard cascade on the same
  held-out set.

## 10. Before saying "done"

Run the relevant eval harness, confirm tests pass and lint is clean, verify the cited-answer
format still holds, and summarize what changed and why. Plan non-trivial work before editing.
