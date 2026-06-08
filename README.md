<div align="center">

# 🔊 Audio Research Assistant

**A cited, source-grounded research assistant for audio & speech.**
Ask a question — it searches **your PDFs and the whole web** (research papers, patents,
GitHub, Wikipedia), reads the sources, and answers with **citations and runnable code**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20app-009688?logo=fastapi&logoColor=white)
![Oracle](https://img.shields.io/badge/Oracle%2023ai-vector%20DB-F80000?logo=oracle&logoColor=white)
![Tests](https://img.shields.io/badge/tests-55%20passing-2ea44f)
![No build step](https://img.shields.io/badge/frontend-no%20build%20step-blue)

</div>

---

## ✨ What it does

- 🌐 **Searches everywhere, automatically** — web (DuckDuckGo), research papers
  (arXiv + Semantic Scholar), Wikipedia, patents (Google Patents), and GitHub
  repos/code. **Works with no API key.**
- 📄 **Your own PDFs too** — upload papers; they're parsed, chunked, embedded, and
  searched **together with every external source on every question**, then merged
  and re-ranked so the best evidence wins.
- 🎯 **Grounded & cited** — every claim cites its source (URL · file:line · page).
  It says so plainly when the sources don't cover something.
- 💻 **Reads papers → writes code** — for "how does X work / implement it", it reads
  the method and produces complete, runnable, **original** code or a small simulation.
- ⚡ **Live, interactive UI** — streaming answers, source cards by type, dark mode,
  conversation history, per-message copy/edit/delete. No build step.
- 🔒 **Safe by design** — SSRF-guarded fetches (no localhost/private IPs), timeouts,
  size caps, on-disk caching; API keys stay server-side and are never logged.

---

## 🧠 How it works

```mermaid
flowchart LR
    Q[Your question] --> L[Your PDFs]
    Q --> EXT[Web · arXiv · Semantic Scholar<br/>Wikipedia · patents · GitHub]
    L --> RR[Merge + re-rank vs. your question<br/>read full PDFs]
    EXT --> RR
    RR --> CTX[Build cited evidence]
    CTX --> LLM[LLM writes a cited answer + code]
    LLM --> UI[Streamed to the browser]
```

| Command | What it does |
|---------|--------------|
| **`python run.py`** | Launch the web app → http://localhost:8600 |
| **`python pipeline.py`** | (Optional) build/refresh the index from local PDFs in `data/papers/` |

---

## 🚀 Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env                # then add your keys (see below)
python run.py                         # → http://localhost:8600
```

<details>
<summary><b>🌐 Web-only mode (no database, fastest to deploy)</b></summary>

The simplest setup — no Oracle, no PDFs. In `.env`:
```
ENABLE_LOCAL_RAG=false
ENABLE_WEB_SEARCH=true
OPENROUTER_API_KEY=sk-or-v1-...       # your LLM (or use Ollama locally)
```
Web search runs on free sources (DuckDuckGo, arXiv, Semantic Scholar, Wikipedia,
GitHub) out of the box. Optionally add `TAVILY_API_KEY` for higher-quality web.
</details>

<details>
<summary><b>📄 With your own papers (searched together with the web)</b></summary>

Adds a local PDF library that's searched alongside every web source. In `.env`:
```
ENABLE_LOCAL_RAG=true
ORACLE_DSN=localhost:1521/FREEPDB1    # Oracle 23ai (e.g. in Docker)
GEMINI_API_KEY=...                    # free embeddings: https://aistudio.google.com/apikey
```
Then start your Oracle container and upload PDFs from the sidebar (**＋ Add papers**).
</details>

> The app does not auto-open a browser — visit **http://localhost:8600** yourself.
> It binds to `127.0.0.1` (this PC only).

---

## 💬 Using it

1. **Ask** anything in the chat box — answers stream live with a speed badge.
2. **Sources** open in a side drawer, tagged by type — 📄 Paper · 🌐 Web · 🔬 Research ·
   ⚖️ Patent · 🐙 GitHub — each with a clickable link, file path/line, or page.
3. **Add papers** (sidebar) to upload one or more PDFs; the indexed count is shown.
4. **Delete** a paper from **Your papers** — this removes **everything**: the PDF, its
   chunks, embeddings, vectors, concept links, and cached parse.
5. **Switch models** (top bar) and toggle **dark mode**.

---

## 🔌 Knowledge sources

| Source | Provider | API key? |
|--------|----------|:--------:|
| Web pages | DuckDuckGo (default) · Tavily / Brave / SerpAPI | ❌ free (key = higher quality) |
| Research papers | arXiv · Semantic Scholar | ❌ free |
| Encyclopedic | Wikipedia | ❌ free |
| Patents | Google Patents | ❌ free |
| Code / repos | GitHub | ❌ free (token raises limits) |
| Your library | Local PDFs → Oracle vector search | needs Oracle + Gemini key |

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_PROVIDER` | `openrouter` | `ollama` (local) or `openrouter` (one key → DeepSeek/Qwen/GPT & 300+) |
| `OPENROUTER_API_KEY` | – | Cloud LLM key (or run Ollama locally) |
| `ENABLE_WEB_SEARCH` | `true` | Automatic external search (web/papers/patents/GitHub) |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (free) · `tavily` · `brave` · `serpapi` |
| `ENABLE_LOCAL_RAG` | `false` | Search your uploaded PDFs first (needs Oracle) |
| `EMBEDDING_PROVIDER` | `google` | `google` (Gemini) or `local` (sentence-transformers) |
| `ENABLE_AUTH` | `false` | Require login (user_id + password); private per-user chats |
| `EXTERNAL_TOP_K` · `EVIDENCE_CHARS_PER_SOURCE` · `ANSWER_MAX_TOKENS` | `20` · `3500` · `4096` | Depth/accuracy knobs |

Full list with comments lives in **`.env.example`**. `.env` is gitignored — never commit it.

---

## 👥 Team login (optional)

Turn the app into a multi-user tool — members sign in and each gets their **own
private conversations**.

```bash
# 1. Enable it in .env
ENABLE_AUTH=true
AUTH_SECRET_KEY=<python -c "import secrets;print(secrets.token_hex(32))">

# 2. Create accounts (admin)
python -m backend.auth.users add alice      # prompts for a password
python -m backend.auth.users list
python -m backend.auth.users passwd alice   # reset a password
python -m backend.auth.users delete alice
```

Members then visit the app, get redirected to **`/login`**, and sign in. Passwords are
stored salted + hashed (PBKDF2-HMAC-SHA256); the session is a signed cookie. Set
`ENABLE_SIGNUP=true` to let members self-register.

---

## 🛠️ Tech stack

**FastAPI** + Uvicorn (SSE streaming) · vanilla **HTML/CSS/JS** (no build) ·
**Oracle 23ai** native vector search · **Gemini** embeddings · **BAAI bge** cross-encoder
reranker · **Docling** + PyMuPDF parsing · **OpenRouter / Ollama** LLMs ·
hybrid retrieval (vector + BM25F + RRF + rerank + MMR + HyDE).

See **[`docs/PIPELINE.md`](docs/PIPELINE.md)** for the full walkthrough and
**[`docs/TECH_STACK.md`](docs/TECH_STACK.md)** for versions.

---

## 🧪 Tests

```bash
pytest                       # fast unit suite — no DB / network / models needed
pyflakes backend webapp      # lint
```

---

<details>
<summary><b>📁 Project structure</b></summary>

```
Audio-research-assistant/
├── run.py                  # launch the web app
├── pipeline.py             # build / refresh the local PDF index
├── backend/
│   ├── external_search/    # web · arXiv · Semantic Scholar · Wikipedia · patents · GitHub
│   ├── retrieval/          # hybrid_retrieve, vector, fusion, HyDE
│   ├── ingestion/          # parse → chunk → embed → incremental
│   ├── llm/                # streaming_provider (Ollama + OpenRouter)
│   ├── common/ · answering/ · database/ · memory/ · evaluation/
│   └── config.py
├── webapp/                 # FastAPI server + chat_logic + static UI (index.html, app.js, styles.css)
├── tests/ · scripts/ · viewer_tool/ · docs/ · data/
└── requirements.txt · .env.example · CHANGELOG.md
```
</details>

<div align="center"><sub>Answers are grounded in real sources and cited — built for honest, verifiable research.</sub></div>
