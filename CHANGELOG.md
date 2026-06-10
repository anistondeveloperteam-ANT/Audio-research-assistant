# Changelog

Running notes of what changed in this project and why — the human-readable
companion to the git history.

> **Keep this updated.** Whenever you make a meaningful change, add a short bullet
> under a dated heading (newest at the top). **Keep it compact:** once a date's
> bullets grow long or a feature is superseded, fold them into a one-line summary.
> This file is the quick "what's the current state and how did we get here" —
> not an exhaustive log (that's `git log`).

---

## 2026-06-10

### Concise model picker: only working models + re-added GPT-5.5
- Trimmed the dropdown to verified-working models only: **Groq llama-3.3-70b / llama-3.1-8b**
  (free), **Gemini 2.5 Flash** (free), **GPT-5.5** (your OpenAI key), plus any local **Ollama**.
  Removed DeepSeek (OpenRouter out of credits → 402) and Gemini 2.0 Flash (daily quota → 429).
- Re-added GPT-5.5 routing (now that a valid key is available): `gpt-*` → OpenAI via
  `OPENAI_CLOUD_KEY`. Paste the ChatGPT/OpenAI key into `OPENAI_CLOUD_KEY` in `.env`, then pick
  "OpenAI · gpt-5.5" in the dropdown. (Dark mode remains the default theme.)

### Removed the GPT/OpenAI-cloud option (dead key)
- Live-tested every model in the picker. GPT-5.5 and GPT-4o both returned 401 "Invalid API
  Key", so the OpenAI-cloud option was removed completely: dropped from the dropdown
  (`CLOUD_MODELS`), the `gpt-*` routing branch and `_provider_name` case, the `OPENAI_CLOUD_KEY`
  env var (.env + .env.example), and the GPT-specific `_request_variants` special-case (the
  generic max_tokens→max_completion_tokens fallback still covers any OpenAI-compatible model).
  Default model is now `gemini-2.5-flash`. Tests updated.
- Working dropdown is now: **Groq** (llama-3.3-70b, llama-3.1-8b — verified), **Gemini**
  (2.5-flash verified; 2.0-flash subject to the free daily quota), **DeepSeek** (chat verified;
  r1 needs credits), and local **Ollama**.

### Free LLMs for agentic work (Groq + Gemini) — from free-llm-api-resources
- Added two FREE, OpenAI-compatible providers to the model router/dropdown, chosen from
  cheahjs/free-llm-api-resources (2026) for chained agentic loops:
  - **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) — the repo's top pick:
    ~1,000 requests/day, very fast, strong reasoning. Needs a free `GROQ_API_KEY`.
  - **Gemini** (`gemini-2.5-flash`, `gemini-2.0-flash`) — reuses the existing
    `GEMINI_API_KEY`; verified working live (no 402 token-budget issues).
- `route_model()` now maps `gemini-*` → Google and the Groq model ids → Groq; the picker
  labels them and shows "(add key)" when the key is missing. Models without a slash-collision
  risk are handled (a Groq id containing `/` still routes to Groq, not OpenRouter).
- Switched the local `.env` active model to `gemini-2.5-flash` so chat works for free out of
  the box (the prior DeepSeek/OpenRouter account was out of credits). Tests cover routing +
  the dropdown.

### Fix "(no answer)" on low-credit OpenRouter accounts
- The provider's 402 "can only afford N tokens" handler had a 256-token floor, so when an
  account could only afford fewer (e.g. 180) the retry kept requesting 256, failed every
  time, and the stream **silently returned empty** — surfaced as "(no answer)" plus a fake
  "0/100, 5 round(s)" verification footer. Now it shrinks to the actual affordable amount
  (small floor) so a near-empty balance still yields a short real answer, and a persistent
  failure **raises** instead of returning empty.
- The agentic chat path now shows a clear, actionable message on a truly empty draft
  (top up credits / use a free local Ollama model / lower `ANSWER_MAX_TOKENS`) instead of
  "(no answer)" + a meaningless verification verdict. Added a mocked regression test.

### LangGraph research engine (optional)
- New `backend/agent/langgraph_research.py`: the deep-research pipeline
  (PLAN → SEARCH → SYNTHESIZE → REFLECT ⟲ → REPORT) is now also available as an explicit
  **LangGraph** `StateGraph` — typed shared state, named nodes, and a conditional edge that
  loops back to SEARCH until the question is covered or the round budget ends. The reasoning
  steps are the proven `research_agent` functions (reused, not re-implemented).
- Opt in with `RESEARCH_ENGINE=langgraph` (the web `/api/research` and the CLI then use it);
  defaults to the built-in loop. LangGraph is an **optional** dependency — if it isn't
  installed the app falls back gracefully. CLI: `python -m backend.agent.langgraph_research "…"`.
- Added `langgraph==1.2.4` to requirements (optional); 5 offline (mocked) tests cover the
  graph flow, the reflect→search loop, the round-budget cap, and the fallback path.

### Code agent works across providers (per-model routing)
- Fixed: with chat on a cloud provider (e.g. DeepSeek/OpenRouter), a coding task sent
  `AGENT_MODEL=qwen2.5-coder:7b` to the wrong endpoint -> "not a valid model ID". Now a
  model override (the agent's model) routes to ITS own endpoint/key by name via
  `route_model()` — so the coder runs on local Ollama even while chat is on OpenRouter.
  Routing is shared between the provider factory and the UI model picker.

### One model picker across providers (Ollama + DeepSeek + GPT-5.5)
- The Model dropdown now lists your local **Ollama** models *and* cloud models —
  **GPT-5.5 / GPT-4o** and **DeepSeek (V3 + R1)** — and switching one rewrites the
  endpoint **and** key automatically (routing inferred from the model name). Cloud keys
  live in `OPENROUTER_API_KEY` (DeepSeek) and `OPENAI_CLOUD_KEY` (GPT); models without a
  key show "(add key)".

### Readable progress + no-emoji answers
- The status line used an unreadable shimmering-gradient text that looked like a blank
  bar during long runs — replaced with **plain, readable status text** (gentle pulse), so
  you can actually see "Searching… / Drafting… / Thinking… / Verifying…" while it works.
- Answers are now **emoji-free**: the system prompt forbids emojis and the renderer also
  strips any emoji the model emits (code blocks untouched) — clean, professional output.

### Visible "Thinking" panel (Claude/ChatGPT style)
- The model's hidden reasoning is now captured and shown in a **collapsible "Thinking"
  panel** above the answer — click to expand/collapse; it streams live (pulsing icon)
  while the model thinks, then settles to "Thought process". Works for reasoning models
  (qwen3, DeepSeek-R1, …) that expose `reasoning_content`; gracefully absent otherwise.
- `stream_chat(..., yield_reasoning=True)` now surfaces reasoning as `{"reasoning": …}`
  chunks; the chat pipeline forwards them as `thinking` events. The answer itself keeps
  streaming token-by-token as before.

### Model dropdown lists local Ollama models
- When `OPENAI_BASE_URL` points at Ollama, the Model picker now lists the models you
  actually have installed (queried from Ollama's `/v1/models`) — e.g.
  `Ollama · qwen3:8b`, `Ollama · qwen2.5-coder:7b` — instead of cloud-only names. Falls
  back to the curated cloud list otherwise; always includes the active + `AGENT_MODEL`.

### Polished answers + UI cleanup
- **Math now renders** — `$…$` / `$$…$$` formulas display properly via KaTeX (the
  markdown parser is protected from mangling the LaTeX first).
- **Removed noisy inline `[n]` citations** from the rendered answer (kept out of code
  blocks), for a clean Claude/ChatGPT-style result.
- **Removed the manual "Review" button** — answers are peer-reviewed automatically
  (`AUTO_REVIEW`); the dead handlers were deleted too.
- Confirmed the code agent uses the local **qwen2.5-coder:7b** (`AGENT_MODEL`).
- Rewrote the **README** to be friendlier and interactive (free local/Gemini model
  options, a flow diagram, collapsible sections).

### Saved-answer reuse (semantic cache) — finished + hardened
- Reuse a prior answer for the same/very-similar question (per user) instead of
  re-searching + re-calling the LLM. Matching is near-exact **lexical** (>=0.97) OR
  **semantic** (embeddings via the existing Gemini embedder, cosine >=0.88, with a
  provider/model tag so only comparable vectors match; graceful lexical fallback).
- **Correctness guard** (`unsafe_to_reuse`): never serve a different question's
  answer even at high similarity — blocks swaps ("A vs B"/"B vs A"), identifier/number
  changes (A100/H100, GPT-4/GPT-5), short-entity subs (TCP/UDP), polarity/antonyms
  (with/without, advantages/disadvantages, encoder/decoder, increase/decrease,
  km/miles), and any single content-word substitution. Adversarially reviewed.
- Caching is gated on real quality (verify verdict passed, code run didn't fail, not a
  post-review rewrite) and stores the answer body only (footers stripped). Per-user
  dedup; edits/deletes invalidate across the user's sessions; the cache is cleared when
  local PDFs are ingested/deleted; time-sensitive questions ("latest", "2026", "state
  of the art") always re-search. A **"From memory"** badge marks reused answers.
- Schema migrated in place (v3) on existing DBs. 14 cache tests (8 new) + full suite pass.

## 2026-06-09

### Automatic peer review after every answer/code (AUTO_REVIEW)
- The manual "Review" step now runs **automatically** after generation. For chat
  answers: the final answer is peer-reviewed, improved once if the verdict is
  major-revision/reject, and a one-line `Auto-review:` verdict is appended. For the
  code agent: the best result is peer-reviewed and the verdict shows in the timeline.
  One extra LLM call; toggle with `AUTO_REVIEW` (default true) — set false on slow
  local models.

### Local Ollama support + per-task model (no new provider code)
- Ollama is OpenAI-compatible, so it works through the existing client by setting
  `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`, and an
  `OPENAI_MODEL` (e.g. `qwen3:8b`). Verified live.
- `get_provider(model=...)` now takes an optional model override, and the autonomous
  code agent uses `AGENT_MODEL` when set (e.g. `qwen2.5-coder:7b`) so chat/research
  and coding can run on different local models. Documented in `.env.example`.

### Added 3 project-fit dev-tooling files from ECC (no bulk install)
- `.claude/agents/fastapi-reviewer.md` — reviews FastAPI routes, async/streaming
  (SSE/NDJSON), validation, auth/session, and SSRF/secret safety.
- `.claude/skills/rag-retrieval-review/SKILL.md` — checklist for retrieval/RAG changes
  (grounding, citations, evidence budget, freshness).
- `.claude/skills/research-first-development/SKILL.md` — research-before-implement loop.
  These are dev-workflow aids (not app runtime); the rest of ECC stays out per CLAUDE.md.

### Autonomous deep-research agent ("just give me results")
- New `backend/agent/research_agent.py`: give it one question and it runs the whole
  job on its own — PLAN (decompose) → EXECUTE (search everywhere each round) →
  REFLECT (find gaps, search more) → WRITE a comprehensive cited report → REVIEW
  (self-critique + one revision). Bounded two-tier memory; self-contained Sources list.
  CLI: `python -m backend.agent.research_agent "question"`. Synthesizes the three
  repos: THINK→EXECUTE→REFLECT + two-tier memory (auto-deep-researcher-24x7),
  the review stage (Awesome-AI-Scientist), lifecycle structure (kimi-code).
- Deep research is now **merged into every chat answer** (no toggle): each question
  is auto-decomposed into a few "angles", all of them are searched across every
  source, and the answer is written from the combined evidence. Evidence is bounded
  and **auto-shrinks** to fit the model's token budget so low-balance accounts still
  get a full answer. (Standalone `/api/research` endpoint kept for direct use.)
- Provider resilience: `stream_chat` now auto-retries on OpenRouter `402`
  "can only afford N tokens" by shrinking the budget — low-balance accounts keep working.
- Verified live end-to-end (real searches + DeepSeek); tests in `tests/test_research_agent.py`.



### OpenAI-only chat LLM (removed OpenRouter + Gemini-chat providers)
- Simplified the LLM layer to a single, clean OpenAI provider: rewrote
  `backend/llm/streaming_provider.py` (one `OpenAIProvider`, no multi-provider
  abstraction) and `webapp/settings.py` (OpenAI model picker only). Removed the
  OpenRouter and Gemini-chat code, env vars, dropdown entries, and tests.
- Gemini stays only for **embeddings** (`GEMINI_API_KEY`) — it is not a chat model.
- Updated `.env.example`, README, and docs to OpenAI-only; verified live + 99 tests pass.
- Follow-up cleanup: dropped the now-redundant `LLM_PROVIDER` env var and the
  multi-provider scaffolding (`MODEL_ENV`, `VALID_PROVIDERS`, `PROVIDER_MODELS`).
  `webapp/settings.py` is now a small OpenAI-only model manager; the model switcher
  API/UI shape is unchanged.

### General "answer from everywhere" + stronger answer quality
- Rewrote the system prompt: no longer audio-specific and no longer biased toward the
  local PDF corpus. It now treats **all source types as equal evidence**, synthesizes
  across web/papers/patents/code/encyclopedic sources, and produces a comprehensive,
  well-structured answer (direct lead + sectioned depth, diverse citations, recency-aware).
- Fresh start: cleared the pre-indexed documents (Oracle papers/chunks, the PDF files,
  the parse cache, and stale external-search cache).

### Save credits via the provider, not by cutting accuracy
- Kept the accuracy-favoring defaults (`AGENTIC_MAX_VERIFY_ROUNDS=3`,
  `EXTERNAL_TOP_K=20`, `EVIDENCE_CHARS_PER_SOURCE=3500`, `AGENTIC_EXTRA_SEARCH_K=8`).
  The recommended way to cut cost is a **cheaper provider** — DeepSeek
  (`deepseek-v4-flash`) is a fraction of a reasoning model's price and has
  aggressive prompt/context caching — so you save credits without losing answer
  quality. (Briefly trimmed these defaults earlier; reverted by request.)

### Review action in the web UI
- Every answer now has a **Review** button (next to Copy). It sends the answer to
  `/api/review` and shows a structured peer review (summary, strengths, weaknesses,
  questions, suggestions, scores, recommendation) as a new message. Verified live.

### Automated peer reviewer (idea from Awesome-AI-Scientist)
- That repo is a reading list (no code); its useful concept is a **review system**.
  Added `backend/answering/reviewer.py` (original code): `review(text)` returns a
  structured peer review — summary, strengths, weaknesses, clarifying questions,
  actionable suggestions, per-criterion scores (novelty/soundness/clarity/significance),
  and a recommendation. CLI: `python -m backend.answering.reviewer <file|text|stdin>`.
  Verified live; tests in `tests/test_reviewer.py` (mocked). 94 tests pass.

## 2026-06-08

### Remove Sources panel + de-emoji login
- Removed the Sources drawer/button/citation chips entirely; `[n]` markers are now
  plain text. Replaced the login page's emojis with clean line-SVG icons.

### Share a public URL (`run.py --share`) + permanent custom domain
- `python run.py --share` exposes a **public `https://…trycloudflare.com`** link anyone
  can open — downloads the Cloudflare tunnel client once into `data/tools/` (gitignored),
  starts the app on `0.0.0.0`, prints the URL, and warns if `ENABLE_AUTH` is off.
- **Permanent / custom domain:** set `CLOUDFLARE_TUNNEL_TOKEN` (from a Cloudflare named
  tunnel in the Zero Trust dashboard) and `--share` serves your app at your own stable
  hostname every run (`CLOUDFLARE_TUNNEL_HOSTNAME` is shown for convenience).
- `--lan` binds to the local network. No paid service required.

### Rebrand: "Research Assistant" (no longer audio-specific)
- Renamed the product from "Audio Research Assistant" to **Research Assistant**
  everywhere user-facing: page titles, sidebar + login branding (new 🔬 search mark,
  not 🔊), welcome copy, example prompts, README/docs, FastAPI title, and code
  docstrings. The general-purpose engine is unchanged; only the framing is now
  topic-agnostic. (Repo folder/URL left as-is to avoid breaking paths.)

### Fresh, latest-first sources + no more truncated answers
- **Recency:** arXiv now fetches newest **and** most-relevant and merges them;
  Semantic Scholar sorts by publication date; GitHub ranks most-recently-updated
  (`GITHUB_REPO_SORT`). A recency boost in the reranker surfaces recent results —
  strongly when the query says "latest/recent/2026". External cache window cut
  24h → 1h so today's papers/repos/pages appear fast. Source cards show the date.
- **Completeness:** raised `ANSWER_MAX_TOKENS` 4096 → 8000 (fixes the truncated-code
  answer) and `AGENTIC_MAX_VERIFY_ROUNDS` 2 → 3 (more draft→verify→refine passes,
  better verification scores). Verified live: arXiv returns relevant + recent. 89 tests pass.

### Per-query sources with prev/next navigation
- Sources are now **per question**. Click a question (or its **Sources · N** chip, or a
  `[n]` citation) and the drawer shows *only that query's* sources, with a header
  naming the question and a **‹ prev / next ›** pager (plus ←/→ keys and a "2 / 5"
  position) to step between every query's sources. Citation chips open the right
  answer's set and jump to the cited source.

### Claude-style agent interface
- When the agent works, it now renders a **step timeline** (like Claude's agent UI)
  instead of flat markdown: each event is a card with an icon, a live
  **spinner → checkmark** status, result badges (e.g. `OK · 0.5s`, `score 100`),
  an expandable body, and a prominent **verified result** card at the end. Code shows
  in the IDE-style panels; steps animate in. Pure client-side rendering (no full
  re-render per token).

### Agent is automatic now (no toggle) + polished user chip
- Removed the manual **Agent** button. Coding/solve tasks ("implement…", "benchmark…",
  "write a function…", "find the fastest algorithm…") **auto-route** to the autonomous
  write→run→verify agent; every other question uses chat, which already verifies its
  answer and runs any code it writes (LLM + agent always work together). A lead line
  tells the user when the agent engaged.
- Signed-in user chip redesigned: a coloured avatar with the member's initial, their
  name, and a clear **Sign out** button.

### Agent pre-execution hook (idea from kimi-code)
- Adopted the *one* fitting idea from `MoonshotAI/kimi-code` (a TypeScript coding-agent
  product — full repo not used): **lifecycle hooks**. New `backend/agent/hooks.py`
  `pre_run(code, task)` audits every program the agent runs to
  `data/logs/agent_audit.jsonl` and applies an optional policy gate
  (`AGENT_BLOCK_PATTERNS` regexes / `AGENT_PRERUN_HOOK` command) that can block a run
  before the Docker sandbox. Default = allow + audit. Loop emits a `blocked` step.
- Original Python implementation (kimi-code is TS; nothing copied). Focused write-up in
  `docs/KIMI_CODE_ADOPTION.md`. 86 tests pass.

### IDE-style code blocks
- Code in answers and Agent mode now renders as a polished panel: syntax
  highlighting (highlight.js, atom-one-dark), a header with a language badge +
  mac-style dots, and a proper Copy button. Replaces the flat, dull code style.
- Fixed an earlier regression where a missing `</script>` (from the dark-mode
  change) swallowed the stylesheet link and left the whole UI unstyled.

### Web "Agent mode" + dark mode by default
- Added an **Agent** toggle in the chat top bar. When on, your message goes to the
  new streaming `/api/agent` endpoint and the THINK→RUN→REFLECT steps render live
  (generated code, sandbox run output, reviewer verdict), ending in the best verified
  program + answer. Verified live end-to-end.
- The UI now defaults to **dark mode** (still remembers your manual choice).

### Agent: two-tier memory + PROJECT_BRIEF steering
- Completed the adoption of the `auto-deep-researcher-24x7` ideas (Apache-2.0, original
  code) into `backend/agent/`: added a **constant-size two-tier memory** (`memory.py` —
  frozen brief + auto-compacting attempt log) so long loops never bloat context, and a
  **PROJECT_BRIEF** interface (`--brief FILE`: goal + if-then decision tree) plus an
  optional mid-run **`--directive FILE`** steer. Example brief in `docs/`.
- `run_agent(task, brief=…, directive_path=…)`; the loop now carries only the brief +
  compacted log + last program (bounded) into each THINK. 83 tests pass.

### Agentic verified web answers
- Upgraded the browser chat path from one-shot retrieval -> answer to a bounded
  agentic loop: draft a cited answer, verify it against the numbered evidence,
  search local/web/research/patents/GitHub again when support is missing, then
  rewrite and return the best checked answer.
- Fenced Python in answers is optionally executed in the existing network-less
  Docker sandbox, and the run result is included in verification. New knobs:
  `ENABLE_AGENTIC_ANSWER_LOOP`, `AGENTIC_MAX_VERIFY_ROUNDS`,
  `AGENTIC_MIN_VERIFY_SCORE`, `AGENTIC_EXTRA_SEARCH_K`, `AGENTIC_SIMULATE_CODE`.
- Added focused offline tests for the agentic answer helpers. 80 tests pass.

### Autonomous research agent (write → run → verify)
- New `backend/agent/`: a THINK→EXECUTE→REFLECT loop that designs a Python program,
  runs it in a **throwaway Docker sandbox** (`--network none`, capped mem/CPU/PIDs,
  wall-clock timeout, `--rm`, code piped via stdin so no host fs is touched), reviews
  the real output, and refines until it has the best *verified* solution.
- `code_runner.py` (sandbox), `loop.py` (the loop, optional web/paper research to seed
  attempt 1), CLI `python -m backend.agent "task"`. Knobs: `AGENT_MAX_ITERS`,
  `AGENT_DOCKER_IMAGE`, `AGENT_RUN_TIMEOUT`, `AGENT_MEM_LIMIT`, `AGENT_CPUS`.
- Verified live (gpt-5.5 + Docker): solved a task in 1 cycle, score 100. Tests:
  `tests/test_agent.py` (fully mocked — no Docker/LLM/network). 74 tests pass.

### Tidy-up: consolidated the data inspector into scripts/
- Moved `viewer_tool/show_my_data.py` → `scripts/show_data.py` (all admin CLIs now
  live under `scripts/`) and made it run from any directory. Refreshed the README
  structure and PIPELINE docs. Reviewed the rest: the pipeline is already cleanly
  named and organized, with no dead code to remove.

### GPT-5 models in the picker + per-model API compatibility
- Added the GPT-5 family to the model dropdown (`gpt-5.5` default, plus `gpt-5.5-pro`,
  `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.1`) alongside `gpt-4.1` / `gpt-4o`.
- `stream_chat` now adapts parameters per model: GPT-5 / o-series use
  `max_completion_tokens` and the default temperature; gpt-4o/4.1 keep `max_tokens` +
  custom temperature — with fallbacks so any current/future model works. Verified
  `gpt-5.5` answers live.

### Claude Code config
- Added a tracked Claude Code setup from the ECC recommendation and the selected
  `affaan-m/ecc` files: root `CLAUDE.md`, local rule overlays, ECC common/python
  reference rules, five reviewer agents, six workflow skills, MIT attribution,
  and a tightened `.claude/settings.json`. AgentShield scan: Grade A, no
  critical/high findings.

### Login required each visit + redesigned login page
- The session cookie is now a **session cookie by default** (clears when the browser
  closes), so users sign in every visit. `SESSION_MAX_AGE=<seconds>` keeps them
  logged in for that long instead.
- Rebuilt `/login` as a polished two-panel page: animated brand hero, segmented
  Sign in / Sign up tabs with a sliding indicator, password show/hide, Caps-Lock
  warning, loading spinner, and friendly errors.

### LLM is now OpenAI-only
- Dropped the Ollama and OpenRouter providers. `streaming_provider.py` is a single
  **OpenAI** client (`OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`);
  default model `gpt-4o`, switchable in the UI (`gpt-4o-mini`, `gpt-4.1`, …).
- Updated settings/model picker, the eval harness examples, `.env(.example)`, and
  the README/PIPELINE/TECH_STACK docs. 68 tests pass.

### Optional Memgraph GraphRAG
- Added an optional `backend.graph_rag` layer that builds a Memgraph graph from
  Oracle papers/chunks/concepts and uses it to expand local retrieval before
  reranking. It is disabled by default (`ENABLE_GRAPH_RAG=false`) and falls back
  cleanly when Memgraph is unavailable.

### Login page: Sign in / Sign up tabs + Forgot password
- Redesigned `/login` with segmented **Sign in / Sign up** tabs (sign-up posts to
  `/api/signup`, gated by `ENABLE_SIGNUP`) and a **Forgot password?** link that
  shows the admin reset step (`python -m backend.auth.users passwd <id>`).
  Self-service email reset is intentionally not included (needs SMTP).

### Multi-user login (optional)
- Added `ENABLE_AUTH`: members sign in at `/login` with a user_id + password and
  each member's conversations are private (per-user `user_id` on the memory store,
  with ownership checks on every session route).
- Passwords are salted + hashed with PBKDF2-HMAC-SHA256 in a SQLite user store
  (`data/auth.db`); sessions are signed cookies (Starlette SessionMiddleware).
- Admin CLI: `python -m backend.auth.users add|list|passwd|delete`.
- Optional self-registration via `ENABLE_SIGNUP`. New login page + in-app sign-out.
- New deps pinned: `itsdangerous`, `python-multipart`. Tests: `tests/test_auth.py`.

### Optional opt-out for the SSRF guard
- Added `EXTERNAL_ALLOW_UNSAFE_URLS` (default **false**). When true it disables the
  SSRF guard so the fetcher may hit localhost / private / internal addresses. It
  adds no public-search reach — keep it false except on a trusted single-user
  machine; never on a public deployment. Tests pin the guard on regardless of env.

### Always search everywhere (no fallback gating)
- External search now runs on **every** question, combined with the local papers —
  not only when the papers miss. One query pulls from your PDFs **and** the web,
  arXiv, Semantic Scholar, Wikipedia, patents, and GitHub; results are merged and
  re-ranked. Verified live: a single query returned 32 sources across all channels.
- Removed the `LOCAL_FOUND_SCORE` fallback gate; updated the README flow.

## 2026-06-06

### PDF upload/manage restored + thorough delete + polished README
- Fixed the upload UI not showing: `chat_logic` now loads `.env` before reading
  `ENABLE_LOCAL_RAG`, and `local_rag_enabled()` is read live. With local RAG on, the
  sidebar shows **＋ Add papers**, the indexed **count**, and **Your papers** (delete).
- **Delete removes everything**: chunks (CLOB embeddings + native vectors), concept
  links, now-orphaned concepts, the papers row, the PDF file, and any cached parse.
- Rewrote `README.md` to be friendly and visual (badges, mermaid flow, collapsible
  quick-start, source/feature tables). Removed model-name mentions of "Claude" from
  the docs.

### Search everywhere, free, no key
- Added **free, keyless** channels so external search works with no API key:
  **DuckDuckGo** general web (POST endpoint), **Semantic Scholar** papers, and
  **Wikipedia** — alongside arXiv, GitHub, patents (Google Patents via the web
  provider), and full online-PDF reading.
- `WEB_SEARCH_PROVIDER=duckduckgo` is the free default; Tavily/Brave/SerpAPI remain
  optional for higher-quality web. `get_web_provider()` always returns a provider
  now (free DDG fallback), so web search is always available.
- Raised breadth: `EXTERNAL_TOP_K=20`, arXiv 6 / Semantic Scholar 6 / web 8 /
  GitHub 5 / Wikipedia 3 / patents 4. `safe_get` gained POST support (for DDG).
- Verified live (no key): a query returned ~20 cited sources spanning web, papers,
  Wikipedia, patents, GitHub, and read PDFs. SSRF guard kept (security, not a
  content restriction). 55 tests pass.

### Deeper, more accurate sources + read full papers → code/simulation
- **Reads full paper PDFs**, not just abstracts: the top arXiv results are
  downloaded and parsed (page-numbered), so the model has the actual methods to
  work from. `ARXIV_READ_PDF_COUNT` (default 3).
- **Much more evidence per source** to the model: `EVIDENCE_CHARS_PER_SOURCE`
  3500 (was ~900), so answers are deeper and accurate.
- **Keeps more sources** (`EXTERNAL_TOP_K` 15; arXiv 6 / GitHub 5 / web 8) and a
  **bigger answer budget** (`ANSWER_MAX_TOKENS` 4096) for complete code / simulations.
- **Better search queries**: a generic `clean_query()` turns long natural-language
  questions into keyword queries for the search APIs (the full question is still
  used for the LLM + re-ranking) — fixes "long question → no results". No domain
  dictionaries; retrieval stays broad. arXiv switched to HTTPS.
- Prompt now explicitly: read the method, then write complete runnable original
  code / a small simulation, cited. Verified: a "explain X and give runnable code"
  query returned cited sources + a 7k-char answer with working code.

### Automatic external search (no toggle) + research papers & patents
- Removed the **Web** toggle button. External search is now **automatic**: if the
  local papers don't answer a question (top relevance < `LOCAL_FOUND_SCORE`), or
  local RAG is off, the assistant automatically falls back to external sources.
- Added two channels: **research papers via arXiv** (free, no key — new
  `research_paper` source type) and **patents** via the web provider (Google
  Patents focus — new `patent` type), alongside the existing web / GitHub /
  online-PDF channels. All de-duped + reranked, each cited with URL/file/page.
- `ENABLE_WEB_SEARCH` defaults **on**; arXiv + GitHub work with **no key**, so the
  fallback is useful even without a paid web key (a key adds web pages + patents).
  `/api/chat` no longer needs a `web_search` flag. New source-card badges
  (Research / Patent). Verified live: a question with no key returned 4 arXiv +
  3 GitHub cited sources. 52 tests pass.

### Web-search-first (production): local RAG now optional & off by default
- Flipped the product to a **web-search assistant**: web/GitHub/online-PDF search
  is the primary, always-on knowledge source. The local Oracle/PDF RAG is now
  **optional** behind `ENABLE_LOCAL_RAG` (default **false**).
- The app now **boots and answers with no Oracle and no torch** — verified
  `import webapp.server` pulls neither at load (heavy local-RAG imports are lazy,
  only when `ENABLE_LOCAL_RAG=true`). Makes it deployable without the database.
- External re-ranking defaults to the **fast lexical scorer**; the cross-encoder
  is opt-in (`EXTERNAL_RERANK_CROSS_ENCODER`) so web-only stays light.
- `/api/chat` defaults `web_search=true`; `/api/config` reports `local_rag_enabled`.
  The UI hides the Add-papers / library controls and adapts the welcome copy in
  web mode. If no source is configured, the answer explains how to enable one.

### External knowledge retrieval (optional web / GitHub / online-PDF search)
- New `backend/external_search/` package: a provider layer (web search via
  Tavily/Brave/SerpAPI, GitHub repo/code via the REST API, online PDF reader) with
  SSRF-guarded + size/timeout-capped HTTP, a TTL disk cache (`data/external_cache/`),
  de-dup, and rerank against the query. Cited `ExternalSource` records carry
  source_type (web/github_repo/github_code/online_pdf/local_pdf) + url/file/line/page.
- Wired into `chat_logic` as a **separate, additive** evidence channel — local PDF
  RAG is unchanged and preferred. A **Web** toggle (top bar) appears when
  `ENABLE_WEB_SEARCH=true` + a provider key is set; default off otherwise. Failures
  are non-blocking (warning toast, local answer continues). Source cards show the
  type badge + URL/file/page. API keys stay server-side; never logged.
- Tests: mocked-network unit tests for URL safety/SSRF, HTML extraction, provider
  parsing, GitHub parsing, PDF-failure handling, dedup, ranking, and formatting.
- Docs + `.env.example` updated; re-added `beautifulsoup4` for HTML extraction.

### One optimized retrieval mode + richer Gemini embeddings
- Removed Fast / Balanced / Deep "research modes". The app now always runs a single
  `DEFAULT_RETRIEVAL_SETTINGS` config (vector/BM25/rerank top-k 24, ≤2 sources per
  paper, up to 12 sources) tuned for high accuracy with good speed. Back-compat kept:
  `normalize_mode()`→`"Default"`, `get_mode_settings()`→the single config,
  `apply_research_mode(mode)` ignores `mode`; `/api/chat` still accepts (and ignores)
  a `mode` field. Architecture unchanged (vector + HyDE + BM25F + RRF + rerank + MMR).
- Gemini embeddings now use light structure: queries as
  `task: question answering | query: …` and documents as
  `title: … | section: … | concepts: … | text: …` (ingestion joins paper title +
  section + audio concepts). Re-embedded the corpus (102 chunks) and re-migrated
  the vector column. Output stays 768-dim, L2-normalized.
- Updated tests (single-mode assertions + `run.py --help`) and docs.

### Dark-mode footer fix + live status dot
- The sidebar footer (the **model** label and **"papers indexed"**) was nearly
  invisible in dark mode — it used `--muted-2` on the dark panel. Bumped those
  labels to `--text-soft` so they're clearly readable in both themes.
- Added a gentle pulse to the status dot so the footer feels alive.

### Launcher cleanup
- Simplified `run.py` to a local-only launcher: removed LAN sharing, firewall setup,
  and the `--share` / `--local` / `--host` CLI paths. Kept `--port` and stale-port
  cleanup.

## 2026-06-05

### Dead-code sweep
- Trimmed `backend/config.py` to only what's imported (PAPERS_DIR + ORACLE_* creds
  + data-dir creation); removed ~10 unused constants that nothing read (the live
  code reads those settings from the environment directly).
- Removed the unused `safe_name()` helper in `pdf_parser.py`.
- Verified: vulture finds no high-confidence dead code; every remaining module is
  either in the live app chain or an intentional `python -m` CLI tool (DB admin,
  eval). All 20 tests pass.

### Q/A distinction + model dropdown fix
- Added **"You" / "Answer" role tags** so questions and answers are instantly
  distinguishable; the answer bubble gets an accent left-stripe and lifts on hover.
- Fixed the **Model dropdown text overlap** (added arrow padding + max-width +
  ellipsis) and shortened labels to drop the redundant `vendor/` prefix
  (e.g. "OpenRouter · deepseek-v4-flash").

### UX cleanup per feedback
- **No startup prompt:** `run.py` is now **local by default** (binds 127.0.0.1) — no
  Windows firewall/permission popup on launch.
- **Delete a question instantly:** removed the confirmation dialog — the trash icon
  deletes the question + its answer immediately.
- **Cleaner top bar:** removed the Mode and Sources-count selectors (sensible
  defaults used instead). Kept just the Model picker, the Sources bar, and the
  dark-mode toggle.
- **Removed the "AI" avatar** next to answers; assistant replies are now clean
  full-width bubbles.
- **Crisper type:** enabled Inter stylistic sets + tighter tracking + grayscale
  smoothing for a less "dull", more premium feel.

### Targeted upgrade: speed, accuracy, interactive UI
Web research confirmed the existing stack (Docling parser + Gemini embeddings) is
already 2026 best-in-class for this hardware, so we improved *on top of* it rather
than rewriting:
- **Speed + accuracy:** benchmarked the candidate models on the indexed papers with
  `evaluate_llm`. **`deepseek/deepseek-v4-flash` won on both** — 94% keypoint
  coverage *and* ~8.6s/answer (≈2× faster than qwen3-32b / deepseek-v4-pro at equal
  accuracy). Set it as the default model everywhere; reordered the dropdown so it's
  first.
- **Accuracy eval, tailored:** rewrote `data/llm_eval_questions.json` to match the
  actual corpus (deep-learning speech enhancement / denoising / dereverberation).
  Measured accuracy jumped from a misleading 19% (generic questions) to **94%** —
  the system was always accurate; the old questions just didn't match the papers.
- **Interactive UI:** live **elapsed-time** counter while generating, a shimmering
  status line, "Found N relevant passages…" status, a **speed + model badge** on each
  finished answer (e.g. "⚡ 8.6s · deepseek-v4-flash"), and source cards that lift on
  hover. No more dull, frozen-looking waits.

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
