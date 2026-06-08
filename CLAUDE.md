# CLAUDE.md - Research Assistant

Project instructions for Claude Code. Keep this file short and trust the code
when docs and implementation disagree.

## Project Shape

- Product: source-grounded research assistant for audio, speech enhancement,
  DSP, papers, web sources, GitHub/code references, and local PDF libraries.
- Runtime stack: Python 3.11, FastAPI, vanilla HTML/CSS/JS, OpenRouter/Ollama,
  external search, optional Oracle local RAG, optional Memgraph GraphRAG.
- Do not add a frontend build step or framework.
- `.env` is local and secret-bearing. Never print it, commit it, or copy values
  into docs/tests/logs.

## Current Entry Points

```powershell
python run.py                           # local web app: http://localhost:8600
python pipeline.py --status             # inspect local PDF index
python pipeline.py                      # build local PDF index when Oracle is on
python -m backend.graph_rag.build_graph # optional Memgraph graph build
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\pyflakes backend webapp tests
```

## Product Contract

- Answers must be grounded in retrieved evidence and cite sources.
- Prefer local paper evidence for project-specific/local-library answers.
- Use external web/research/GitHub sources for current or outside-library facts.
- If evidence is missing or conflicting, say so plainly.
- Never invent paper titles, URLs, metrics, line numbers, or citations.

## Development Rules

- Read the relevant files before editing.
- Keep changes small and compatible with existing patterns.
- Do not overwrite user changes in a dirty worktree.
- Add/update tests for behavior changes.
- Use the repo's `.venv` for verification on Windows.
- For retrieval, GraphRAG, external search, auth, or prompt changes, run the
  focused tests plus the full suite when practical.

## Security Rules

- Keep SSRF protections on by default.
- Do not execute downloaded code.
- Do not weaken auth, session, URL-safety, or secret-handling code.
- Avoid broad shell permissions, destructive git commands, and local-network
  exposure unless explicitly requested.

## Useful Project Rules

Claude Code should also load the focused project overlays in:

- `.claude/rules/common.md`
- `.claude/rules/python.md`

Selected ECC reference rules are available in:

- `.claude/rules/common/`
- `.claude/rules/python/`

Use the reviewer agents in `.claude/agents/` for non-trivial diffs, and the
workflow skills in `.claude/skills/` when the task matches. These selected
reference agents/rules/skills are imported from `affaan-m/ecc` under the MIT
license copied at `.claude/ECC_LICENSE`.
