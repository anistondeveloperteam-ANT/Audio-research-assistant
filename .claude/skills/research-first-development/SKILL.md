---
name: research-first-development
description: "Before writing new code, research existing solutions (GitHub code search, primary docs, package registries) and prefer adopting a proven approach over hand-rolling. Keeps net-new code minimal and battle-tested."
origin: ECC (adapted)
---

# Research-First Development Skill

A short discipline: **research before you implement.** Most "new" code already exists in a
better-tested form. This skill is the front door to the project's development workflow
(`.claude/rules/common/development-workflow.md`) — run it before any non-trivial feature.

## When to Use

- Before implementing a new capability, utility, integration, or algorithm.
- Before adding a dependency or hand-writing something a library already does.
- When a request says "take from this repo" / "implement like X" — confirm what's reusable first.

## The Loop

1. **GitHub first.** Search for existing implementations and patterns:
   `gh search code` / `gh search repos`. Read 1–3 real examples before designing.
2. **Primary docs second.** Confirm API behavior and version specifics from the vendor docs
   (or Context7) — not memory.
3. **Registries.** Check PyPI/npm for a maintained library before writing utility code.
4. **Broader web only if needed.** Use web search after the first three are insufficient.
5. **Decide:** adopt > port > wrap > write-new. Prefer a proven approach that meets the bar.

## Apply It Honestly (project rules)

- **Take ideas, not whole repos.** Port the useful 5–20% as *original* code; cite the source
  and respect its license. Do **not** bulk-import large packs (see `CLAUDE.md`).
- **Reuse this project's own building blocks** before adding new ones: `get_provider()`,
  `hybrid_retrieve`, `gather_external_evidence`, `format_evidence`, `TwoTierMemory`,
  the Docker `code_runner`, the reviewer. Don't duplicate them.
- **Keep it small and compatible.** Match existing patterns; add/adjust tests; verify with the
  repo's `.venv` (`pytest`, `pyflakes`) before claiming done.

## Anti-Patterns

- Writing a parser/cache/retry/HTTP client from scratch when a stdlib/library exists.
- Copying code verbatim from a repository (license + drift risk) instead of reimplementing.
- Cloning an entire framework/pack for one helper.
- Implementing before reading even one real-world example of the same problem.

## Output

A 3–5 line "research note" before coding: what already exists, what you'll reuse vs. write,
the chosen approach, and any license/attribution. Then implement the minimal original piece.
