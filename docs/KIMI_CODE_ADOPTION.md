# Borrowing from Kimi Code — One Idea, Done Right

**Repo reviewed:** `MoonshotAI/kimi-code` · **Date:** 2026-06-08
**Scope of this document:** *only* the change we adopted from that repo — nothing
else in the project is described or altered here.

---

## 1. What Kimi Code is (and why the full repo is the wrong fit)

Kimi Code is a **TypeScript terminal AI‑coding‑agent product** — a pnpm monorepo with
a TUI framework (`pi‑tui`), a plugin marketplace, ACP/MCP editor integrations, and
built‑in agent skills. It is a *whole product*, not a library of small liftable files.

> **Verdict: do not use the full repo.** It is TypeScript (our project is Python), it
> is tightly coupled to its own monorepo / terminal‑UI / plugin system, and almost
> everything it does (terminal UI, IDE protocols, plugin store) does not apply to a
> Python **web + agent** application.

### Component‑by‑component verdict

| Kimi Code part | Verdict | Why |
|----------------|---------|-----|
| `apps/` CLI + `pi‑tui` TUI | **SKIP** | Terminal UI; we have a web UI. |
| Plugin marketplace / `.agents/skills` | **SKIP** | Heavy extensibility we don't need. |
| ACP / MCP editor integrations | **SKIP** | IDE‑to‑agent protocols; irrelevant here. |
| `packages/` shared TS libs | **SKIP** | Language + architecture mismatch. |
| **Lifecycle hooks** (execution gates) | **TAKE (as an idea)** | Small, language‑agnostic, and fits our agent's sandbox step. |

---

## 2. The one idea worth taking: **lifecycle hooks**

Kimi Code describes *“lifecycle hooks — local command‑execution gates for
audit/automation.”* In plain terms: **before the agent runs something, a hook gets a
say** — it can record the action and allow or block it.

Our agent already writes a Python program every cycle and runs it in a **throwaway,
network‑less Docker sandbox**. That sandbox is the real safety boundary. The hook adds
two things on top, cheaply:

1. **Audit** — a permanent, reviewable record of *every* program the agent ran.
2. **Gate** — an optional policy that can **block** a run before it starts.

### How the function works

```
agent writes a program
        │
        ▼
  pre_run(code, task)   ◄── the lifecycle hook
        │
        ├── 1. AUDIT  → append one JSON line (time, task, code hash, length, decision)
        │
        ├── 2. POLICY → block if a configured regex matches the code
        │            → OR run an external gate command (code on stdin); non‑zero = block
        │
        ▼
   allowed?  ── no ──►  skip the sandbox; tell the agent "blocked" → it rewrites
        │
       yes
        ▼
   run in Docker sandbox  (as before)
```

The policy is **ALLOW by default** — the container is still the primary protection, so
the gate is opt‑in defense‑in‑depth, while the audit log is always on.

---

## 3. The change / improvement we made

A single, self‑contained module plus a 6‑line splice into the loop. **Nothing else in
the project changed.**

| File | Change |
|------|--------|
| `backend/agent/hooks.py` | **New.** `pre_run(code, task) -> HookDecision`: audits the program, applies the policy, returns allow/block. |
| `backend/agent/loop.py` | Calls `pre_run(...)` right before the Docker run; on block it emits a `blocked` event and feeds the reason back so the agent rewrites. |
| CLI + web UI | Render a `blocked` step (`[BLOCKED] …` / 🛡️ chip). |
| `tests/test_agent.py` | Tests: default‑allow + audit written, regex block, external‑hook block, loop never executes a blocked program. |

### Before → After

| | Before | After |
|---|--------|-------|
| **Visibility** | No record of what code the agent executed. | Every run is logged to `data/logs/agent_audit.jsonl` (time, task, code hash, length, decision). |
| **Control** | The only gate was the sandbox itself. | A policy can **block** a run before it starts (regex blocklist or an external gate command). |
| **Safety posture** | Sandbox only. | Sandbox **+** audit **+** optional pre‑execution gate (defense in depth). |

### Configuration (all optional, in `.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `AGENT_AUDIT_LOG` | `data/logs/agent_audit.jsonl` | Where runs are logged. Empty = auditing off. |
| `AGENT_BLOCK_PATTERNS` | *(empty)* | Comma‑separated regexes; a match blocks the run. |
| `AGENT_PRERUN_HOOK` | *(empty)* | A shell command; the code is piped to its stdin; a non‑zero exit blocks the run. |

---

## 4. Why this is the right amount to borrow

- **Small & original.** One ~90‑line Python module, written from scratch (Kimi Code is
  TypeScript — nothing was copied), implementing only the *idea* of a hook.
- **Fits our architecture.** It slots onto the existing sandbox step; no new
  dependency, no new service, no UI rebuild.
- **Useful for a research assistant.** The audit log makes the agent's actions
  transparent and reviewable — valuable on its own, separate from any blocking.

> **Bottom line:** from a 100‑file TypeScript product we took exactly one concept —
> a pre‑execution lifecycle hook — and implemented it as a tiny, original Python module.
> That is the difference between *using* a repo and *being trapped by* it.
