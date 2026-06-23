# Router: reasoning vs. code — route by intent, not by "math is involved"

**Date:** 2026-06-20
**Status:** Approved (brainstorming) → implementing inline

## Problem

The router that decides *code agent* vs *reasoning/retrieval* misclassifies
quantitative reasoning questions as code tasks. A self-contained calculation
("How much storage for 3 minutes of 44.1 kHz 16-bit stereo audio? Show your
reasoning") gets sent to the autonomous code agent, which writes and runs Python,
instead of being answered directly with worked steps.

The trigger is wrong: *presence of numbers / a required numeric answer* is treated
as *needs the code agent*. Code intent means the **user wants a program / script /
simulation produced** — not that the answer happens to contain a number.

## Root cause

The router has two layers, unioned in
`backend/answering/task_classifier.py::classify()` as
`code_task = llm.code_task or regex.code_task`:

1. **Regex** (`backend/answering/code_intent.py::is_code_intent`) keys off coding
   *nouns/verbs* (`code`, `script`, `implement`, `simulate`, …). It does **not**
   fire on a bare calculation word problem — no code noun → `False`. The UI
   fast-path `webapp/static/app.js::looksLikeCodingTask` mirrors it. **Innocent.**
2. **Semantic LLM** (`task_classifier.py::_SYSTEM_PROMPT`) is the culprit. Its
   first sentence instructs the model to flag anything that asks to *"WRITE, RUN,
   SIMULATE, BENCHMARK, MODEL, **COMPUTE** … produce working code/executed
   results."* Listing **COMPUTE/MODEL** as code triggers makes "compute the storage
   for 3 min of audio" → `code_task=true` → union → code agent.

## Fix (one file)

Rewrite `_SYSTEM_PROMPT` so the classifier decides **what the user wants** — a
program produced/run vs. an answer/explanation — and never decides on the basis of
numbers, formulas, or a required numeric result.

- **`code_task = false` (reasoning):** "compute / calculate / derive / how much /
  how many / what is the value of / show your reasoning / explain / prove /
  estimate" — a quantity a person can work out by arithmetic, unit math, algebra,
  or a closed-form formula. Numbers/formulas alone never make it code.
- **`code_task = true` (code agent):** the user explicitly wants software —
  write/give/build code·script·program·function, implement/refactor/debug/
  benchmark, run/simulate/model a system — **or** the computation genuinely needs
  execution (large / iterative / data-driven beyond hand reasoning).
- Add 3 worked examples; keep the `deterministic | simulation | numeric_algorithm`
  verification taxonomy for genuine code tasks. Single-pass, 80-token budget
  (no added latency).

## Deliberately unchanged

- The regex backstop + the `looksLikeCodingTask` UI mirror (already conservative on
  numbers — no change needed).
- The union semantics (`llm or regex`) — keeps high recall for genuine code the LLM
  might hesitate on; the fix only stops reasoning questions being misrouted, it does
  not weaken the code/simulation route.
- All fallbacks (LLM unavailable / timeout / bad JSON → regex), independent
  verification, honest labeling.

## Testing

New `tests/test_routing_numbers_not_code.py` (offline, deterministic):

- (a) calculation word problems route to **reasoning** — regex layer (real, semantic
  off) **and** semantic layer (mocked LLM returning the corrected verdict);
- (b) a genuine "write a program / simulation" request still routes to the **code
  agent**;
- (c) numbers / formulas alone do not trigger the code path;
- prompt-content assertions pinning the new "wants a program vs. an answer" framing
  and that numbers/`show your reasoning` are not code (the deterministic proxy for
  the model's behavior — matches how this repo already tests LLM prompts).

Existing `test_code_intent.py` / `test_task_classifier.py` stay green (they mock the
provider, so the prompt rewrite cannot break them).

## Demo

3 questions: a calculation word problem → reasoning; "Write Python to …" → code; a
document-fact question → retrieval.

## Verify

```
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pyflakes backend webapp tests
```
