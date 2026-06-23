# Source-relevance gate — don't force irrelevant retrieved sources into answers

**Date:** 2026-06-20
**Status:** Approved (brainstorming) → implementing inline

## Problem

Retrieval runs and its results are used to ground/cite the answer regardless of
whether they actually address the question. A reasoning-answerable question (e.g. an
audio-storage *calculation*) that retrieves 8 topically-audio-but-irrelevant papers
gets an answer bent to fit those papers, with a spurious citation — instead of the
correct reasoned answer.

## Root cause

CRAG grades only **local** evidence by reranker score and decides *where* to search.
But:
- **External results have no relevance gate** — `_extend_unique(items, ext_items)`
  adds whatever web/arXiv/etc. returned, and those flow into the grounded draft.
- The draft `SYSTEM_PROMPT` says "synthesize the numbered sources" + "cite every
  non-trivial claim [1]…", so the model bends to and cites whatever was retrieved.
- The two existing guards miss this: `_is_evidence_refusal` only fires when the model
  *refuses* (here it complies); `_relevant_sources` only filters the *displayed* panel
  to whatever was *cited* (a cited-but-irrelevant source survives).
- The failure mode is *topically-similar-but-irrelevant*, which **reranker scores
  cannot catch** (high topical similarity, wrong content).

## Fix

### New unit — `backend/answering/relevance_gate.py`

`relevant_source_indices(provider, *, question, items, max_items=12) -> set[int]` —
one bounded, low-token LLM call. Shown the question + a short snippet of each top-N
source, it returns strict JSON `{"relevant": [source numbers that DIRECTLY address
this question]}`. The prompt is explicitly strict: same broad topic ≠ relevant.

- **Fail-OPEN:** disabled / provider unavailable / empty / unparseable / exception →
  return ALL indices (a transient hiccup must never silently strip grounding).
- Sources beyond `max_items` are unjudged → kept (never drop an unseen source).
- `relevance_gate_enabled()` — live env read of `SOURCE_RELEVANCE_GATE` (default on).
- No new dependency.

### Wiring — `webapp/chat_logic.py`

After `items` is assembled from local+external and **before** the existing
`if not items:` branch, run `_apply_relevance_gate(items, q, trace)`:

- Narrow `items` to the relevant subset (a status line reports the drop).
- If the subset is **empty**, `items` becomes `[]` and falls through into the
  EXISTING `if not items:` logic — freshness-sensitive → honest "couldn't find current
  information"; otherwise → `_reasoning_fallback` (reason it out, empty sources, no
  citation). Irrelevant text never enters the draft prompt, so the answer can't be
  bent to it or cite it.

### Contract reinforcement — `SYSTEM_PROMPT`

Add one clause: use ONLY sources that directly address the question; a citation must
directly support its claim; if a source doesn't, don't cite it and don't bend the
answer to it; if none address the question, ignore them and reason, saying the sources
weren't relevant.

## Kept intact

CRAG (decides *where* to search) — the gate is the final usability filter on *what was
found*. `_relevant_sources` (display) and `_is_evidence_refusal` reroute remain as
secondary nets. Direct-reasoning route, code agent, independent verification, honest
labeling — untouched. The gate is fail-open, so existing tests whose mocks don't model
the relevance call keep all sources (current behavior).

## Testing

New `tests/test_source_relevance_gate.py` (offline, deterministic, provider mocked):

- unit: keep-subset / none-relevant→empty / all-relevant / unseen-kept; fail-open on
  disabled / unavailable / bad-JSON / exception; `_RELEVANCE_SYSTEM` content;
- end-to-end via `stream_chat_events`: (a) irrelevant sources → discarded → reasoned
  answer, no citation; (b) a genuinely supporting source → kept + cited; (c) mixed →
  relevant kept, irrelevant dropped;
- `SYSTEM_PROMPT` content (cite-only-supporting / don't-bend clause).

## Demo

3 questions: reasoning-answerable + irrelevant corpus → reasoned/no-cite; genuinely
supported → cited; mixed → relevant kept, irrelevant dropped.

## Verify

```
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pyflakes backend webapp tests
```

## Cost

One extra bounded LLM call per retrieval-grounded answer (gated, fail-open, top-N
capped).
