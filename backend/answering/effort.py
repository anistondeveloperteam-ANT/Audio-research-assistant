"""
effort.py — Scale research effort to the question's actual complexity.

The research pipeline used to do MAXIMUM work on every question: plan several "angle" sub-questions,
search every source channel for each, and run multiple verify->rewrite loops with broad re-searches —
slow and API-expensive even for a one-line factual question. This module gauges, with a cheap
DETERMINISTIC heuristic (no LLM call, so no latency and no token cost), how much work a question
actually needs and returns an effort budget the pipeline obeys:

  - simple / single-intent   -> 0 planned angles, 1 verify pass (no rewrite re-search)
  - genuinely complex / multi -> up to the configured angle/loop caps

The gauge is BIASED TO SIMPLE: heavy multi-angle research is the exception (only on clear complexity
signals), not the default. The caps are passed IN by the caller, so the existing fast/deep knobs stay
the single source of truth for the ceilings — this module only decides 0-vs-cap. No new dependency.
"""
from __future__ import annotations

import os
from typing import NamedTuple


def effort_scaling_enabled() -> bool:
    """Live env read (default ON). When off, the pipeline keeps the legacy full-budget behaviour —
    every question planned to the mode's angle cap and the mode's full verify-loop cap — so the
    effort gauge can be disabled per deploy or in tests without touching anything else."""
    return os.getenv("EFFORT_SCALING", "true").strip().lower() not in ("0", "false", "no", "off")

# Multi-part / depth signals. Matched as lowercase substrings against the space-padded question, so
# tokens like " vs " only fire as whole words. The bias is intentionally toward SIMPLE: a question is
# complex only when it clearly asks for a comparison, an enumeration, multiple parts, or explicit depth.
_COMPLEX_CUES = (
    "compare", "comparison", "contrast", " vs ", " vs.", "versus",
    "difference between", "differences between", "similarities and differences",
    "pros and cons", "advantages and disadvantages", "benefits and drawbacks",
    "trade-off", "tradeoff", "trade off", "trade-offs", "tradeoffs",
    "as well as", "in addition to", "and also",
    "list of", "types of", "kinds of", "categories of", "examples of",
    "several", "various", "each of",
    "comprehensive", "in depth", "in-depth", "deep dive", "deep-dive",
    "thorough", "detailed analysis", "step by step", "step-by-step",
    "overview of", "survey of",
)

# A question with more than this many words is treated as complex even with no explicit cue (a long,
# detailed ask is rarely single-intent). Two or more "?" => multiple questions stacked together. Two or
# more " and " => an enumeration of distinct items (one " and " is common in ordinary single questions).
_LONG_WORDS = 30
_MIN_QUESTION_MARKS = 2
_MIN_ANDS = 2


class Effort(NamedTuple):
    """The work budget for one question.

    angles    -- extra "angle" sub-questions to plan (0 = answer the literal question only)
    max_loops -- verify->rewrite passes allowed (1 = a single pass, no re-search loop)
    label     -- "simple" | "complex", for logging / UI
    """
    angles: int
    max_loops: int
    label: str


def is_complex(question: str) -> bool:
    """True only when the question clearly needs multi-angle research: a comparison, an enumeration,
    several parts, or explicit depth. Deterministic and biased to False (simple); no LLM call."""
    ql = (question or "").strip().lower()
    if not ql:
        return False
    padded = f" {ql} "
    if any(cue in padded for cue in _COMPLEX_CUES):
        return True
    if ql.count("?") >= _MIN_QUESTION_MARKS:
        return True
    if padded.count(" and ") >= _MIN_ANDS:
        return True
    if len(ql.split()) > _LONG_WORDS:
        return True
    return False


def assess_effort(question: str, *, angle_cap: int, loop_cap: int) -> Effort:
    """Gauge how much work `question` needs and return a budget bounded by the caller's caps.

    A simple, single-intent question gets NO planned angles and a SINGLE verify pass (no rewrite
    re-search) — typically a few sources and a couple of model calls. A genuinely complex / multi-part
    question gets the full configured budget (angle_cap angles, loop_cap verify passes). The returned
    effort NEVER exceeds the caps the caller passes in, so the fast/deep ceilings still apply."""
    angle_cap = max(0, int(angle_cap))
    loop_cap = max(1, int(loop_cap))
    if is_complex(question):
        return Effort(angles=angle_cap, max_loops=loop_cap, label="complex")
    return Effort(angles=0, max_loops=1, label="simple")
