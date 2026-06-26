"""
experience.py — the "learns day by day" layer (Phase 1: experience / lessons memory).

The assistant LEARNS from its own runs without any model training: every time it CORRECTS itself
(an arithmetic override, a conclusion-matches-work reconcile, a verify->refine rewrite) or the user
REGENERATES to a better answer, a short, GENERALISABLE lesson is distilled and stored. Before drafting
a future SIMILAR question, the top lessons are recalled (scored relevance x recency x confidence) and
injected into the prompt, so the agent stops repeating the same mistakes and matches what the user
prefers. Lessons that produce a verified answer are reinforced; stale ones decay and are pruned.

Deterministic by construction (no extra LLM call to distil — the lesson is built from signals already
computed), recency-weighted (latest wins; old lessons fade), and SAFE: lessons never leak another
answer's content (a preference lesson captures only the answer's SHAPE), lessons are GENERALISABLE
guidance (never a fact) so a number/identifier change is fine, and recall only fires on genuinely
similar questions (a relevance floor high enough to exclude the word-overlap/different-intent band).
Mistake-lessons are captured ONLY from runs that ended verified. Gated by EXPERIENCE_MEMORY (default
on) and fail-open everywhere — it can never break an answer.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.answering import tuning as _tuning


def experience_enabled() -> bool:
    return os.getenv("EXPERIENCE_MEMORY", "true").strip().lower() not in ("0", "false", "no", "off")


def _float_env(name: str, default: float, lo: float) -> float:
    try:
        return max(lo, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _min_relevance() -> float:
    # High enough that a word-overlapping but DIFFERENT-INTENT question can't pull a lesson (the
    # 0.4-0.6 cross-topic band), low enough that the SAME question with a different number/identifier
    # still generalises (those score ~0.85+). Env default; the eval-gated tuner may override it.
    return _tuning.tuned("EXPERIENCE_MIN_RELEVANCE", _float_env("EXPERIENCE_MIN_RELEVANCE", 0.62, 0.0))


def _top_k() -> int:
    try:
        base = max(0, int(os.getenv("EXPERIENCE_TOP_K", "3")))
    except (TypeError, ValueError):
        base = 3
    return int(_tuning.tuned("EXPERIENCE_TOP_K", base))


def _half_life_days() -> float:
    return _tuning.tuned("EXPERIENCE_HALF_LIFE_DAYS", _float_env("EXPERIENCE_HALF_LIFE_DAYS", 30.0, 1.0))


# ---------------------------------------------------------------------------
# Distil a lesson from signals the answering pipeline already produced.
# ---------------------------------------------------------------------------
def lesson_from_outcome(question: str, *, corrections: Sequence[Any] = (),
                        reconciled: bool = False, rewritten: bool = False) -> Optional[str]:
    """A GENERALISABLE 'what went wrong -> the fix' lesson, or None if nothing was corrected. Built
    only from deterministic signals: an arithmetic override (`corrections`), a conclusion-matches-work
    reconcile (`reconciled`), or a verify->refine rewrite (`rewritten`)."""
    parts: List[str] = []
    if corrections:
        parts.append("compute any arithmetic in code and state EXACTLY that computed value "
                     "(a prior answer on a question like this mis-stated a number)")
    if reconciled:
        parts.append("make the final stated result match the answer's own derivation "
                     "(a prior answer here contradicted its own work)")
    if rewritten:
        parts.append("ground each claim and self-check before finalizing "
                     "(a prior draft here failed verification and needed a rewrite)")
    if not parts:
        return None
    return "On questions like this, " + "; ".join(parts) + "."


_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,3}\s|\n\s*[-*]\s|\n\s*\d+\.\s")


def _answer_shape(answer: str) -> List[str]:
    """Deterministic SHAPE features of a preferred answer (length / code / structure) — captured
    instead of the answer's content, so a preference lesson can never leak wrong facts into a later
    answer; it only nudges the format the user prefers."""
    text = answer or ""
    n = len(text)
    feats = ["a detailed, in-depth answer" if n > 1500
             else ("a short, concise answer" if n < 400 else "a medium-length answer")]
    if "```" in text:
        feats.append("with runnable code")
    if _HEADING_RE.search(text):
        feats.append("organized into clear sections / bullet points")
    return feats


def preference_lesson(question: str, answer: str) -> Optional[str]:
    """A GENERALISABLE style preference distilled from a regeneration the user kept — SHAPE only, no
    content. None when the answer is too thin to read a shape from."""
    if not (answer or "").strip() or len((answer or "").strip()) < 40:
        return None
    return ("The user regenerated a similar question and preferred "
            + ", ".join(_answer_shape(answer)) + " — aim for that shape here (style only).")


# ---------------------------------------------------------------------------
# Recall (inject into the draft prompt) + capture (after the answer is finalized).
# ---------------------------------------------------------------------------
def format_lessons_block(lessons: Sequence[Dict[str, Any]]) -> str:
    """The prompt block injected before drafting. Empty when there are no relevant lessons."""
    items = [(L.get("content") or "").strip()[:300] for L in lessons if (L.get("content") or "").strip()]
    if not items:
        return ""
    lines = ["LEARNED FROM EXPERIENCE — guidance distilled from earlier answers on similar questions. "
             "Apply each ONLY where it genuinely fits THIS question (most relevant first):"]
    lines += ["- " + it for it in items]
    return "\n" + "\n".join(lines) + "\n"


def recall(mem, *, user_id: str, question: str,
           query_embedding: Optional[List[float]] = None,
           query_meta: Optional[str] = None) -> Tuple[str, List[int]]:
    """Return (prompt_block, lesson_ids) for THIS question. Fail-open: ("", []) on disabled / any
    error. The ids let the caller reinforce the lessons that helped, after a verified answer."""
    if not experience_enabled():
        return "", []
    try:
        lessons = mem.recall_lessons(
            user_id=user_id, question=question, query_embedding=query_embedding,
            query_meta=query_meta, min_relevance=_min_relevance(), top_k=_top_k(),
            half_life_days=_half_life_days())
    except Exception:                                  # noqa: BLE001 - recall must never break a draft
        return "", []
    return format_lessons_block(lessons), [int(L["id"]) for L in lessons if L.get("id") is not None]


def capture_outcome(mem, *, user_id: str, question: str, answer: str,
                    corrections: Sequence[Any] = (), reconciled: bool = False,
                    rewritten: bool = False, regenerated: bool = False, verified: bool = True,
                    query_embedding: Optional[List[float]] = None,
                    query_meta: Optional[str] = None, logic_version: int = 0) -> None:
    """Learn from a finished answer: store a MISTAKE lesson when the pipeline corrected itself, and a
    PREFERENCE lesson when the user regenerated to a verified answer. Fail-open; never raises."""
    if not experience_enabled():
        return
    try:
        # Only learn from a run that ENDED WELL: a mistake-lesson from an answer that still failed
        # verification is a "fix" that didn't actually work — storing it would teach from a bad run.
        mistake = lesson_from_outcome(question, corrections=corrections, reconciled=reconciled,
                                      rewritten=rewritten) if verified else None
        if mistake:
            mem.record_lesson(user_id=user_id, kind="mistake", question=question, content=mistake,
                              source="auto_correction", embedding=query_embedding,
                              embedding_meta=query_meta, confidence=1.0, logic_version=logic_version)
        if regenerated and verified:
            pref = preference_lesson(question, answer)
            if pref:
                mem.record_lesson(user_id=user_id, kind="preference", question=question, content=pref,
                                  source="regeneration", embedding=query_embedding,
                                  embedding_meta=query_meta, confidence=1.0, logic_version=logic_version)
    except Exception:                                  # noqa: BLE001 - learning must never break a turn
        pass
