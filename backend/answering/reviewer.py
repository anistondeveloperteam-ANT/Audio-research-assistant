"""
Automated peer reviewer.

Give it a piece of technical writing — a paper, abstract, proposal, or your own
draft — and it returns a structured, constructive review: summary, strengths,
weaknesses, clarifying questions, concrete suggestions, per-criterion scores, and
a recommendation. It is the "review" stage of an AI-research workflow.

    python -m backend.answering.reviewer paper.txt
    python -m backend.answering.reviewer "paste an abstract here"
    type draft.md | python -m backend.answering.reviewer        # from stdin

Idea (a "review system") adapted from the Awesome-AI-Scientist survey
(https://github.com/ResearAI/Awesome-AI-Scientist) — a reading list, so this is an
original implementation, not copied code. Uses the project's configured LLM.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from backend.llm.streaming_provider import get_provider

MAX_CHARS = int(os.getenv("REVIEW_MAX_INPUT_CHARS", "14000"))
REVIEW_MAX_TOKENS = int(os.getenv("REVIEW_MAX_TOKENS", "2200"))

_SYSTEM = (
    "You are a rigorous but fair peer reviewer for scientific and technical work. "
    "Given a piece of writing (paper, abstract, proposal, or draft), write a constructive "
    "review. Be specific and actionable — refer to concrete parts of the text, not vague "
    "generalities. Reply with ONLY a JSON object, no prose, in exactly this shape:\n"
    '{"summary": "2-3 sentences on what the work claims and does",\n'
    ' "strengths": ["..."], "weaknesses": ["..."],\n'
    ' "questions": ["clarifying questions for the authors"],\n'
    ' "suggestions": ["concrete, actionable improvements"],\n'
    ' "scores": {"novelty": 1-10, "soundness": 1-10, "clarity": 1-10, "significance": 1-10},\n'
    ' "recommendation": "accept" | "minor revision" | "major revision" | "reject",\n'
    ' "confidence": 1-5}'
)

# When a task/topic is supplied, the reviewer FIRST judges topical relevance so that
# off-topic work (however clean) cannot be waved through as "verified".
_SYSTEM_RELEVANCE = (
    "You are a rigorous peer reviewer. You are given THE USER'S REQUEST / TOPIC and a piece of "
    "work produced for it (an answer and/or a program). FIRST judge TOPICAL RELEVANCE: does the "
    "work actually address that request and topic? Work that solves an UNRELATED problem must get "
    "a low relevance score and recommendation 'reject', no matter how clean or correct it is. "
    "Reply with ONLY a JSON object, no prose:\n"
    '{"summary": "2-3 sentences", "strengths": ["..."], "weaknesses": ["..."],\n'
    ' "suggestions": ["concrete, actionable improvements"],\n'
    ' "scores": {"relevance": 1-10, "soundness": 1-10, "clarity": 1-10, "significance": 1-10},\n'
    ' "recommendation": "accept" | "minor revision" | "major revision" | "reject", "confidence": 1-5}'
)

RELEVANCE_MIN = int(os.getenv("REVIEW_RELEVANCE_MIN", "6"))


def is_relevant(rev: Dict[str, Any]) -> bool:
    """False only when the reviewer clearly flagged the work as off-topic. A missing
    relevance signal is treated as relevant (don't block when the gate can't judge)."""
    if not rev or rev.get("error"):
        return True
    rel = (rev.get("scores") or {}).get("relevance")
    if rel is not None:
        try:
            if int(rel) < RELEVANCE_MIN:
                return False
        except (TypeError, ValueError):
            pass
    return rev.get("recommendation") != "reject"


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def review(text: str, task: str = "") -> Dict[str, Any]:
    """Return a structured peer review of `text` (empty dict on failure). When `task`
    (the user's request/topic) is given, the review also scores topical relevance."""
    text = (text or "").strip()
    if not text:
        return {}
    provider = get_provider()
    if not provider.is_available:
        message = getattr(
            provider,
            "unavailable_message",
            lambda: "LLM not available - set the selected provider API key in .env.",
        )()
        return {"error": message}
    task = (task or "").strip()
    system = _SYSTEM_RELEVANCE if task else _SYSTEM
    user = ((f"THE USER'S REQUEST / TOPIC:\n{task[:3000]}\n\n" if task else "")
            + f"Review the following work:\n\n{text[:MAX_CHARS]}")
    raw = "".join(provider.stream_chat(
        [{"role": "user", "content": user}], system=system,
        max_tokens=REVIEW_MAX_TOKENS, temperature=0.2)).strip()
    out = _parse_json(raw)
    if not out:
        return {"error": "Could not parse the review.", "raw": raw[:500]}
    return out


# ----------------------------------------------------------------------
def _main() -> int:
    import sys
    from pathlib import Path

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    arg = " ".join(sys.argv[1:]).strip()
    if arg and Path(arg).is_file():
        text = Path(arg).read_text(encoding="utf-8", errors="ignore")
    elif arg:
        text = arg
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Give some text, a file path, or pipe text in. Nothing to review.")
        return 2

    print("Reviewing…\n")
    r = review(text)
    if r.get("error"):
        print("Error:", r["error"])
        return 1

    def _section(title, items):
        if items:
            print(f"{title}:")
            for it in items:
                print(f"  - {it}")
            print()

    print("=" * 60)
    print("SUMMARY:", r.get("summary", ""))
    print("=" * 60 + "\n")
    _section("STRENGTHS", r.get("strengths"))
    _section("WEAKNESSES", r.get("weaknesses"))
    _section("QUESTIONS", r.get("questions"))
    _section("SUGGESTIONS", r.get("suggestions"))
    sc = r.get("scores") or {}
    if sc:
        print("SCORES (1-10): " + " · ".join(f"{k}={v}" for k, v in sc.items()))
    print(f"RECOMMENDATION: {r.get('recommendation', '?')}  (confidence {r.get('confidence', '?')}/5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
