"""
Code-intent detection — the single source of truth for routing a query to the autonomous
code agent (write -> run in sandbox -> verify against generated tests) instead of the
prose/citation answer pipeline.

Domain-independent: it keys off *coding* verbs/nouns ("code", "implement", "python script",
"simulate", "code for X"), never off any specific algorithm or library. Mirror these patterns
in webapp/static/app.js::looksLikeCodingTask so the UI fast-path and the server agree.
"""
from __future__ import annotations

import re

# Strong coding verbs — on their own these signal "build/run code", any domain.
_STRONG = re.compile(
    r"\b(implement|simulate|simulation|benchmark|refactor|debug|optimi[sz]e|leetcode)\b")

# A request verb followed (within a short window) by a code noun: "give me ... code",
# "write a ... function", "generate a ... script".
_REQUEST_CODE = re.compile(
    r"\b(write|give|gen|generate|show|build|create|make|provide|produce|need|want)\b"
    r".{0,40}\b(code|script|program|function|implementation|snippet)\b")

# "python" next to a code noun in either order: "python code", "code ... in python".
_PYTHON_CODE = re.compile(
    r"\bpython\b.{0,40}\b(code|script|program|function|implementation|snippet|class)\b"
    r"|\b(code|script|program|function|implementation|snippet|class)\b.{0,40}\bpython\b")

# A code noun with a purpose preposition: "code for X", "script that ...", "implementation of X".
_CODE_PREP = re.compile(
    r"\b(code|script|snippet)\s+(for|to|that|which)\b|\bimplementation\s+(of|for)\b")


def is_code_intent(query: str) -> bool:
    """True when the query asks for code to be written/run (route to the code agent)."""
    s = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+# ]", " ", (query or "").lower())) + " "
    return bool(
        _STRONG.search(s)
        or _REQUEST_CODE.search(s)
        or _PYTHON_CODE.search(s)
        or _CODE_PREP.search(s)
    )


# Strong CALCULATION / REASONING cues: the user wants the question worked through and ANSWERED, not a
# program written. "show your reasoning/work/steps", "how much/many", "explain/derive/prove/estimate",
# "what is the value", "step by step". Note "show <your/the/me> work" (your steps) — NOT "show X
# working" (demonstrate X), so a code task like "show MVDR working on signals" is not mistaken for one.
_REASONING_CUE = re.compile(
    r"\bhow (?:much|many)\b"
    r"|\bshow (?:(?:your|the|all|my|our|us|me)\s+){1,2}(?:full\s+)?"
    r"(?:reasoning|work|working|working out|steps?|calculation|calculations|math)\b"
    r"|\bstep[ -]by[ -]step\b"
    r"|\bwhat(?:'?s| is) the value\b"
    r"|\bwork(?:\s+(?:it|this|them))?\s+out\b"
    r"|\b(?:explain|derive|prove|estimate)\b")

# Any explicit code-production signal. If ANY of these appear the question is NOT a pure reasoning
# question (the user wants software), so the reasoning veto must not fire. Kept to UNAMBIGUOUS code
# words — NOT 'function'/'class'/'algorithm', which also appear in maths/CS-theory questions (those
# are guarded by is_code_intent instead).
_CODE_WORD = re.compile(
    r"\b(code|coding|script|program|programme|pseudocode|implement\w*|simulate|simulation|"
    r"python|javascript|c\+\+|c#|runnable|sandbox|compile)\b")


def is_reasoning_question(query: str) -> bool:
    """True when the query is a CALCULATION / REASONING question to be ANSWERED by working through it
    (prose, with steps) — NOT a program to be written or run. Requires a strong reasoning cue AND the
    ABSENCE of any code-production signal. The mere presence of numbers, a formula, or a needed numeric
    result does NOT make a question a code task, so this VETOES code routing for such questions —
    regardless of what a semantic classifier guesses. A genuine 'write a function ...' carries a code
    word/verb and is therefore never a reasoning question."""
    s = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+# ]", " ", (query or "").lower())) + " "
    if _CODE_WORD.search(s) or is_code_intent(query):
        return False
    return bool(_REASONING_CUE.search(s))


# A request to SHOW THE WORKING (the user wants steps, not a looked-up fact): "show your work/reasoning/
# steps", "showing your working", "step by step".
_SHOW_WORK = re.compile(
    r"\bshow(?:ing)?(?:\s+\w+){0,3}\s+"
    r"(?:reasoning|work|working|working out|steps?|calculation|calculations|math)\b"
    r"|\bstep[ -]by[ -]step\b")

# Explicit arithmetic in the query itself: "17*23", "17 x 23", "12 times 11". Checked on the RAW text
# because the operators are stripped by the word-level sanitizer used elsewhere.
_ARITH = re.compile(r"\d\s*[+*/×·÷x]\s*\d|\b\d+\s+times\s+\d+\b")


def is_self_contained_calculation(query: str) -> bool:
    """A SELF-CONTAINED CALCULATION the model should WORK OUT directly (compute once, show the steps) and
    that should SKIP retrieval — so a calculation is never dragged through document/web search or given
    irrelevant citations. It requires a reasoning cue, an actual number, AND a STRONG calculation signal:
    either an explicit request to SHOW THE WORKING / step-by-step, or explicit arithmetic in the query.

    The strong-signal requirement is deliberate: a bare 'how much / how many ...' is NOT enough, because
    factual lookups ('how much funding did X get in 2023', 'how many people attended Y') use exactly that
    phrasing and an incidental year supplies a digit — those must still RETRIEVE, not be answered from
    un-sourced reasoning. Broader 'explain X' concept questions (no number) also keep the retrieval path."""
    if not is_reasoning_question(query):
        return False
    s = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+# ]", " ", (query or "").lower())) + " "
    if not re.search(r"\d", s):
        return False
    return bool(_SHOW_WORK.search(s) or _ARITH.search((query or "").lower()))
