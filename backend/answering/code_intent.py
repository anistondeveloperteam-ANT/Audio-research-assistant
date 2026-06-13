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
