"""
arithmetic_check.py — deterministic arithmetic self-check for reasoning/calculation answers.

A calculation answer should not STATE a numeric equality that is literally false (e.g. "12 x 12 = 100").
This module finds simple "A op B = C" equalities and, when C is wrong, corrects it to the recomputed
value. It runs on EVERY reasoning answer, so it is SAFE-BY-CONSTRUCTION — it only touches cases where
the intent is unambiguous, and SKIPS everything where "A op B = C" might not be plain binary arithmetic:

  * operands must be plain INTEGERS (a decimal like "3.2" is often a figure/section number, and "1,5"
    is a European decimal — both are skipped);
  * the operator is + * / only (NOT subtraction — "3 - 5 = 2" legitimately means the magnitude |3-5|);
  * the right-hand side must not be a percentage ("3 / 4 = 75%" is a fraction→percent, not 3/4=75) and
    must not be comma-grouped ("1000 * 2 = 2,000" is correct);
  * an operand that is the tail of a longer chain ("... * 180 / 8 = ...") is skipped — the "=" applies
    to the whole chain.

It recomputes the arithmetic itself (never an LLM), so within those guarded cases it can only make a
stated equality TRUE. It deliberately misses many real errors (decimals, subtraction, percentages) —
those are the model's responsibility (the prompt requires it to verify equalities) — but it never
corrupts a correct or innocent line.
"""
from __future__ import annotations

import re
from typing import List, NamedTuple, Tuple

# <int> <op> <int> = <number>. Operands are plain integers (optional sign); the operator excludes '-'
# (subtraction is ambiguous with |a-b|); the RHS may be a decimal (so legitimate rounding is accepted)
# but must NOT be immediately followed by a word char, '.', '%' (percentage) or ',' (thousands group).
_OPND = r"-?\d+"
_OP = r"[+*/x×·÷]"
_RHS = r"-?\d+(?:\.\d+)?"
_EQ_RE = re.compile(rf"(?<![\w.])({_OPND})\s*({_OP})\s*({_OPND})\s*=\s*({_RHS})(?![\w.%,])")


class Equality(NamedTuple):
    text: str
    stated: float
    actual: float
    rhs_span: Tuple[int, int]
    decimals: int


def _to_int(tok: str):
    try:
        return int(tok)
    except (ValueError, TypeError):
        return None


def _apply(a: int, op: str, b: int):
    if op in "*x×·":
        return a * b
    if op in "/÷":
        return None if b == 0 else a / b
    if op == "+":
        return a + b
    return None


def _decimals(tok: str) -> int:
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def _format(value: float, decimals: int) -> str:
    if decimals == 0 or float(value).is_integer():
        return str(int(round(value)))
    return f"{round(value, decimals):.{decimals}f}"


def false_equalities(text: str) -> List[Equality]:
    """Every guarded 'A op B = C' in `text` whose stated C does not equal A op B at the shown precision."""
    out: List[Equality] = []
    for m in _EQ_RE.finditer(text or ""):
        # SAFETY: skip if the left operand is mid-expression (the tail of a chain like "* 180 / 8 = ..."),
        # where the "=" applies to the whole chain, not this pair.
        before = (text[: m.start()]).rstrip()
        if before and before[-1] in "+-*/x×·÷0123456789.,":
            continue
        a, op, b = _to_int(m.group(1)), m.group(2), _to_int(m.group(3))
        if a is None or b is None:
            continue
        try:
            c = float(m.group(4))
        except ValueError:
            continue
        actual = _apply(a, op, b)
        if actual is None:
            continue
        dp = _decimals(m.group(4))
        if abs(round(actual, dp) - c) > 1e-9:        # wrong even after rounding to the shown precision
            out.append(Equality(m.group(0), c, actual, m.span(4), dp))
    return out


def fix_false_equalities(text: str) -> str:
    """Return `text` with every guarded, literally-false 'A op B = C' corrected so C equals A op B
    (recomputed). Conservative by construction (see module docstring): it never rewrites an ambiguous
    line, so it cannot corrupt correct or innocent prose."""
    bad = false_equalities(text)
    if not bad:
        return text
    fixed = text
    for eq in sorted(bad, key=lambda e: e.rhs_span[0], reverse=True):   # right-to-left keeps spans valid
        start, end = eq.rhs_span
        fixed = fixed[:start] + _format(eq.actual, eq.decimals) + fixed[end:]
    return fixed
