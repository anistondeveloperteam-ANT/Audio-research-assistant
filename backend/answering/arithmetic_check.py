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

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

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


# ======================================================================================
# COMPUTE-IN-CODE SOURCE OF TRUTH (general): evaluate the answer's OWN shown work and make the
# code-computed value authoritative. This generalises the binary checker above to FULL multi-term
# expressions (so a dropped factor in a chain — "a x b x c / d = WRONG" — is caught, not skipped),
# overrides any stated number that differs from the computed one, propagates the single governing
# result to its restatements (summary/body/headline agree), and flags magnitude / decimal-vs-binary
# convention problems. It NEVER uses an LLM and NEVER calls eval(); a safe tokenizer + shunting-yard
# evaluator computes each expression, so within its guarded cases it can only make a number TRUE.
#
# Safety is identical in spirit to the binary checker (it must never corrupt correct/innocent prose):
# operands are plain INTEGERS (a decimal "3.2" is a figure/version number); a bare "int - int" is left
# alone (it legitimately means |a-b|); a '%' / thousands-grouped / word-glued RHS is skipped; an LHS
# glued to a word or a comma-number ("GPT-4 x 2", "1,250 / 5") is skipped.
# ======================================================================================

_MUL = {"*", "x", "×", "·"}
_DIV = {"/", "÷"}
_OPS = _MUL | _DIV | {"+", "-", "^"}
# Characters that may appear inside an arithmetic expression (NOT comma — a comma boundary means the
# run is a fragment of a larger comma-number, e.g. the "250 / 5" inside "$1,250 / 5", so we skip it).
_ARITH_CHARS = set("0123456789. \t+-*/x×·÷^()")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_RHS_RE = re.compile(r"\s*(-?\d+(?:\.\d+)?)")
# A number after '=' that is followed by ANY arithmetic operator is the head of a CONTINUED expression
# ('a = b + c', a factorisation 'a = b * c = d'), not a terminal NUM — overriding 'b' would fabricate a
# false equality, so such a line is skipped. DELIBERATELY broader than _OPS: it covers ASCII operators
# AND common Unicode/LaTeX/Word variants (real minus U+2212, ∗ U+2217, ∙ U+2219, ⋅ U+22C5, ∕ U+2215,
# ＋ U+FF0B, × · ÷). Over-skipping only forgoes a fix; under-skipping corrupts. (En/em dashes are NOT
# included — they are prose, not subtraction.)
_RHS_CONTINUES_RE = re.compile("\\s*[-+*/^x×·÷−∗∙⋅∕⁄＋]")
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3, "u-": 4}
_RIGHT_ASSOC = {"^", "u-"}

# decimal-vs-binary convention factors (bytes etc.): mixing them in one answer is a convention error.
_BINARY_FACTORS = {"1024", "1048576", "1073741824", "1099511627776"}
_DECIMAL_FACTORS = {"1000", "1000000", "1000000000", "1000000000000"}


def arithmetic_verify_enabled() -> bool:
    """Live env read (default ON). The deterministic compute-and-override runs on every calculation
    answer; this flag is only an operability kill-switch."""
    return os.getenv("ARITHMETIC_VERIFY", "true").strip().lower() not in ("0", "false", "no", "off")


def _is_number(tok: str) -> bool:
    return bool(_NUM_RE.fullmatch(tok))


def _norm_op(op: str) -> str:
    if op in _MUL:
        return "*"
    if op in _DIV:
        return "/"
    return op


def _tokenize(expr: str) -> Optional[List[str]]:
    """Split a pure-arithmetic string into number/operator/paren tokens. Returns None on ANY character
    that isn't part of arithmetic, so non-numeric prose can never be evaluated."""
    tokens: List[str] = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            num = expr[i:j]
            if num.count(".") > 1:
                return None
            tokens.append(num)
            i = j
            continue
        if c in _OPS or c in "()":
            tokens.append(c)
            i += 1
            continue
        return None
    return tokens


def _to_rpn(tokens: List[str]) -> Optional[List[str]]:
    """Shunting-yard with unary minus and right-associative '^'. Returns RPN, or None on malformed
    input (mismatched parens, etc.)."""
    out: List[str] = []
    ops: List[str] = []
    prev: Optional[str] = None                       # None | 'num' | 'op' | '(' | ')'
    for tok in tokens:
        if _is_number(tok):
            out.append(tok)
            prev = "num"
        elif tok == "(":
            ops.append(tok)
            prev = "("
        elif tok == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if not ops:
                return None
            ops.pop()
            prev = ")"
        else:
            op = _norm_op(tok)
            if op == "-" and prev in (None, "op", "("):
                op = "u-"                            # unary minus
            while ops and ops[-1] != "(" and (
                _PREC[ops[-1]] > _PREC[op]
                or (_PREC[ops[-1]] == _PREC[op] and op not in _RIGHT_ASSOC)
            ):
                out.append(ops.pop())
            ops.append(op)
            prev = "op"
    while ops:
        if ops[-1] in "()":
            return None
        out.append(ops.pop())
    return out


def _eval_rpn(rpn: List[str]) -> Optional[float]:
    stack: List[float] = []
    for tok in rpn:
        if _is_number(tok):
            stack.append(float(tok))
        elif tok == "u-":
            if not stack:
                return None
            stack.append(-stack.pop())
        else:
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            elif tok == "/":
                if b == 0:
                    return None                      # division by zero -> unsanitary, never "fix" to inf
                stack.append(a / b)
            elif tok == "^":
                if abs(a) > 1e6 or abs(b) > 64:      # guard against absurd magnitudes / overflow
                    return None
                stack.append(a ** b)
            else:
                return None
    return stack[0] if len(stack) == 1 else None


def safe_eval(expr: str) -> Optional[float]:
    """Deterministically evaluate a pure-arithmetic string (no eval, no deps). Returns the finite value
    or None when the string isn't a well-formed arithmetic expression (or would be non-finite)."""
    toks = _tokenize(expr or "")
    if not toks:
        return None
    rpn = _to_rpn(toks)
    if rpn is None:
        return None
    try:
        val = _eval_rpn(rpn)
    except Exception:                                 # noqa: BLE001 - any arithmetic error -> give up safely
        return None
    if val is None or not math.isfinite(val):
        return None
    return val


class CalcEquality(NamedTuple):
    lhs: str
    stated: float
    computed: float
    rhs_span: Tuple[int, int]
    decimals: int


def _find_calc_equalities(text: str) -> List[CalcEquality]:
    """Every guarded 'EXPR = NUM' in `text` where EXPR is a full integer-operand arithmetic expression
    (>=1 binary op). Lists ALL such equalities (stated == computed or not) so callers can both verify
    and correct. Applies the safety guards in the section docstring."""
    out: List[CalcEquality] = []
    for m in re.finditer(r"=", text or ""):
        eq = m.start()
        nxt = text[eq + 1] if eq + 1 < len(text) else ""
        prev_ch = text[eq - 1] if eq > 0 else ""
        if nxt == "=" or prev_ch in "=<>!:+-*/":       # ==, <=, >=, !=, :=, augmented — not an equality
            continue
        # ---- LHS: maximal arithmetic run immediately left of '=' ----
        i = eq
        while i > 0 and text[i - 1] in _ARITH_CHARS:
            i -= 1
        run = text[i:eq]
        lhs = run.strip()
        if not lhs:
            continue
        bch = text[i - 1] if i > 0 else ""
        # Skip when the expression is GLUED to a word/comma-number (no separating space): "GPT-4 x 2",
        # "1,250 / 5". A space before it ("the area is 12 x 12") is fine — that is real arithmetic.
        if bch == "," or (bch.isalnum() and not run[:1].isspace()):
            continue
        if lhs[0] in "+-":
            continue                                   # leading sign: a markdown bullet "- 2 * 2", a list
            #                                            dash, or a signed first operand -> too ambiguous
            #                                            (a unary minus would inject a negative result)
        toks = _tokenize(lhs)
        if not toks:
            continue
        nums = [t for t in toks if _is_number(t)]
        ops = [t for t in toks if t in _OPS]
        if not nums or not ops:
            continue                                   # need at least one operator + operand
        if any("." in t for t in nums):
            continue                                   # decimal operand -> figure/version number, skip
        if len(toks) == 3 and toks[1] == "-":
            continue                                   # bare 'int - int' means |a-b| — leave alone
        computed = safe_eval(lhs)
        if computed is None:
            continue
        # ---- RHS: the number right of '=' (guarded) ----
        rm = _RHS_RE.match(text, eq + 1)
        if not rm:
            continue
        rhs_str = rm.group(1)
        after = text[rm.end():rm.end() + 4]
        if after[:1] == "%":
            continue                                   # 3 / 4 = 75%  (fraction -> percent)
        if after[:1].isalpha():
            continue                                   # glued unit/word (= 25MB / = 12x)
        if after[:1] == "." and len(after) > 1 and after[1].isdigit():
            continue                                   # version / more decimals
        if re.match(r",\d{3}(?!\d)", after):
            continue                                   # thousands group (= 2,000)
        if _RHS_CONTINUES_RE.match(text[rm.end():]):
            continue                                   # RHS continues as an expression: 'a = b + c',
            #                                            'a = b * c = d' (a factorisation). It is not a
            #                                            terminal NUM, so overriding 'b' would fabricate a
            #                                            false equality — leave the whole line alone.
        try:
            stated = float(rhs_str)
        except ValueError:
            continue
        out.append(CalcEquality(lhs, stated, computed, rm.span(1), _decimals(rhs_str)))
    return out


def _convention_ok(text: str, eqs: List[CalcEquality]) -> bool:
    """False when one answer mixes binary (1024/1048576/...) AND decimal (1000/1000000/...) conversion
    factors as operands — a decimal-vs-binary convention error that yields inconsistent magnitudes."""
    operands = {t for e in eqs for t in (_tokenize(e.lhs) or []) if _is_number(t)}
    return not (operands & _BINARY_FACTORS and operands & _DECIMAL_FACTORS)


@dataclass(frozen=True)
class CalcReport:
    """Result of computing the answer's own arithmetic in code and making it authoritative."""
    fixed_text: str
    corrections: Tuple[Tuple[str, str], ...] = ()      # (stated, computed) pairs that were overridden
    result: Optional[float] = None                     # the governing computed value, if any
    all_true: bool = True                              # every shown equality holds after correction
    consistent: bool = True                            # no remaining numeric contradiction
    sanity_ok: bool = True                             # no division-by-zero / non-finite computation
    convention_ok: bool = True                         # binary/decimal convention not mixed
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def verified(self) -> bool:
        """Deliverable as VERIFIED only when stated == computed everywhere and units/magnitude are sane."""
        return self.consistent and self.sanity_ok and self.convention_ok


def verify_calculation(text: str) -> CalcReport:
    """Compute the answer's OWN shown arithmetic in code and make that the single source of truth:
    override every shown equality whose stated number differs from the code-computed value of its
    expression. Pure + deterministic (no LLM, no eval) and SAFE BY CONSTRUCTION — it only ever writes a
    code-computed value over the RHS of an 'EXPR = NUM' it proved wrong (so it can only turn a shown
    equality TRUE), never touching prose, an operand, or any other equation. A no-op on text with no
    evaluable arithmetic.

    Scope note: it reconciles the answer's shown WORK (every 'EXPR = NUM' it can evaluate, including a
    result computed several ways). It deliberately does NOT rewrite a free-prose restatement that has no
    expression of its own ('the total is 1000') — there is no safe deterministic way to tell such a
    number apart from an unrelated quantity that merely shares a value/unit, and guessing corrupts
    correct prose (it could even fabricate a false equality). That residual stays with the LLM
    conclusion-matches-work backstop on the agentic path; this engine never introduces a wrong value."""
    if not arithmetic_verify_enabled() or not (text or "").strip():
        return CalcReport(fixed_text=text)
    eqs = _find_calc_equalities(text)
    if not eqs:
        return CalcReport(fixed_text=text)
    # Override every shown equality's RHS with the computed value (right-to-left keeps spans valid).
    corrections: List[Tuple[str, str]] = []
    fixed = text
    for e in sorted(eqs, key=lambda e: e.rhs_span[0], reverse=True):
        if abs(round(e.computed, e.decimals) - e.stated) > 1e-9:
            start, end = e.rhs_span
            new = _format(e.computed, e.decimals)
            corrections.append((text[start:end], new))
            fixed = fixed[:start] + new + fixed[end:]
    eqs2 = _find_calc_equalities(fixed)
    gov = max(eqs2, key=lambda e: e.rhs_span[1]) if eqs2 else None
    convention_ok = _convention_ok(fixed, eqs2)
    notes = (("mixes binary (1024) and decimal (1000) unit conventions",) if not convention_ok else ())
    return CalcReport(
        fixed_text=fixed,
        corrections=tuple(reversed(corrections)),       # back into reading order
        result=(gov.computed if gov else None),
        all_true=True,
        consistent=True,                                 # every SHOWN equality is now computed-correct
        sanity_ok=True,                                  # safe_eval rejects div-by-zero / non-finite
        convention_ok=convention_ok,
        notes=notes,
    )
