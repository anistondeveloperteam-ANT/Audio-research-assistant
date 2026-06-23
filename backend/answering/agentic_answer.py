"""
Agentic answer helpers for the web chat path.

This is deliberately not a free-form desktop agent. It keeps the product contract:
answers are grounded in retrieved evidence, verification is LLM-based against that
evidence, and generated Python is executed only in the existing locked-down Docker
sandbox.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any, Dict, List, Optional


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false")
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def agentic_loop_enabled() -> bool:
    return env_flag("ENABLE_AGENTIC_ANSWER_LOOP", default=True)


def max_verify_rounds() -> int:
    try:
        return max(1, min(5, int(os.getenv("AGENTIC_MAX_VERIFY_ROUNDS", "3"))))
    except ValueError:
        return 3


def max_deep_loops() -> int:
    """Hard cap on the agentic verify->rewrite loop (it also early-stops on a passing verdict
    or one with no concrete gap). Distinct from max_verify_rounds() so DEEP mode can run fewer
    loops without changing the mode's verify-round semantics. Default 2."""
    try:
        return max(1, min(5, int(os.getenv("DEEP_MAX_LOOPS", "2"))))
    except ValueError:
        return 2


def min_verify_score() -> int:
    try:
        return max(0, min(100, int(os.getenv("AGENTIC_MIN_VERIFY_SCORE", "80"))))
    except ValueError:
        return 80


def simulate_code_enabled() -> bool:
    return env_flag("AGENTIC_SIMULATE_CODE", default=True)


def auto_review_enabled() -> bool:
    """Automatically peer-review the final answer/code (the 'Review' step, run for you)."""
    return env_flag("AUTO_REVIEW", default=True)


def independent_verify_enabled() -> bool:
    """Independent confirmation layer (default on): before an answer is labelled 'verified', confirm it
    by a route that does NOT share its derivation (re-derive from scratch + unit/magnitude/limiting-case
    sanity). Self-consistent self-tests are the DEPENDENT layer; this adds the independent one on top."""
    return env_flag("AGENTIC_INDEPENDENT_VERIFY", default=True)


def complete_text(
    provider: Any,
    messages: List[Dict[str, str]],
    *,
    system: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> str:
    return "".join(
        provider.stream_chat(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    ).strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse strict JSON, or the first JSON object embedded in a model reply."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_PY_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S | re.I)
_ANY_FENCE = re.compile(r"```\s*\n(.*?)```", re.S)


def extract_python_blocks(answer: str) -> List[str]:
    """Return fenced Python code blocks from an answer, longest first."""
    blocks = [b.strip() for b in _PY_FENCE.findall(answer or "") if b.strip()]
    if not blocks:
        # Conservative fallback for plain fenced code that looks like Python.
        for block in _ANY_FENCE.findall(answer or ""):
            code = block.strip()
            if re.search(r"\b(import|from|def|class|print|if __name__)\b", code):
                blocks.append(code)
    blocks.sort(key=len, reverse=True)
    return blocks


def python_blocks_in_order(answer: str) -> List[str]:
    """Fenced ```python / ```py blocks in DOCUMENT ORDER (top to bottom). Unlike
    extract_python_blocks this does NOT sort by length and does NOT fall back to non-python
    fences — so 'the last block' is the canonical final program, never a ```text output dump."""
    return [b.strip() for b in _PY_FENCE.findall(answer or "") if b.strip()]


def run_best_python_block(answer: str) -> Optional[Dict[str, Any]]:
    """Run the longest Python code block in Docker, if enabled and available.

    Returns None when there is no code block. Never executes on the host.
    """
    blocks = extract_python_blocks(answer)
    if not blocks or not simulate_code_enabled():
        return None

    code = blocks[0]
    if len(code) > 25_000:
        return {
            "attempted": False,
            "ok": False,
            "summary": "Python block was too large to run safely.",
            "stdout": "",
            "stderr": "",
            "error": "code block too large",
        }

    try:
        from backend.agent.code_runner import docker_available, run_python
    except Exception as exc:
        return {
            "attempted": False,
            "ok": False,
            "summary": f"Sandbox runner unavailable: {exc}",
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }

    if not docker_available():
        return {
            "attempted": False,
            "ok": False,
            "summary": "Docker is not running, so generated Python was not executed.",
            "stdout": "",
            "stderr": "",
            "error": "docker unavailable",
        }

    try:
        timeout = int(os.getenv("AGENTIC_SIMULATION_TIMEOUT", os.getenv("AGENT_RUN_TIMEOUT", "30")))
    except ValueError:
        timeout = 30
    result = run_python(code, timeout=timeout)
    return {
        "attempted": True,
        "ok": bool(result.ok),
        "summary": result.summary,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
    }


_VERIFY_SYSTEM = (
    "You are a strict evidence verifier for a cited research assistant. "
    "Check whether the answer is fully supported by the numbered evidence, cites "
    "source numbers correctly, addresses the question, and does not invent facts. "
    "If a sandbox run result is provided, use it to judge generated Python or "
    "simulation claims. Reply with ONLY JSON:\n"
    '{"ok": true|false, "score": 0-100, "needs_more_search": true|false, '
    '"followup_query": "short search query if more evidence is needed", '
    '"feedback": "specific corrections needed", '
    '"missing_evidence": ["gap"], "citation_issues": ["issue"]}'
)

# ORIGIN-INDEPENDENT quality judge for an answer produced from the model's OWN reasoning (no retrieved
# evidence). It judges CORRECTNESS / COMPLETENESS / HONESTY — NOT whether sources are cited — so a
# correct self-contained answer is not failed merely for lacking citations, and a wrong or fact-
# fabricating one is still caught.
_VERIFY_REASONING_SYSTEM = (
    "You are a strict ANSWER-QUALITY judge for a research assistant. This answer was produced from the "
    "model's OWN REASONING / GENERAL KNOWLEDGE — there is no retrieved evidence, because the question is "
    "self-contained or answerable without external documents. Judge the answer ON ITS MERITS, NOT by "
    "whether it cites sources: is it CORRECT (internally consistent; the reasoning/derivation is sound "
    "and, where it matters, shown; any calculation checks out), COMPLETE (addresses the whole question), "
    "and HONEST (it does NOT fabricate specific external facts, figures, dates, or citations it cannot "
    "know)? Do NOT require citations for a self-contained answer. Score HIGH when it is correct and "
    "complete. Score LOW when it is wrong, incomplete, or invents external facts — and if the question "
    "GENUINELY needs up-to-date or external information the model cannot supply, set needs_more_search="
    "true with a followup_query. Reply with ONLY JSON:\n"
    '{"ok": true|false, "score": 0-100, "needs_more_search": true|false, '
    '"followup_query": "short search query if external facts are required", '
    '"feedback": "specific corrections needed", '
    '"missing_evidence": ["gap"], "citation_issues": []}'
)

# Draft prompt for the reasoning basis: answer a solvable question from knowledge/derivation, never
# refuse for lack of sources, never fabricate external facts.
REASONING_ANSWER_SYSTEM = (
    "You are an expert research assistant. Answer the user's question DIRECTLY and CORRECTLY using your "
    "own knowledge and step-by-step reasoning — there is no retrieved document for this question, and "
    "that is fine: a correct answer from reasoning is exactly what is wanted. SHOW your reasoning / "
    "derivation / calculation so the answer is checkable, then give a clear final answer. Do NOT refuse "
    "for lack of sources, and do NOT invent specific external facts, figures, dates, or citations you "
    "cannot verify — if part of the question truly needs current or external data you don't have, say so "
    "plainly for that part and answer the rest. Prefer an honest 'I'm not certain' over a confident "
    "guess on anything you cannot derive."
)


def verify_answer(
    provider: Any,
    *,
    question: str,
    evidence: str,
    answer: str,
    run_info: Optional[Dict[str, Any]] = None,
    basis: str = "evidence",
) -> Dict[str, Any]:
    """Judge an answer's quality. `basis="evidence"` (default) checks grounding + citations against the
    numbered evidence; `basis="reasoning"` (or no evidence) judges correctness/completeness/honesty of a
    self-contained answer WITHOUT requiring citations — the SAME quality bar, origin-independent."""
    run_text = ""
    if run_info:
        run_text = (
            "\n\nSANDBOX RUN RESULT:\n"
            f"summary: {run_info.get('summary', '')}\n"
            f"stdout:\n{run_info.get('stdout') or '(none)'}\n"
            f"stderr:\n{run_info.get('stderr') or '(none)'}\n"
            f"error: {run_info.get('error') or '(none)'}"
        )
    reasoning = basis == "reasoning" or not (evidence or "").strip()
    system = _VERIFY_REASONING_SYSTEM if reasoning else _VERIFY_SYSTEM
    evidence_section = "" if reasoning else f"NUMBERED EVIDENCE:\n{evidence}\n\n"
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{evidence_section}"
        f"ANSWER TO VERIFY:\n{answer}"
        f"{run_text}"
    )
    raw = complete_text(
        provider,
        [{"role": "user", "content": user}],
        system=system,
        max_tokens=int(os.getenv("AGENTIC_VERIFY_MAX_TOKENS", "1200")),
        temperature=0.0,
    )
    verdict = parse_json_object(raw)
    verdict.setdefault("ok", False)
    verdict.setdefault("score", 0)
    verdict.setdefault("needs_more_search", False)
    verdict.setdefault("followup_query", "")
    verdict.setdefault("feedback", DEFAULT_FEEDBACK)
    verdict.setdefault("missing_evidence", [])
    verdict.setdefault("citation_issues", [])
    try:
        verdict["score"] = max(0, min(100, int(verdict.get("score", 0))))
    except (TypeError, ValueError):
        verdict["score"] = 0
    verdict["ok"] = bool(verdict.get("ok"))
    verdict["needs_more_search"] = bool(verdict.get("needs_more_search"))
    return verdict


def verification_passed(verdict: Dict[str, Any]) -> bool:
    return bool(verdict.get("ok")) and int(verdict.get("score", 0)) >= min_verify_score()


# INDEPENDENT verification: confirm the answer by a route that does NOT share its derivation, so an
# error baked into the answer's own assumptions (a missed unit conversion, a wrong factor, an
# implausible magnitude) cannot pass undetected — a self-derived check inherits that very error.
_INDEPENDENT_VERIFY_SYSTEM = (
    "You are an INDEPENDENT checker. You are given a QUESTION and a PROPOSED ANSWER. Do NOT trust or "
    "reuse HOW the answer was derived — SOLVE THE PROBLEM YOURSELF FROM SCRATCH, by a DIFFERENT method "
    "or decomposition than the answer appears to use, then COMPARE your independent result to the "
    "answer's conclusion. ALSO run assumption-level SANITY CHECKS that a same-assumptions test would "
    "MISS:\n"
    "1. UNIT CONSISTENCY — track units end to end; the final unit must be exactly what the question "
    "asks (e.g. Hz x bits x seconds = bits, then / 8 = bytes, then / 1e6 = MB). A dropped, extra, or "
    "wrong conversion is a DISAGREEMENT.\n"
    "2. ORDER OF MAGNITUDE / PLAUSIBILITY — is the result physically and numerically plausible: not off "
    "by ~10x / 1000x, not negative or zero where impossible, within any obvious bound the problem "
    "implies?\n"
    "3. LIMITING / KNOWN CASES — does it behave correctly at a boundary, or match a well-known reference "
    "value (a standard constant, a textbook figure)?\n"
    "A flaw SHARED between the answer and a same-assumption check surfaces as a MISMATCH between YOUR "
    "independent derivation (or a sanity check) and the answer — call it out specifically. Judge "
    "AGREEMENT on the substantive result, allowing reasonable rounding/tolerance. If the question is a "
    "matter of opinion or has nothing independently checkable, set agrees to null (do NOT force a "
    "verdict). Reply with ONLY JSON:\n"
    '{"agrees": true|false|null, "independent_answer": "your from-scratch result", '
    '"issues": ["unit / magnitude / limiting-case / mismatch problems"], "confidence": 0-100}'
)


def independent_check(provider: Any, *, question: str, answer: str) -> Dict[str, Any]:
    """INDEPENDENT confirmation by a route that does NOT share the answer's derivation: re-derive the
    answer from scratch + assumption-level sanity (unit consistency, order-of-magnitude/plausibility,
    limiting/known cases). Returns {agrees: True|False|None, independent_answer, issues, confidence}.
    agrees=None means NO independent confirmation exists (an opinion, nothing checkable, or a hiccup) ->
    the answer must NOT be labelled 'verified' on the dependent check alone. Fail-OPEN to None; never
    raises."""
    if not independent_verify_enabled() or not (answer or "").strip():
        return {"agrees": None, "independent_answer": "", "issues": [], "confidence": 0}
    user = (f"QUESTION:\n{question}\n\nPROPOSED ANSWER (do NOT reuse its derivation; solve it yourself, "
            f"then compare):\n{answer}")
    try:
        raw = complete_text(
            provider, [{"role": "user", "content": user}], system=_INDEPENDENT_VERIFY_SYSTEM,
            max_tokens=int(os.getenv("AGENTIC_VERIFY_MAX_TOKENS", "1200")), temperature=0.0)
    except Exception:                       # noqa: BLE001 - independence is best-effort, never fatal
        return {"agrees": None, "independent_answer": "", "issues": [], "confidence": 0}
    v = parse_json_object(raw)
    raw_agrees = v.get("agrees", None)
    if raw_agrees is True or (isinstance(raw_agrees, str) and raw_agrees.strip().lower() == "true"):
        agrees: Optional[bool] = True
    elif raw_agrees is False or (isinstance(raw_agrees, str) and raw_agrees.strip().lower() == "false"):
        agrees = False
    else:
        agrees = None
    issues = v.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    try:
        conf = max(0, min(100, int(v.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0
    return {"agrees": agrees, "independent_answer": str(v.get("independent_answer", ""))[:2000],
            "issues": [str(x) for x in issues][:6], "confidence": conf}


def is_truly_verified(dependent_passed: bool, independent: Optional[Dict[str, Any]] = None) -> bool:
    """SELF-CONSISTENT != VERIFIED. An answer is 'verified' ONLY when the dependent check passes AND an
    INDEPENDENT check AGREES. A dependent pass with no independent confirmation (agrees None) or a
    refutation (agrees False) is NOT verified — show the answer with honest confidence instead. With the
    independent layer disabled, fall back to the legacy dependent-only result."""
    if not dependent_passed:
        return False
    if not independent_verify_enabled():
        return True
    return bool(independent and independent.get("agrees") is True)


# Placeholder used when the verifier returns nothing usable — NOT actionable guidance.
DEFAULT_FEEDBACK = "Verifier did not return a usable verdict."


def has_concrete_gap(verdict: Dict[str, Any]) -> bool:
    """True only when the verifier named a SPECIFIC, fixable problem — missing evidence, a
    citation issue, or a follow-up search to run — not merely a sub-threshold score. The
    verify->rewrite loop early-stops when a non-passing verdict has no concrete gap, since a
    rewrite would only chase a marginal score with nothing actionable to fix."""
    if verdict.get("missing_evidence") or verdict.get("citation_issues"):
        return True
    return bool(verdict.get("needs_more_search")) or bool((verdict.get("followup_query") or "").strip())


def has_actionable_feedback(verdict: Dict[str, Any]) -> bool:
    """True when the verifier gave SPECIFIC prose corrections (e.g. 'soften the overstated claim
    about [2]') even though it named no structured gap. Such a defect is fixable from the existing
    evidence via a guided rewrite, so the loop must not early-stop on the FIRST such verdict —
    skipping it would ship a known-flawed draft. Excludes the empty/placeholder default."""
    fb = (verdict.get("feedback") or "").strip()
    return bool(fb) and fb != DEFAULT_FEEDBACK


def followup_query(question: str, verdict: Dict[str, Any]) -> str:
    query = (verdict.get("followup_query") or "").strip()
    if query:
        return query[:240]
    missing = verdict.get("missing_evidence") or []
    if isinstance(missing, Iterable) and not isinstance(missing, (str, bytes)):
        gap = " ".join(str(x) for x in list(missing)[:3]).strip()
    else:
        gap = str(missing).strip()
    return f"{question} {gap}".strip()[:240]


def build_revision_message(
    *,
    question: str,
    evidence: str,
    previous_answer: str,
    verdict: Dict[str, Any],
    run_info: Optional[Dict[str, Any]] = None,
) -> str:
    run_text = ""
    if run_info:
        run_text = (
            "\n\nSandbox result for generated Python:\n"
            f"{run_info.get('summary', '')}\n"
            f"stdout:\n{run_info.get('stdout') or '(none)'}\n"
            f"stderr:\n{run_info.get('stderr') or '(none)'}"
        )
    return (
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n\n{evidence}\n\n"
        f"Previous draft:\n{previous_answer}\n\n"
        f"Verifier feedback:\n{verdict.get('feedback', '')}\n"
        f"Missing evidence: {verdict.get('missing_evidence', [])}\n"
        f"Citation issues: {verdict.get('citation_issues', [])}"
        f"{run_text}\n\n"
        "Rewrite the final answer. Use only the numbered evidence, fix every issue, "
        "cite claims with [n], and keep any code complete and runnable."
    )


def verification_footer(
    *,
    verdict: Optional[Dict[str, Any]],
    rounds: int,
    run_info: Optional[Dict[str, Any]] = None,
) -> str:
    bits: List[str] = []
    if verdict:
        status = "passed" if verification_passed(verdict) else "completed with remaining caveats"
        bits.append(f"evidence check {status} ({int(verdict.get('score', 0))}/100, {rounds} round(s))")
    if run_info:
        bits.append(f"sandbox run: {run_info.get('summary', 'not run')}")
    if not bits:
        return ""
    return "\n\nVerification: " + "; ".join(bits) + "."
