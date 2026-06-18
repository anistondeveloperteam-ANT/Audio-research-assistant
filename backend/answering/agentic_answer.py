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


def verify_answer(
    provider: Any,
    *,
    question: str,
    evidence: str,
    answer: str,
    run_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_text = ""
    if run_info:
        run_text = (
            "\n\nSANDBOX RUN RESULT:\n"
            f"summary: {run_info.get('summary', '')}\n"
            f"stdout:\n{run_info.get('stdout') or '(none)'}\n"
            f"stderr:\n{run_info.get('stderr') or '(none)'}\n"
            f"error: {run_info.get('error') or '(none)'}"
        )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"NUMBERED EVIDENCE:\n{evidence}\n\n"
        f"ANSWER TO VERIFY:\n{answer}"
        f"{run_text}"
    )
    raw = complete_text(
        provider,
        [{"role": "user", "content": user}],
        system=_VERIFY_SYSTEM,
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
