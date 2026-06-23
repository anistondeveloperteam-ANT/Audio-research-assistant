"""Suggest config thresholds from MEASURED eval results — print-only, never auto-applied.

This computes RECOMMENDATIONS (a value + the evidence behind it) for tuning knobs such as
CRAG_STRONG_MIN. It NEVER writes .env or changes behaviour: a human reviews the suggestion and applies
it via the normal .env edit. Tuning stays evidence-driven but human-approved (no silent self-tuning).
"""
from __future__ import annotations

from typing import Any, Dict, List

# Targets a healthy grader should clear. Conservative, documented, easy to change.
_SKIP_PRECISION_TARGET = 0.95   # when a STRONG grade skips web search, it must be right this often
_RECALL_TARGET = 0.70           # the grader should still surface enough true STRONG/PARTIAL cases


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def suggest_thresholds(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return threshold recommendations from measured metrics. Each item: var, current, recommended,
    change, evidence, action. PURE — no I/O, no env writes, no behaviour change.

    Recognised metric keys (all optional): crag_strong_min, crag_partial_min, crag_skip_precision,
    crag_recall."""
    out: List[Dict[str, Any]] = []

    strong = metrics.get("crag_strong_min")
    skip_prec = metrics.get("crag_skip_precision")
    if strong is not None and skip_prec is not None:
        if skip_prec < _SKIP_PRECISION_TARGET:
            rec = round(_clamp(float(strong) + 0.03, 0.30, 0.80), 3)
            out.append({
                "var": "CRAG_STRONG_MIN", "current": float(strong), "recommended": rec,
                "change": round(rec - float(strong), 3),
                "evidence": (f"STRONG-grade skip precision is {skip_prec:.1%} "
                             f"(target >= {_SKIP_PRECISION_TARGET:.0%}); raising the strong-evidence "
                             "floor makes the grade skip web search less eagerly."),
                "action": f"Human: set CRAG_STRONG_MIN={rec} in .env (review first).",
            })
        else:
            out.append({
                "var": "CRAG_STRONG_MIN", "current": float(strong), "recommended": float(strong),
                "change": 0.0,
                "evidence": f"STRONG-grade skip precision {skip_prec:.1%} meets target; no change.",
                "action": "No change recommended.",
            })

    partial = metrics.get("crag_partial_min")
    recall = metrics.get("crag_recall")
    if partial is not None and recall is not None:
        if recall < _RECALL_TARGET:
            rec = round(_clamp(float(partial) - 0.02, 0.10, 0.60), 3)
            out.append({
                "var": "CRAG_PARTIAL_MIN", "current": float(partial), "recommended": rec,
                "change": round(rec - float(partial), 3),
                "evidence": (f"Grader recall is {recall:.1%} (target >= {_RECALL_TARGET:.0%}); lowering "
                             "the partial-relevance floor lets more true-relevant chunks through."),
                "action": f"Human: set CRAG_PARTIAL_MIN={rec} in .env (review first).",
            })
        else:
            out.append({
                "var": "CRAG_PARTIAL_MIN", "current": float(partial), "recommended": float(partial),
                "change": 0.0,
                "evidence": f"Grader recall {recall:.1%} meets target; no change.",
                "action": "No change recommended.",
            })
    return out


def format_suggestions(suggestions: List[Dict[str, Any]]) -> str:
    """Render suggestions as a human-readable block. The header makes the no-auto-apply contract
    explicit."""
    lines = ["PARAMETER SUGGESTIONS (evidence-based; NOT auto-applied - a human edits .env)", ""]
    if not suggestions:
        lines.append("  No measured metrics provided — nothing to suggest.")
        return "\n".join(lines)
    for s in suggestions:
        arrow = "no change" if s["change"] == 0 else f"{s['current']} -> {s['recommended']}"
        lines.append(f"  {s['var']}: {arrow}")
        lines.append(f"     evidence: {s['evidence']}")
        lines.append(f"     {s['action']}")
        lines.append("")
    return "\n".join(lines)
