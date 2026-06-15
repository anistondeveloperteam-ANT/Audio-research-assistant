"""
measure_evidence_grader.py  --  Reproducible measurement of the CRAG evidence grader
(backend/answering/evidence_grader.grade_evidence).

The grader maps the reranker scores of the retrieved PDF chunks to an action grade:
STRONG (answer from the PDFs, skip external search) / PARTIAL (keep PDFs + search the web) /
NONE (drop local, go fully external). This script measures it the same way
backend/evaluation/measure_classifiers.py measures the routers: a 3-class confusion matrix,
per-class precision/recall/F1 with 95% Wilson CIs, macro/weighted averages, AND the
external-skip rate (how often a STRONG grade correctly answered from the library without
spending a web search).

The grader is pure and deterministic (it only reads scores — no LLM, no network), so every
number here is exact and reproducible. The labeled set is curated to include boundary cases the
fixed thresholds are KNOWN to disagree with a human on, so the metrics are honest, not trivially
100%.

Run:  python -m backend.evaluation.measure_evidence_grader            # -> docs/CRAG_GRADING.md
      python -m backend.evaluation.measure_evidence_grader --out X.md
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from backend.answering.evidence_grader import (  # noqa: E402
    NONE,
    PARTIAL,
    STRONG,
    crag_partial_min,
    crag_strong_count,
    crag_strong_min,
    grade_evidence,
)

CLASSES = [STRONG, PARTIAL, NONE]

# ----------------------------------------------------------------------
# Labeled set (ground truth): (description, chunk_reranker_scores, expected_grade).
# Each chunk is modeled by its reranker score — the only signal the grader reads — so the
# fixtures stay compact and honest. A few rows are deliberate boundary DISAGREEMENTS (the human
# action differs from what the fixed thresholds produce) so precision/recall are realistic.
# ----------------------------------------------------------------------
GRADES: List[Tuple[str, List[float], str]] = [
    # --- clear STRONG: enough high-relevance chunks ---
    ("MVDR fully covered (3 strong chunks)", [0.86, 0.74, 0.61, 0.40], STRONG),
    ("two chunks exactly at the bar", [0.60, 0.55], STRONG),
    ("dense coverage", [0.91, 0.83, 0.70, 0.55, 0.33], STRONG),
    # --- clear PARTIAL: some relevant, but thin ---
    ("one strong + filler", [0.66, 0.20], PARTIAL),
    ("two borderline-relevant", [0.41, 0.34], PARTIAL),
    ("single solid chunk only", [0.72, 0.18, 0.05], PARTIAL),
    ("just above the floor", [0.31, 0.30], PARTIAL),
    # --- clear NONE: nothing relevant ---
    ("nothing clears the floor", [0.24, 0.11], NONE),
    ("empty retrieval", [], NONE),
    ("one near-miss chunk", [0.29], NONE),
    # --- honest boundary DISAGREEMENTS (threshold vs human action) ---
    # 3 decent-but-sub-0.55 chunks: a human reads this as well-covered (STRONG); the grader, lacking
    # 2 chunks >= 0.55, calls it PARTIAL -> safe under-grade (it still searches the web).
    ("three decent sub-bar chunks", [0.54, 0.52, 0.50], STRONG),
    # exactly 2 chunks barely over the bar: the grader calls STRONG and skips external; a cautious
    # human might want a web check (PARTIAL) -> an over-grade the skip-rate section surfaces.
    ("two chunks barely over the bar", [0.57, 0.56], PARTIAL),
]


# ----------------------------------------------------------------------
# Metrics (no third-party deps; same Wilson interval as measure_classifiers.py)
# ----------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def _confusion(data: List[Tuple[str, List[float], str]]):
    """Run grade_evidence over the labeled set; return (confusion_matrix, misses)."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    misses: List[Tuple[str, str, str]] = []
    for desc, scores, label in data:
        items = [{"score": s} for s in scores]
        pred = grade_evidence(items)
        cm[idx[label]][idx[pred]] += 1
        if pred != label:
            misses.append((desc, label, pred))
    return cm, misses


def _per_class(cm) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, float], Dict[str, float]]:
    per: Dict[str, Dict[str, float]] = {}
    correct = total = 0
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]
        col = sum(cm[r][i] for r in range(len(CLASSES)))
        row = sum(cm[i])
        prec = tp / col if col else 0.0
        rec = tp / row if row else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": row,
                  "ci_recall": _wilson(tp, row)}
        correct += tp
        total += row
    acc = correct / total if total else 0.0
    macro = {k: sum(per[c][k] for c in CLASSES) / len(CLASSES) for k in ("precision", "recall", "f1")}
    wt = {k: (sum(per[c][k] * per[c]["support"] for c in CLASSES) / total if total else 0.0)
          for k in ("precision", "recall", "f1")}
    return per, acc, macro, wt


def skip_stats(data: List[Tuple[str, List[float], str]]) -> Dict[str, float]:
    """External-skip accounting: a STRONG grade skips the web search. Report how often we skip,
    and how reliable that skip is (precision of skipping = correct STRONG / all predicted STRONG)."""
    skipped = correct_skip = 0
    for _desc, scores, label in data:
        pred = grade_evidence([{"score": s} for s in scores])
        if pred == STRONG:
            skipped += 1
            if label == STRONG:
                correct_skip += 1
    n = len(data)
    return {
        "n": n, "skipped": skipped, "correct_skip": correct_skip,
        "skip_rate": skipped / n if n else 0.0,
        "skip_precision": correct_skip / skipped if skipped else 0.0,
        "ci_skip_precision": _wilson(correct_skip, skipped),
    }


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ci(t: Tuple[float, float]) -> str:
    return f"[{t[0] * 100:.1f}-{t[1] * 100:.1f}%]"


def build_report() -> str:
    cm, misses = _confusion(GRADES)
    per, acc, macro, wt = _per_class(cm)
    skip = skip_stats(GRADES)

    head = "| actual \\ pred | " + " | ".join(CLASSES) + " | support |\n"
    head += "|---|" + "|".join([":---:"] * len(CLASSES)) + "|:---:|\n"
    for i, c in enumerate(CLASSES):
        head += f"| **{c}** | " + " | ".join(str(cm[i][j]) for j in range(len(CLASSES)))
        head += f" | {sum(cm[i])} |\n"

    pc = "| class | precision | recall | F1 | support |\n|---|---|---|---|---|\n"
    for c in CLASSES:
        p = per[c]
        pc += (f"| {c} | {_pct(p['precision'])} | {_pct(p['recall'])}  95% CI {_ci(p['ci_recall'])} "
               f"| {_pct(p['f1'])} | {p['support']} |\n")
    pc += (f"| **macro avg** | {_pct(macro['precision'])} | {_pct(macro['recall'])} | {_pct(macro['f1'])} | - |\n"
           f"| **weighted avg** | {_pct(wt['precision'])} | {_pct(wt['recall'])} | {_pct(wt['f1'])} | - |\n")

    miss = "\n".join(f"- `{d}` -- actual **{lbl}**, predicted **{pr}**" for d, lbl, pr in misses) \
        or "- _none_"

    skip_block = (
        f"A **STRONG** grade answers from the library and skips the web search entirely (the "
        f"adaptive win). Over the labeled set:\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| External searches skipped (STRONG) | {skip['skipped']}/{skip['n']} ({_pct(skip['skip_rate'])}) |\n"
        f"| Skip precision (STRONG that was truly STRONG) | {_pct(skip['skip_precision'])}  "
        f"95% CI {_ci(skip['ci_skip_precision'])} |\n\n"
        f"Skip precision < 100% means some skips were over-confident (answered from the PDFs when a "
        f"web check was warranted) — the lever is `CRAG_STRONG_MIN` / `CRAG_STRONG_COUNT`.")

    return "\n".join([
        "# CRAG Evidence-Grader Measurement Report",
        "",
        "> **Auto-generated** by `python -m backend.evaluation.measure_evidence_grader` — every "
        "number is computed by running `grade_evidence` on a labeled set. The grader is pure and "
        "deterministic (reads reranker scores only), so this is exact and reproducible.",
        "",
        f"**Thresholds in effect:** STRONG = >= {crag_strong_count()} chunks at score "
        f">= {crag_strong_min()}; PARTIAL = any chunk >= {crag_partial_min()}; else NONE.",
        "",
        f"**Overall accuracy (micro-F1): {_pct(acc)}** over {len(GRADES)} labeled cases.",
        "",
        "## Confusion matrix (rows = actual action, cols = predicted)",
        "",
        head,
        "## Per-class metrics",
        "",
        pc,
        "## External-skip rate",
        "",
        skip_block,
        "",
        f"## Misclassified ({len(misses)})",
        "",
        miss,
        "",
    ])


def main(argv: List[str]) -> int:
    out = "docs/CRAG_GRADING.md"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    report = build_report()
    dest = Path(__file__).resolve().parents[2] / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
