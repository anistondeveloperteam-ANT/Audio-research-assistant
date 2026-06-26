"""
evaluate_retrieval.py  --  Batch 5 (Better Evaluation Harness)

Backward compatible with your existing data/evaluation_questions.json
(question + expected_terms). Adds proper IR metrics on top:

  - term_recall (your original "score") -- kept for continuity
  - recall@k        for k in {1, 3, 5, 10}
  - precision@k     for k in {1, 3, 5, 10}
  - MRR             (mean reciprocal rank of the first hit)
  - nDCG@k          for k in {3, 5, 10}
  - hit@1, hit@5    (got at least one expected term in top-N)

Plus useful diagnostics:
  - per-question breakdown with which terms hit and at what rank
  - chunk-type distribution across top-10 results
  - paper-diversity (how many distinct papers in top-10)
  - timing stats (mean, p50, p95)
  - failure analysis: questions that scored < 0.5

Optional mode comparison (off by default, opt in via flag):
  python backend/evaluate_retrieval.py --compare-modes
  -> Runs Fast / Balanced / Deep back-to-back and prints a table.

Run:
  python backend/evaluate_retrieval.py
  python backend/evaluate_retrieval.py --top-k 8
  python backend/evaluate_retrieval.py --compare-modes
  python backend/evaluate_retrieval.py --mode deep
  python backend/evaluate_retrieval.py --quiet
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]

# Ensure project root is importable so `import backend.*` resolves when this
# module is run directly (e.g. `python -m backend.evaluation.evaluate_retrieval`).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_FILE = ROOT / "data" / "evaluation_questions.json"
REPORT_FILE = ROOT / "data" / "extracted" / "retrieval_eval_report.json"
MODES_REPORT_FILE = ROOT / "data" / "extracted" / "retrieval_eval_modes_report.json"


# ----------------------------------------------------------------------
# Imports for retrieval + mode binding
# ----------------------------------------------------------------------

def import_retriever():
    try:
        from backend.retrieval.hybrid_retrieve import hybrid_retrieve
        return hybrid_retrieve
    except Exception as exc:
        raise RuntimeError(
            "Could not import hybrid_retrieve from backend/hybrid_retrieve.py. "
            "Run this from project root with your .venv activated."
        ) from exc


def try_import_mode_binding():
    """Optional. Used only with --compare-modes or --mode flags. Returns a callable `bind(mode)` that
    binds the run profile to THIS process's request context (no process-global env mutation)."""
    try:
        from backend.answering.research_modes import resolve_research_mode
        from backend.common import request_context as _rc

        def _bind(mode=None):
            _rc.set_request_settings(resolve_research_mode(mode))
        return _bind
    except Exception:
        return None


# ----------------------------------------------------------------------
# Question loading (backward-compatible)
# ----------------------------------------------------------------------

def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {QUESTIONS_FILE}")

    raw = QUESTIONS_FILE.read_text(encoding="utf-8", errors="ignore").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("")
        print("ERROR: data\\evaluation_questions.json is not valid JSON.")
        print(f"JSON error: {exc}")
        print("")
        raise SystemExit(1)

    if not isinstance(data, list):
        raise ValueError("evaluation_questions.json must contain one JSON array.")

    cleaned = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            print(f"Skipping invalid item #{i}: not an object")
            continue
        question = str(item.get("question", "")).strip()
        expected_terms = item.get("expected_terms", [])
        if not question:
            print(f"Skipping item #{i}: missing question")
            continue
        if not isinstance(expected_terms, list):
            expected_terms = []
        cleaned.append({
            "question": question,
            "expected_terms": [str(x).strip() for x in expected_terms if str(x).strip()],
        })

    if not cleaned:
        raise ValueError("No valid questions in evaluation_questions.json")
    return cleaned


# ----------------------------------------------------------------------
# Per-result text helpers
# ----------------------------------------------------------------------

def result_text(result: Dict[str, Any]) -> str:
    fields = [
        result.get("title"),
        result.get("paper"),
        result.get("section"),
        result.get("section_name"),
        result.get("concepts"),
        result.get("audio_concepts"),
        result.get("chunk_type"),
        result.get("type"),
        result.get("text"),
        result.get("chunk_text"),
        result.get("preview"),
    ]
    return " ".join(str(x or "") for x in fields).lower()


def result_contains_any_term(result: Dict[str, Any], terms: List[str]) -> bool:
    """A result is 'relevant' if it contains ANY expected term."""
    if not terms:
        return False
    text = result_text(result)
    return any(t.lower() in text for t in terms)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def term_recall(results: List[Dict[str, Any]],
                expected_terms: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Backward-compatible 'score': fraction of expected_terms that
    appear anywhere in the top results' concatenated text.
    """
    if not expected_terms:
        return 0.0, [], []
    full_text = "\n".join(result_text(r) for r in results)
    hits = []
    misses = []
    for term in expected_terms:
        if term.lower() in full_text:
            hits.append(term)
        else:
            misses.append(term)
    score = len(hits) / len(expected_terms)
    return score, hits, misses


def first_hit_rank(results: List[Dict[str, Any]],
                   expected_terms: List[str]) -> Optional[int]:
    """1-indexed rank of the first relevant result, or None."""
    if not expected_terms:
        return None
    for i, r in enumerate(results, 1):
        if result_contains_any_term(r, expected_terms):
            return i
    return None


def recall_at_k(results: List[Dict[str, Any]],
                expected_terms: List[str], k: int) -> float:
    """Fraction of expected_terms that show up in the top-k results."""
    if not expected_terms:
        return 0.0
    top = results[:k]
    text = "\n".join(result_text(r) for r in top)
    hits = sum(1 for t in expected_terms if t.lower() in text)
    return hits / len(expected_terms)


def precision_at_k(results: List[Dict[str, Any]],
                   expected_terms: List[str], k: int) -> float:
    """Fraction of top-k results that contain any expected term."""
    if not expected_terms or k <= 0:
        return 0.0
    top = results[:k]
    if not top:
        return 0.0
    relevant = sum(1 for r in top if result_contains_any_term(r, expected_terms))
    return relevant / len(top)


def mrr(results: List[Dict[str, Any]], expected_terms: List[str]) -> float:
    """Mean reciprocal rank of the FIRST relevant result. 0 if none."""
    rank = first_hit_rank(results, expected_terms)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(results: List[Dict[str, Any]],
              expected_terms: List[str], k: int) -> float:
    """
    Binary relevance nDCG@k. Relevance = 1 if result contains any
    expected term, 0 otherwise. Ideal DCG = sum 1/log2(i+1) for the
    first min(k, expected_count) positions, treated as 'best case'.
    """
    if not expected_terms or k <= 0:
        return 0.0
    rels = [1 if result_contains_any_term(r, expected_terms) else 0
            for r in results[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal_count = min(k, sum(rels) or k)
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ----------------------------------------------------------------------
# Result summary helpers
# ----------------------------------------------------------------------

def safe_number(value: Any) -> Any:
    try:
        return round(float(value), 4)
    except Exception:
        return value


def source_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": result.get("title") or result.get("paper"),
        "section": result.get("section") or result.get("section_name"),
        "pages": [result.get("page_start"), result.get("page_end")],
        "chunk_type": result.get("chunk_type") or result.get("type"),
        "concepts": result.get("concepts") or result.get("audio_concepts"),
        "hybrid_score": safe_number(result.get("hybrid_score")),
        "rerank_score": safe_number(result.get("rerank_score")),
        "preview": str(result.get("text") or result.get("chunk_text") or "")[:500],
    }


def chunk_type_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for r in results:
        t = r.get("chunk_type") or r.get("type") or "unknown"
        counts[str(t)] += 1
    return dict(counts)


def paper_diversity(results: List[Dict[str, Any]]) -> int:
    papers = set()
    for r in results:
        title = (r.get("title") or r.get("paper") or "").strip()
        if title:
            papers.add(title)
    return len(papers)


# ----------------------------------------------------------------------
# Retrieval call
# ----------------------------------------------------------------------

def call_retriever(retriever, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    try:
        results = retriever(question, top_k=top_k)
    except TypeError:
        results = retriever(question)
    if results is None:
        return []
    if isinstance(results, dict):
        if "results" in results:
            results = results["results"]
        elif "sources" in results:
            results = results["sources"]
        else:
            results = [results]
    return list(results)


# ----------------------------------------------------------------------
# Single-pass evaluation
# ----------------------------------------------------------------------

def run_single_pass(retriever,
                    questions: List[Dict[str, Any]],
                    top_k: int = 10,
                    quiet: bool = False,
                    label: str = "") -> Dict[str, Any]:
    per_question = []
    times: List[float] = []

    metric_sums = {
        "term_recall": 0.0,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "precision_at_5": 0.0,
        "precision_at_10": 0.0,
        "mrr": 0.0,
        "ndcg_at_3": 0.0,
        "ndcg_at_5": 0.0,
        "ndcg_at_10": 0.0,
        "hit_at_1": 0.0,
        "hit_at_5": 0.0,
        "paper_diversity": 0.0,
    }

    if not quiet:
        head = f"AUDIO RAG RETRIEVAL EVALUATION{(' [' + label + ']') if label else ''}"
        print("=" * 100)
        print(head)
        print("=" * 100)
        print("Questions:", len(questions))
        print("Top-k:", top_k)
        print("Questions file:", QUESTIONS_FILE)
        print("")

    for index, item in enumerate(questions, 1):
        question = item["question"]
        expected_terms = item.get("expected_terms", [])

        if not quiet:
            print("-" * 100)
            print(f"Question {index}: {question}")

        start = time.time()
        error = None
        try:
            results = call_retriever(retriever, question, top_k=top_k)
        except Exception as exc:
            results = []
            error = str(exc)
        elapsed = time.time() - start
        times.append(elapsed)

        tr_score, hits, misses = term_recall(results, expected_terms)
        rank = first_hit_rank(results, expected_terms)
        r1 = recall_at_k(results, expected_terms, 1)
        r3 = recall_at_k(results, expected_terms, 3)
        r5 = recall_at_k(results, expected_terms, 5)
        r10 = recall_at_k(results, expected_terms, 10)
        p1 = precision_at_k(results, expected_terms, 1)
        p3 = precision_at_k(results, expected_terms, 3)
        p5 = precision_at_k(results, expected_terms, 5)
        p10 = precision_at_k(results, expected_terms, 10)
        mrr_val = mrr(results, expected_terms)
        ndcg3 = ndcg_at_k(results, expected_terms, 3)
        ndcg5 = ndcg_at_k(results, expected_terms, 5)
        ndcg10 = ndcg_at_k(results, expected_terms, 10)
        h1 = 1.0 if rank is not None and rank <= 1 else 0.0
        h5 = 1.0 if rank is not None and rank <= 5 else 0.0
        diversity = paper_diversity(results)

        metric_sums["term_recall"] += tr_score
        metric_sums["recall_at_1"] += r1
        metric_sums["recall_at_3"] += r3
        metric_sums["recall_at_5"] += r5
        metric_sums["recall_at_10"] += r10
        metric_sums["precision_at_1"] += p1
        metric_sums["precision_at_3"] += p3
        metric_sums["precision_at_5"] += p5
        metric_sums["precision_at_10"] += p10
        metric_sums["mrr"] += mrr_val
        metric_sums["ndcg_at_3"] += ndcg3
        metric_sums["ndcg_at_5"] += ndcg5
        metric_sums["ndcg_at_10"] += ndcg10
        metric_sums["hit_at_1"] += h1
        metric_sums["hit_at_5"] += h5
        metric_sums["paper_diversity"] += diversity

        if not quiet:
            print(f"Time: {elapsed:.2f}s  term_recall: {tr_score:.3f}  "
                  f"first_hit_rank: {rank}  mrr: {mrr_val:.3f}  "
                  f"ndcg@5: {ndcg5:.3f}  papers: {diversity}")
            if hits:
                print("Hits:", hits)
            if misses:
                print("Misses:", misses)
            if error:
                print("ERROR:", error)
            elif results:
                top = results[0]
                print("Top source:",
                      f"{top.get('title') or top.get('paper') or 'Unknown'} | "
                      f"section={top.get('section') or top.get('section_name')} | "
                      f"type={top.get('chunk_type') or top.get('type')}")
            else:
                print("Top source: none")

        per_question.append({
            "question": question,
            "expected_terms": expected_terms,
            "time_seconds": round(elapsed, 3),
            "term_recall": round(tr_score, 3),
            "first_hit_rank": rank,
            "recall_at_1": round(r1, 3),
            "recall_at_3": round(r3, 3),
            "recall_at_5": round(r5, 3),
            "recall_at_10": round(r10, 3),
            "precision_at_1": round(p1, 3),
            "precision_at_3": round(p3, 3),
            "precision_at_5": round(p5, 3),
            "precision_at_10": round(p10, 3),
            "mrr": round(mrr_val, 3),
            "ndcg_at_3": round(ndcg3, 3),
            "ndcg_at_5": round(ndcg5, 3),
            "ndcg_at_10": round(ndcg10, 3),
            "hit_at_1": int(h1),
            "hit_at_5": int(h5),
            "paper_diversity": diversity,
            "chunk_type_distribution": chunk_type_distribution(results),
            "hits": hits,
            "misses": misses,
            "error": error,
            "top_sources": [source_summary(r) for r in results[:5]],
        })

    n = len(questions)
    averages = {k: round(v / max(n, 1), 4) for k, v in metric_sums.items()}

    times_sorted = sorted(times)
    timing = {
        "mean_seconds": round(statistics.fmean(times), 3) if times else 0.0,
        "p50_seconds": round(times_sorted[len(times_sorted) // 2], 3) if times else 0.0,
        "p95_seconds": round(times_sorted[int(0.95 * (len(times_sorted) - 1))], 3) if times else 0.0,
        "max_seconds": round(max(times), 3) if times else 0.0,
        "total_seconds": round(sum(times), 3),
    }

    failure_threshold = 0.5
    failures = [q for q in per_question
                if (q["term_recall"] < failure_threshold)
                or (q["first_hit_rank"] is None)]

    return {
        "label": label or "default",
        "top_k": top_k,
        "question_count": n,
        "averages": averages,
        "timing": timing,
        "failures": [
            {
                "question": q["question"],
                "term_recall": q["term_recall"],
                "first_hit_rank": q["first_hit_rank"],
                "misses": q["misses"],
            }
            for q in failures
        ],
        "results": per_question,
    }


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

def print_summary(report: Dict[str, Any]) -> None:
    a = report["averages"]
    t = report["timing"]
    print("")
    print("=" * 100)
    print("FINAL RESULT" + (f" [{report.get('label')}]" if report.get("label") else ""))
    print("=" * 100)
    print(f"Questions:           {report['question_count']}")
    print(f"Top-k:               {report['top_k']}")
    print("")
    print("Retrieval metrics (averaged across questions)")
    print(f"  term_recall (compat):  {a['term_recall']:.3f}")
    print(f"  recall@1:              {a['recall_at_1']:.3f}")
    print(f"  recall@3:              {a['recall_at_3']:.3f}")
    print(f"  recall@5:              {a['recall_at_5']:.3f}")
    print(f"  recall@10:             {a['recall_at_10']:.3f}")
    print(f"  precision@1:           {a['precision_at_1']:.3f}")
    print(f"  precision@3:           {a['precision_at_3']:.3f}")
    print(f"  precision@5:           {a['precision_at_5']:.3f}")
    print(f"  precision@10:          {a['precision_at_10']:.3f}")
    print(f"  MRR:                   {a['mrr']:.3f}")
    print(f"  nDCG@3:                {a['ndcg_at_3']:.3f}")
    print(f"  nDCG@5:                {a['ndcg_at_5']:.3f}")
    print(f"  nDCG@10:               {a['ndcg_at_10']:.3f}")
    print(f"  hit@1:                 {a['hit_at_1']:.3f}")
    print(f"  hit@5:                 {a['hit_at_5']:.3f}")
    print(f"  paper diversity avg:   {a['paper_diversity']:.2f}")
    print("")
    print("Timing")
    print(f"  mean:  {t['mean_seconds']:.2f}s")
    print(f"  p50:   {t['p50_seconds']:.2f}s")
    print(f"  p95:   {t['p95_seconds']:.2f}s")
    print(f"  max:   {t['max_seconds']:.2f}s")
    print(f"  total: {t['total_seconds']:.2f}s")
    print("")

    if report.get("failures"):
        print(f"Weak questions ({len(report['failures'])} of {report['question_count']}):")
        for f in report["failures"][:20]:
            r = f["term_recall"]
            rank = f["first_hit_rank"]
            rank_str = str(rank) if rank else "none"
            print(f"  [r={r:.2f} rank={rank_str}] {f['question']}")
            if f["misses"]:
                print(f"    missing terms: {f['misses']}")
        if len(report["failures"]) > 20:
            print(f"  (+{len(report['failures']) - 20} more)")
        print("")

    avg = a["term_recall"]
    if avg >= 0.85:
        status = "STRONG"
    elif avg >= 0.70:
        status = "GOOD"
    elif avg >= 0.55:
        status = "ACCEPTABLE"
    else:
        status = "WEAK"
    print(f"Overall status: {status}")
    print("=" * 100)


def _report_markdown(report: Dict[str, Any]) -> str:
    avg = report.get("averages", {})
    tim = report.get("timing", {})
    L = ["# Retrieval Eval Report", "",
         f"_{report.get('label', 'default')} · top_k={report.get('top_k')} · "
         f"{report.get('question_count')} questions_", "",
         "## Recall / ranking", "", "| metric | value |", "|---|---|"]
    for k in ("term_recall", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5", "paper_diversity"):
        if k in avg:
            L.append(f"| {k} | {avg[k]:.3f} |")
    L += ["", "## Latency (seconds)", "", "| metric | value |", "|---|---|"]
    for k in ("mean_seconds", "p50_seconds", "p95_seconds", "max_seconds", "total_seconds"):
        if k in tim:
            L.append(f"| {k} | {tim[k]:.2f} |")
    failures = report.get("failures") or []
    if failures:
        L += ["", f"## Weak questions ({len(failures)})", ""]
        for f in failures:
            miss = ", ".join((f.get("misses") or [])[:5])
            L.append(f"- r={f.get('term_recall', 0):.2f} — {f.get('question', '')[:80]}"
                     + (f"  _(missing: {miss})_" if miss else ""))
    return "\n".join(L) + "\n"


def save_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    print(f"Report saved: {path}")
    print(f"Markdown:     {md_path}")


# ----------------------------------------------------------------------
# Mode comparison
# ----------------------------------------------------------------------

def run_mode_comparison(retriever,
                        questions: List[Dict[str, Any]],
                        top_k: int = 10) -> Dict[str, Any]:
    apply_mode = try_import_mode_binding()
    if apply_mode is None:
        print("WARN: could not import research_modes.apply_research_mode.")
        print("      Falling back to a single Balanced run.")
        balanced = run_single_pass(retriever, questions, top_k=top_k,
                                   quiet=True, label="balanced (no binding)")
        return {"modes": [balanced]}

    modes = ["fast", "balanced", "deep"]
    reports = []

    for mode in modes:
        try:
            apply_mode(mode)
        except Exception as exc:
            print(f"WARN: apply_research_mode({mode!r}) failed: {exc}")
            continue

        # Bust caches so each mode starts fresh
        try:
            import backend.retrieval.hybrid_retrieve as hr  # type: ignore
            hr._chunks_cache = None  # noqa: SLF001
            hr._bm25_cache = None    # noqa: SLF001
        except Exception:
            pass

        print(f"\n>>> Running mode: {mode.upper()}")
        rep = run_single_pass(retriever, questions, top_k=top_k,
                              quiet=True, label=mode)
        reports.append(rep)
        a = rep["averages"]
        t = rep["timing"]
        print(f"  term_recall={a['term_recall']:.3f}  "
              f"recall@5={a['recall_at_5']:.3f}  "
              f"MRR={a['mrr']:.3f}  "
              f"nDCG@5={a['ndcg_at_5']:.3f}  "
              f"hit@5={a['hit_at_5']:.3f}  "
              f"mean_time={t['mean_seconds']:.2f}s")

    print("")
    print("=" * 100)
    print("MODE COMPARISON TABLE")
    print("=" * 100)
    header = f"{'Mode':<10} {'term_rec':>8} {'r@5':>6} {'MRR':>6} {'nDCG@5':>7} {'hit@5':>6} {'time(s)':>8}"
    print(header)
    print("-" * len(header))
    for rep in reports:
        a = rep["averages"]
        t = rep["timing"]
        row = (
            f"{rep['label']:<10} "
            f"{a['term_recall']:>8.3f} "
            f"{a['recall_at_5']:>6.3f} "
            f"{a['mrr']:>6.3f} "
            f"{a['ndcg_at_5']:>7.3f} "
            f"{a['hit_at_5']:>6.3f} "
            f"{t['mean_seconds']:>8.2f}"
        )
        print(row)
    print("=" * 100)

    return {"modes": reports}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio RAG retrieval evaluator with IR metrics.",
    )
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many results to retrieve per question (default 10).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-question output, show only summary.")
    args = parser.parse_args()

    questions = load_questions()
    retriever = import_retriever()

    # There is one optimized retrieval config now — always bind it before running.
    apply_mode = try_import_mode_binding()
    if apply_mode is not None:
        try:
            apply_mode(None)
        except Exception as exc:
            print(f"WARN: apply_research_mode() failed: {exc}")

    report = run_single_pass(retriever, questions, top_k=args.top_k,
                             quiet=args.quiet, label="Default")
    print_summary(report)
    save_report(report, REPORT_FILE)


if __name__ == "__main__":
    main()
