"""
evaluate_llm.py -- measure how accurately an LLM answers questions about your
indexed papers, end to end (retrieval + answer), so you can compare models and
decide which one to keep.

Run from the project root (Oracle DB must be up and papers indexed):

    # Score the model currently selected in .env
    python -m backend.evaluation.evaluate_llm

    # Compare several models head-to-head (provider:model, comma-separated)
    python -m backend.evaluation.evaluate_llm --models "openai:gpt-4o,openai:gpt-4o-mini"

    # Add an LLM-as-judge correctness score (costs extra API calls)
    python -m backend.evaluation.evaluate_llm --judge --judge-model openai:gpt-4o

    # Test the model's raw knowledge, ignoring retrieval
    python -m backend.evaluation.evaluate_llm --no-retrieval

What it scores, per question:
  - keypoint coverage : fraction of the expected key points found in the answer
  - cited             : did the answer cite its sources with [n]?
  - judge (optional)  : 0-100 correctness graded by a strong "judge" model

It prints a scorecard for each model, a ranked comparison, and saves a JSON
report under data/. Questions live in data/llm_eval_questions.json -- edit that
file so the key points match the papers you have indexed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

# Reuse the EXACT prompt + evidence logic the live web app uses, so the score
# reflects the real system rather than a parallel implementation.
from webapp.chat_logic import (  # noqa: E402
    SYSTEM_PROMPT, format_evidence, build_user_message, public_source,
)
from backend.retrieval.hybrid_retrieve import hybrid_retrieve  # noqa: E402
from backend.answering.research_modes import apply_research_mode  # noqa: E402
from backend.llm.streaming_provider import get_provider  # noqa: E402

QUESTIONS_PATH = ROOT / "data" / "llm_eval_questions.json"


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def _keypoint_hit(answer_norm: str, key_point: str) -> bool:
    """A key point matches if ANY of its 'a|b|c' alternatives appears."""
    for alt in key_point.split("|"):
        alt = _norm(alt).strip()
        if alt and alt in answer_norm:
            return True
    return False


def score_answer(answer: str, key_points: List[str]) -> Tuple[float, bool, List[str]]:
    a = _norm(answer)
    hits = [kp for kp in key_points if _keypoint_hit(a, kp)]
    coverage = len(hits) / len(key_points) if key_points else 0.0
    cited = bool(re.search(r"\[\d+\]", answer))
    missed = [kp for kp in key_points if kp not in hits]
    return coverage, cited, missed


# ----------------------------------------------------------------------
# Provider construction + generation
# ----------------------------------------------------------------------
def build_provider(spec: Optional[str]):
    """spec is 'openai:model' or just 'model' (None -> current .env model).
    Sets OPENAI_MODEL and uses the real get_provider() factory."""
    if not spec:
        return get_provider()
    _, _, tail = spec.partition(":")
    model = (tail or spec).strip()
    if model:
        os.environ["OPENAI_MODEL"] = model
    return get_provider()


def generate(provider, question: str, mode: str, top_k: int,
             use_retrieval: bool, max_tokens: int = 2048) -> Tuple[str, List[Dict[str, Any]], float]:
    sources: List[Dict[str, Any]] = []
    if use_retrieval:
        try:
            apply_research_mode(mode)
        except Exception:
            pass
        results = hybrid_retrieve(question, top_k=top_k) or []
        sources = [public_source(r, i) for i, r in enumerate(results, 1)]
        user_msg = build_user_message(question, format_evidence(results))
        system = SYSTEM_PROMPT
    else:
        user_msg = question
        system = "You are an expert in audio signal processing. Answer precisely."

    t0 = time.time()
    parts: List[str] = []
    for chunk in provider.stream_chat(
        [{"role": "user", "content": user_msg}],
        system=system, max_tokens=max_tokens, temperature=0.2,
    ):
        parts.append(chunk)
    return "".join(parts).strip(), sources, time.time() - t0


def judge_answer(judge, question: str, key_points: List[str], answer: str) -> Optional[int]:
    prompt = (
        f"Question: {question}\n\n"
        f"Key points a correct answer should cover:\n- " + "\n- ".join(key_points) + "\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Score the answer from 0 to 100 for correctness and completeness against "
        "the key points. Reply with ONLY the number."
    )
    out = "".join(judge.stream_chat(
        [{"role": "user", "content": prompt}],
        system="You are a strict grader. Output only an integer 0-100.",
        max_tokens=8, temperature=0.0,
    ))
    m = re.search(r"\d{1,3}", out)
    return min(100, int(m.group())) if m else None


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def evaluate_model(spec: Optional[str], questions: List[Dict[str, Any]],
                   mode: str, top_k: int, use_retrieval: bool,
                   judge=None, max_tokens: int = 2048) -> Dict[str, Any]:
    provider = build_provider(spec)
    label = f"{provider.name}:{provider.model}"
    print(f"\n{'=' * 70}\nMODEL: {label}\n{'=' * 70}")
    if not provider.is_available:
        print(f"  [skipped] {provider.unavailable_message()}")
        return {"model": label, "available": False, "results": []}

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        key_points = q.get("key_points", [])
        try:
            answer, sources, secs = generate(provider, question, mode, top_k, use_retrieval, max_tokens)
        except Exception as exc:
            print(f"  Q{i}: generation failed: {exc}")
            rows.append({"question": question, "error": str(exc), "coverage": 0.0,
                         "cited": False, "judge": None, "seconds": 0.0})
            continue
        coverage, cited, missed = score_answer(answer, key_points)
        jscore = judge_answer(judge, question, key_points, answer) if judge else None
        rows.append({
            "question": question, "coverage": round(coverage, 3), "cited": cited,
            "judge": jscore, "seconds": round(secs, 1), "n_sources": len(sources),
            "missed_key_points": missed, "answer": answer,
        })
        jtxt = f"  judge {jscore:>3}" if jscore is not None else ""
        print(f"  Q{i}: coverage {coverage * 100:5.0f}%  cited {'Y' if cited else 'n'}"
              f"{jtxt}  ({secs:.1f}s)  {question[:48]}")

    scored = [r for r in rows if "error" not in r]
    n = len(scored) or 1
    agg = {
        "model": label,
        "available": True,
        "questions": len(rows),
        "avg_coverage": round(sum(r["coverage"] for r in scored) / n, 3),
        "citation_rate": round(sum(1 for r in scored if r["cited"]) / n, 3),
        "avg_seconds": round(sum(r["seconds"] for r in scored) / n, 1),
        "errors": sum(1 for r in rows if "error" in r),
        "results": rows,
    }
    judged = [r["judge"] for r in scored if r["judge"] is not None]
    if judged:
        agg["avg_judge"] = round(sum(judged) / len(judged), 1)
    return agg


def main() -> int:
    p = argparse.ArgumentParser(description="Measure & compare LLM answer accuracy on your papers.")
    p.add_argument("--models", default="", help="Comma-separated provider:model specs. Default: the .env model.")
    p.add_argument("--questions", default=str(QUESTIONS_PATH), help="Path to the questions JSON.")
    p.add_argument("--mode", default="Balanced", help="Retrieval mode (Fast/Balanced/Deep).")
    p.add_argument("--top-k", type=int, default=8, help="Sources to retrieve per question.")
    p.add_argument("--max-tokens", type=int, default=2048, help="Answer token budget (match the app; reasoning models need room).")
    p.add_argument("--limit", type=int, default=0, help="Only evaluate the first N questions (0 = all).")
    p.add_argument("--no-retrieval", action="store_true", help="Test raw model knowledge (skip retrieval).")
    p.add_argument("--judge", action="store_true", help="Also score with an LLM judge (extra API calls).")
    p.add_argument("--judge-model", default="", help="provider:model for the judge (default: the first evaluated model).")
    args = p.parse_args()

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = data.get("questions", data if isinstance(data, list) else [])
    if args.limit > 0:
        questions = questions[:args.limit]
    if not questions:
        print("No questions found. Add some to", args.questions)
        return 1

    specs: List[Optional[str]] = [s.strip() for s in args.models.split(",") if s.strip()] or [None]

    judge = None
    if args.judge:
        judge_spec = args.judge_model or (specs[0] if specs[0] else None)
        judge = build_provider(judge_spec)
        print(f"Judge model: {judge.name}:{judge.model}")

    use_retrieval = not args.no_retrieval
    print(f"Questions: {len(questions)} | retrieval: {use_retrieval} | mode: {args.mode} | top_k: {args.top_k}")

    reports = [evaluate_model(spec, questions, args.mode, args.top_k, use_retrieval, judge, args.max_tokens)
               for spec in specs]

    # Ranked comparison
    ranked = sorted([r for r in reports if r.get("available")],
                    key=lambda r: (r.get("avg_judge", r["avg_coverage"] * 100)), reverse=True)
    print(f"\n{'=' * 70}\nSCORECARD  (higher is better)\n{'=' * 70}")
    header = f"{'model':<40} {'coverage':>9} {'cited':>6} {'judge':>6} {'sec':>6}"
    print(header); print("-" * len(header))
    for r in ranked:
        judge_txt = f"{r.get('avg_judge', '—'):>6}" if "avg_judge" in r else f"{'—':>6}"
        print(f"{r['model']:<40} {r['avg_coverage'] * 100:>8.0f}% "
              f"{r['citation_rate'] * 100:>5.0f}% {judge_txt} {r['avg_seconds']:>6.1f}")
    if ranked:
        print(f"\nBest: {ranked[0]['model']}")

    out_path = ROOT / "data" / f"llm_eval_results_{int(time.time())}.json"
    out_path.write_text(json.dumps({"config": vars(args), "reports": reports}, indent=2), encoding="utf-8")
    print(f"\nSaved detailed report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
