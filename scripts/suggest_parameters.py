"""CLI: suggest config thresholds from measured eval results. Print-only — NEVER edits .env.

Usage:
  python scripts/suggest_parameters.py --skip-precision 0.92 --recall 0.74
  python scripts/suggest_parameters.py --metrics path/to/metrics.json

It reads the CURRENT thresholds from config and prints a recommendation + the evidence. A human applies
any change by editing .env. The system never self-tunes its behaviour.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Suggest config thresholds from measured eval (print-only; never edits .env).")
    p.add_argument("--metrics", type=str, default=None,
                   help="JSON file of measured metrics (crag_skip_precision, crag_recall, ...).")
    p.add_argument("--skip-precision", type=float, default=None,
                   help="Measured STRONG-grade skip precision (0..1).")
    p.add_argument("--recall", type=float, default=None, help="Measured grader recall (0..1).")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass
    from backend.answering.evidence_grader import crag_strong_min, crag_partial_min
    from backend.evaluation.param_suggest import suggest_thresholds, format_suggestions

    metrics = {"crag_strong_min": crag_strong_min(), "crag_partial_min": crag_partial_min()}
    if args.metrics:
        try:
            metrics.update(json.loads(Path(args.metrics).read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"ERROR reading --metrics: {exc}", file=sys.stderr)
            return 2
    if args.skip_precision is not None:
        metrics["crag_skip_precision"] = args.skip_precision
    if args.recall is not None:
        metrics["crag_recall"] = args.recall

    if "crag_skip_precision" not in metrics and "crag_recall" not in metrics:
        print("No measured metrics given. Measure first, e.g.:\n"
              "  python -m backend.evaluation.measure_evidence_grader\n"
              "then pass --skip-precision / --recall (or --metrics metrics.json).\n")
        print(f"Current: CRAG_STRONG_MIN={metrics['crag_strong_min']}, "
              f"CRAG_PARTIAL_MIN={metrics['crag_partial_min']}")
        return 0
    print(format_suggestions(suggest_thresholds(metrics)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
