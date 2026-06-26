"""
self_tuner.py — the EVAL-GATED self-tuning loop (Phase 3).

Closes the loop that `param_suggest.py` left open — but SAFELY. For each tunable threshold it tries a
small step up/down (within the registry's bounds), measures each candidate with the eval harness, and
ADOPTS a value ONLY when it provably improves the metric by a margin. Everything else is rejected. This
is coordinate ascent gated by an offline eval — never a blind change.

Safety contract (mirrors the owner's "no silent self-tuning" rule):
  * Default is PROPOSE-ONLY — it records what it *would* change but does NOT alter behaviour.
  * It only APPLIES when SELF_TUNING is explicitly on (or apply=True is passed).
  * Every trial (proposed or applied) is recorded in `tuning_events` for audit.
  * Adopted values are bounded (clamped to the tunable's [lo, hi]) and fully REVERSIBLE
    (`tuning.clear_overrides(mem)` reverts everything in one call).
  * The tuner runs OFFLINE (CLI / scheduled), never on the request path.

The optimizer core (`tune`) takes an injected `evaluate_fn(overrides) -> float` (higher is better), so
it is fully unit-testable with a synthetic metric. `make_llm_evaluator` / `make_retrieval_evaluator`
wire `evaluate_fn` to the real harnesses for the owner's offline runs.
"""
from __future__ import annotations

import argparse
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from backend.answering import tuning

logger = logging.getLogger(__name__)

EvaluateFn = Callable[[Dict[str, float]], float]


def self_tuning_enabled() -> bool:
    """Default OFF: applying tuned config is opt-in. Proposing is always allowed."""
    return os.getenv("SELF_TUNING", "").strip().lower() in ("1", "true", "yes", "on")


def _candidates(t: "tuning.Tunable", cur: float) -> List[float]:
    """The neighbour values to try for one tunable: one step down, one step up, clamped to bounds and
    deduped, excluding the current value."""
    out: List[float] = []
    for cand in (cur - t.step, cur + t.step):
        c = tuning.clamp(t, cand)
        if c != tuning.clamp(t, cur) and c not in out:
            out.append(c)
    return out


@contextmanager
def pinned(overrides: Dict[str, float]):
    """Make `overrides` the live config for the duration of one evaluation, then restore. The pipeline's
    own `refresh(mem)` is suppressed while pinned so it cannot reload over the candidate."""
    tuning.pin(overrides)
    try:
        yield
    finally:
        tuning.unpin()


def tune(*, mem, evaluate_fn: EvaluateFn, names: Optional[List[str]] = None, margin: float = 0.0,
         rounds: int = 1, apply: Optional[bool] = None, source: str = "self_tuner") -> Dict[str, Any]:
    """Eval-gated coordinate ascent over the tunables.

    For each name, evaluate the current value and its neighbours; keep the best ONLY if it beats the
    current metric by more than `margin`. Repeat for `rounds` passes. Returns a summary with the
    baseline metric, the accepted changes, and whether they were applied.

    `apply` defaults to `self_tuning_enabled()`: when False the run is PROPOSE-ONLY (records trials,
    changes nothing); when True the adopted values are persisted and made live. `evaluate_fn(overrides)`
    must return a score where HIGHER is better."""
    do_apply = self_tuning_enabled() if apply is None else bool(apply)
    names = [n for n in (names or tuning.tunable_names()) if n in tuning.TUNABLES]

    current: Dict[str, float] = dict(tuning.current_overrides())   # start from any live overrides
    base_metric = float(evaluate_fn(current))
    start_metric = base_metric
    changes: List[Dict[str, Any]] = []

    for _ in range(max(1, int(rounds))):
        for name in names:
            t = tuning.TUNABLES[name]
            cur_val = current.get(name, t.default)
            best_val, best_metric = cur_val, base_metric
            for cand in _candidates(t, cur_val):
                trial = dict(current)
                trial[name] = cand
                metric = float(evaluate_fn(trial))
                accepted = metric > best_metric + margin
                mem.record_tuning_event(
                    name=name, old_value=best_val, new_value=cand, metric_before=best_metric,
                    metric_after=metric, accepted=accepted, applied=(accepted and do_apply),
                    note=("apply" if do_apply else "propose"))
                if accepted:
                    best_val, best_metric = cand, metric
            if best_val != cur_val:
                current[name] = best_val
                changes.append({"name": name, "from": cur_val, "to": best_val,
                                "metric_before": base_metric, "metric_after": best_metric})
                base_metric = best_metric
                if do_apply:
                    tuning.set_override(mem, name, best_val, source=source)

    if do_apply:
        tuning.refresh(mem, force=True)               # make the whole adopted set live at once

    return {"applied": do_apply, "start_metric": round(start_metric, 4),
            "final_metric": round(base_metric, 4), "changes": changes, "names": names}


# ----------------------------------------------------------------------
# Production evaluators (lazy — only needed for a real offline tuning run).
# ----------------------------------------------------------------------
def make_retrieval_evaluator(*, metric: str = "ndcg_at_5", top_k: int = 8,
                             mode: str = "balanced") -> EvaluateFn:
    """`evaluate_fn` that scores a candidate config by running the RETRIEVAL eval harness with the
    overrides pinned. Needs Oracle + data/evaluation_questions.json. Returns averages[metric]."""
    def _ev(overrides: Dict[str, float]) -> float:
        from backend.evaluation.evaluate_retrieval import (import_retriever, load_questions,
                                                           run_single_pass)
        with pinned(overrides):
            retriever = import_retriever()
            report = run_single_pass(retriever, load_questions(), top_k=top_k, label=mode)
        return float((report.get("averages") or {}).get(metric, 0.0))
    return _ev


def make_llm_evaluator(*, metric: str = "avg_coverage", model: Optional[str] = None,
                       judge: bool = False) -> EvaluateFn:
    """`evaluate_fn` that scores a candidate config by running the full-answer LLM eval with the
    overrides pinned. Needs an LLM provider + data/llm_eval_questions.json. Returns report[metric]
    (e.g. avg_coverage, or avg_judge when judge=True)."""
    def _ev(overrides: Dict[str, float]) -> float:
        from backend.evaluation.evaluate_llm import evaluate_model, load_questions
        with pinned(overrides):
            report = evaluate_model(model, load_questions(), judge=judge)
        return float(report.get(metric, 0.0) or 0.0)
    return _ev


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval-gated self-tuning of pipeline thresholds.")
    p.add_argument("--apply", action="store_true",
                   help="persist adopted values (requires SELF_TUNING=on); default is propose-only")
    p.add_argument("--harness", choices=["retrieval", "llm"], default="retrieval",
                   help="which eval harness gates the tuning")
    p.add_argument("--metric", default=None, help="metric key to maximise (harness-specific default)")
    p.add_argument("--names", default=None, help="comma-separated tunables to tune (default: all)")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--margin", type=float, default=0.0, help="min metric gain to adopt a change")
    p.add_argument("--reset", action="store_true", help="clear ALL overrides (revert to defaults) and exit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    from backend.memory.store import MemoryStore, default_db_path
    from pathlib import Path
    mem = MemoryStore(default_db_path(Path(__file__).resolve().parents[2]))

    if args.reset:
        n = tuning.clear_overrides(mem)
        print(f"Reverted {n} tuned override(s) to defaults.")
        return 0

    if args.harness == "llm":
        evaluate_fn = make_llm_evaluator(metric=(args.metric or "avg_coverage"))
    else:
        evaluate_fn = make_retrieval_evaluator(metric=(args.metric or "ndcg_at_5"))

    names = [s.strip() for s in args.names.split(",")] if args.names else None
    if args.apply and not self_tuning_enabled():
        print("Refusing to --apply: set SELF_TUNING=on to allow persisting tuned config. "
              "Running PROPOSE-ONLY instead.\n")
    result = tune(mem=mem, evaluate_fn=evaluate_fn, names=names, rounds=args.rounds,
                  margin=args.margin, apply=(args.apply and self_tuning_enabled()))

    mode = "APPLIED" if result["applied"] else "PROPOSED (not applied)"
    print(f"\nSelf-tuning {mode}.  metric {result['start_metric']} -> {result['final_metric']}")
    if not result["changes"]:
        print("  No change improved the metric — config left as-is.")
    for c in result["changes"]:
        print(f"  {c['name']}: {c['from']} -> {c['to']}  "
              f"({c['metric_before']:.4f} -> {c['metric_after']:.4f})")
    if not result["applied"] and result["changes"]:
        print("\n  (propose-only — re-run with SELF_TUNING=on --apply to persist these.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
