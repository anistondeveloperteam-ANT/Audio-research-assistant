"""
measure_classifiers.py  --  Honest, reproducible measurement of the decision classifiers in this
system, with full confusion matrices and every standard metric.

This RAG assistant has no single trained model; its measurable "models" are the deterministic
decision components that route and guard every request:

  1. Code-intent router      backend/answering/code_intent.is_code_intent        (binary)
  2. Task-type classifier    backend/answering/task_classifier.infer_task_type   (3-class)
  3. Query-sanity gate       backend/answering/query_sanity.check_query_sanity   (binary)
  4. Answer-reuse safety     backend/memory/store.unsafe_to_reuse                (binary)

We measure the DETERMINISTIC layer (CODE_INTENT_SEMANTIC=false) so the numbers are exact and
reproducible offline — no LLM, no network. (In production the semantic LLM layer is UNIONed on
top of the code-intent regex to lift recall; that path needs a live model and is not part of this
offline measurement.)

Run:  python -m backend.evaluation.measure_classifiers            # prints + writes docs/MEASUREMENT.md
      python -m backend.evaluation.measure_classifiers --out X.md
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, List, Tuple

# Deterministic, offline: measure the always-on regex/heuristic layer (no LLM call).
os.environ.setdefault("CODE_INTENT_SEMANTIC", "false")

from backend.answering.code_intent import is_code_intent          # noqa: E402
from backend.answering.query_sanity import check_query_sanity     # noqa: E402
from backend.answering.task_classifier import infer_task_type     # noqa: E402
from backend.memory.store import unsafe_to_reuse                  # noqa: E402


# ----------------------------------------------------------------------
# Labeled evaluation sets (ground truth). Curated to include cases the
# deterministic layer is KNOWN to get wrong, so the numbers are honest.
# ----------------------------------------------------------------------

# 1) CODE-INTENT — label 1 = should route to the code agent, 0 = prose answer.
CODE_INTENT: List[Tuple[str, int]] = [
    ("implement quicksort", 1),
    ("write a function that sorts a list", 1),
    ("give me RTF-MVDR python code", 1),
    ("python code for the FFT", 1),
    ("benchmark mergesort vs quicksort", 1),
    ("simulate a damped pendulum", 1),
    ("refactor this loop", 1),
    ("generate an implementation of Dijkstra", 1),
    ("show me a script that scrapes a page", 1),
    ("give me code to price a European option with Black-Scholes", 1),
    ("debug this segfault in my program", 1),
    ("optimize this function for speed", 1),
    ("make a function to reverse a string", 1),
    ("write python to FFT a chirp", 1),                 # known regex miss (no code-noun)
    ("model SIR epidemic spread", 1),                   # known regex miss ('model' not a verb)
    ("price a European option", 1),                     # known regex miss
    ("show MVDR working on synthetic signals", 1),      # known regex miss
    ("estimate pi with monte carlo", 1),                # known regex miss
    ("create a class for a binary tree", 1),            # known regex miss ('class' w/o python)
    ("What is MVDR beamforming?", 0),
    ("Explain the Black-Scholes model", 0),
    ("How does transformer attention work?", 0),
    ("Compare Raft and Paxos", 0),
    ("Summarize the latest diffusion-model papers", 0),
    ("What does the function of the hippocampus involve?", 0),   # 'function' must NOT trigger
    ("Give me the latest papers on speech enhancement", 0),
    ("Why is the sky blue?", 0),
    ("Define entropy in information theory", 0),
    ("What are the pros and cons of microservices?", 0),
    ("Describe how RAG works", 0),
    ("Who invented the FFT?", 0),
    ("Is quicksort stable?", 0),
    ("What is the time complexity of mergesort?", 0),
]

# 2) TASK-TYPE — only on genuine code tasks. 3 classes.
TASK_TYPE: List[Tuple[str, str]] = [
    ("implement quicksort", "deterministic"),
    ("reverse a string in place", "deterministic"),
    ("merge two sorted lists", "deterministic"),
    ("parse a CSV into records", "deterministic"),
    ("implement Dijkstra shortest path", "deterministic"),
    ("count word frequencies in a sentence", "deterministic"),
    ("compute the FFT of a signal", "numeric_algorithm"),
    ("price a European call with Black-Scholes", "numeric_algorithm"),
    ("compute eigenvalues of a matrix", "numeric_algorithm"),
    ("gradient descent to minimize a function", "numeric_algorithm"),
    ("design an MVDR beamformer", "numeric_algorithm"),
    ("apply a convolution filter to an image", "numeric_algorithm"),
    ("solve a linear system Ax = b", "numeric_algorithm"),       # known heuristic miss
    ("simulate a damped pendulum", "simulation"),
    ("estimate pi with a monte carlo simulation", "simulation"),
    ("model SIR epidemic spread", "simulation"),
    ("a random walk on a 2D grid", "simulation"),
    ("n-body gravity simulation", "simulation"),
    ("simulate rolling two dice 10000 times", "simulation"),     # known heuristic miss ('dice')
]

# 3) QUERY-SANITY — label 1 = legitimate (should pass), 0 = gibberish (should be refused).
SANITY: List[Tuple[str, int]] = [
    ("What is MVDR beamforming?", 1),
    ("Explain the Black-Scholes model", 1),
    ("How does RAG improve answers?", 1),
    ("implement quicksort in python", 1),
    ("compute the FFT of a chirp signal", 1),
    ("Compare Raft and Paxos and when to use each", 1),
    ("What is 6*7?", 1),
    ("How do I price a European option?", 1),
    ("asdf qwer zxcv", 0),
    ("buoh", 0),
    ("aaaaaaaa", 0),
    ("lolololol", 0),
    ("?????", 0),
    ("x", 0),
    ("   ", 0),
    ("zxcvbnm", 0),
]

# 4) ANSWER-REUSE SAFETY — unsafe_to_reuse(a, b): label 1 = UNSAFE (must NOT reuse a's answer for
#    b — different meaning), 0 = SAFE (true rephrase, reuse is fine).
REUSE: List[Tuple[str, str, int]] = [
    ("A100 vs H100 performance", "A100 vs V100 performance", 1),     # identifier change
    ("advantages of TCP", "disadvantages of TCP", 1),               # polarity flip
    ("encrypt a file in python", "decrypt a file in python", 1),    # contrast group
    ("upsample an audio signal", "downsample an audio signal", 1),  # contrast group
    ("Raft vs Paxos", "Paxos vs Raft", 1),                          # argument swap
    ("convert km to miles", "convert miles to km", 1),              # unit swap
    ("benefits of microservices", "drawbacks of microservices", 1),  # contrast group
    ("what is GPT", "what is BERT", 1),                             # short-entity change
    ("How does MVDR reduce noise?", "How does MVDR beamforming reduce noise?", 0),
    ("Summarize the Raft consensus paper", "Give a summary of the Raft consensus paper", 0),
    ("What is gradient descent?", "Explain gradient descent", 0),
    ("Best way to parse JSON in Python", "How to parse JSON in Python", 0),
    ("How does the FFT work?", "How does the fast Fourier transform work?", 0),
    ("Explain reinforcement learning", "Explain reinforcement learning to me", 0),
]


# ----------------------------------------------------------------------
# Metrics (no third-party deps)
# ----------------------------------------------------------------------
def _binary(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity / TPR
    spec = tn / (tn + fp) if (tn + fp) else 0.0         # TNR
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    def fbeta(b: float) -> float:
        b2 = b * b
        den = b2 * prec + rec
        return (1 + b2) * prec * rec / den if den else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    po = acc
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return {
        "support": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "prevalence": (tp + fn) / n if n else 0.0,
        "accuracy": acc, "balanced_accuracy": (rec + spec) / 2,
        "precision": prec, "recall_sensitivity_tpr": rec, "specificity_tnr": spec,
        "npv": npv, "f1": fbeta(1.0), "f0.5": fbeta(0.5), "f2": fbeta(2.0),
        "fpr": fpr, "fnr": fnr, "mcc": mcc, "cohen_kappa": kappa,
    }


def _run_binary(fn: Callable[[str], bool], data: List[Tuple[str, int]]):
    tp = fp = tn = fn_ = 0
    misses = []
    for text, label in data:
        pred = 1 if fn(text) else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
            misses.append((text, label, pred))
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn_ += 1
            misses.append((text, label, pred))
    return _binary(tp, fp, tn, fn_), misses


def _run_multiclass(fn: Callable[[str], str], data: List[Tuple[str, str]], classes: List[str]):
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]
    misses = []
    for text, label in data:
        pred = fn(text)
        if pred not in idx:
            pred = classes[0]
        cm[idx[label]][idx[pred]] += 1
        if pred != label:
            misses.append((text, label, pred))
    per = {}
    correct = total = 0
    for c in classes:
        i = idx[c]
        tp = cm[i][i]
        col = sum(cm[r][i] for r in range(len(classes)))
        row = sum(cm[i])
        prec = tp / col if col else 0.0
        rec = tp / row if row else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": row}
        correct += tp
        total += row
    acc = correct / total if total else 0.0
    macro = {k: sum(per[c][k] for c in classes) / len(classes) for k in ("precision", "recall", "f1")}
    wt = {k: (sum(per[c][k] * per[c]["support"] for c in classes) / total if total else 0.0)
          for k in ("precision", "recall", "f1")}
    return cm, per, acc, macro, wt, misses


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _binary_section(title: str, fn_name: str, m: Dict[str, float], misses) -> str:
    cm = (
        "|  | **Pred: Positive** | **Pred: Negative** |\n"
        "|---|:---:|:---:|\n"
        f"| **Actual: Positive** | TP = {m['tp']} | FN = {m['fn']} |\n"
        f"| **Actual: Negative** | FP = {m['fp']} | TN = {m['tn']} |\n"
    )
    rows = [
        ("Support (N)", str(m["support"])), ("Prevalence (positives)", _pct(m["prevalence"])),
        ("**Accuracy**", _pct(m["accuracy"])), ("Balanced accuracy", _pct(m["balanced_accuracy"])),
        ("**Precision** (PPV)", _pct(m["precision"])),
        ("**Recall** (Sensitivity / TPR)", _pct(m["recall_sensitivity_tpr"])),
        ("Specificity (TNR)", _pct(m["specificity_tnr"])), ("NPV", _pct(m["npv"])),
        ("**F1**", _pct(m["f1"])), ("F0.5", _pct(m["f0.5"])), ("F2", _pct(m["f2"])),
        ("FPR (fall-out)", _pct(m["fpr"])), ("FNR (miss rate)", _pct(m["fnr"])),
        ("**MCC** (Matthews, −1..1)", f"{m['mcc']:.3f}"),
        ("Cohen's κ (0..1)", f"{m['cohen_kappa']:.3f}"),
    ]
    met = "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    miss = "\n".join(f"- `{t}` — actual **{'pos' if lbl else 'neg'}**, predicted "
                     f"**{'pos' if pr else 'neg'}**" for t, lbl, pr in misses) or "- _none_ ✅"
    return (f"### {title}\n\n`{fn_name}` · binary classifier\n\n"
            f"**Confusion matrix**\n\n{cm}\n**Metrics**\n\n{met}\n\n"
            f"**Misclassified ({len(misses)})**\n\n{miss}\n")


def _multiclass_section(title, fn_name, classes, cm, per, acc, macro, wt, misses) -> str:
    head = "| actual ╲ pred | " + " | ".join(classes) + " | support |\n"
    head += "|---|" + "|".join([":---:"] * len(classes)) + "|:---:|\n"
    body = ""
    for i, c in enumerate(classes):
        body += f"| **{c}** | " + " | ".join(str(cm[i][j]) for j in range(len(classes)))
        body += f" | {sum(cm[i])} |\n"
    pc = "| class | precision | recall | F1 | support |\n|---|---|---|---|---|\n"
    for c in classes:
        p = per[c]
        pc += f"| {c} | {_pct(p['precision'])} | {_pct(p['recall'])} | {_pct(p['f1'])} | {p['support']} |\n"
    pc += (f"| **macro avg** | {_pct(macro['precision'])} | {_pct(macro['recall'])} | {_pct(macro['f1'])} | — |\n"
           f"| **weighted avg** | {_pct(wt['precision'])} | {_pct(wt['recall'])} | {_pct(wt['f1'])} | — |\n")
    miss = "\n".join(f"- `{t}` — actual **{lbl}**, predicted **{pr}**" for t, lbl, pr in misses) or "- _none_ ✅"
    return (f"### {title}\n\n`{fn_name}` · {len(classes)}-class classifier\n\n"
            f"**Confusion matrix** (rows = actual, cols = predicted)\n\n{head}{body}\n"
            f"**Overall accuracy (micro-F1): {_pct(acc)}**\n\n{pc}\n"
            f"**Misclassified ({len(misses)})**\n\n{miss}\n")


def build_report() -> str:
    ci_m, ci_miss = _run_binary(is_code_intent, CODE_INTENT)
    sn_m, sn_miss = _run_binary(lambda q: check_query_sanity(q).ok, SANITY)
    rz_m, rz_miss = _run_binary(lambda pair: unsafe_to_reuse(*pair),
                                [((a, b), lbl) for a, b, lbl in REUSE])
    tt_classes = ["deterministic", "numeric_algorithm", "simulation"]
    tt = _run_multiclass(infer_task_type, TASK_TYPE, tt_classes)

    summary = (
        "| Classifier | Type | Accuracy | Precision | Recall | F1 | MCC |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| Code-intent router | binary | {_pct(ci_m['accuracy'])} | {_pct(ci_m['precision'])} | "
        f"{_pct(ci_m['recall_sensitivity_tpr'])} | {_pct(ci_m['f1'])} | {ci_m['mcc']:.3f} |\n"
        f"| Task-type classifier | 3-class | {_pct(tt[2])} | {_pct(tt[4]['precision'])} | "
        f"{_pct(tt[4]['recall'])} | {_pct(tt[4]['f1'])} | — |\n"
        f"| Query-sanity gate | binary | {_pct(sn_m['accuracy'])} | {_pct(sn_m['precision'])} | "
        f"{_pct(sn_m['recall_sensitivity_tpr'])} | {_pct(sn_m['f1'])} | {sn_m['mcc']:.3f} |\n"
        f"| Answer-reuse safety | binary | {_pct(rz_m['accuracy'])} | {_pct(rz_m['precision'])} | "
        f"{_pct(rz_m['recall_sensitivity_tpr'])} | {_pct(rz_m['f1'])} | {rz_m['mcc']:.3f} |\n"
    )

    return "\n".join([
        "# 📐 Classifier Measurement Report",
        "",
        "> **Auto-generated** by `python -m backend.evaluation.measure_classifiers` — every number "
        "below is computed by running the real code on a labeled set, not hand-written. "
        "Regenerate any time; results are deterministic and fully offline.",
        "",
        "## What is measured (and what isn't)",
        "",
        "This is a Retrieval-Augmented-Generation assistant: the final *answer* quality is the LLM's, "
        "and is graded separately (faithfulness / relevancy / retrieval recall — see "
        "[RAG_BASELINE.md](RAG_BASELINE.md) and `backend/evaluation/evaluate_*`). What this report "
        "measures are the **deterministic decision classifiers** that route and guard every request "
        "— the parts of the system that *do* have a ground truth and a confusion matrix.",
        "",
        "Measured on the **deterministic layer** (`CODE_INTENT_SEMANTIC=false`) so the numbers are "
        "exact and reproducible with no LLM/network. In production a semantic LLM classifier is "
        "**unioned on top** of the code-intent regex to raise recall (it recovers the phrasings "
        "marked as misses below); that path needs a live model and isn't part of this offline run.",
        "",
        "## Summary",
        "",
        summary,
        "## Metric glossary",
        "",
        "| Term | Meaning |",
        "|---|---|",
        "| **Confusion matrix** | Counts of TP / FP / TN / FN (predicted vs actual). |",
        "| **Accuracy** | (TP+TN)/N — overall correctness. |",
        "| **Precision (PPV)** | TP/(TP+FP) — of things flagged positive, how many really are. |",
        "| **Recall (Sensitivity, TPR)** | TP/(TP+FN) — of real positives, how many were caught. |",
        "| **Specificity (TNR)** | TN/(TN+FP) — of real negatives, how many were left alone. |",
        "| **F1 / F0.5 / F2** | Harmonic mean of P&R (F0.5 favors precision, F2 favors recall). |",
        "| **Balanced accuracy** | (Recall+Specificity)/2 — fair when classes are imbalanced. |",
        "| **MCC** | Matthews correlation (−1..1); 1 = perfect, 0 = random — robust on imbalance. |",
        "| **Cohen's κ** | Agreement above chance (0..1). |",
        "| **FPR / FNR** | False-positive / false-negative rate. |",
        "| **Prevalence / Support** | Share of positives / number of examples. |",
        "",
        "---",
        "",
        "## 1. Code-intent router — *does this question need the code agent?*",
        "",
        _binary_section("Code-intent router", "backend/answering/code_intent.is_code_intent",
                        ci_m, ci_miss),
        "> 🟢 **Read:** high **precision** (it rarely sends a prose question to the agent) with "
        "lower **recall** — the deterministic regex misses differently-phrased code tasks "
        "(*\"model SIR…\"*, *\"price a European option\"*). Those exact misses are why the semantic "
        "LLM layer is unioned on top in production: it catches them while precision stays high.",
        "",
        "---",
        "",
        "## 2. Task-type classifier — *how should the answer be verified?*",
        "",
        _multiclass_section("Task-type classifier", "backend/answering/task_classifier.infer_task_type",
                            tt_classes, *tt),
        "> 🟢 **Read:** picks the verification strategy (exact-output vs domain-invariants vs "
        "simulation properties). Misses are conservative — an unrecognized phrasing falls back to "
        "`deterministic`, the safest default.",
        "",
        "---",
        "",
        "## 3. Query-sanity gate — *is this a real question or gibberish?*",
        "",
        _binary_section("Query-sanity gate", "backend/answering/query_sanity.check_query_sanity",
                        sn_m, sn_miss),
        "> 🟢 **Read:** a cheap first-pass filter (no ML) that blocks keyboard-mash before it reaches "
        "retrieval + the LLM. Tuned to favor letting real questions through (high recall on legit).",
        "",
        "---",
        "",
        "## 4. Answer-reuse safety — *is it safe to reuse a cached answer?*",
        "",
        _binary_section("Answer-reuse safety", "backend/memory/store.unsafe_to_reuse",
                        rz_m, rz_miss),
        "> 🟢 **Read:** the guard that blocks a cache hit when two similar-looking questions actually "
        "differ (an A↔B swap, A100↔H100, *with*↔*without*). It errs toward **blocking** (high recall "
        "on UNSAFE) — a wrong reuse is worse than a recompute.",
        "",
        "---",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m backend.evaluation.measure_classifiers      # regenerates this file",
        "```",
        "",
        "Labeled sets live in `backend/evaluation/measure_classifiers.py` (curated to include known "
        "edge cases). Extend them and re-run to track the numbers over time.",
        "",
    ])


def main(argv: List[str]) -> int:
    try:                                   # Windows consoles default to cp1252; the report has emoji
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out = "docs/MEASUREMENT.md"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    report = build_report()
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    dest = root / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
