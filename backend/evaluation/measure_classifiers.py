"""
measure_classifiers.py  --  Full, reproducible measurement of the decision classifiers in this
system: confusion matrices, every standard metric, 95% confidence intervals, a per-domain
breakdown, and a LIVE regex-vs-(regex ∪ LLM) comparison for the router.

This RAG assistant has no single trained model; its measurable "models" are the deterministic
decision components that route and guard every request:

  1. Code-intent router      backend/answering/code_intent.is_code_intent        (binary)
  2. Task-type classifier    backend/answering/task_classifier.infer_task_type   (3-class)
  3. Query-sanity gate       backend/answering/query_sanity.check_query_sanity   (binary)
  4. Answer-reuse safety     backend/memory/store.unsafe_to_reuse                (binary)
  5. Freshness detector      webapp/chat_logic._freshness_sensitive             (binary)

The default run measures the DETERMINISTIC layer (no LLM/network) so numbers are exact and
reproducible. With the live LLM available it ALSO measures the semantic router (regex ∪ LLM) to
quantify the recall lift. Answer/retrieval quality (LLM-side) is graded separately — see
docs/RAG_BASELINE.md and backend/evaluation/evaluate_*.

Run:  python -m backend.evaluation.measure_classifiers              # full run -> docs/MEASUREMENT.md
      python -m backend.evaluation.measure_classifiers --no-semantic
      python -m backend.evaluation.measure_classifiers --out X.md
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("CODE_INTENT_SEMANTIC", "false")   # deterministic baseline by default

from backend.answering.code_intent import is_code_intent          # noqa: E402
from backend.answering.query_sanity import check_query_sanity     # noqa: E402
from backend.answering.task_classifier import infer_task_type     # noqa: E402
from backend.memory.store import unsafe_to_reuse                  # noqa: E402


def _freshness(q: str) -> bool:
    from webapp.chat_logic import _freshness_sensitive            # lazy (avoids import coupling)
    return _freshness_sensitive(q)


# ----------------------------------------------------------------------
# Labeled evaluation sets (ground truth). Curated to include cases the
# deterministic layer is KNOWN to get wrong, so the numbers are honest.
# ----------------------------------------------------------------------

# 1) CODE-INTENT — (text, label, domain). label 1 = route to the code agent, 0 = prose.
CODE_INTENT: List[Tuple[str, int, str]] = [
    ("implement quicksort", 1, "algorithms"),
    ("write a function that sorts a list", 1, "algorithms"),
    ("generate an implementation of Dijkstra", 1, "algorithms"),
    ("implement binary search", 1, "algorithms"),
    ("benchmark mergesort vs quicksort", 1, "algorithms"),
    ("create a class for a binary tree", 1, "algorithms"),
    ("make a function to reverse a string", 1, "string/data"),
    ("count word frequencies in a sentence in python", 1, "string/data"),
    ("parse this CSV and print the columns", 1, "string/data"),
    ("write code to compute factorial of 10", 1, "math"),
    ("python code for the FFT", 1, "numeric"),
    ("give me RTF-MVDR python code", 1, "numeric"),
    ("write python to FFT a chirp", 1, "numeric"),
    ("show MVDR working on synthetic signals", 1, "numeric"),
    ("compute the eigenvalues of this matrix", 1, "numeric"),
    ("give me code to price a European option with Black-Scholes", 1, "finance"),
    ("price a European option", 1, "finance"),
    ("simulate a damped pendulum", 1, "simulation"),
    ("model SIR epidemic spread", 1, "simulation"),
    ("estimate pi with monte carlo", 1, "simulation"),
    ("simulate rolling two dice and report the average", 1, "simulation"),
    ("implement a logistic regression from scratch", 1, "ml"),
    ("build a neural network in numpy", 1, "ml"),
    ("show me a script that scrapes a page", 1, "web"),
    ("write a python script to download a file", 1, "web"),
    ("refactor this loop", 1, "general"),
    ("debug this segfault in my program", 1, "general"),
    ("optimize this function for speed", 1, "general"),
    ("What is MVDR beamforming?", 0, "—"),
    ("Explain the Black-Scholes model", 0, "—"),
    ("How does transformer attention work?", 0, "—"),
    ("Compare Raft and Paxos", 0, "—"),
    ("Summarize the latest diffusion-model papers", 0, "—"),
    ("What does the function of the hippocampus involve?", 0, "—"),
    ("Give me the latest papers on speech enhancement", 0, "—"),
    ("Why is the sky blue?", 0, "—"),
    ("Define entropy in information theory", 0, "—"),
    ("What are the pros and cons of microservices?", 0, "—"),
    ("Describe how RAG works", 0, "—"),
    ("Who invented the FFT?", 0, "—"),
    ("Is quicksort stable?", 0, "—"),
    ("What is the time complexity of mergesort?", 0, "—"),
    ("When was Python created?", 0, "—"),
    ("What is gradient descent?", 0, "—"),
    ("Explain how a hash table works", 0, "—"),
    ("What is the difference between TCP and UDP?", 0, "—"),
    ("How do neural networks learn?", 0, "—"),
    ("Recommend papers on echo cancellation", 0, "—"),
]

# 2) TASK-TYPE — code tasks only, 3 classes.
TASK_TYPE: List[Tuple[str, str]] = [
    ("implement quicksort", "deterministic"),
    ("reverse a string in place", "deterministic"),
    ("merge two sorted lists", "deterministic"),
    ("parse a CSV into records", "deterministic"),
    ("implement Dijkstra shortest path", "deterministic"),
    ("count word frequencies in a sentence", "deterministic"),
    ("implement binary search", "deterministic"),
    ("compute factorial of n", "deterministic"),
    ("validate that a list is sorted", "deterministic"),
    ("group records by a key", "deterministic"),
    ("compute the FFT of a signal", "numeric_algorithm"),
    ("price a European call with Black-Scholes", "numeric_algorithm"),
    ("compute eigenvalues of a matrix", "numeric_algorithm"),
    ("gradient descent to minimize a function", "numeric_algorithm"),
    ("design an MVDR beamformer", "numeric_algorithm"),
    ("apply a convolution filter to an image", "numeric_algorithm"),
    ("invert a matrix", "numeric_algorithm"),
    ("interpolate a curve through points", "numeric_algorithm"),
    ("compute the DFT of a sequence", "numeric_algorithm"),
    ("solve a linear system Ax = b", "numeric_algorithm"),        # known heuristic miss
    ("simulate a damped pendulum", "simulation"),
    ("estimate pi with a monte carlo simulation", "simulation"),
    ("model SIR epidemic spread", "simulation"),
    ("a random walk on a 2D grid", "simulation"),
    ("n-body gravity simulation", "simulation"),
    ("simulate rolling two dice 10000 times", "simulation"),
    ("a stochastic SIS model", "simulation"),
    ("particle diffusion simulation", "simulation"),
]

# 3) QUERY-SANITY — label 1 = legitimate, 0 = gibberish/too-short.
SANITY: List[Tuple[str, int]] = [
    ("What is MVDR beamforming?", 1),
    ("Explain the Black-Scholes model", 1),
    ("How does RAG improve answers?", 1),
    ("implement quicksort in python", 1),
    ("compute the FFT of a chirp signal", 1),
    ("Compare Raft and Paxos and when to use each", 1),
    ("What is 6*7?", 1),
    ("How do I price a European option?", 1),
    ("How do I implement an LRU cache?", 1),
    ("What's the difference between BM25 and TF-IDF?", 1),
    ("Explain reciprocal rank fusion", 1),
    ("compute eigenvalues of a 3x3 matrix", 1),
    ("asdf qwer zxcv", 0),
    ("buoh", 0),
    ("aaaaaaaa", 0),
    ("lolololol", 0),
    ("?????", 0),
    ("x", 0),
    ("   ", 0),
    ("zxcvbnm", 0),
    ("qwertyuiop", 0),
    ("hjkl hjkl hjkl", 0),
    ("....", 0),
    ("zzzzzz", 0),
]

# 4) ANSWER-REUSE SAFETY — unsafe_to_reuse(a, b): label 1 = UNSAFE (must NOT reuse), 0 = SAFE.
REUSE: List[Tuple[str, str, int]] = [
    ("A100 vs H100 performance", "A100 vs V100 performance", 1),
    ("advantages of TCP", "disadvantages of TCP", 1),
    ("encrypt a file in python", "decrypt a file in python", 1),
    ("upsample an audio signal", "downsample an audio signal", 1),
    ("Raft vs Paxos", "Paxos vs Raft", 1),
    ("convert km to miles", "convert miles to km", 1),
    ("benefits of microservices", "drawbacks of microservices", 1),
    ("what is GPT", "what is BERT", 1),
    ("sort ascending", "sort descending", 1),
    ("python 2 vs python 3", "python 3 vs python 4", 1),
    ("increase the learning rate", "decrease the learning rate", 1),
    ("compress a file", "decompress a file", 1),
    ("synchronous IO", "asynchronous IO", 1),
    ("How does MVDR reduce noise?", "How does MVDR beamforming reduce noise?", 0),
    ("Summarize the Raft consensus paper", "Give a summary of the Raft consensus paper", 0),
    ("What is gradient descent?", "Explain gradient descent", 0),
    ("Explain reinforcement learning", "Explain reinforcement learning to me", 0),
    ("Explain the transformer architecture", "Explain the transformer architecture in detail", 0),
    ("What is overfitting?", "Define overfitting", 0),
    ("How does a hash map work?", "How do hash maps work?", 0),
    ("Best way to parse JSON in Python", "How to parse JSON in Python", 0),
]

# 5) FRESHNESS — label 1 = time-sensitive (bypass cache), 0 = stable.
FRESHNESS: List[Tuple[str, int]] = [
    ("latest papers on speech enhancement", 1),
    ("what's new in Python 3.13", 1),
    ("current state of the art in ASR", 1),
    ("recent advances in RAG", 1),
    ("newest GPU benchmarks", 1),
    ("what happened at NeurIPS 2024", 1),
    ("today's top ML news", 1),
    ("state-of-the-art object detection", 1),
    ("up-to-date FastAPI tutorial", 1),
    ("What is MVDR beamforming?", 0),
    ("Explain quicksort", 0),
    ("How does TCP work?", 0),
    ("Define entropy", 0),
    ("Compare Raft and Paxos", 0),
    ("implement binary search", 0),
    ("How does gradient descent converge?", 0),
    ("What is reciprocal rank fusion?", 0),
    ("Explain the attention mechanism", 0),
]


# ----------------------------------------------------------------------
# Metrics (no third-party deps)
# ----------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion — better than normal approx on small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def _binary(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    def fbeta(b: float) -> float:
        b2 = b * b
        den = b2 * prec + rec
        return (1 + b2) * prec * rec / den if den else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    po, pe = acc, ((((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return {
        "support": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "prevalence": (tp + fn) / n if n else 0.0,
        "accuracy": acc, "balanced_accuracy": (rec + spec) / 2,
        "precision": prec, "recall_sensitivity_tpr": rec, "specificity_tnr": spec,
        "npv": npv, "f1": fbeta(1.0), "f0.5": fbeta(0.5), "f2": fbeta(2.0),
        "fpr": fpr, "fnr": fnr, "mcc": mcc, "cohen_kappa": kappa,
        "ci_accuracy": _wilson(tp + tn, n),
        "ci_precision": _wilson(tp, tp + fp),
        "ci_recall": _wilson(tp, tp + fn),
    }


def _run_binary(fn: Callable[[str], bool], data: List[Tuple]):
    tp = fp = tn = fn_ = 0
    misses = []
    for row in data:
        text, label = row[0], row[1]
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


def _run_multiclass(fn, data, classes):
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
    per, correct, total = {}, 0, 0
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


def _domain_recall():
    """Per-domain recall on the code-intent positives (where the regex layer wins / loses)."""
    by: Dict[str, List[int]] = {}
    for text, label, dom in CODE_INTENT:
        if label != 1:
            continue
        by.setdefault(dom, []).append(1 if is_code_intent(text) else 0)
    return {d: (sum(v), len(v)) for d, v in sorted(by.items())}


def _measure_semantic():
    """LIVE: regex ∪ LLM router. Returns dict with union metrics + how many predictions the LLM
    changed vs regex + which regex-misses it recovered. None if the provider is unavailable."""
    from backend.llm.streaming_provider import get_provider
    prev = os.environ.get("CODE_INTENT_SEMANTIC")
    os.environ["CODE_INTENT_SEMANTIC"] = "true"
    try:
        import backend.answering.task_classifier as tc
        tc.clear_cache()
        if not get_provider().is_available:
            return None
        tp = fp = tn = fn_ = 0
        changed, recovered, new_fp = 0, [], []
        for text, label, _dom in CODE_INTENT:
            regex_pred = 1 if is_code_intent(text) else 0
            if regex_pred == 1:
                pred = 1                       # union = regex OR LLM: a regex hit needs no LLM call
            else:
                try:
                    pred = 1 if tc.is_code_task(text) else 0
                except Exception:
                    pred = regex_pred
            if pred != regex_pred:
                changed += 1
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
                if regex_pred == 0:
                    new_fp.append(text)
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn_ += 1
            if label == 1 and regex_pred == 0 and pred == 1:
                recovered.append(text)
        return {"m": _binary(tp, fp, tn, fn_), "changed": changed,
                "recovered": recovered, "new_fp": new_fp}
    except Exception:
        return None
    finally:
        if prev is None:
            os.environ.pop("CODE_INTENT_SEMANTIC", None)
        else:
            os.environ["CODE_INTENT_SEMANTIC"] = prev


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ci(t: Tuple[float, float]) -> str:
    return f"[{t[0] * 100:.1f}–{t[1] * 100:.1f}%]"


def _binary_section(title: str, fn_name: str, m: Dict[str, float], misses) -> str:
    cm = (
        "|  | **Pred: Positive** | **Pred: Negative** |\n"
        "|---|:---:|:---:|\n"
        f"| **Actual: Positive** | TP = {m['tp']} | FN = {m['fn']} |\n"
        f"| **Actual: Negative** | FP = {m['fp']} | TN = {m['tn']} |\n"
    )
    rows = [
        ("Support (N)", str(m["support"])), ("Prevalence (positives)", _pct(m["prevalence"])),
        ("**Accuracy**", f"{_pct(m['accuracy'])}  95% CI {_ci(m['ci_accuracy'])}"),
        ("Balanced accuracy", _pct(m["balanced_accuracy"])),
        ("**Precision** (PPV)", f"{_pct(m['precision'])}  95% CI {_ci(m['ci_precision'])}"),
        ("**Recall** (Sensitivity / TPR)", f"{_pct(m['recall_sensitivity_tpr'])}  95% CI {_ci(m['ci_recall'])}"),
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
            f"**Confusion matrix**\n\n{cm}\n**Metrics** (95% CI = Wilson score interval)\n\n{met}\n\n"
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


def build_report(semantic: bool = False) -> str:
    ci_m, ci_miss = _run_binary(is_code_intent, CODE_INTENT)
    sn_m, sn_miss = _run_binary(lambda q: check_query_sanity(q).ok, SANITY)
    rz_m, rz_miss = _run_binary(lambda pair: unsafe_to_reuse(*pair),
                                [((a, b), lbl) for a, b, lbl in REUSE])
    fr_m, fr_miss = _run_binary(_freshness, FRESHNESS)
    tt_classes = ["deterministic", "numeric_algorithm", "simulation"]
    tt = _run_multiclass(infer_task_type, TASK_TYPE, tt_classes)
    sem = _measure_semantic() if semantic else None

    summary = (
        "| Classifier | Type | N | Accuracy | Precision | Recall | F1 | MCC |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| Code-intent router (regex) | binary | {ci_m['support']} | {_pct(ci_m['accuracy'])} | "
        f"{_pct(ci_m['precision'])} | {_pct(ci_m['recall_sensitivity_tpr'])} | {_pct(ci_m['f1'])} | {ci_m['mcc']:.3f} |\n"
    )
    if sem:
        s = sem["m"]
        summary += (f"| Code-intent router (regex ∪ LLM) | binary | {s['support']} | {_pct(s['accuracy'])} | "
                    f"{_pct(s['precision'])} | {_pct(s['recall_sensitivity_tpr'])} | {_pct(s['f1'])} | {s['mcc']:.3f} |\n")
    summary += (
        f"| Task-type classifier | 3-class | {len(TASK_TYPE)} | {_pct(tt[2])} | "
        f"{_pct(tt[4]['precision'])} | {_pct(tt[4]['recall'])} | {_pct(tt[4]['f1'])} | — |\n"
        f"| Query-sanity gate | binary | {sn_m['support']} | {_pct(sn_m['accuracy'])} | {_pct(sn_m['precision'])} | "
        f"{_pct(sn_m['recall_sensitivity_tpr'])} | {_pct(sn_m['f1'])} | {sn_m['mcc']:.3f} |\n"
        f"| Answer-reuse safety | binary | {rz_m['support']} | {_pct(rz_m['accuracy'])} | {_pct(rz_m['precision'])} | "
        f"{_pct(rz_m['recall_sensitivity_tpr'])} | {_pct(rz_m['f1'])} | {rz_m['mcc']:.3f} |\n"
        f"| Freshness detector | binary | {fr_m['support']} | {_pct(fr_m['accuracy'])} | {_pct(fr_m['precision'])} | "
        f"{_pct(fr_m['recall_sensitivity_tpr'])} | {_pct(fr_m['f1'])} | {fr_m['mcc']:.3f} |\n"
    )

    # Per-domain recall (code-intent positives, regex layer)
    dom = _domain_recall()
    dom_tbl = "| domain | recall | caught / total |\n|---|---|---|\n" + "\n".join(
        f"| {d} | {_pct(k / n)} | {k}/{n} |" for d, (k, n) in dom.items())

    # Semantic comparison block
    if sem is None:
        sem_block = ("_The live semantic router was not run (pass the default — `--semantic` — with a "
                     "reachable LLM provider). The regex baseline above is always-on regardless._")
    else:
        s = sem["m"]
        rec_lift = s["recall_sensitivity_tpr"] - ci_m["recall_sensitivity_tpr"]
        recovered = "\n".join(f"- `{t}`" for t in sem["recovered"]) or "- _none_"
        newfp = "\n".join(f"- `{t}`" for t in sem["new_fp"]) or "- _none_ ✅ (precision preserved)"
        note = ("" if sem["changed"] else
                "\n\n> ⚠️ The LLM changed **0** predictions at measurement time (likely throttled / "
                "rate-limited), so the union collapsed to the regex baseline. Re-run when the provider "
                "has fresh quota to see the lift.")
        sem_block = (
            f"Run live on the same {ci_m['support']} examples. The router uses **regex ∪ LLM** "
            f"(a regex hit is always positive; the LLM can add positives the regex missed).\n\n"
            f"| | regex only | regex ∪ LLM |\n|---|---|---|\n"
            f"| Accuracy | {_pct(ci_m['accuracy'])} | {_pct(s['accuracy'])} |\n"
            f"| Precision | {_pct(ci_m['precision'])} | {_pct(s['precision'])} |\n"
            f"| Recall | {_pct(ci_m['recall_sensitivity_tpr'])} | {_pct(s['recall_sensitivity_tpr'])} |\n"
            f"| F1 | {_pct(ci_m['f1'])} | {_pct(s['f1'])} |\n"
            f"| MCC | {ci_m['mcc']:.3f} | {s['mcc']:.3f} |\n\n"
            f"**Recall lift from the LLM: {rec_lift * 100:+.1f} pts** · predictions changed vs regex: "
            f"**{sem['changed']}**{note}\n\n"
            f"**Regex-misses recovered by the LLM ({len(sem['recovered'])})**\n\n{recovered}\n\n"
            f"**New false positives introduced by the LLM**\n\n{newfp}\n")

    return "\n".join([
        "# 📐 Classifier Measurement Report",
        "",
        "> **Auto-generated** by `python -m backend.evaluation.measure_classifiers` — every number "
        "is computed by running the real code on a labeled set, not hand-written. Deterministic and "
        "reproducible; the semantic section is a live LLM run captured as-is.",
        "",
        "## What is measured (and what isn't)",
        "",
        "This is a Retrieval-Augmented-Generation assistant: the final *answer* quality is the LLM's "
        "and is graded separately (faithfulness / relevancy / retrieval recall — see "
        "[RAG_BASELINE.md](RAG_BASELINE.md) and `backend/evaluation/evaluate_*`). What this report "
        "measures are the **deterministic decision classifiers** that route and guard every request "
        "— the parts that have a ground truth and a confusion matrix.",
        "",
        "The deterministic layer (regex/heuristics) is measured offline with no LLM/network, so it is "
        "exact and reproducible. The **code-intent router** is *also* measured live as **regex ∪ LLM** "
        "(its production configuration) to quantify the recall lift.",
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
        "| **95% CI** | Wilson score interval — the plausible range given the sample size. |",
        "| **FPR / FNR** | False-positive / false-negative rate. |",
        "| **Prevalence / Support** | Share of positives / number of examples. |",
        "",
        "---",
        "",
        "## 1. Code-intent router — *does this question need the code agent?*",
        "",
        _binary_section("Code-intent router (regex layer)",
                        "backend/answering/code_intent.is_code_intent", ci_m, ci_miss),
        "**Per-domain recall** (code positives only, regex layer):\n\n" + dom_tbl + "\n",
        "#### Production router: regex ∪ LLM (live)\n",
        sem_block,
        "",
        "> 🟢 **Read:** the regex layer has very high **precision** (it rarely sends a prose question to "
        "the agent) but lower **recall** on differently-phrased tasks — so production unions an LLM on "
        "top to recover them while keeping precision high.",
        "",
        "---",
        "",
        "## 2. Task-type classifier — *how should the answer be verified?*",
        "",
        _multiclass_section("Task-type classifier", "backend/answering/task_classifier.infer_task_type",
                            tt_classes, *tt),
        "> 🟢 **Read:** picks the verification strategy (exact-output vs domain-invariants vs simulation "
        "properties). Misses fall back to `deterministic`, the safest default.",
        "",
        "---",
        "",
        "## 3. Query-sanity gate — *is this a real question or gibberish?*",
        "",
        _binary_section("Query-sanity gate", "backend/answering/query_sanity.check_query_sanity",
                        sn_m, sn_miss),
        "> 🟢 **Read:** a cheap first-pass filter (no ML) that blocks keyboard-mash before retrieval + the LLM.",
        "",
        "---",
        "",
        "## 4. Answer-reuse safety — *is it safe to reuse a cached answer?*",
        "",
        _binary_section("Answer-reuse safety", "backend/memory/store.unsafe_to_reuse", rz_m, rz_miss),
        "> 🟢 **Read:** blocks a cache hit when two similar-looking questions actually differ (A↔B swap, "
        "A100↔H100, *with*↔*without*). It errs toward **blocking** (high recall on UNSAFE) — a wrong "
        "reuse is worse than a recompute, so a few conservative over-blocks are by design.",
        "",
        "---",
        "",
        "## 5. Freshness detector — *should this bypass the cache and re-search?*",
        "",
        _binary_section("Freshness detector", "webapp/chat_logic._freshness_sensitive", fr_m, fr_miss),
        "> 🟢 **Read:** routes time-sensitive questions (*latest, today, 2024, state-of-the-art*) around "
        "the cache so they always re-search; errs toward bypassing (a stale 'latest' answer is worse).",
        "",
        "---",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m backend.evaluation.measure_classifiers              # full run (incl. live LLM)",
        "python -m backend.evaluation.measure_classifiers --no-semantic",
        "```",
        "",
        "Labeled sets live in `backend/evaluation/measure_classifiers.py` (curated to include known "
        "edge cases). Extend them and re-run to track the numbers over time.",
        "",
    ])


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out = "docs/MEASUREMENT.md"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    semantic = "--no-semantic" not in argv
    report = build_report(semantic=semantic)
    from pathlib import Path
    dest = Path(__file__).resolve().parents[2] / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
