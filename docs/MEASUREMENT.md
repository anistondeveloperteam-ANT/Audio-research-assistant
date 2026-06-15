# 📐 Classifier Measurement Report

> **Auto-generated** by `python -m backend.evaluation.measure_classifiers` — every number below is computed by running the real code on a labeled set, not hand-written. Regenerate any time; results are deterministic and fully offline.

## What is measured (and what isn't)

This is a Retrieval-Augmented-Generation assistant: the final *answer* quality is the LLM's, and is graded separately (faithfulness / relevancy / retrieval recall — see [RAG_BASELINE.md](RAG_BASELINE.md) and `backend/evaluation/evaluate_*`). What this report measures are the **deterministic decision classifiers** that route and guard every request — the parts of the system that *do* have a ground truth and a confusion matrix.

Measured on the **deterministic layer** (`CODE_INTENT_SEMANTIC=false`) so the numbers are exact and reproducible with no LLM/network. In production a semantic LLM classifier is **unioned on top** of the code-intent regex to raise recall (it recovers the phrasings marked as misses below); that path needs a live model and isn't part of this offline run.

## Summary

| Classifier | Type | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Code-intent router | binary | 81.8% | 100.0% | 68.4% | 81.3% | 0.692 |
| Task-type classifier | 3-class | 94.7% | 95.5% | 94.7% | 94.7% | — |
| Query-sanity gate | binary | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| Answer-reuse safety | binary | 78.6% | 72.7% | 100.0% | 84.2% | 0.603 |

## Metric glossary

| Term | Meaning |
|---|---|
| **Confusion matrix** | Counts of TP / FP / TN / FN (predicted vs actual). |
| **Accuracy** | (TP+TN)/N — overall correctness. |
| **Precision (PPV)** | TP/(TP+FP) — of things flagged positive, how many really are. |
| **Recall (Sensitivity, TPR)** | TP/(TP+FN) — of real positives, how many were caught. |
| **Specificity (TNR)** | TN/(TN+FP) — of real negatives, how many were left alone. |
| **F1 / F0.5 / F2** | Harmonic mean of P&R (F0.5 favors precision, F2 favors recall). |
| **Balanced accuracy** | (Recall+Specificity)/2 — fair when classes are imbalanced. |
| **MCC** | Matthews correlation (−1..1); 1 = perfect, 0 = random — robust on imbalance. |
| **Cohen's κ** | Agreement above chance (0..1). |
| **FPR / FNR** | False-positive / false-negative rate. |
| **Prevalence / Support** | Share of positives / number of examples. |

---

## 1. Code-intent router — *does this question need the code agent?*

### Code-intent router

`backend/answering/code_intent.is_code_intent` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 13 | FN = 6 |
| **Actual: Negative** | FP = 0 | TN = 14 |

**Metrics**

| Metric | Value |
|---|---|
| Support (N) | 33 |
| Prevalence (positives) | 57.6% |
| **Accuracy** | 81.8% |
| Balanced accuracy | 84.2% |
| **Precision** (PPV) | 100.0% |
| **Recall** (Sensitivity / TPR) | 68.4% |
| Specificity (TNR) | 100.0% |
| NPV | 70.0% |
| **F1** | 81.3% |
| F0.5 | 91.5% |
| F2 | 73.0% |
| FPR (fall-out) | 0.0% |
| FNR (miss rate) | 31.6% |
| **MCC** (Matthews, −1..1) | 0.692 |
| Cohen's κ (0..1) | 0.648 |

**Misclassified (6)**

- `write python to FFT a chirp` — actual **pos**, predicted **neg**
- `model SIR epidemic spread` — actual **pos**, predicted **neg**
- `price a European option` — actual **pos**, predicted **neg**
- `show MVDR working on synthetic signals` — actual **pos**, predicted **neg**
- `estimate pi with monte carlo` — actual **pos**, predicted **neg**
- `create a class for a binary tree` — actual **pos**, predicted **neg**

> 🟢 **Read:** high **precision** (it rarely sends a prose question to the agent) with lower **recall** — the deterministic regex misses differently-phrased code tasks (*"model SIR…"*, *"price a European option"*). Those exact misses are why the semantic LLM layer is unioned on top in production: it catches them while precision stays high.

---

## 2. Task-type classifier — *how should the answer be verified?*

### Task-type classifier

`backend/answering/task_classifier.infer_task_type` · 3-class classifier

**Confusion matrix** (rows = actual, cols = predicted)

| actual ╲ pred | deterministic | numeric_algorithm | simulation | support |
|---|:---:|:---:|:---:|:---:|
| **deterministic** | 6 | 0 | 0 | 6 |
| **numeric_algorithm** | 1 | 6 | 0 | 7 |
| **simulation** | 0 | 0 | 6 | 6 |

**Overall accuracy (micro-F1): 94.7%**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| deterministic | 85.7% | 100.0% | 92.3% | 6 |
| numeric_algorithm | 100.0% | 85.7% | 92.3% | 7 |
| simulation | 100.0% | 100.0% | 100.0% | 6 |
| **macro avg** | 95.2% | 95.2% | 94.9% | — |
| **weighted avg** | 95.5% | 94.7% | 94.7% | — |

**Misclassified (1)**

- `solve a linear system Ax = b` — actual **numeric_algorithm**, predicted **deterministic**

> 🟢 **Read:** picks the verification strategy (exact-output vs domain-invariants vs simulation properties). Misses are conservative — an unrecognized phrasing falls back to `deterministic`, the safest default.

---

## 3. Query-sanity gate — *is this a real question or gibberish?*

### Query-sanity gate

`backend/answering/query_sanity.check_query_sanity` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 8 | FN = 0 |
| **Actual: Negative** | FP = 0 | TN = 8 |

**Metrics**

| Metric | Value |
|---|---|
| Support (N) | 16 |
| Prevalence (positives) | 50.0% |
| **Accuracy** | 100.0% |
| Balanced accuracy | 100.0% |
| **Precision** (PPV) | 100.0% |
| **Recall** (Sensitivity / TPR) | 100.0% |
| Specificity (TNR) | 100.0% |
| NPV | 100.0% |
| **F1** | 100.0% |
| F0.5 | 100.0% |
| F2 | 100.0% |
| FPR (fall-out) | 0.0% |
| FNR (miss rate) | 0.0% |
| **MCC** (Matthews, −1..1) | 1.000 |
| Cohen's κ (0..1) | 1.000 |

**Misclassified (0)**

- _none_ ✅

> 🟢 **Read:** a cheap first-pass filter (no ML) that blocks keyboard-mash before it reaches retrieval + the LLM. Tuned to favor letting real questions through (high recall on legit).

---

## 4. Answer-reuse safety — *is it safe to reuse a cached answer?*

### Answer-reuse safety

`backend/memory/store.unsafe_to_reuse` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 8 | FN = 0 |
| **Actual: Negative** | FP = 3 | TN = 3 |

**Metrics**

| Metric | Value |
|---|---|
| Support (N) | 14 |
| Prevalence (positives) | 57.1% |
| **Accuracy** | 78.6% |
| Balanced accuracy | 75.0% |
| **Precision** (PPV) | 72.7% |
| **Recall** (Sensitivity / TPR) | 100.0% |
| Specificity (TNR) | 50.0% |
| NPV | 100.0% |
| **F1** | 84.2% |
| F0.5 | 76.9% |
| F2 | 93.0% |
| FPR (fall-out) | 50.0% |
| FNR (miss rate) | 0.0% |
| **MCC** (Matthews, −1..1) | 0.603 |
| Cohen's κ (0..1) | 0.533 |

**Misclassified (3)**

- `('Summarize the Raft consensus paper', 'Give a summary of the Raft consensus paper')` — actual **neg**, predicted **pos**
- `('Best way to parse JSON in Python', 'How to parse JSON in Python')` — actual **neg**, predicted **pos**
- `('How does the FFT work?', 'How does the fast Fourier transform work?')` — actual **neg**, predicted **pos**

> 🟢 **Read:** the guard that blocks a cache hit when two similar-looking questions actually differ (an A↔B swap, A100↔H100, *with*↔*without*). It errs toward **blocking** (high recall on UNSAFE) — a wrong reuse is worse than a recompute.

---

## Reproduce

```bash
python -m backend.evaluation.measure_classifiers      # regenerates this file
```

Labeled sets live in `backend/evaluation/measure_classifiers.py` (curated to include known edge cases). Extend them and re-run to track the numbers over time.
