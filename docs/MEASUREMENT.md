# 📐 Classifier Measurement Report

> **Auto-generated** by `python -m backend.evaluation.measure_classifiers` — every number is computed by running the real code on a labeled set, not hand-written. Deterministic and reproducible; the semantic section is a live LLM run captured as-is.

## What is measured (and what isn't)

This is a Retrieval-Augmented-Generation assistant: the final *answer* quality is the LLM's and is graded separately (faithfulness / relevancy / retrieval recall — see [RAG_BASELINE.md](RAG_BASELINE.md) and `backend/evaluation/evaluate_*`). What this report measures are the **deterministic decision classifiers** that route and guard every request — the parts that have a ground truth and a confusion matrix.

The deterministic layer (regex/heuristics) is measured offline with no LLM/network, so it is exact and reproducible. The **code-intent router** is *also* measured live as **regex ∪ LLM** (its production configuration) to quantify the recall lift.

## Summary

| Classifier | Type | N | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|
| Code-intent router (regex) | binary | 48 | 79.2% | 100.0% | 64.3% | 78.3% | 0.655 |
| Code-intent router (regex ∪ LLM) | binary | 48 | 87.5% | 100.0% | 78.6% | 88.0% | 0.777 |
| Task-type classifier | 3-class | 28 | 96.4% | 96.8% | 96.4% | 96.4% | — |
| Query-sanity gate | binary | 24 | 95.8% | 100.0% | 91.7% | 95.7% | 0.920 |
| Answer-reuse safety | binary | 21 | 85.7% | 81.2% | 100.0% | 89.7% | 0.713 |
| Freshness detector | binary | 18 | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |

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
| **95% CI** | Wilson score interval — the plausible range given the sample size. |
| **FPR / FNR** | False-positive / false-negative rate. |
| **Prevalence / Support** | Share of positives / number of examples. |

---

## 1. Code-intent router — *does this question need the code agent?*

### Code-intent router (regex layer)

`backend/answering/code_intent.is_code_intent` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 18 | FN = 10 |
| **Actual: Negative** | FP = 0 | TN = 20 |

**Metrics** (95% CI = Wilson score interval)

| Metric | Value |
|---|---|
| Support (N) | 48 |
| Prevalence (positives) | 58.3% |
| **Accuracy** | 79.2%  95% CI [65.7–88.3%] |
| Balanced accuracy | 82.1% |
| **Precision** (PPV) | 100.0%  95% CI [82.4–100.0%] |
| **Recall** (Sensitivity / TPR) | 64.3%  95% CI [45.8–79.3%] |
| Specificity (TNR) | 100.0% |
| NPV | 66.7% |
| **F1** | 78.3% |
| F0.5 | 90.0% |
| F2 | 69.2% |
| FPR (fall-out) | 0.0% |
| FNR (miss rate) | 35.7% |
| **MCC** (Matthews, −1..1) | 0.655 |
| Cohen's κ (0..1) | 0.600 |

**Misclassified (10)**

- `create a class for a binary tree` — actual **pos**, predicted **neg**
- `count word frequencies in a sentence in python` — actual **pos**, predicted **neg**
- `parse this CSV and print the columns` — actual **pos**, predicted **neg**
- `write python to FFT a chirp` — actual **pos**, predicted **neg**
- `show MVDR working on synthetic signals` — actual **pos**, predicted **neg**
- `compute the eigenvalues of this matrix` — actual **pos**, predicted **neg**
- `price a European option` — actual **pos**, predicted **neg**
- `model SIR epidemic spread` — actual **pos**, predicted **neg**
- `estimate pi with monte carlo` — actual **pos**, predicted **neg**
- `build a neural network in numpy` — actual **pos**, predicted **neg**

**Per-domain recall** (code positives only, regex layer):

| domain | recall | caught / total |
|---|---|---|
| algorithms | 83.3% | 5/6 |
| finance | 50.0% | 1/2 |
| general | 100.0% | 3/3 |
| math | 100.0% | 1/1 |
| ml | 50.0% | 1/2 |
| numeric | 40.0% | 2/5 |
| simulation | 50.0% | 2/4 |
| string/data | 33.3% | 1/3 |
| web | 100.0% | 2/2 |

#### Production router: regex ∪ LLM (live)

Run live on the same 48 examples. The router uses **regex ∪ LLM** (a regex hit is always positive; the LLM can add positives the regex missed).

| | regex only | regex ∪ LLM |
|---|---|---|
| Accuracy | 79.2% | 87.5% |
| Precision | 100.0% | 100.0% |
| Recall | 64.3% | 78.6% |
| F1 | 78.3% | 88.0% |
| MCC | 0.655 | 0.777 |

**Recall lift from the LLM: +14.3 pts** · predictions changed vs regex: **4**

**Regex-misses recovered by the LLM (4)**

- `create a class for a binary tree`
- `count word frequencies in a sentence in python`
- `parse this CSV and print the columns`
- `price a European option`

**New false positives introduced by the LLM**

- _none_ ✅ (precision preserved)


> 🟢 **Read:** the regex layer has very high **precision** (it rarely sends a prose question to the agent) but lower **recall** on differently-phrased tasks — so production unions an LLM on top to recover them while keeping precision high.

---

## 2. Task-type classifier — *how should the answer be verified?*

### Task-type classifier

`backend/answering/task_classifier.infer_task_type` · 3-class classifier

**Confusion matrix** (rows = actual, cols = predicted)

| actual ╲ pred | deterministic | numeric_algorithm | simulation | support |
|---|:---:|:---:|:---:|:---:|
| **deterministic** | 10 | 0 | 0 | 10 |
| **numeric_algorithm** | 1 | 9 | 0 | 10 |
| **simulation** | 0 | 0 | 8 | 8 |

**Overall accuracy (micro-F1): 96.4%**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| deterministic | 90.9% | 100.0% | 95.2% | 10 |
| numeric_algorithm | 100.0% | 90.0% | 94.7% | 10 |
| simulation | 100.0% | 100.0% | 100.0% | 8 |
| **macro avg** | 97.0% | 96.7% | 96.7% | — |
| **weighted avg** | 96.8% | 96.4% | 96.4% | — |

**Misclassified (1)**

- `solve a linear system Ax = b` — actual **numeric_algorithm**, predicted **deterministic**

> 🟢 **Read:** picks the verification strategy (exact-output vs domain-invariants vs simulation properties). Misses fall back to `deterministic`, the safest default.

---

## 3. Query-sanity gate — *is this a real question or gibberish?*

### Query-sanity gate

`backend/answering/query_sanity.check_query_sanity` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 11 | FN = 1 |
| **Actual: Negative** | FP = 0 | TN = 12 |

**Metrics** (95% CI = Wilson score interval)

| Metric | Value |
|---|---|
| Support (N) | 24 |
| Prevalence (positives) | 50.0% |
| **Accuracy** | 95.8%  95% CI [79.8–99.3%] |
| Balanced accuracy | 95.8% |
| **Precision** (PPV) | 100.0%  95% CI [74.1–100.0%] |
| **Recall** (Sensitivity / TPR) | 91.7%  95% CI [64.6–98.5%] |
| Specificity (TNR) | 100.0% |
| NPV | 92.3% |
| **F1** | 95.7% |
| F0.5 | 98.2% |
| F2 | 93.2% |
| FPR (fall-out) | 0.0% |
| FNR (miss rate) | 8.3% |
| **MCC** (Matthews, −1..1) | 0.920 |
| Cohen's κ (0..1) | 0.917 |

**Misclassified (1)**

- `Explain reciprocal rank fusion` — actual **pos**, predicted **neg**

> 🟢 **Read:** a cheap first-pass filter (no ML) that blocks keyboard-mash before retrieval + the LLM.

---

## 4. Answer-reuse safety — *is it safe to reuse a cached answer?*

### Answer-reuse safety

`backend/memory/store.unsafe_to_reuse` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 13 | FN = 0 |
| **Actual: Negative** | FP = 3 | TN = 5 |

**Metrics** (95% CI = Wilson score interval)

| Metric | Value |
|---|---|
| Support (N) | 21 |
| Prevalence (positives) | 61.9% |
| **Accuracy** | 85.7%  95% CI [65.4–95.0%] |
| Balanced accuracy | 81.2% |
| **Precision** (PPV) | 81.2%  95% CI [57.0–93.4%] |
| **Recall** (Sensitivity / TPR) | 100.0%  95% CI [77.2–100.0%] |
| Specificity (TNR) | 62.5% |
| NPV | 100.0% |
| **F1** | 89.7% |
| F0.5 | 84.4% |
| F2 | 95.6% |
| FPR (fall-out) | 37.5% |
| FNR (miss rate) | 0.0% |
| **MCC** (Matthews, −1..1) | 0.713 |
| Cohen's κ (0..1) | 0.674 |

**Misclassified (3)**

- `('Summarize the Raft consensus paper', 'Give a summary of the Raft consensus paper')` — actual **neg**, predicted **pos**
- `('How does a hash map work?', 'How do hash maps work?')` — actual **neg**, predicted **pos**
- `('Best way to parse JSON in Python', 'How to parse JSON in Python')` — actual **neg**, predicted **pos**

> 🟢 **Read:** blocks a cache hit when two similar-looking questions actually differ (A↔B swap, A100↔H100, *with*↔*without*). It errs toward **blocking** (high recall on UNSAFE) — a wrong reuse is worse than a recompute, so a few conservative over-blocks are by design.

---

## 5. Freshness detector — *should this bypass the cache and re-search?*

### Freshness detector

`webapp/chat_logic._freshness_sensitive` · binary classifier

**Confusion matrix**

|  | **Pred: Positive** | **Pred: Negative** |
|---|:---:|:---:|
| **Actual: Positive** | TP = 9 | FN = 0 |
| **Actual: Negative** | FP = 0 | TN = 9 |

**Metrics** (95% CI = Wilson score interval)

| Metric | Value |
|---|---|
| Support (N) | 18 |
| Prevalence (positives) | 50.0% |
| **Accuracy** | 100.0%  95% CI [82.4–100.0%] |
| Balanced accuracy | 100.0% |
| **Precision** (PPV) | 100.0%  95% CI [70.1–100.0%] |
| **Recall** (Sensitivity / TPR) | 100.0%  95% CI [70.1–100.0%] |
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

> 🟢 **Read:** routes time-sensitive questions (*latest, today, 2024, state-of-the-art*) around the cache so they always re-search; errs toward bypassing (a stale 'latest' answer is worse).

---

## Reproduce

```bash
python -m backend.evaluation.measure_classifiers              # full run (incl. live LLM)
python -m backend.evaluation.measure_classifiers --no-semantic
```

Labeled sets live in `backend/evaluation/measure_classifiers.py` (curated to include known edge cases). Extend them and re-run to track the numbers over time.
