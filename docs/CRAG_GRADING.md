# CRAG Evidence-Grader Measurement Report

> **Auto-generated** by `python -m backend.evaluation.measure_evidence_grader` — every number is computed by running `grade_evidence` on a labeled set. The grader is pure and deterministic (reads reranker scores only), so this is exact and reproducible.

**Thresholds in effect:** STRONG = >= 2 chunks at score >= 0.55; PARTIAL = any chunk >= 0.3; else NONE.

**Overall accuracy (micro-F1): 83.3%** over 12 labeled cases.

## Confusion matrix (rows = actual action, cols = predicted)

| actual \ pred | STRONG | PARTIAL | NONE | support |
|---|:---:|:---:|:---:|:---:|
| **STRONG** | 3 | 1 | 0 | 4 |
| **PARTIAL** | 1 | 4 | 0 | 5 |
| **NONE** | 0 | 0 | 3 | 3 |

## Per-class metrics

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| STRONG | 75.0% | 75.0%  95% CI [30.1-95.4%] | 75.0% | 4 |
| PARTIAL | 80.0% | 80.0%  95% CI [37.6-96.4%] | 80.0% | 5 |
| NONE | 100.0% | 100.0%  95% CI [43.8-100.0%] | 100.0% | 3 |
| **macro avg** | 85.0% | 85.0% | 85.0% | - |
| **weighted avg** | 83.3% | 83.3% | 83.3% | - |

## External-skip rate

A **STRONG** grade answers from the library and skips the web search entirely (the adaptive win). Over the labeled set:

| metric | value |
|---|---|
| External searches skipped (STRONG) | 4/12 (33.3%) |
| Skip precision (STRONG that was truly STRONG) | 75.0%  95% CI [30.1-95.4%] |

Skip precision < 100% means some skips were over-confident (answered from the PDFs when a web check was warranted) — the lever is `CRAG_STRONG_MIN` / `CRAG_STRONG_COUNT`.

## Misclassified (2)

- `three decent sub-bar chunks` -- actual **STRONG**, predicted **PARTIAL**
- `two chunks barely over the bar` -- actual **PARTIAL**, predicted **STRONG**
