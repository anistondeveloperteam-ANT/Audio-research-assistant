# RAG Improvement — Baseline (recorded 2026-06-12)

Measured on `main` (with this session's uncommitted observability / PHASE-3 / Crawl4AI work in the tree), before any RAG-improvement changes.

## 1. Tests — `pytest -q`
- **183 passed, 3 skipped.** (The 3 skips are the opt-in DeepEval quality gates.)
- The reported failure — `tests/test_rtf_mvdr.py` importing `backend.audio.rtf_mvdr` — is **already resolved**: that WIP test file has been removed and `backend/audio` does not exist. No collection error remains, so "fix test collection" is a no-op (details in the plan).

## 2. Index — `pipeline.py --status`
| metric | value |
|---|---|
| PDFs in `data/papers/` | 3 |
| Oracle | reachable (localhost:1521/FREEPDB1) |
| Indexed papers | 3 |
| **Indexed chunks** | **64** |
| Chunks with vector | 64 |
| turbovec cache | enabled, valid |

→ The corpus is tiny (3 PDFs). This is the **dominant** cause of weak broad-question coverage.

## 3. Retrieval eval — `evaluate_retrieval --top-k 8 --quiet`
- **Overall: WEAK** — 6 of 8 questions weak.
- **Recall:** most questions term_recall ≈ 0.2–0.4 (≈0.3 mean), matching the previously reported ~0.4.
- **Latency: SLOW** — mean **12.26s**, p50 10.0s, p95 19.5s, max 32.5s, total 98s for 8 queries.
- Missing terms are audio-specific (DOA, MVDR, LCMV, GSC, PESQ, STOI, beamforming). With only 3 PDFs the corpus **cannot** cover them, so low recall here is mostly a **corpus-size** problem, not purely retrieval logic.

## Key takeaways → drives the plan
1. **Test problem already fixed** (file removed) — no `backend.audio` needed; implementing an audio-specific MVDR module would also contradict the "broad coverage, not audio-only" goal.
2. **Retrieval is slow (~12s/query)** → profile + parallelize the independent stages (vector / HyDE / BM25 / graph) in `hybrid_retrieve.py`; keep rerank + MMR correct.
3. **Recall is low mostly because the corpus is tiny (64 chunks)** → broaden the library + add a coverage report + ingestion checklist (adding *relevant, diverse* PDFs, not random chunk-count inflation).
4. **Eval questions are narrow audio topics** → expand the eval set toward broad audio/speech/ML/eval/datasets/architectures coverage.
