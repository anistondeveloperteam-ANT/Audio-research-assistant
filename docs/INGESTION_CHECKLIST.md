# Ingestion Checklist — add PDFs that *broaden coverage*

The corpus is currently tiny (a few PDFs / ~64 chunks), which is the main reason broad
questions get weak recall. The goal when adding PDFs is **wide, balanced coverage** across
audio, speech, signal processing, ML, evaluation, datasets, and research methods — **not**
randomly inflating the chunk count with more papers on the same few topics.

## Before you add

1. **See what's missing.** Run the coverage report:
   ```bash
   .venv\Scripts\python.exe pipeline.py --corpus-report
   ```
   Open `data/extracted/corpus_coverage_report.md` and look at:
   - **Missing topics** (`:x:`) — broad domains with **0** papers. Fill these first.
   - **Under-represented topics** — domains with only 1 paper. Add a second, different one.
   - **Duplicate titles** — don't re-add papers already indexed.

2. **Pick diverse, high-value PDFs.** Prefer survey/overview papers and a few strong primary
   papers per domain (one survey covers more ground than five narrow papers on one method).
   Spread across the domains in the report, not just DOA/MVDR/beamforming.

3. **Prefer text-based PDFs.** Scanned/image-only PDFs fall back to OCR (`ENABLE_OCR`), which
   is slower and lower quality. A born-digital PDF chunks far better.

## Add + index

4. Drop the PDFs into **`data/papers/`** (one file per paper; a clear filename helps the
   auto-title).
5. Build the index (incremental only processes new/changed PDFs):
   ```bash
   .venv\Scripts\python.exe pipeline.py --incremental
   ```
   (Oracle must be running — e.g. the `oracle-ai-db` Docker container, `FREEPDB1:1521`.)

## Verify it improved coverage (not just count)

6. **Check ingestion succeeded** — no paper should land with 0 chunks:
   ```bash
   .venv\Scripts\python.exe pipeline.py --status          # totals
   .venv\Scripts\python.exe pipeline.py --corpus-report    # gaps + per-paper chunk counts
   ```
   If a paper shows **0 chunks** (listed under "Failed ingestions"), the PDF likely didn't
   parse — try a text-based version or enable OCR.
7. **Spot-check chunk quality** for a new paper:
   ```bash
   .venv\Scripts\python.exe pipeline.py --inspect-chunks <paper_id>
   ```
   Chunks should be coherent passages with sensible sections — not headers/footers or garbage.
8. **Measure recall** on the broad eval set (and add domain questions to
   `data/evaluation_questions.json` for topics you now cover):
   ```bash
   .venv\Scripts\python.exe -m backend.evaluation.evaluate_retrieval --top-k 8 --quiet
   ```
   Compare `term_recall` and the weak-question list in
   `data/extracted/retrieval_eval_report.md` before vs after.

## Rules of thumb

- **Coverage > count.** 10 PDFs spanning 10 domains beat 50 PDFs on 2 domains.
- **Fill `Missing topics` first**, then strengthen `Under-represented` ones.
- **No duplicates.** Re-indexing the same paper adds chunks without adding coverage.
- **Re-run `--corpus-report` after every batch** so additions are deliberate.
