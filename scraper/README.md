# PDF scraper

Download PDFs from a list of **direct PDF URLs** into a review folder, separate
from the indexing pipeline. Workflow:

```
scrape  ->  review downloads/  ->  move the good ones into data/papers/  ->  index
```

## Usage

From the project root (use the project venv, e.g. `.\.venv\Scripts\python.exe`):

```bash
# URLs straight on the command line
python -m scraper.scrape_pdfs https://example.com/a.pdf https://example.com/b.pdf

# or a file with one URL per line ('#' comments and blank lines are ignored)
python -m scraper.scrape_pdfs --urls-file scraper/urls.txt

# re-runnable: skip URLs whose target file already exists
python -m scraper.scrape_pdfs -f scraper/urls.txt --skip-existing
```

Downloads land in `scraper/downloads/` by default (override with `--out DIR`).

### Options

| Flag | Meaning |
|------|---------|
| `-f`, `--urls-file PATH` | Read URLs from a text file (one per line). |
| `-o`, `--out DIR` | Output folder (default `scraper/downloads/`). |
| `--timeout SECONDS` | Per-request timeout (default 30). |
| `--overwrite` | Replace a same-named file instead of writing a unique `name (1).pdf`. |
| `--skip-existing` | Skip a URL whose target name already exists (idempotent re-runs). |
| `--url-names` | Name files from the URL/`Content-Disposition` instead of the PDF's title. |

## What it guards against

- **Not-a-PDF responses:** the body must contain the `%PDF` header, so HTML
  "404 / login" pages served with a `200` are rejected, not saved.
- **Proper filenames:** the saved name is taken from the PDF's own embedded title
  (the descriptive "proper name"), falling back to `Content-Disposition`, then the
  URL. So an arXiv link like `.../pdf/2505.03442` is saved as
  `Knowledge Distillation for Speech Denoising.pdf`, not `2505.03442.pdf`. All names
  are sanitized to be safe on Windows. Pass `--url-names` to keep the URL/id name.
- **Clobbering:** an existing file is never overwritten unless you pass
  `--overwrite`; otherwise a unique `name (1).pdf` is used.
- **One bad URL aborting the batch:** failures are reported per URL; the rest
  still download. The process exits non-zero if anything failed.

## Add the downloads to the search index

After reviewing, move the PDFs you want into `data/papers/` and run:

```bash
python pipeline.py --incremental
```

## Notes

- Each file is buffered in memory before being written (fine for typical papers;
  not intended for multi-GB files).
- This tool only downloads **direct** PDF links. It does not crawl pages or
  search — give it URLs that point straight at a `.pdf`.
