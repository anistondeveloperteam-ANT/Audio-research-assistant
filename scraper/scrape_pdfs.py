#!/usr/bin/env python
"""scrape_pdfs.py -- download PDFs from a list of direct URLs into a review folder.

You give it direct ``.pdf`` URLs (on the command line and/or in a text file) and
it downloads each one into ``scraper/downloads/`` (override with ``--out``). Each
response is validated to actually be a PDF (``%PDF`` header) before it is saved,
so HTML "not found" pages served with a 200 are rejected instead of polluting
the folder. Filenames come from ``Content-Disposition`` when present, otherwise
from the URL, and are sanitized so they are safe on Windows.

Usage:
    # one or more URLs on the command line
    python -m scraper.scrape_pdfs https://example.com/paper.pdf https://example.com/other.pdf

    # or a text file, one URL per line ('#' comments and blank lines ignored)
    python -m scraper.scrape_pdfs --urls-file scraper/urls.txt

    # re-runnable: skip URLs whose target file already exists
    python -m scraper.scrape_pdfs -f scraper/urls.txt --skip-existing

After reviewing the downloads, move the ones you want into ``data/papers/`` and
run ``python pipeline.py --incremental`` to add them to the search index.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import unquote, urlparse

import requests

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "downloads"
DEFAULT_TIMEOUT = 30.0
_CHUNK_SIZE = 64 * 1024
_MAX_NAME_LEN = 200
# A plain, honest User-Agent. Some servers reject the default requests UA.
_USER_AGENT = "Mozilla/5.0 (compatible; audio-research-assistant pdf-scraper/1.0)"
# PDF files must contain the "%PDF" marker within the first bytes (per the PDF spec
# it appears in the first 1024 bytes), so checking the head is a reliable type guard.
_PDF_MAGIC = b"%PDF"
_PDF_MAGIC_WINDOW = 1024
_MIN_TITLE_LEN = 4  # shorter "titles" are almost always junk metadata
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")

_session: requests.Session | None = None


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of attempting to download one URL.

    status is one of: "ok" (saved), "skipped" (already present), "failed" (error).
    """

    url: str
    status: str
    path: str | None = None
    message: str = ""
    num_bytes: int = 0


# ----------------------------------------------------------------------
# Pure helpers (no network)
# ----------------------------------------------------------------------
def read_urls(path: Path) -> list[str]:
    """Read URLs from a text file: one per line, ignoring blanks and '#' comments.

    Duplicate URLs are removed while preserving first-seen order.
    """
    urls: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return list(dict.fromkeys(urls))


def sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in Windows filenames and trim length."""
    name = _ILLEGAL_CHARS.sub("_", name)
    name = name.replace("\r", "_").replace("\n", "_")
    name = name.strip().strip(".")  # Windows dislikes trailing dots/spaces
    return name[:_MAX_NAME_LEN]


def filename_from_url(url: str) -> str:
    """Derive a safe ``*.pdf`` filename from a URL's path (or host as a fallback)."""
    parsed = urlparse(url)
    raw = os.path.basename(unquote(parsed.path)).strip()
    name = sanitize_filename(raw) or sanitize_filename(parsed.netloc) or "download"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _filename_from_disposition(disposition: str) -> str:
    """Extract a filename from a Content-Disposition header value (RFC 6266/5987)."""
    if not disposition:
        return ""
    # filename*=UTF-8''percent%20encoded.pdf  -- preferred when present
    star = re.search(r"filename\*\s*=\s*[^']*''([^;]+)", disposition, re.IGNORECASE)
    if star:
        return unquote(star.group(1).strip())
    quoted = re.search(r'filename\s*=\s*"([^"]+)"', disposition, re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()
    bare = re.search(r"filename\s*=\s*([^;]+)", disposition, re.IGNORECASE)
    if bare:
        return bare.group(1).strip().strip('"')
    return ""


def _ensure_pdf_ext(name: str) -> str:
    """Append '.pdf' unless the name already ends with it (case-insensitive)."""
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def _clean_title(title: str) -> str:
    """Normalize a PDF metadata title; return '' if it is empty or obviously junk."""
    title = _WHITESPACE.sub(" ", title or "").strip()
    if len(title) < _MIN_TITLE_LEN or not re.search(r"[A-Za-z]", title):
        return ""
    return title


def _pdf_title_pymupdf(body: bytes) -> str:
    """Read the /Title from PDF metadata via PyMuPDF; '' if unavailable."""
    try:
        import fitz  # PyMuPDF
    except Exception:  # noqa: BLE001 -- parser is optional; degrade gracefully
        return ""
    try:
        with fitz.open(stream=body, filetype="pdf") as doc:
            return (doc.metadata or {}).get("title", "") or ""
    except Exception:  # noqa: BLE001 -- broken/unreadable PDF
        return ""


def _pdf_title_pypdf(body: bytes) -> str:
    """Fallback: read the /Title from PDF metadata via pypdf; '' if unavailable."""
    try:
        import io

        from pypdf import PdfReader
    except Exception:  # noqa: BLE001
        return ""
    try:
        meta = PdfReader(io.BytesIO(body)).metadata
        return (meta.title if meta and meta.title else "") or ""
    except Exception:  # noqa: BLE001
        return ""


def pdf_title(body: bytes) -> str:
    """Best-effort human-readable title from a PDF's embedded metadata ('' if none)."""
    return _clean_title(_pdf_title_pymupdf(body) or _pdf_title_pypdf(body))


def filename_from_response(
    url: str,
    headers: Mapping[str, str],
    body: bytes | None = None,
    *,
    use_pdf_title: bool = True,
) -> str:
    """Choose a filename for a downloaded PDF.

    Priority: the PDF's own embedded title (the most descriptive "proper name") ->
    the server's Content-Disposition filename -> the URL. The title step is skipped
    when ``use_pdf_title`` is False or no ``body`` is given.
    """
    if use_pdf_title and body:
        title = sanitize_filename(pdf_title(body))
        if title:
            return _ensure_pdf_ext(title)
    disposition = headers.get("Content-Disposition") or headers.get("content-disposition") or ""
    name = sanitize_filename(_filename_from_disposition(disposition))
    if name:
        return _ensure_pdf_ext(name)
    return filename_from_url(url)


def looks_like_pdf(head: bytes) -> bool:
    """True if the leading bytes contain the ``%PDF`` marker (tolerates a BOM/whitespace)."""
    return _PDF_MAGIC in head[:_PDF_MAGIC_WINDOW]


def unique_path(dest_dir: Path, filename: str) -> Path:
    """Return a non-existing path for ``filename`` in ``dest_dir`` (adds ' (n)' if needed)."""
    target = dest_dir / filename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ----------------------------------------------------------------------
# Download (network)
# ----------------------------------------------------------------------
def _default_session() -> requests.Session:
    """Lazily create one shared session so connections are reused across URLs."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def download_pdf(
    url: str,
    dest_dir: Path | str,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    skip_existing: bool = False,
    use_pdf_title: bool = True,
    chunk_size: int = _CHUNK_SIZE,
) -> DownloadResult:
    """Download one PDF URL into ``dest_dir``, validating it is really a PDF.

    Returns a DownloadResult; never raises for expected failures (network errors,
    bad status, non-PDF body) so a single bad URL does not abort a batch.
    """
    dest_dir = Path(dest_dir)
    session = session or _default_session()

    if skip_existing:
        existing = dest_dir / filename_from_url(url)
        if existing.exists():
            return DownloadResult(url, "skipped", path=str(existing), message="already exists")

    try:
        resp = session.get(
            url, stream=True, timeout=timeout, headers={"User-Agent": _USER_AGENT}
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 -- report any request failure, keep going
        return DownloadResult(url, "failed", message=f"request failed: {exc}")

    try:
        body = b"".join(chunk for chunk in resp.iter_content(chunk_size=chunk_size) if chunk)
    except Exception as exc:  # noqa: BLE001 -- mid-stream read error
        return DownloadResult(url, "failed", message=f"download error: {exc}")
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass

    if not looks_like_pdf(body):
        return DownloadResult(url, "failed", message="response is not a PDF (no %PDF header)")

    filename = filename_from_response(
        url, getattr(resp, "headers", {}) or {}, body, use_pdf_title=use_pdf_title
    )
    target = dest_dir / filename
    if target.exists():
        if skip_existing:
            return DownloadResult(url, "skipped", path=str(target), message="already exists")
        if not overwrite:
            target = unique_path(dest_dir, filename)

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    except OSError as exc:
        return DownloadResult(url, "failed", message=f"write error: {exc}")

    return DownloadResult(url, "ok", path=str(target), num_bytes=len(body))


def scrape(
    urls: list[str],
    dest_dir: Path | str,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    skip_existing: bool = False,
    use_pdf_title: bool = True,
    on_progress: Callable[[int, int, DownloadResult], None] | None = None,
) -> list[DownloadResult]:
    """Download every URL in ``urls`` into ``dest_dir``; return one result per URL."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = session or _default_session()

    results: list[DownloadResult] = []
    total = len(urls)
    for index, url in enumerate(urls, start=1):
        result = download_pdf(
            url,
            dest_dir,
            session=session,
            timeout=timeout,
            overwrite=overwrite,
            skip_existing=skip_existing,
            use_pdf_title=use_pdf_title,
        )
        results.append(result)
        if on_progress is not None:
            on_progress(index, total, result)
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_progress(index: int, total: int, result: DownloadResult) -> None:
    tag = {"ok": "OK  ", "skipped": "SKIP", "failed": "FAIL"}.get(result.status, "??  ")
    print(f"[{index}/{total}] {tag} {result.url}")
    detail = result.path if result.status == "ok" else result.message
    if detail:
        print(f"           -> {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download PDFs from a list of direct URLs into a review folder."
    )
    parser.add_argument("urls", nargs="*", help="Direct PDF URLs (in addition to --urls-file).")
    parser.add_argument(
        "-f", "--urls-file", type=Path,
        help="Text file with one URL per line ('#' comments and blank lines ignored).",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output folder (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace a file if the same name exists (default: write a unique '(n)' name).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a URL whose target name already exists (idempotent re-runs).",
    )
    parser.add_argument(
        "--url-names", action="store_true",
        help="Name files from the URL/Content-Disposition instead of the PDF's title.",
    )
    args = parser.parse_args(argv)

    urls: list[str] = []
    if args.urls_file:
        if not args.urls_file.exists():
            print(f"ERROR: urls file not found: {args.urls_file}", file=sys.stderr)
            return 2
        urls.extend(read_urls(args.urls_file))
    urls.extend(u.strip() for u in args.urls if u.strip())
    urls = list(dict.fromkeys(urls))

    if not urls:
        print("No URLs given. Pass URLs as arguments or use --urls-file PATH.", file=sys.stderr)
        return 2

    print(f"Downloading {len(urls)} PDF(s) -> {args.out}")
    results = scrape(
        urls,
        args.out,
        timeout=args.timeout,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        use_pdf_title=not args.url_names,
        on_progress=_print_progress,
    )

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = [r for r in results if r.status == "failed"]
    print(f"\nDone: {ok} downloaded, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  - {r.url}\n      {r.message}")
    print(f"\nReview the PDFs in {args.out}, then move the ones you want into data/papers/")
    print("and run:  python pipeline.py --incremental")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
