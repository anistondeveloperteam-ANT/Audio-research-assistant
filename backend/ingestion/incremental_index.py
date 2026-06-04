
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACTED_DIR = DATA_DIR / "extracted"

MANIFEST_FILE = EXTRACTED_DIR / "incremental_index_manifest.json"
INDEX_STATE_FILE = EXTRACTED_DIR / "index_state.json"
LAST_INDEX_LOG = EXTRACTED_DIR / "last_index_update.log"

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def scan_library() -> Dict[str, Dict[str, Any]]:
    library: Dict[str, Dict[str, Any]] = {}
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        stat = pdf.stat()
        library[pdf.name] = {
            "name": pdf.name,
            "path": str(pdf.relative_to(ROOT)),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "sha256": sha256_file(pdf),
        }
    return library


def library_signature(library: Dict[str, Dict[str, Any]]) -> str:
    payload = [f"{name}:{item.get('size')}:{item.get('sha256')}" for name, item in sorted(library.items())]
    return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()


def diff_library(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, List[str]]:
    old_files = old.get("files", {}) if isinstance(old, dict) else {}
    added, changed, deleted = [], [], []

    for name, info in new.items():
        previous = old_files.get(name)
        if not previous:
            added.append(name)
        elif previous.get("sha256") != info.get("sha256"):
            changed.append(name)

    for name in old_files:
        if name not in new:
            deleted.append(name)

    return {"added": added, "changed": changed, "deleted": deleted}


def run_cmd(args: List[str], timeout: int = 7200) -> Tuple[bool, str]:
    started = time.time()
    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, shell=False)
    elapsed = round(time.time() - started, 2)
    output = f"$ {' '.join(args)}\nTime: {elapsed} sec\nExit code: {result.returncode}\n\n{result.stdout or ''}"
    if result.stderr:
        output += "\n[STDERR]\n" + result.stderr
    return result.returncode == 0, output


def run_full_pipeline() -> Tuple[bool, str, str | None]:
    logs: List[str] = []
    steps = [
        ("Reading and structuring papers", [sys.executable, "-m", "backend.ingestion.ingest_papers"], 3600),
        ("Building search memory", [sys.executable, "-m", "backend.ingestion.embed_chunks"], 7200),
        ("Updating vector search", [sys.executable, "-m", "backend.database.vector_migration"], 3600),
    ]

    for label, cmd, timeout in steps:
        ok, out = run_cmd(cmd, timeout=timeout)
        logs.append("\n" + "=" * 100)
        logs.append(label)
        logs.append("=" * 100)
        logs.append(out)
        if not ok:
            return False, "\n".join(logs), label
    return True, "\n".join(logs), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Run indexing even if PDFs are unchanged.")
    parser.add_argument("--status", action="store_true", help="Show incremental index status only.")
    args = parser.parse_args()

    started = time.time()
    os.environ["PARSER_MODE"] = "auto"
    os.environ["ENABLE_DOCLING"] = "true"
    os.environ["ENABLE_MARKER"] = "false"
    os.environ["ENABLE_OCR"] = "true"

    old_manifest = read_json(MANIFEST_FILE, {"files": {}})
    current = scan_library()
    diff = diff_library(old_manifest, current)
    sig = library_signature(current)

    if args.status:
        print("PDF count:", len(current))
        print("Added:", diff["added"])
        print("Changed:", diff["changed"])
        print("Deleted:", diff["deleted"])
        print("Signature:", sig)
        return

    logs = [
        "Incremental indexing controller v1",
        f"Time: {now_iso()}",
        f"PDF count: {len(current)}",
        f"Added: {diff['added']}",
        f"Changed: {diff['changed']}",
        f"Deleted: {diff['deleted']}",
        f"Force: {args.force}",
    ]

    if not current:
        state = {
            "last_index_time": now_iso(),
            "ok": False,
            "skipped": True,
            "reason": "No PDFs found in data/papers",
            "pdf_count": 0,
            "library_signature": sig,
            "seconds": round(time.time() - started, 2),
        }
        write_json(INDEX_STATE_FILE, state)
        LAST_INDEX_LOG.write_text("\n".join(logs), encoding="utf-8")
        print("No PDFs found in data/papers.")
        raise SystemExit(1)

    has_changes = bool(diff["added"] or diff["changed"] or diff["deleted"])

    if not has_changes and not args.force:
        logs.append("")
        logs.append("No new/changed/deleted PDFs detected. Skipping parse/embed/vector update.")
        logs.append("Fast path complete.")
        state = {
            "last_index_time": old_manifest.get("last_index_time") or now_iso(),
            "last_checked_time": now_iso(),
            "ok": True,
            "skipped": True,
            "reason": "No library changes detected",
            "pdf_count": len(current),
            "library_signature": sig,
            "seconds": round(time.time() - started, 2),
            "added": [],
            "changed": [],
            "deleted": [],
            "log_path": str(LAST_INDEX_LOG),
        }
        write_json(INDEX_STATE_FILE, state)
        LAST_INDEX_LOG.write_text("\n".join(logs), encoding="utf-8")
        print("FAST PATH: no PDF changes. Index already up to date.")
        print("PDF count:", len(current))
        print("Time:", state["seconds"], "sec")
        return

    ok, pipeline_log, failed_step = run_full_pipeline()
    logs.append("")
    logs.append(pipeline_log)
    elapsed = round(time.time() - started, 2)

    state = {
        "last_index_time": now_iso(),
        "ok": ok,
        "skipped": False,
        "failed_step": failed_step,
        "seconds": elapsed,
        "pdf_count": len(current),
        "library_signature": sig,
        "added": diff["added"],
        "changed": diff["changed"],
        "deleted": diff["deleted"],
        "log_path": str(LAST_INDEX_LOG),
    }

    if ok:
        manifest = {"last_index_time": now_iso(), "library_signature": sig, "files": current}
        write_json(MANIFEST_FILE, manifest)

    write_json(INDEX_STATE_FILE, state)
    LAST_INDEX_LOG.write_text("\n".join(logs), encoding="utf-8")

    if ok:
        print("INCREMENTAL INDEX COMPLETE")
        print("PDF count:", len(current))
        print("Added:", diff["added"])
        print("Changed:", diff["changed"])
        print("Deleted:", diff["deleted"])
        print("Time:", elapsed, "sec")
    else:
        print("INCREMENTAL INDEX FAILED")
        print("Failed step:", failed_step)
        print("Log:", LAST_INDEX_LOG)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
