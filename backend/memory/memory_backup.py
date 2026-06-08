"""
memory_backup.py  --  AudioLab AI memory export/import

A portable, defensive backup system for the memory.db SQLite database
plus optional auxiliary files (.env config, eval reports).

Design principles:
  - HUMAN-READABLE: exports to JSON, not opaque SQLite blob
  - VERSIONED: every export carries a schema version stamp
  - CHECKSUMED: SHA256 on every file -- import detects corruption
  - SAFE: secrets in .env are masked before export
  - REVERSIBLE: import auto-backs up the existing memory.db first
  - SELECTIVE: import can choose which sections to restore
  - COMPRESSED: gzip the JSON so files stay small

Export bundle layout (.tar.gz):
  manifest.json              # schema_version, timestamps, checksums
  memory/
      sessions.json
      turns.json
      facts.json
      stats.json             # for sanity-check during import
  config/
      env.masked.txt         # .env with secrets replaced by <MASKED>
  reports/
      *.txt                  # optional, all eval reports
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sqlite3
import tarfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = 1
EXPORT_FILE_MAGIC = "audiolab-ai-memory-export"

# Pattern: SECRET_KEY=value or API_KEY=value etc. anything that looks
# like a secret gets masked.
SECRET_KEY_PATTERN = re.compile(
    r"^(\w*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)\w*)\s*=\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------

@dataclass
class ExportSummary:
    """What's inside an export. Returned by export_memory() and
    by inspect_export() before an import."""
    schema_version: int = EXPORT_SCHEMA_VERSION
    audiolab_version: str = "Phase 2 -- Batch 15"
    exported_at: str = ""
    exported_at_unix: float = 0.0
    n_sessions: int = 0
    n_turns: int = 0
    n_facts_global: int = 0
    n_facts_session: int = 0
    includes_env: bool = False
    includes_reports: bool = False
    n_report_files: int = 0
    checksums: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ImportPlan:
    """Returned by import_memory(dry_run=True). Tells you what would
    happen without actually doing it."""
    will_replace_sessions: int = 0
    will_keep_sessions: int = 0
    new_sessions: int = 0
    new_turns: int = 0
    new_global_facts: int = 0
    new_session_facts: int = 0
    conflicts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _now_unix() -> float:
    return time.time()


def _mask_env(text: str) -> str:
    """Replace secret-looking values in an .env file with <MASKED>.
    Preserves keys and comments so the file is still usable as a template."""
    def replacer(m):
        key = m.group(1)
        # If the value is empty, leave it; if it's already masked, leave it
        val = m.group(2).strip()
        if not val or val == "<MASKED>":
            return f"{key}={val}"
        return f"{key}=<MASKED>"
    return SECRET_KEY_PATTERN.sub(replacer, text)


def _safe_get_cursor(conn: sqlite3.Connection):
    """Return a cursor with row_factory enabled."""
    conn.row_factory = sqlite3.Row
    return conn.cursor()


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------

def export_memory(
    memory_db_path: Path,
    output_path: Path,
    project_root: Optional[Path] = None,
    include_env: bool = True,
    include_reports: bool = True,
    mask_secrets: bool = True,
) -> ExportSummary:
    """Export memory.db (and optional auxiliary files) to a portable
    .tar.gz bundle at output_path.

    Args:
        memory_db_path : path to data/memory.db
        output_path    : where to write the .tar.gz file
        project_root   : path to the project root (used to find .env, reports)
        include_env    : include .env (masked) in the bundle
        include_reports: include data/reports/*.txt
        mask_secrets   : redact API keys / passwords in .env

    Returns:
        ExportSummary describing what was exported.
    """
    memory_db_path = Path(memory_db_path)
    output_path = Path(output_path)

    if not memory_db_path.exists():
        raise FileNotFoundError(f"memory.db not found at {memory_db_path}")

    if not memory_db_path.is_file():
        raise ValueError(f"{memory_db_path} is not a file")

    summary = ExportSummary()
    summary.exported_at = _now_iso()
    summary.exported_at_unix = _now_unix()
    summary.includes_env = bool(include_env and project_root)
    summary.includes_reports = bool(include_reports and project_root)

    # 1. Open the DB read-only (won't change the source)
    conn = sqlite3.connect(f"file:{memory_db_path}?mode=ro", uri=True)
    try:
        cur = _safe_get_cursor(conn)

        # Dump sessions
        cur.execute("SELECT * FROM sessions ORDER BY created_at ASC")
        sessions = _rows_to_dicts(cur.fetchall())
        summary.n_sessions = len(sessions)

        # Dump turns
        cur.execute(
            "SELECT * FROM turns ORDER BY session_id, turn_index ASC"
        )
        turns = _rows_to_dicts(cur.fetchall())
        summary.n_turns = len(turns)

        # Dump facts
        cur.execute("SELECT * FROM facts ORDER BY scope, session_id, key")
        facts = _rows_to_dicts(cur.fetchall())
        summary.n_facts_global = sum(1 for f in facts if f["scope"] == "global")
        summary.n_facts_session = sum(1 for f in facts if f["scope"] == "session")

        stats = {
            "n_sessions": summary.n_sessions,
            "n_turns": summary.n_turns,
            "n_facts_global": summary.n_facts_global,
            "n_facts_session": summary.n_facts_session,
        }
    finally:
        conn.close()

    # 2. Build payload bytes (each section as JSON bytes -- gzipped later)
    payloads: Dict[str, bytes] = {}

    sessions_json = json.dumps(sessions, indent=2, ensure_ascii=False).encode("utf-8")
    turns_json = json.dumps(turns, indent=2, ensure_ascii=False).encode("utf-8")
    facts_json = json.dumps(facts, indent=2, ensure_ascii=False).encode("utf-8")
    stats_json = json.dumps(stats, indent=2).encode("utf-8")

    payloads["memory/sessions.json"] = sessions_json
    payloads["memory/turns.json"] = turns_json
    payloads["memory/facts.json"] = facts_json
    payloads["memory/stats.json"] = stats_json

    # 3. Optional: .env (masked)
    if include_env and project_root:
        env_path = Path(project_root) / ".env"
        if env_path.exists():
            try:
                env_text = env_path.read_text(encoding="utf-8")
                if mask_secrets:
                    env_text = _mask_env(env_text)
                # Preamble warning
                preamble = (
                    "# AudioLab AI memory export -- .env (MASKED)\n"
                    "# Secret values (API keys, tokens, passwords) have been\n"
                    "# replaced with <MASKED>. Fill them in again after restore.\n"
                    "#\n"
                )
                env_text = preamble + env_text
                payloads["config/env.masked.txt"] = env_text.encode("utf-8")
            except Exception as exc:
                summary.notes += f"\n.env read failed: {exc}"

    # 4. Optional: eval reports
    if include_reports and project_root:
        reports_dir = Path(project_root) / "data" / "reports"
        if reports_dir.exists() and reports_dir.is_dir():
            for report_file in sorted(reports_dir.iterdir()):
                if not report_file.is_file():
                    continue
                # Only include text-like reports, skip anything weird
                if report_file.suffix.lower() not in (".txt", ".md", ".json", ".csv"):
                    continue
                if report_file.stat().st_size > 5 * 1024 * 1024:  # 5MB cap
                    summary.notes += (
                        f"\nSkipped report {report_file.name} (>5MB)"
                    )
                    continue
                try:
                    payloads[f"reports/{report_file.name}"] = report_file.read_bytes()
                    summary.n_report_files += 1
                except Exception as exc:
                    summary.notes += f"\nReport read failed: {exc}"

    # 5. Compute checksums of each payload
    for name, data in payloads.items():
        summary.checksums[name] = _sha256_bytes(data)

    # 6. Build manifest
    manifest = {
        "magic": EXPORT_FILE_MAGIC,
        "schema_version": EXPORT_SCHEMA_VERSION,
        **{k: v for k, v in asdict(summary).items()
           if k != "checksums"},
        "checksums": summary.checksums,
    }
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")

    # 7. Write the tar.gz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, mode="w:gz", compresslevel=6) as tf:
        # Manifest first (so a quick `tar -t` shows it)
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_json)
        info.mtime = int(summary.exported_at_unix)
        tf.addfile(info, io.BytesIO(manifest_json))
        # Each payload
        for name, data in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(summary.exported_at_unix)
            tf.addfile(info, io.BytesIO(data))

    return summary


# ----------------------------------------------------------------------
# INSPECT  (read a bundle without applying it)
# ----------------------------------------------------------------------

def inspect_export(bundle_path: Path) -> Dict[str, Any]:
    """Read the manifest from an export bundle. Verifies magic + checksums.

    Returns the manifest dict. Raises ValueError if the bundle is invalid
    or corrupt.
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    manifest = None
    payloads: Dict[str, bytes] = {}

    try:
        with tarfile.open(bundle_path, mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if member.name == "manifest.json":
                    try:
                        manifest = json.loads(data.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ValueError(
                            f"Bundle manifest is corrupt (invalid JSON): {exc}"
                        ) from None
                else:
                    payloads[member.name] = data
    except tarfile.TarError as exc:
        raise ValueError(
            f"Bundle is not a valid tar.gz archive: {exc}"
        ) from None
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"Bundle cannot be read (file may be corrupt or truncated): {exc}"
        ) from None

    if manifest is None:
        raise ValueError(
            "Bundle is missing manifest.json -- not an AudioLab AI export"
        )
    if manifest.get("magic") != EXPORT_FILE_MAGIC:
        raise ValueError(
            f"Bundle magic mismatch (got {manifest.get('magic')!r}). "
            "Not an AudioLab AI export."
        )
    bundle_version = manifest.get("schema_version", 0)
    if bundle_version > EXPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Bundle schema_version={bundle_version} is newer than this "
            f"installation supports (max {EXPORT_SCHEMA_VERSION}). Upgrade "
            "your Research Assistant installation."
        )

    # Verify checksums
    expected = manifest.get("checksums", {})
    bad = []
    for name, expect in expected.items():
        actual = _sha256_bytes(payloads.get(name, b""))
        if actual != expect:
            bad.append(name)
    if bad:
        raise ValueError(
            "Checksum mismatch in bundle (file corruption?):\n  "
            + "\n  ".join(bad)
        )

    return manifest


# ----------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------

def import_memory(
    bundle_path: Path,
    memory_db_path: Path,
    mode: str = "merge",
    include_sessions: bool = True,
    include_facts: bool = True,
    dry_run: bool = False,
    backups_dir: Optional[Path] = None,
) -> ImportPlan:
    """Import a memory export bundle into the existing memory.db.

    Args:
        bundle_path      : path to the .tar.gz export
        memory_db_path   : path to data/memory.db to import into
        mode             : 'merge' (keep existing + add new, skip dup IDs)
                           or 'replace' (wipe everything first, then load)
        include_sessions : import sessions + turns
        include_facts    : import facts
        dry_run          : if True, return a plan without writing anything
        backups_dir      : where to auto-backup the existing memory.db
                           (defaults to memory_db_path.parent / "backups")

    Returns:
        ImportPlan describing what was/would be done.
    """
    bundle_path = Path(bundle_path)
    memory_db_path = Path(memory_db_path)

    if mode not in ("merge", "replace"):
        raise ValueError(f"mode must be 'merge' or 'replace', got {mode!r}")

    # Validate the bundle first
    inspect_export(bundle_path)

    # Read payloads
    payloads: Dict[str, bytes] = {}
    with tarfile.open(bundle_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or member.name == "manifest.json":
                continue
            f = tf.extractfile(member)
            if f is not None:
                payloads[member.name] = f.read()

    bundle_sessions = json.loads(payloads.get("memory/sessions.json", b"[]"))
    bundle_turns = json.loads(payloads.get("memory/turns.json", b"[]"))
    bundle_facts = json.loads(payloads.get("memory/facts.json", b"[]"))

    plan = ImportPlan()

    # If DB doesn't exist, init it from schema
    if not memory_db_path.exists():
        memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        _init_empty_db(memory_db_path)

    # Auto-backup the existing DB (unless dry_run)
    if not dry_run:
        bk_dir = Path(backups_dir) if backups_dir else memory_db_path.parent / "backups"
        bk_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_file = bk_dir / f"memory_pre_import_{ts}.db"
        shutil.copy2(memory_db_path, backup_file)
        plan.backup_path = str(backup_file)

    conn = sqlite3.connect(str(memory_db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # Read current DB state for the plan
        cur.execute("SELECT id FROM sessions")
        existing_session_ids = {r["id"] for r in cur.fetchall()}

        cur.execute("SELECT scope, session_id, key FROM facts")
        existing_facts = {(r["scope"], r["session_id"], r["key"]) for r in cur.fetchall()}

        # ---- Sessions + turns ----
        if include_sessions:
            if mode == "replace":
                if not dry_run:
                    cur.execute("DELETE FROM turns")
                    cur.execute("DELETE FROM sessions")
                plan.will_replace_sessions = len(existing_session_ids)
                plan.will_keep_sessions = 0
            else:
                # merge: keep existing, skip duplicates
                plan.will_keep_sessions = len(existing_session_ids)

            for s in bundle_sessions:
                if mode == "merge" and s["id"] in existing_session_ids:
                    continue
                plan.new_sessions += 1
                if not dry_run:
                    cur.execute(
                        "INSERT OR REPLACE INTO sessions "
                        "(id, title, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (s["id"], s["title"], s["created_at"], s["updated_at"])
                    )

            # Turns -- always tied to session_id; merge mode skips turns
            # whose session was skipped (already exists)
            for t in bundle_turns:
                if mode == "merge" and t["session_id"] in existing_session_ids:
                    continue
                plan.new_turns += 1
                if not dry_run:
                    cur.execute(
                        "INSERT INTO turns "
                        "(session_id, turn_index, role, content, sources_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (t["session_id"], t["turn_index"], t["role"],
                         t["content"], t.get("sources_json"), t["created_at"])
                    )

        # ---- Facts ----
        if include_facts:
            if mode == "replace":
                if not dry_run:
                    cur.execute("DELETE FROM facts")
                # plan: we wipe all existing facts in replace mode
            for f in bundle_facts:
                key = (f["scope"], f.get("session_id"), f["key"])
                if mode == "merge" and key in existing_facts:
                    plan.conflicts.append(
                        f"Fact already exists, kept original: "
                        f"{f['scope']}/{f.get('session_id', '-')}/{f['key']}"
                    )
                    continue
                if f["scope"] == "global":
                    plan.new_global_facts += 1
                else:
                    plan.new_session_facts += 1
                if not dry_run:
                    cur.execute(
                        "INSERT OR REPLACE INTO facts "
                        "(scope, session_id, key, value, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (f["scope"], f.get("session_id"), f["key"],
                         f["value"], f["created_at"], f["updated_at"])
                    )

        if not dry_run:
            conn.commit()
    except Exception:
        if not dry_run:
            conn.rollback()
        raise
    finally:
        conn.close()

    return plan


def _init_empty_db(memory_db_path: Path) -> None:
    """Create an empty memory.db with the standard schema."""
    conn = sqlite3.connect(str(memory_db_path))
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            TEXT PRIMARY KEY,
            title         TEXT NOT NULL DEFAULT 'New conversation',
            created_at    REAL NOT NULL,
            updated_at    REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS turns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            turn_index    INTEGER NOT NULL,
            role          TEXT NOT NULL,
            content       TEXT NOT NULL,
            sources_json  TEXT,
            created_at    REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session
            ON turns(session_id, turn_index);

        CREATE TABLE IF NOT EXISTS facts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            scope         TEXT NOT NULL CHECK (scope IN ('session', 'global')),
            session_id    TEXT,
            key           TEXT NOT NULL,
            value         TEXT NOT NULL,
            created_at    REAL NOT NULL,
            updated_at    REAL NOT NULL,
            UNIQUE (scope, session_id, key)
        );

        CREATE INDEX IF NOT EXISTS idx_facts_scope_session
            ON facts(scope, session_id);
        """)
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------
# CLI helpers (used by the batch scripts)
# ----------------------------------------------------------------------

def cli_export(project_root: Path, output_path: Optional[Path] = None) -> Path:
    """Command-line export. Returns the actual output path used."""
    project_root = Path(project_root)
    memory_db = project_root / "data" / "memory.db"

    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        export_dir = project_root / "data" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output_path = export_dir / f"audiolab_memory_{ts}.tar.gz"

    summary = export_memory(
        memory_db_path=memory_db,
        output_path=Path(output_path),
        project_root=project_root,
        include_env=True,
        include_reports=True,
        mask_secrets=True,
    )

    size_kb = output_path.stat().st_size / 1024.0
    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"File:                {output_path}")
    print(f"Size:                {size_kb:.1f} KB")
    print(f"Exported at:         {summary.exported_at}")
    print(f"Sessions:            {summary.n_sessions}")
    print(f"Turns:               {summary.n_turns}")
    print(f"Long-term facts:     {summary.n_facts_global}")
    print(f"Session facts:       {summary.n_facts_session}")
    print(f"Includes .env (masked): {summary.includes_env}")
    print(f"Includes reports:    {summary.includes_reports} ({summary.n_report_files} files)")
    if summary.notes:
        print(f"Notes:               {summary.notes.strip()}")
    print()
    return output_path


def cli_import(project_root: Path, bundle_path: Path,
               mode: str = "merge", dry_run: bool = False) -> ImportPlan:
    """Command-line import."""
    project_root = Path(project_root)
    bundle_path = Path(bundle_path)
    memory_db = project_root / "data" / "memory.db"

    print()
    print("=" * 60)
    if dry_run:
        print("IMPORT  --  DRY RUN (no changes will be made)")
    else:
        print(f"IMPORT  --  mode = {mode.upper()}")
    print("=" * 60)
    print(f"Bundle:              {bundle_path}")

    manifest = inspect_export(bundle_path)
    print(f"Exported at:         {manifest.get('exported_at')}")
    print(f"Sessions:            {manifest.get('n_sessions')}")
    print(f"Turns:               {manifest.get('n_turns')}")
    print(f"Long-term facts:     {manifest.get('n_facts_global')}")
    print(f"Session facts:       {manifest.get('n_facts_session')}")
    print()

    plan = import_memory(
        bundle_path=bundle_path,
        memory_db_path=memory_db,
        mode=mode,
        include_sessions=True,
        include_facts=True,
        dry_run=dry_run,
    )

    print("Plan:")
    print(f"  New sessions:      {plan.new_sessions}")
    print(f"  New turns:         {plan.new_turns}")
    print(f"  New global facts:  {plan.new_global_facts}")
    print(f"  New session facts: {plan.new_session_facts}")
    if plan.conflicts:
        print(f"  Conflicts kept:    {len(plan.conflicts)}")
    if plan.backup_path:
        print(f"  Pre-import backup: {plan.backup_path}")
    if dry_run:
        print()
        print("(dry run -- no changes were actually written)")
    print()
    return plan
