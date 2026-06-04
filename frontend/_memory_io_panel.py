"""
_memory_io_panel.py  --  AudioLab AI memory export/import UI

Renders inside the sidebar as a "Memory" section with two expanders:
  - Export   : one-click download of the current memory.db bundle
  - Import   : upload a bundle, preview, then merge or replace

Designed to match the v7 brand (white main + navy sidebar + sky-blue accents).
All HTML used in st.markdown is SINGLE-LINE (no </div> bug).
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import streamlit as st

try:
    from . import _theme as T
    from ._components import sidebar_section
except ImportError:
    import _theme as T
    from _components import sidebar_section


# Backend import (works regardless of how chat_ui.py imports things)
try:
    from backend.memory_io import (
        export_memory, import_memory, inspect_export,
        ExportSummary, ImportPlan, EXPORT_SCHEMA_VERSION,
    )
except ImportError:
    # Fallback: backend may not be on path; chat_ui.py adds it
    from memory_io import (
        export_memory, import_memory, inspect_export,
        ExportSummary, ImportPlan, EXPORT_SCHEMA_VERSION,
    )


def render_memory_io_panel(project_root: Path) -> None:
    """Render the Memory section in the sidebar."""
    project_root = Path(project_root)
    memory_db = project_root / "data" / "memory.db"

    sidebar_section("Memory")
    _render_export_expander(project_root, memory_db)
    _render_import_expander(project_root, memory_db)


# ----------------------------------------------------------------------
# EXPORT EXPANDER
# ----------------------------------------------------------------------

def _render_export_expander(project_root: Path, memory_db: Path) -> None:
    with st.sidebar.expander("Export memory", expanded=False):
        st.markdown(
            f'<div style="font-size:0.8rem;color:{T.COLOR_SIDEBAR_TEXT_MUTED};'
            f'line-height:1.4;margin-bottom:0.5rem;">'
            f'Download all your conversations and facts as a portable backup. '
            f'Safe to share &mdash; API keys are masked.'
            f'</div>',
            unsafe_allow_html=True,
        )

        include_env = st.checkbox(
            "Include .env (secrets masked)",
            value=True,
            key="memio_inc_env",
            help="Include your .env config file. Secret values (API_KEY, TOKEN, "
                 "PASSWORD) will be replaced with <MASKED> before export.",
        )
        include_reports = st.checkbox(
            "Include eval reports",
            value=True,
            key="memio_inc_reports",
            help="Include text files from data/reports/",
        )

        if st.button("Build export", key="memio_build_export",
                     use_container_width=True):
            if not memory_db.exists():
                st.error("No memory.db found. Nothing to export.")
                return

            try:
                # Build bundle in-memory so we can offer a direct download
                # (no leftover files cluttering data/exports)
                ts = time.strftime("%Y%m%d_%H%M%S")
                temp_path = project_root / "data" / "exports" / f"audiolab_memory_{ts}.tar.gz"
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                with st.spinner("Building bundle..."):
                    summary = export_memory(
                        memory_db_path=memory_db,
                        output_path=temp_path,
                        project_root=project_root,
                        include_env=include_env,
                        include_reports=include_reports,
                        mask_secrets=True,
                    )

                # Store in session_state so download button persists across reruns
                st.session_state["memio_export_path"] = str(temp_path)
                st.session_state["memio_export_summary"] = summary

            except FileNotFoundError as exc:
                st.error(f"Memory DB missing: {exc}")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

        # Persistent summary + download button (across reruns)
        if "memio_export_path" in st.session_state:
            path = Path(st.session_state["memio_export_path"])
            summary: ExportSummary = st.session_state.get("memio_export_summary")
            if path.exists():
                size_kb = path.stat().st_size / 1024.0
                _render_export_summary_pill(summary, size_kb)

                with open(path, "rb") as f:
                    st.download_button(
                        label="Download .tar.gz",
                        data=f.read(),
                        file_name=path.name,
                        mime="application/gzip",
                        key="memio_dl",
                        use_container_width=True,
                    )

                st.caption(
                    f"Saved to: data/exports/{path.name}"
                )


def _render_export_summary_pill(summary, size_kb: float) -> None:
    """Compact summary card after a successful export."""
    if summary is None:
        return
    html = (
        f'<div style="background:{T.COLOR_SIDEBAR_BG_ELEV};'
        f'border:1px solid {T.COLOR_SUCCESS_BORDER};'
        f'border-left:3px solid {T.COLOR_SUCCESS};'
        f'border-radius:{T.RADIUS_MD};padding:{T.SPACE_SM} {T.SPACE_MD};'
        f'margin:{T.SPACE_SM} 0;font-size:0.78rem;'
        f'color:{T.COLOR_SIDEBAR_TEXT};line-height:1.5;">'
        f'<div style="font-weight:600;color:{T.COLOR_SUCCESS};'
        f'margin-bottom:4px;letter-spacing:0.04em;'
        f'text-transform:uppercase;font-size:0.7rem;">'
        f'Export ready &middot; {size_kb:.1f} KB</div>'
        f'<div>{summary.n_sessions} sessions &middot; '
        f'{summary.n_turns} turns &middot; '
        f'{summary.n_facts_global + summary.n_facts_session} facts</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# IMPORT EXPANDER
# ----------------------------------------------------------------------

def _render_import_expander(project_root: Path, memory_db: Path) -> None:
    with st.sidebar.expander("Import memory", expanded=False):
        st.markdown(
            f'<div style="font-size:0.8rem;color:{T.COLOR_SIDEBAR_TEXT_MUTED};'
            f'line-height:1.4;margin-bottom:0.5rem;">'
            f'Restore from a previous export. Your current memory is '
            f'auto-backed up before any changes.'
            f'</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Bundle (.tar.gz)",
            type=["gz", "tar"],
            key="memio_upload",
            label_visibility="collapsed",
        )

        if uploaded is None:
            return

        # Persist the uploaded bytes to a temp file so the import functions
        # (which expect a path) can read it
        temp_dir = project_root / "data" / "exports"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"_uploaded_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Inspect (validates magic + checksums)
        try:
            manifest = inspect_export(temp_path)
        except ValueError as exc:
            st.error(f"Bundle invalid: {exc}")
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        except Exception as exc:
            st.error(f"Could not read bundle: {exc}")
            return

        _render_bundle_preview(manifest)

        # Import options
        mode = st.radio(
            "Mode",
            options=["merge", "replace"],
            index=0,
            key="memio_mode",
            horizontal=True,
            help="MERGE: keep existing data + add new entries (skip duplicate IDs). "
                 "REPLACE: wipe all existing memory first, then load the bundle.",
        )

        c1, c2 = st.columns(2)
        with c1:
            include_sessions = st.checkbox(
                "Sessions + turns", value=True, key="memio_inc_sess"
            )
        with c2:
            include_facts = st.checkbox(
                "Facts", value=True, key="memio_inc_facts"
            )

        # Dry-run button
        if st.button("Preview changes", key="memio_dryrun",
                     use_container_width=True):
            try:
                plan = import_memory(
                    bundle_path=temp_path,
                    memory_db_path=memory_db,
                    mode=mode,
                    include_sessions=include_sessions,
                    include_facts=include_facts,
                    dry_run=True,
                )
                st.session_state["memio_plan"] = plan
            except Exception as exc:
                st.error(f"Dry-run failed: {exc}")

        # Show last plan if present
        plan = st.session_state.get("memio_plan")
        if plan is not None:
            _render_import_plan(plan, mode)

            # Real import button -- requires a confirmation phrase
            st.markdown(
                f'<div style="font-size:0.75rem;color:{T.COLOR_SIDEBAR_TEXT_MUTED};'
                f'margin:0.5rem 0 0.25rem 0;">'
                f'Type <code>IMPORT</code> to confirm:</div>',
                unsafe_allow_html=True,
            )
            confirm = st.text_input(
                "Confirm", key="memio_confirm",
                label_visibility="collapsed",
                placeholder="IMPORT",
            )

            if st.button("Apply import", key="memio_apply",
                         use_container_width=True, type="primary",
                         disabled=(confirm != "IMPORT")):
                try:
                    with st.spinner("Importing..."):
                        real_plan = import_memory(
                            bundle_path=temp_path,
                            memory_db_path=memory_db,
                            mode=mode,
                            include_sessions=include_sessions,
                            include_facts=include_facts,
                            dry_run=False,
                        )
                    st.success(
                        f"Import complete. "
                        f"{real_plan.new_sessions} new sessions, "
                        f"{real_plan.new_turns} new turns, "
                        f"{real_plan.new_global_facts + real_plan.new_session_facts} new facts."
                    )
                    if real_plan.backup_path:
                        bk = Path(real_plan.backup_path).name
                        st.caption(f"Pre-import backup: data/backups/{bk}")
                    # Clear state so the user can do another import
                    for k in ("memio_plan", "memio_upload", "memio_confirm"):
                        st.session_state.pop(k, None)
                    # Clean up temp
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                    st.info("Refresh the page to see your imported sessions.")
                except Exception as exc:
                    st.error(f"Import failed: {exc}")


def _render_bundle_preview(manifest: dict) -> None:
    """Compact summary of what's inside an uploaded bundle."""
    n_sess = manifest.get("n_sessions", 0)
    n_turns = manifest.get("n_turns", 0)
    n_facts = manifest.get("n_facts_global", 0) + manifest.get("n_facts_session", 0)
    exported_at = manifest.get("exported_at", "unknown")
    schema_v = manifest.get("schema_version", "?")

    html = (
        f'<div style="background:{T.COLOR_SIDEBAR_BG_ELEV};'
        f'border:1px solid {T.COLOR_SIDEBAR_BORDER};'
        f'border-left:3px solid {T.COLOR_ACCENT};'
        f'border-radius:{T.RADIUS_MD};padding:{T.SPACE_SM} {T.SPACE_MD};'
        f'margin:{T.SPACE_SM} 0;font-size:0.78rem;'
        f'color:{T.COLOR_SIDEBAR_TEXT};line-height:1.5;">'
        f'<div style="font-weight:600;color:{T.COLOR_ACCENT};'
        f'margin-bottom:4px;letter-spacing:0.04em;'
        f'text-transform:uppercase;font-size:0.7rem;">'
        f'Bundle valid &middot; schema v{schema_v}</div>'
        f'<div>{n_sess} sessions &middot; {n_turns} turns &middot; {n_facts} facts</div>'
        f'<div style="color:{T.COLOR_SIDEBAR_TEXT_MUTED};margin-top:2px;">'
        f'Exported: {exported_at}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_import_plan(plan, mode: str) -> None:
    """Compact summary of what an import would do (from dry-run)."""
    total_new = (plan.new_sessions + plan.new_turns
                 + plan.new_global_facts + plan.new_session_facts)
    color = T.COLOR_WARNING if mode == "replace" else T.COLOR_ACCENT

    conflicts_line = ""
    if plan.conflicts:
        conflicts_line = (
            f'<div style="color:{T.COLOR_SIDEBAR_TEXT_MUTED};margin-top:2px;">'
            f'Conflicts kept: {len(plan.conflicts)}</div>'
        )

    html = (
        f'<div style="background:{T.COLOR_SIDEBAR_BG_ELEV};'
        f'border:1px solid {T.COLOR_SIDEBAR_BORDER};'
        f'border-left:3px solid {color};'
        f'border-radius:{T.RADIUS_MD};padding:{T.SPACE_SM} {T.SPACE_MD};'
        f'margin:{T.SPACE_SM} 0;font-size:0.78rem;'
        f'color:{T.COLOR_SIDEBAR_TEXT};line-height:1.5;">'
        f'<div style="font-weight:600;color:{color};'
        f'margin-bottom:4px;letter-spacing:0.04em;'
        f'text-transform:uppercase;font-size:0.7rem;">'
        f'Preview &middot; {mode.upper()}</div>'
        f'<div>+ {plan.new_sessions} sessions, {plan.new_turns} turns</div>'
        f'<div>+ {plan.new_global_facts + plan.new_session_facts} facts</div>'
        f'{conflicts_line}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
