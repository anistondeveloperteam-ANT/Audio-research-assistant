
"""
market_ui.py — Market UI (formerly streamlit_app.py)
Audio Research Paper Assistant

User workflow:
1. User uploads PDFs.
2. PDF count updates immediately.
3. App learns automatically ONLY when new/changed PDFs are detected.
4. User asks any question.
5. Previous sources/results clear automatically when the question changes or new PDF is uploaded.
6. Similar-question memory works silently in backend; it is not shown to normal users.

Run:
    python run.py --market          # http://localhost:8501
"""

from __future__ import annotations

import base64

import hashlib
import json
import os
import re
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACTED_DIR = DATA_DIR / "extracted"

LAST_INDEX_LOG = EXTRACTED_DIR / "last_index_update.log"
INDEX_STATE_FILE = EXTRACTED_DIR / "index_state.json"
QUESTION_MEMORY_FILE = EXTRACTED_DIR / "question_memory.json"

for path in [ROOT, BACKEND]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.answering.research_modes import apply_research_mode as apply_mode_profile, get_mode_settings
from backend.answering.prompt_quality import enhance_result_prompt

load_dotenv(ROOT / ".env")

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Audio Research Paper Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: #151923;
}
.block-container {
    max-width: 1280px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.hero {
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
    padding: 1.35rem 1.4rem;
    border-radius: 22px;
    margin-bottom: 1rem;
}
.card {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
    padding: 1rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}
.answer-card {
    border: 1px solid rgba(255,255,255,0.10);
    background: #111827;
    padding: 1.15rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}
.source-card {
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.035);
    padding: 0.95rem;
    border-radius: 16px;
    margin-bottom: 0.8rem;
}
.small-muted {
    color: #9ca3af;
    font-size: 0.9rem;
}
.green { color: #22c55e; font-weight: 700; }
.amber { color: #f59e0b; font-weight: 700; }
.red { color: #ef4444; font-weight: 700; }
.pill {
    display: inline-block;
    padding: 0.20rem 0.55rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.08);
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
}
.stButton > button {
    border-radius: 12px;
    height: 2.8rem;
}
textarea {
    border-radius: 14px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(args: List[str], timeout: int = 3600) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        output = (result.stdout or "") + ("\n[STDERR]\n" + result.stderr if result.stderr else "")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired as exc:
        return False, f"Command timed out: {' '.join(args)}\n{exc}"
    except Exception as exc:
        return False, f"Command failed: {' '.join(args)}\n{exc}"


def read_text(path: Path, default: str = "") -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return default


def write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    return default


def count_pdfs() -> int:
    return len(list(PAPERS_DIR.glob("*.pdf")))


def file_upload_key(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def sha256_file_for_signature(path: Path) -> str:
    """
    Same hash logic used by backend/incremental_index.py.
    This prevents false library-changed messages when the same PDF is uploaded again.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_library_signature() -> str:
    """
    Must match backend/incremental_index.py:
    name + size + sha256.
    """
    items = []
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        try:
            stat = pdf.stat()
            digest = sha256_file_for_signature(pdf)
            items.append(f"{pdf.name}:{stat.st_size}:{digest}")
        except Exception:
            items.append(pdf.name)
    return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()

def load_index_state() -> Dict[str, Any]:
    return read_json(INDEX_STATE_FILE, {})


def apply_research_mode(mode: str) -> Dict[str, Any]:
    return apply_mode_profile(mode)


def save_uploaded_pdfs(uploaded_files) -> List[str]:
    """
    Saves only new or changed PDFs.
    Returns saved filenames.
    """
    saved_names = []

    for uploaded in uploaded_files or []:
        if not uploaded.name.lower().endswith(".pdf"):
            continue

        key = file_upload_key(uploaded)
        if key in st.session_state.processed_upload_keys:
            continue

        target = PAPERS_DIR / uploaded.name
        new_bytes = uploaded.getbuffer()

        should_write = True
        if target.exists() and target.stat().st_size == len(new_bytes):
            should_write = False

        if should_write:
            target.write_bytes(new_bytes)
            saved_names.append(uploaded.name)

        st.session_state.processed_upload_keys.add(key)

    if saved_names:
        clear_current_answer()

    return saved_names


def library_needs_learning() -> bool:
    state = load_index_state()
    return state.get("library_signature") != current_library_signature() or not state.get("ok")


def run_index_pipeline(progress_container=None) -> Dict[str, Any]:
    """
    Incremental learning path.
    Calls backend/incremental_index.py, which skips parsing/embedding if PDFs are unchanged.
    """
    started = time.time()

    progress = progress_container.progress(0) if progress_container else None
    note = progress_container.empty() if progress_container else None

    if note:
        note.info("Checking papers...")
    if progress:
        progress.progress(15)

    ok, output = run_cmd([sys.executable, "-m", "backend.ingestion.incremental_index"], timeout=7200)

    if progress:
        progress.progress(100)

    LAST_INDEX_LOG.write_text(output, encoding="utf-8", errors="ignore")
    state = read_json(INDEX_STATE_FILE, {})

    seconds = round(time.time() - started, 2)
    skipped = "FAST PATH" in output or state.get("skipped") is True

    if note:
        if ok:
            if skipped:
                note.success("Papers are up to date.")
            else:
                note.success(f"Papers checked in {seconds} sec.")
        else:
            note.error("Learning failed. Open Developer tools.")

    return {
        "ok": ok,
        "failed_step": state.get("failed_step"),
        "seconds": seconds,
        "logs": output,
        "state": state,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------
# Silent question learning cache
# ---------------------------------------------------------------------

QUESTION_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for", "from", "by",
    "is", "are", "was", "were", "what", "which", "how", "why", "can", "i", "me", "my",
    "please", "tell", "about", "latest", "best", "good", "give", "show",
}


def normalize_question(question: str) -> str:
    question = question.lower()
    question = re.sub(r"[^a-z0-9\s\-]", " ", question)
    question = re.sub(r"\s+", " ", question).strip()
    return question


def question_terms(question: str) -> set:
    toks = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", normalize_question(question))
    return {t for t in toks if t not in QUESTION_STOPWORDS and len(t) > 2}


def similarity(q1: str, q2: str) -> float:
    a = question_terms(q1)
    b = question_terms(q2)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def load_question_memory() -> List[Dict[str, Any]]:
    data = read_json(QUESTION_MEMORY_FILE, [])
    return data if isinstance(data, list) else []


def save_question_memory(memory: List[Dict[str, Any]]) -> None:
    memory = sorted(memory, key=lambda x: x.get("last_used_at", ""), reverse=True)[:300]
    write_json(QUESTION_MEMORY_FILE, memory)


def find_similar_memory(question: str, threshold: float | None = None) -> Optional[Dict[str, Any]]:
    memory = load_question_memory()
    if threshold is None:
        try:
            threshold = float(os.getenv("QUESTION_MEMORY_THRESHOLD", "0.72"))
        except Exception:
            threshold = 0.72
    current_sig = current_library_signature()

    best = None
    best_score = 0.0

    for item in memory:
        if item.get("library_signature") != current_sig:
            continue
        score = similarity(question, item.get("question", ""))
        if score > best_score:
            best_score = score
            best = item

    if best and best_score >= threshold:
        best["similarity"] = round(best_score, 3)
        best["last_used_at"] = now_iso()
        save_question_memory(memory)
        return best

    return None


def store_question_memory(question: str, result: Dict[str, Any]) -> None:
    memory = load_question_memory()
    current_sig = current_library_signature()

    prompt_path = result.get("manual_prompt_path") or str(EXTRACTED_DIR / "latest_manual_prompt.txt")
    context_path = result.get("context_path") or str(EXTRACTED_DIR / "latest_context.txt")

    entry = {
        "question": question,
        "normalized": normalize_question(question),
        "terms": sorted(question_terms(question)),
        "created_at": now_iso(),
        "last_used_at": now_iso(),
        "library_signature": current_sig,
        "mode": result.get("mode", "manual"),
        "source_count": result.get("source_count") or result.get("sources_used") or "—",
        "time_seconds": result.get("time_seconds"),
        "manual_prompt_path": prompt_path,
        "context_path": context_path,
    }

    new_memory = []
    replaced = False
    for item in memory:
        if similarity(question, item.get("question", "")) >= 0.90 and item.get("library_signature") == current_sig:
            new_memory.append(entry)
            replaced = True
        else:
            new_memory.append(item)

    if not replaced:
        new_memory.insert(0, entry)

    save_question_memory(new_memory)


# ---------------------------------------------------------------------
# Ask + sources
# ---------------------------------------------------------------------

def ask_backend(question: str, mode: str, use_cache: bool = True) -> Dict[str, Any]:
    apply_research_mode(mode)
    started = time.time()

    if use_cache:
        cached = find_similar_memory(question)
        if cached:
            cached["from_memory"] = True
            cached["time_seconds"] = round(time.time() - started, 2)
            return cached

    try:
        from backend.answering.answer_orchestrator import run_research_question
        result = run_research_question(question)
        if isinstance(result, dict):
            result["time_seconds"] = round(time.time() - started, 2)
            result["from_memory"] = False
            store_question_memory(question, result)
            return result
    except Exception as exc:
        orchestrator_error = str(exc)
    else:
        orchestrator_error = ""

    try:
        from backend.answering.evidence_builder import build_evidence_for_question
        result = build_evidence_for_question(question)
        if isinstance(result, dict):
            result["time_seconds"] = round(time.time() - started, 2)
            result.setdefault("mode", "manual")
            result["from_memory"] = False
            store_question_memory(question, result)
            return result
    except Exception as exc:
        evidence_error = str(exc)
    else:
        evidence_error = ""

    result = {
        "mode": "manual",
        "time_seconds": round(time.time() - started, 2),
        "manual_prompt_path": str(EXTRACTED_DIR / "latest_manual_prompt.txt"),
        "context_path": str(EXTRACTED_DIR / "latest_context.txt"),
        "answer_text": "",
        "source_count": "—",
        "sources": [],
        "from_memory": False,
        "error": f"Could not call backend orchestrator. orchestrator={orchestrator_error}; evidence={evidence_error}",
    }
    store_question_memory(question, result)
    return result


def collect_sources_from_context(context_path: Optional[str]) -> List[Dict[str, Any]]:
    if not context_path:
        return []

    text = read_text(Path(context_path))
    if not text:
        return []

    blocks = []
    parts = text.split("[SOURCE ")
    for part in parts[1:]:
        body = "[SOURCE " + part
        lines = body.splitlines()
        title = "Uploaded paper"
        section = "Evidence"
        pages = ""
        concepts = ""

        for line in lines[:10]:
            lower = line.lower()
            if "paper:" in lower or "title:" in lower:
                title = line.split(":", 1)[-1].strip()
            elif "section:" in lower:
                section = line.split(":", 1)[-1].strip()
            elif "pages:" in lower:
                pages = line.split(":", 1)[-1].strip()
            elif "concepts:" in lower:
                concepts = line.split(":", 1)[-1].strip()

        blocks.append({
            "title": title,
            "section": section,
            "pages": pages,
            "concepts": concepts,
            "preview": body[:1200],
        })

        if len(blocks) >= 12:
            break

    return blocks


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.info("Sources will appear after you ask a question.")
        return

    for i, src in enumerate(sources, 1):
        title = src.get("title") or src.get("paper") or "Unknown paper"
        section = src.get("section") or src.get("section_name") or "Unknown section"
        pages = src.get("pages") or [src.get("page_start"), src.get("page_end")]
        concepts = src.get("concepts") or src.get("audio_concepts") or ""
        preview = src.get("preview") or src.get("text") or src.get("chunk_text") or ""

        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        st.markdown(f"**[{i}] {title}**")
        st.markdown(f'<div class="small-muted">Section: {section} · Pages: {pages}</div>', unsafe_allow_html=True)

        if concepts:
            for concept in str(concepts).replace("[", "").replace("]", "").replace("'", "").split(","):
                concept = concept.strip()
                if concept:
                    st.markdown(f'<span class="pill">{concept}</span>', unsafe_allow_html=True)

        with st.expander("Preview evidence"):
            st.write(str(preview)[:1200])

        st.markdown("</div>", unsafe_allow_html=True)


def prompt_and_context(result: Dict[str, Any]) -> Tuple[str, str]:
    prompt_path = Path(result.get("manual_prompt_path") or EXTRACTED_DIR / "latest_manual_prompt.txt")
    context_path = Path(result.get("context_path") or EXTRACTED_DIR / "latest_context.txt")
    return read_text(prompt_path), read_text(context_path)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "question" not in st.session_state:
    st.session_state.question = ""
if "last_question_text" not in st.session_state:
    st.session_state.last_question_text = ""
if "last_question_answered" not in st.session_state:
    st.session_state.last_question_answered = ""
if "processed_upload_keys" not in st.session_state:
    st.session_state.processed_upload_keys = set()


# Clear previous result when user changes query.
if st.session_state.question != st.session_state.last_question_text:
    if st.session_state.last_question_text:
        st.session_state.last_result = None
    st.session_state.last_question_text = st.session_state.question


def clear_current_answer() -> None:
    try:
        st.session_state.last_result = None
    except Exception:
        pass
    try:
        st.session_state.last_question_answered = ""
    except Exception:
        pass

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎧 Paper Library")

    pdf_count_placeholder = st.empty()
    pdf_count_placeholder.metric("PDFs learned", count_pdfs())

    index_state = load_index_state()
    if index_state.get("ok"):
        st.caption(f"Last learned: {index_state.get('last_index_time')}")
    elif index_state:
        st.error("Learning needs attention")
    else:
        st.info("Upload PDFs to begin.")

    research_mode = st.selectbox(
        "Answer depth",
        ["Fast", "Balanced", "Deep"],
        index=1,
        help="Fast = quickest. Balanced = recommended. Deep = strongest local retrieval with fresh evidence.",
    )
    active_mode_settings = apply_research_mode(research_mode)
    st.caption(active_mode_settings.get("description", ""))

    st.divider()

    st.markdown("### Upload PDFs")
    uploaded = st.file_uploader(
        "Drop PDF research papers here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        saved = save_uploaded_pdfs(uploaded)
        pdf_count_placeholder.metric("PDFs learned", count_pdfs())

        if saved:
            st.success(f"{len(saved)} PDF(s) uploaded. Learning now...")
            with st.status("Learning uploaded papers...", expanded=False) as status:
                result = run_index_pipeline(st)
                if result["ok"]:
                    status.update(label="Ready to ask questions", state="complete")
                    st.success("Papers checked.")
                    pdf_count_placeholder.metric("PDFs learned", count_pdfs())
                else:
                    status.update(label="Learning failed", state="error")
                    st.error("Learning failed. Open Developer tools below.")
        else:
            # Same file uploaded again or Streamlit rerun.
            # Do not show confusing "Library changed" unless it is truly changed.
            if not library_needs_learning():
                st.success("Papers are up to date.")
            else:
                st.info("Checking library status...")
                with st.status("Checking papers...", expanded=False) as status:
                    result = run_index_pipeline(st)
                    if result.get("ok"):
                        if result.get("skipped"):
                            status.update(label="Papers are up to date", state="complete")
                            st.success("Papers are up to date.")
                        else:
                            status.update(label="Papers checked", state="complete")
                            st.success("Papers checked.")
                        pdf_count_placeholder.metric("PDFs learned", count_pdfs())
                    else:
                        status.update(label="Learning failed", state="error")
                        st.error("Learning failed. Open Developer tools below.")

    st.divider()
    st.caption("The assistant improves repeated/similar questions silently in the background.")

    with st.expander("Developer tools"):
        if st.button("Relearn all papers", use_container_width=True):
            result = run_index_pipeline(st)
            if not result["ok"]:
                st.code(result["logs"])

        if st.button("Clear backend question memory", use_container_width=True):
            save_question_memory([])
            st.success("Backend question memory cleared.")

        st.caption("Index log")
        st.text_area("last_index_update.log", read_text(LAST_INDEX_LOG, "No log yet."), height=220)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
  <h1>🎧 Audio Research Paper Assistant</h1>
  <div class="small-muted">
    Ask questions from your uploaded audio/DSP research papers.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Ask", "Sources"])

with tabs[0]:
    st.session_state.question = st.text_area(
        "Ask any question",
        value=st.session_state.question,
        height=125,
        placeholder="Example: What are the latest methods for DOA estimation?",
    )

    c1, c2, c3 = st.columns([1, 1, 4])
    ask_clicked = c1.button("Ask", type="primary", use_container_width=True)
    clear_clicked = c2.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.question = ""
        st.session_state.last_question_text = ""
        try:
            clear_current_answer()
        except NameError:
            st.session_state.last_result = None
            st.session_state.last_question_answered = ""
        st.rerun()

    if ask_clicked:
        q = st.session_state.question.strip()
        try:
            clear_current_answer()
        except NameError:
            st.session_state.last_result = None
            st.session_state.last_question_answered = ""
        if not q:
            st.warning("Please enter a question.")
        elif count_pdfs() == 0:
            st.warning("Upload PDFs first. Learning will start automatically.")
        else:
            if library_needs_learning():
                with st.status("Learning new papers first...", expanded=False) as status:
                    idx = run_index_pipeline(st)
                    if idx["ok"]:
                        status.update(label="Papers checked", state="complete")
                    else:
                        status.update(label="Learning failed", state="error")
                        st.error("Learning failed. Open Developer tools in sidebar.")
                        st.stop()

            with st.status("Thinking...", expanded=False) as status:
                status.update(label="Retrieving evidence...", state="running")
                result = ask_backend(q, research_mode, use_cache=(os.getenv("USE_QUESTION_MEMORY", "true").lower() == "true"))
                try:
                    result = enhance_result_prompt(result)
                except Exception as prompt_quality_error:
                    result = dict(result or {})
                    result["prompt_quality_error"] = str(prompt_quality_error)
                status.update(label="Response ready", state="complete")

            st.session_state.last_result = result
            st.session_state.last_question_answered = q

    result = st.session_state.last_result

    if result:
        st.markdown("### Result")

        mode = result.get("mode", "manual")
        source_count = result.get("source_count") or result.get("sources_used") or "—"
        time_s = result.get("time_seconds", "—")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mode", mode)
        m2.metric("Sources", source_count)
        m3.metric("Time", f"{time_s} sec" if isinstance(time_s, (int, float)) else str(time_s))

        prompt_text, context_text = prompt_and_context(result)
        answer_text = result.get("answer_text") or ""

        if answer_text and mode != "manual":
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(answer_text)
            st.markdown("</div>", unsafe_allow_html=True)
        elif prompt_text:
            st.markdown(
                '<div class="answer-card"><b>Clean source-grounded prompt is ready.</b><br>'
                'Free mode: copy this clean prompt into Claude/ChatGPT web. Paid API later will generate the final answer here automatically.</div>',
                unsafe_allow_html=True,
            )
            st.text_area("Clean research prompt", value=prompt_text, height=420)
            st.download_button("Download prompt", prompt_text, file_name="audio_research_prompt.txt", mime="text/plain")
        else:
            st.warning("No prompt/answer was generated. Open Developer tools for logs.")

        if context_text:
            st.download_button("Download retrieved evidence", context_text, file_name="retrieved_evidence.txt", mime="text/plain")

        if result.get("error"):
            with st.expander("Developer note"):
                st.warning(result["error"])
    else:
        st.markdown(
            '<div class="card">Enter a question and click Ask. Ask a question to generate fresh evidence. Sources refresh automatically.</div>',
            unsafe_allow_html=True,
        )

with tabs[1]:
    st.markdown("### Source evidence")
    result = st.session_state.last_result or {}
    sources = result.get("sources") or result.get("source_cards") or []
    if not sources and result:
        sources = collect_sources_from_context(result.get("context_path") or str(EXTRACTED_DIR / "latest_context.txt"))
    render_sources(sources)
