"""
chat_ui.py  --  AudioLab AI v2 (Phase 2 UI polish, blue+white compact)

Same feature set as Batch 12B + autoimport, dressed in the AudioLab AI v2
brand: white main area, dark navy sidebar, sky-blue accent, compact density,
Crimson Text serif headings preserved.

All features preserved.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Path setup: ROOT must be importable so `import backend.*` resolves; THIS_DIR
# so the local `_chat_ui_utils` / `_components` helpers load by bare name.
ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
for _p in (ROOT, THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

from _chat_ui_utils import (
    compute_file_hash,
    list_existing_pdf_hashes,
    safe_pdf_target,
    list_ollama_models,
    update_env_var,
    run_ingestion,
)
from backend.memory.store import MemoryStore, default_db_path
from backend.tools.web_search import search_web, format_results_for_llm as format_web_for_llm
from backend.tools.code_executor import run_code as run_sandbox_code

# AudioLab AI v2 branding modules
import importlib.util as _ilu


def _load_brand_module(name: str):
    """Load sibling _*.py modules in this non-package directory.

    HOTFIX: explicitly clear __package__ so relative imports never trigger.
    Some Streamlit configurations were treating frontend/ as a package
    via stale __pycache__, causing 'attempted relative import with no
    known parent package' errors. Setting __package__ = '' makes the
    try/except fallback in _components.py reliably take the absolute path.
    """
    path = THIS_DIR / f"{name}.py"
    if not path.exists():
        raise ImportError(
            f"Required brand module not found: {path}. "
            f"Did the install complete successfully?"
        )
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    # Explicitly mark as top-level module (NOT part of a package) so any
    # 'from . import X' in the module raises ImportError predictably and
    # the absolute-import fallback in those modules kicks in.
    mod.__package__ = ""
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        # Give the user a clear message instead of the cryptic Streamlit
        # double-traceback. Most likely cause is stale __pycache__.
        raise ImportError(
            f"Failed to load brand module {name!r}: {type(exc).__name__}: {exc}. "
            f"Try deleting frontend/__pycache__ and restarting."
        ) from exc
    return mod


_theme = _load_brand_module("_theme")
_styles = _load_brand_module("_styles")
_logo = _load_brand_module("_logo")
_components = _load_brand_module("_components")


# ----------------------------------------------------------------------
# Page (AudioLab AI v2 brand: white + navy sidebar + sky blue)
# ----------------------------------------------------------------------
_components.configure_page()
_styles.inject_global_styles()


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_retriever():
    from backend.retrieval.hybrid_retrieve import hybrid_retrieve
    return hybrid_retrieve


@st.cache_resource(show_spinner=False)
def get_mode_applier():
    try:
        from backend.answering.research_modes import apply_research_mode
        return apply_research_mode
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass
    from backend.llm.provider import get_provider
    return get_provider()


@st.cache_resource(show_spinner=False)
def get_memory() -> MemoryStore:
    return MemoryStore(default_db_path(ROOT))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def format_evidence_for_llm(results, max_chars_per_source=900):
    if not results:
        return "(no retrieved sources)"
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("paper") or "Untitled"
        section = r.get("section") or r.get("section_name") or "?"
        ps = r.get("page_start") or "?"
        pe = r.get("page_end") or "?"
        text = (r.get("text") or r.get("chunk_text") or "").strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rsplit(" ", 1)[0] + "..."
        parts.append(f"[{i}] {title} -- {section} (pages {ps}-{pe})\n{text}")
    return "\n\n".join(parts)


SYSTEM_PROMPT_BASE = """You are an expert research assistant for audio signal processing and audio AI.

Answer the user using ONLY the numbered source excerpts provided in the user message. Be technical, precise, and concise.

CRITICAL HONESTY RULES (read carefully):
- If the user's question is unclear, ambiguous, or appears to be gibberish, DO NOT GUESS. Say: "I don't understand the question -- could you rephrase?"
- If the retrieved sources DO NOT actually address the user's question, SAY SO PLAINLY. Do not paraphrase unrelated sources into something that sounds relevant. Example acceptable response: "The retrieved sources don't directly answer this. They cover [X, Y] but not your specific question about [Z]. Could you rephrase, or upload a paper that covers this topic?"
- NEVER invent facts, statistics, or paper titles. NEVER cite a source for a claim that isn't actually in that source.
- It is better to say "I don't know" or "the sources don't cover this" than to hallucinate a confident answer.

Citation rules:
- Local paper excerpts are numbered [1], [2], [3]... -- cite these for claims drawn from the user's uploaded library
- Web search results (if present) are numbered [W1], [W2], [W3]... -- cite these for claims drawn from arXiv / Semantic Scholar
- If a claim is supported by both a local source and a web source, prefer the local citation
- If the sources do not cover something, state that plainly. Do not invent facts.

Code execution:
- When the user asks for a simulation, plot, calculation, or "show me a graph", emit a single Python code block in a ```python ... ``` fence.
- The code block runs in a sandbox with numpy, scipy, matplotlib, pandas, math, statistics. No file I/O, no network, no os/subprocess.
- Generate plots with matplotlib.pyplot. Don't call plt.show() -- the runner captures figures automatically. Use plt.figure() to start a new figure.
- Keep code self-contained; don't reference variables from prior turns.
- After the code block, briefly say what the user should see in the output.
- For pure-explanation questions, no code is needed.

CRITICAL when using dsp_toolkit:
- All toolkit functions are PRE-LOADED in the sandbox. You do NOT need
  to write `from dsp_toolkit import ...` -- just call the functions directly.
  Writing the import is also fine (does no harm).
- Use EXACT function names. The valid names are:
  simulate_ula_signals, steering_vector_ula, sample_covariance,
  array_factor, angle_grid, delay_and_sum, mvdr_beamformer,
  lcmv_beamformer, music_doa, srp_phat_doa, plot_beam_pattern,
  plot_doa_spectrum, mic_geometry_circular, mic_geometry_planar_rect,
  steering_vector_arbitrary, simulate_array_signals,
  delay_and_sum_arbitrary, mvdr_arbitrary, broadband_simulate,
  broadband_delay_and_sum, broadband_mvdr, simulate_room_recording,
  pesq_score, stoi_score, gcc_phat, gcc_phat_doa_pair.
- DO NOT invent variants like 'mvdr_beamform' (use mvdr_beamformer)
  or 'steering_vector_uta' (use steering_vector_ula).
- Pass kwargs explicitly: snr_db=20, not just 20.
- Convention: rx_signals.shape == (n_sensors, n_snapshots). Always.
- See the "DSP toolkit available" section below for full signatures.

Other rules:
- When the user asks about "latest" work, prefer web sources if available; otherwise note the corpus limit.
- Use proper technical terms (MVDR, LCMV, PESQ, STOI, DOA, etc.) where the sources do.
- Default to a conversational tone. Use headings or bullet points only if the user asks for them.
- Keep the answer focused. Two to four short paragraphs is usually right.
"""


def build_system_prompt(memory_block: str, include_dsp_toolkit: bool = False) -> str:
    prompt = SYSTEM_PROMPT_BASE
    if memory_block:
        prompt += (
            "\n\nContext from memory (use this to personalize your answer; "
            + "do not repeat it verbatim back to the user):\n"
            + memory_block
        )
    if include_dsp_toolkit:
        try:
            from backend.tools.dsp_toolkit import describe_api
            prompt += (
                "\n\n=== DSP toolkit available in the code sandbox ===\n"
                "When the user asks for a simulation involving beamforming, "
                "direction-of-arrival estimation, or array signal processing, "
                "PREFER calling these pre-built functions instead of writing "
                "the math from scratch. They are dimension-checked and verified. "
                "Import them with `from dsp_toolkit import ...`.\n\n"
                + describe_api()
            )
        except Exception:
            # If dsp_toolkit isn't importable, just skip the section
            pass
    return prompt


def build_user_message(question: str, evidence: str, web_evidence: str = "") -> str:
    msg = (
        f"Question: {question}\n\n"
        f"Retrieved evidence from uploaded papers:\n\n"
        f"{evidence}\n\n"
    )
    if web_evidence:
        msg += (
            f"Additional evidence from arXiv / Semantic Scholar web search:\n\n"
            f"{web_evidence}\n\n"
        )
    msg += (
        f"Answer the question above using only the evidence above. "
        f"Cite local sources with [n] and web sources with [Wn]."
    )
    return msg


def short(text, n):
    if not text:
        return ""
    s = str(text)
    return s if len(s) <= n else s[: n - 1] + "..."


def humanize_time(epoch: float) -> str:
    try:
        return time.strftime("%b %d, %H:%M", time.localtime(epoch))
    except Exception:
        return ""


def extract_python_blocks(text: str) -> list:
    """Return a list of python code blocks from the LLM output.
    Looks for ```python ... ``` and ``` ... ``` fences (the latter
    only if the content looks plausibly like Python).
    """
    import re
    blocks = []
    # Explicit python fence
    for m in re.finditer(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        code = m.group(1).rstrip()
        if code.strip():
            blocks.append(code)
    # Unmarked fences: only accept if it imports numpy/scipy/etc. or
    # has obvious Python syntax (avoids false positives on JSON dumps)
    if not blocks:
        for m in re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL):
            code = m.group(1).rstrip()
            if not code.strip():
                continue
            lower = code.lower()
            looks_python = (
                "import numpy" in lower
                or "import scipy" in lower
                or "import matplotlib" in lower
                or "import pandas" in lower
                or code.lstrip().startswith(("import ", "from ", "def ", "for ", "if ", "print(", "x ="))
            )
            if looks_python:
                blocks.append(code)
    return blocks


# ----------------------------------------------------------------------
# Initialize memory + session state
# ----------------------------------------------------------------------
mem = get_memory()

if "session_id" not in st.session_state:
    # Try to resume the most recent session; if none, create one
    sessions = mem.list_sessions(limit=1)
    if sessions:
        st.session_state.session_id = sessions[0]["id"]
    else:
        st.session_state.session_id = mem.create_session()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "ingestion_in_progress" not in st.session_state:
    st.session_state.ingestion_in_progress = False


def switch_session(new_session_id: str) -> None:
    st.session_state.session_id = new_session_id


def new_session() -> None:
    sid = mem.create_session()
    st.session_state.session_id = sid


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    # Branded sidebar header on navy background
    _components.render_sidebar_brand()

    # --- Sessions ----------------------------------------------------
    st.divider()
    _components.sidebar_section("Conversations")
    if st.button("+ New conversation", use_container_width=True, type="primary"):
        new_session()
        st.rerun()

    sessions = mem.list_sessions(limit=20)
    current_sid = st.session_state.session_id
    if sessions:
        labels = []
        ids = []
        for s in sessions:
            label = f"{s['title']}  ({humanize_time(s['updated_at'])})"
            labels.append(label)
            ids.append(s["id"])
        try:
            current_idx = ids.index(current_sid)
        except ValueError:
            current_idx = 0
        picked = st.radio(
            "Pick a conversation",
            options=labels,
            index=current_idx,
            key="session_radio",
            label_visibility="collapsed",
        )
        picked_id = ids[labels.index(picked)]
        if picked_id != current_sid:
            switch_session(picked_id)
            # When switching, clear the rename field so it re-seeds with
            # the new session's title on next render
            st.session_state.pop("rename_input", None)
            st.rerun()

        # Rename / delete the current session
        # v19 fix: control expander open-state via session_state so we can
        # auto-close it after Save/Delete.
        _rename_open = st.session_state.get("_rename_expander_open", False)
        with st.expander("Rename or delete this conversation", expanded=_rename_open):
            current = mem.get_session(current_sid)
            _current_title = current["title"] if current else ""

            # v19 fix: the Title field starts EMPTY so the user types a
            # fresh new name (not the old one). The current title is shown
            # only as a faded placeholder hint, not as editable text.
            new_title = st.text_input(
                "Title",
                value="",
                placeholder="Type a new title...",
                key="rename_input",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", use_container_width=True):
                    _t = (new_title or "").strip()
                    if _t:
                        mem.rename_session(current_sid, _t)
                        st.session_state["_rename_expander_open"] = False
                        st.session_state.pop("rename_input", None)
                        st.session_state["_flash_msg"] = ("success", "Renamed.")
                        st.rerun()
                    else:
                        st.session_state["_flash_msg"] = ("warning", "Type a title first.")
                        st.rerun()
            with col2:
                if st.button("Delete", type="secondary", use_container_width=True):
                    mem.delete_session(current_sid)
                    # Pick the next available session or create one
                    remaining = mem.list_sessions(limit=1)
                    if remaining:
                        switch_session(remaining[0]["id"])
                    else:
                        new_session()
                    st.session_state["_rename_expander_open"] = False
                    st.session_state.pop("rename_input", None)
                    st.session_state["_flash_msg"] = ("success", "Deleted.")
                    st.rerun()

        # v19: show a brief TOAST that auto-dismisses after ~4 seconds
        # (replaces the persistent st.success/st.warning boxes that stuck
        # on screen). Toasts appear bottom-right and fade on their own.
        _flash = st.session_state.pop("_flash_msg", None)
        if _flash:
            _kind, _text = _flash
            st.toast(_text)

    # --- Retrieval settings ------------------------------------------
    st.divider()
    _components.sidebar_section("Retrieval")
    mode = st.selectbox("Mode", options=["balanced", "fast", "deep"], index=0)
    top_k = st.slider("Sources to retrieve", 3, 15, 8)

    # v17: tuck advanced retrieval toggles into a collapsible expander
    # so the sidebar isn't an endless scroll. Defaults preserved.
    with st.expander("Advanced retrieval options", expanded=False):
        show_thinking = st.checkbox("Show retrieval details", value=True)
        use_web = st.checkbox(
            "Web search",
            value=False,
            help="Adds external papers as additional [Wn] citations. Slower (3-8s extra).",
        )
        if use_web:
            web_max = st.slider("Web results", 3, 10, 5)
        else:
            web_max = 0

        use_code = st.checkbox(
            "Code generation",
            value=False,
            help="When the LLM emits a ```python block, run it in a sandbox and show output + plots. 30s timeout.",
        )
        if use_code:
            code_timeout = st.slider("Code timeout (sec)", 5, 60, 30)
        else:
            code_timeout = 30

    # --- LLM ---------------------------------------------------------
    st.divider()
    _components.sidebar_section("Language Model")

    # --- Batch 12C HOTFIX: read CURRENT values from .env file directly ---
    # Bug we are fixing: os.getenv() reads from process environment which
    # was loaded when Streamlit started. After we write to .env, the process
    # environment doesn't auto-refresh, causing the saved value to mismatch
    # the dropdown -> infinite rerun loop.
    # Solution: read from .env file directly when checking for changes.
    def _read_env_file_value(key: str, default: str = "") -> str:
        env_path = ROOT / ".env"
        if not env_path.exists():
            return default
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
        except Exception:
            pass
        return default

    # --- Batch 12C: provider switcher (above existing logic) ---
    # Read current provider from .env FILE (not os.environ which is stale)
    _current_provider = _read_env_file_value("LLM_PROVIDER", "openai").lower()
    _provider_choices = ["openai", "ollama"]
    try:
        _prov_idx = _provider_choices.index(_current_provider)
    except ValueError:
        _prov_idx = 0
    _new_provider = st.selectbox(
        "Provider",
        options=_provider_choices,
        index=_prov_idx,
        format_func=lambda x: {
            "openai": "OpenAI (paid)",
            "ollama": "Ollama (free, local)",
        }.get(x, x),
        key="provider_selector_12c",
    )
    if _new_provider != _current_provider:
        try:
            update_env_var(ROOT / ".env", "LLM_PROVIDER", _new_provider)
            # CRITICAL: also update os.environ in this process so the change
            # takes effect immediately without restart
            os.environ["LLM_PROVIDER"] = _new_provider
            st.cache_resource.clear()
            # Mark that we just switched so the success message shows once
            st.session_state["_just_switched_provider"] = _new_provider
            st.rerun()
        except Exception as exc:
            st.error(f"Could not write .env: {exc}")

    # Show toast ONCE after a switch (auto-dismisses, no stale banner)
    if st.session_state.get("_just_switched_provider"):
        _switched_to = st.session_state.pop("_just_switched_provider")
        st.toast(f"Provider switched to {_switched_to}")

    # --- OpenAI model selector (only shown when provider is openai) ---
    if _new_provider == "openai":
        try:
            from backend.llm.multi_provider import (
                OPENAI_AVAILABLE_MODELS,
                OPENAI_DEFAULT_MODEL,
                get_provider as _get_provider,
                test_connection as _test_connection,
            )
            from backend.llm.cost_tracker import (
                format_usd as _format_usd,
                get_today_cost as _get_today_cost,
            )
            _12c_ok = True
        except ImportError:
            try:
                from backend.llm.multi_provider import (
                    OPENAI_AVAILABLE_MODELS,
                    OPENAI_DEFAULT_MODEL,
                    get_provider as _get_provider,
                    test_connection as _test_connection,
                )
                from backend.llm.cost_tracker import (
                    format_usd as _format_usd,
                    get_today_cost as _get_today_cost,
                )
                _12c_ok = True
            except ImportError:
                _12c_ok = False

        if _12c_ok:
            # Same fix: read current model from .env FILE, not os.environ
            _saved_model = _read_env_file_value("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
            try:
                _model_idx = OPENAI_AVAILABLE_MODELS.index(_saved_model)
            except ValueError:
                _model_idx = 0
            _new_model = st.selectbox(
                "OpenAI model",
                options=OPENAI_AVAILABLE_MODELS,
                index=_model_idx,
                key="openai_model_selector_12c",
            )
            st.session_state["openai_model"] = _new_model
            if _new_model != _saved_model:
                try:
                    update_env_var(ROOT / ".env", "OPENAI_MODEL", _new_model)
                    # CRITICAL: update os.environ so the change is immediate
                    os.environ["OPENAI_MODEL"] = _new_model
                except Exception:
                    pass  # not fatal

            # Check if API key is set
            _openai_provider = _get_provider("openai")
            if _openai_provider.is_available():
                st.success("API key detected")
                # Show today's spend
                _today = _get_today_cost()
                st.caption(f"Today's spend: **{_format_usd(_today)}**")

                # Test connection button
                if st.button("Test connection", help="Sends a 5-token test (cost: ~$0.0001)", use_container_width=True):
                    with st.spinner("Calling OpenAI..."):
                        _test_result = _test_connection("openai", model=_new_model)
                    # v19: results show as auto-dismissing toasts (vanish ~4s),
                    # not as persistent boxes that stick on screen.
                    if _test_result.error:
                        _e = _test_result.error
                        if "429" in _e or "quota" in _e.lower() or "RateLimit" in _e:
                            st.toast("OpenAI: no credit (429)")
                            # Brief detail as a small caption (not a big box).
                            st.caption(
                                "No usable credit. Add billing at platform.openai.com, "
                                "or switch Provider to Ollama (free)."
                            )
                        elif "auth" in _e.lower() or "401" in _e or "invalid" in _e.lower():
                            st.toast("OpenAI key invalid")
                            st.caption(
                                "Check .env: OPENAI_API_KEY=sk-... (no quotes, no spaces)."
                            )
                        else:
                            st.toast("OpenAI test failed")
                            st.caption(_e[:160])
                    else:
                        st.toast("OpenAI OK")
                        st.caption(
                            f"Replied '{_test_result.text}' -- "
                            f"{_format_usd(_test_result.cost_usd)}"
                        )
            else:
                st.warning("OPENAI_API_KEY not set in .env")
                st.caption("Add OPENAI_API_KEY=sk-... to your .env file, then restart")
        else:
            st.warning("Batch 12C provider module not installed yet")

    # --- existing Ollama-specific UI (only shown when provider is ollama) ---
    try:
        llm = get_llm()
        ok = llm.is_available
        if _new_provider == "ollama":
            st.markdown(f"**Provider:** `{llm.name}`")

        if llm.name == "ollama" and _new_provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            available = list_ollama_models(host=host)
            current = llm.model

            col_a, col_b = st.columns([4, 1])
            with col_a:
                if available:
                    options = list(dict.fromkeys([current] + available))
                    try:
                        idx = options.index(current)
                    except ValueError:
                        idx = 0
                    selected = st.selectbox(
                        "Model", options=options, index=idx,
                        key="ollama_model_selector",
                    )
                    if selected != current:
                        try:
                            update_env_var(ROOT / ".env", "OLLAMA_MODEL", selected)
                            st.cache_resource.clear()
                            st.success(f"Switched to `{selected}`")
                            time.sleep(0.6)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not write .env: {exc}")
                else:
                    st.markdown(f"**Model:** `{current}`")
                    st.caption("Run `ollama list` in cmd; restart UI to refresh.")
            with col_b:
                st.write("")
                st.write("")
                if st.button(":arrows_counterclockwise:", help="Refresh model list", use_container_width=True):
                    st.rerun()
        elif _new_provider == "ollama":
            st.markdown(f"**Model:** `{llm.model}`")

        if _new_provider == "ollama":
            if ok:
                st.success("Available")
            else:
                st.warning("Not available")
    except Exception as e:
        st.error(f"LLM config error: {e}")
        llm = None

    # --- PDF upload --------------------------------------------------
    st.divider()
    with st.expander(":inbox_tray: Upload paper", expanded=False):
        st.caption("Drag a PDF. Duplicates detected by content hash, not filename.")
        uploaded = st.file_uploader(
            "PDF file", type=["pdf"], accept_multiple_files=False,
            key=f"paper_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )
        if uploaded is not None and not st.session_state.ingestion_in_progress:
            try:
                data = uploaded.getvalue()
                size_kb = len(data) // 1024
                file_hash = compute_file_hash(data)
                papers_dir = ROOT / "data" / "papers"
                existing = list_existing_pdf_hashes(papers_dir)
                if file_hash in existing:
                    st.warning(f"Already in library: **{existing[file_hash]}**")
                else:
                    st.info(f":page_facing_up: `{uploaded.name}` -- {size_kb} KB. New paper.")
                    if st.button("Ingest this PDF", type="primary", use_container_width=True):
                        st.session_state.ingestion_in_progress = True
                        try:
                            target_path = safe_pdf_target(papers_dir, uploaded.name)
                            target_path.write_bytes(data)
                            st.success(f"Saved to `data/papers/{target_path.name}`")
                            status = st.status(
                                f"Parsing + chunking + embedding `{target_path.name}` ...",
                                expanded=True,
                            )
                            log_placeholder = status.empty()
                            log_lines = []
                            def on_line(line):
                                log_lines.append(line)
                                log_placeholder.code("\n".join(log_lines[-12:]))
                            code, message, _all = run_ingestion(ROOT, on_line=on_line)
                            if code == 0:
                                status.update(label=f"`{target_path.name}` ready to query!", state="complete")
                                st.cache_resource.clear()
                                st.session_state.uploader_key += 1
                                time.sleep(1.2)
                                st.rerun()
                            else:
                                status.update(label=f"Ingestion failed: {message}", state="error")
                        finally:
                            st.session_state.ingestion_in_progress = False
            except Exception as exc:
                st.error(f"Upload error: {exc}")

    # --- Memory ------------------------------------------------------
    st.divider()
    with st.expander("Long-term memory", expanded=False):
        st.caption(
            "Facts here are persisted across all sessions. Use this for "
            "your research focus, what you care about, project context, etc."
        )
        global_facts = mem.list_facts("global")
        if not global_facts:
            st.caption("_No long-term facts yet._")
        else:
            st.caption(f"**{len(global_facts)} fact(s)** — click a key to expand")
            # v23: each fact is a COMPACT collapsible item. The key is the
            # title (scannable at a glance); the full value is hidden until
            # you click to expand. A delete button sits inside each one.
            for fact in global_facts:
                with st.expander(fact["key"], expanded=False):
                    st.markdown(fact["value"])
                    if st.button("Delete this fact",
                                 key=f"del_global_{fact['key']}",
                                 use_container_width=True):
                        mem.delete_fact("global", fact["key"])
                        st.session_state["_flash_msg"] = ("success", "Fact deleted.")
                        st.rerun()

        st.markdown("---")
        # v23: bulletproof field clearing. Instead of popping session_state
        # (which was unreliable -- the widget re-seeded), we use a NONCE in
        # the widget key. After a successful save we bump the nonce, so the
        # next render creates BRAND-NEW empty widgets. New key = empty field,
        # every time, guaranteed.
        _nonce = st.session_state.get("_fact_field_nonce", 0)
        new_key = st.text_input(
            "Key", placeholder="e.g. research focus",
            key=f"new_global_key_{_nonce}",
        )
        new_val = st.text_area(
            "Value",
            placeholder="e.g. low-latency MVDR beamforming for embedded systems",
            key=f"new_global_val_{_nonce}", height=72,
        )
        if st.button("Add / update fact", key="add_global_fact",
                     use_container_width=True):
            if new_key.strip() and new_val.strip():
                mem.upsert_fact("global", new_key.strip(), new_val.strip())
                # Bump the nonce -> next render uses fresh empty widgets
                st.session_state["_fact_field_nonce"] = _nonce + 1
                st.session_state["_flash_msg"] = ("success", "Fact saved.")
                st.rerun()
            else:
                st.session_state["_flash_msg"] = ("warning", "Both key and value are required.")
                st.rerun()

    with st.expander("Notes for this conversation", expanded=False):
        st.caption("Cleared when you clear this conversation. Survives within the session.")
        session_facts = mem.list_facts("session", current_sid)
        if not session_facts:
            st.caption("_No session notes yet._")
        else:
            for fact in session_facts:
                with st.expander(fact["key"], expanded=False):
                    st.markdown(fact["value"])
                    if st.button("Delete this note",
                                 key=f"del_session_{fact['key']}",
                                 use_container_width=True):
                        mem.delete_fact("session", fact["key"], current_sid)
                        st.session_state["_flash_msg"] = ("success", "Note deleted.")
                        st.rerun()

    # --- Memory stats + clear button ---------------------------------
    st.divider()
    stats = mem.stats()
    st.caption(
        f"DB: {stats['sessions']} sessions  |  {stats['turns']} turns  |  "
        f"{stats['global_facts']} long-term  |  {stats['session_facts']} session facts"
    )
    if st.button("Clear this conversation", use_container_width=True):
        mem.clear_turns(current_sid)
        mem.clear_session_facts(current_sid)
        st.rerun()


# ----------------------------------------------------------------------
# Main area (white)
# ----------------------------------------------------------------------
current_session = mem.get_session(current_sid)
title = current_session["title"] if current_session else "Audio Research"

# Branded header at top of main area
_provider_label = os.getenv("LLM_PROVIDER", "openai").lower()
_components.render_header(status_pill=(_provider_label, "info"))

# Conversation title as h2 in serif (no subtitle clutter)
st.markdown(
    f'<h2 style="font-family:{_theme.FONT_HEADING}; '
    f'color:{_theme.COLOR_TEXT}; margin-top:{_theme.SPACE_SM}; '
    f'margin-bottom:{_theme.SPACE_LG};">{title}</h2>',
    unsafe_allow_html=True,
)

# Render conversation from DB (not from session state -- DB is authoritative)
turns = mem.get_turns(current_sid)
for turn in turns:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        sources = turn.get("sources") or []
        if sources:
            local_srcs = [s for s in sources if s.get("_kind") not in ("web", "code_run")]
            web_srcs   = [s for s in sources if s.get("_kind") == "web"]
            code_srcs  = [s for s in sources if s.get("_kind") == "code_run"]
            if local_srcs:
                with st.expander(f"{len(local_srcs)} sources used"):
                    for i, src in enumerate(local_srcs, 1):
                        title_s = src.get("title") or src.get("paper") or "Untitled"
                        section = src.get("section") or "?"
                        ps = src.get("page_start")
                        pe = src.get("page_end")
                        st.markdown(f"**[{i}] {title_s}**")
                        st.caption(f"Section: {section}  |  Pages: {ps}-{pe}")
                        st.text(short(src.get("text") or src.get("chunk_text") or "", 500))
            if web_srcs:
                with st.expander(f"{len(web_srcs)} web sources used"):
                    for i, w in enumerate(web_srcs, 1):
                        title_w = w.get("title") or "Untitled"
                        venue = w.get("venue") or w.get("source") or ""
                        year = w.get("year")
                        url = w.get("url") or ""
                        st.markdown(f"**[W{i}] {title_w}**")
                        st.caption(f"{venue}{', ' + str(year) if year else ''}")
                        if url:
                            st.markdown(f"[{url}]({url})")
                        st.text(short(w.get("abstract") or "", 500))
            if code_srcs:
                with st.expander(f"{len(code_srcs)} code block run"):
                    for cr in code_srcs:
                        st.markdown(f"**Block {cr.get('block_index', '?')}**  "
                                    f"({'OK' if cr.get('ok') else 'FAIL'}, "
                                    f"{cr.get('elapsed_sec', 0):.2f}s, "
                                    f"{cr.get('n_plots', 0)} plots)")
                        st.code(cr.get("code") or "", language="python")
                        if cr.get("stdout"):
                            st.text(cr["stdout"])
                        if not cr.get("ok") and cr.get("error"):
                            st.caption(f"Error: {cr['error']}")


question = st.chat_input(
    "Ask a question, request a simulation, or search the literature..."
)

if question:
    # ------- Query sanity check (v8) -------
    # Catch obvious nonsense BEFORE retrieval + LLM. Prevents the model
    # from hallucinating confident answers to gibberish like "buoh".
    try:
        from backend.answering.query_sanity import check_query_sanity
    except ImportError:
        try:
            from backend.answering.query_sanity import check_query_sanity
        except ImportError:
            check_query_sanity = None  # fail open if module missing

    if check_query_sanity is not None:
        sanity = check_query_sanity(question)
        if not sanity.ok:
            # Show the user's message and the polite refusal, no LLM call
            mem.append_turn(current_sid, "user", question)
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                st.warning(sanity.user_message)
            mem.append_turn(
                current_sid, "assistant",
                f"[Query rejected: {sanity.reason}]  {sanity.user_message}"
            )
            st.stop()

    # Auto-rename a fresh conversation using the first user turn
    if current_session and current_session["title"] == "New conversation":
        mem.rename_session(current_sid, question[:60])

    mem.append_turn(current_sid, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # ------- Retrieval -------
        results = []
        apply_mode = get_mode_applier()
        if apply_mode is not None:
            try:
                apply_mode(mode)
            except Exception:
                pass

        retrieve_status = st.status(
            f"Retrieving sources (mode: {mode}) ...", expanded=False,
        )
        with retrieve_status:
            try:
                retriever = get_retriever()
                t0 = time.time()
                results = retriever(question, top_k=top_k)
                elapsed = time.time() - t0
                retrieve_status.update(
                    label=f"Found {len(results)} sources in {elapsed:.1f}s",
                    state="complete",
                )
            except Exception as e:
                retrieve_status.update(label="Retrieval failed", state="error")
                st.error(f"Retrieval error: {e}")

        if show_thinking and results:
            with st.expander(f"Top {min(3, len(results))} sources"):
                for i, r in enumerate(results[:3], 1):
                    title_s = r.get("title") or "Untitled"
                    section = r.get("section") or "?"
                    rerank = r.get("rerank_score")
                    suffix = f"  (rerank {rerank:.3f})" if isinstance(rerank, (int, float)) else ""
                    st.markdown(f"**[{i}] {title_s}** -- {section}{suffix}")
                    st.text(short(r.get("text") or "", 350))

        # ------- Web search (optional) -------
        web_results = []
        if use_web:
            web_status = st.status(
                "Searching arXiv + Semantic Scholar ...",
                expanded=False,
            )
            with web_status:
                try:
                    tw0 = time.time()
                    web_results = search_web(question, max_results=web_max)
                    we = time.time() - tw0
                    web_status.update(
                        label=f"Web: found {len(web_results)} external papers in {we:.1f}s",
                        state="complete",
                    )
                except Exception as ex:
                    web_status.update(label=f"Web search failed: {ex}", state="error")
                    web_results = []

            if show_thinking and web_results:
                with st.expander(f"Top {min(3, len(web_results))} web results"):
                    for i, w in enumerate(web_results[:3], 1):
                        title_w = w.get("title") or "Untitled"
                        venue = w.get("venue") or w.get("source") or ""
                        year = w.get("year")
                        suffix = f"  ({venue}{', ' + str(year) if year else ''})"
                        st.markdown(f"**[W{i}] {title_w}**{suffix}")
                        st.text(short(w.get("abstract") or "", 350))

        # ------- LLM with memory-aware system prompt -------
        memory_block = mem.build_memory_block(current_sid)
        system_prompt = build_system_prompt(memory_block, include_dsp_toolkit=use_code)
        evidence_text = format_evidence_for_llm(results)
        web_evidence_text = format_web_for_llm(web_results) if web_results else ""
        user_msg_with_evidence = build_user_message(question, evidence_text, web_evidence_text)
        recent = mem.get_recent_turns(current_sid, n_messages=6)
        # The just-added user turn is in `recent`; replace its content with
        # the evidence-augmented version, since that's what the LLM should see.
        if recent and recent[-1]["role"] == "user":
            recent[-1] = {"role": "user", "content": user_msg_with_evidence}
        else:
            recent.append({"role": "user", "content": user_msg_with_evidence})

        if llm is None or not llm.is_available:
            lines = ["*LLM is not available right now -- showing the top retrieved sources directly.*", ""]
            for i, r in enumerate(results[:5], 1):
                title_s = r.get("title") or "Untitled"
                section = r.get("section") or "?"
                lines.append(f"**[{i}] {title_s}** -- {section}")
                lines.append(short(r.get("text") or "", 450))
                lines.append("")
            answer = "\n".join(lines) if results else "*No sources found and no LLM available.*"
            st.markdown(answer)
        else:
            # ------- Batch 12C: provider routing -------
            # If user picked OpenAI in the sidebar, use the new provider abstraction
            # with auto-fallback to Ollama. Otherwise, use the existing Ollama
            # streaming path (preserved for backward compat).
            placeholder = st.empty()
            full = ""

            if _provider_label == "openai":
                # NEW: OpenAI path via provider abstraction
                try:
                    from backend.llm.multi_provider import generate_with_fallback
                except ImportError:
                    try:
                        from backend.llm.multi_provider import generate_with_fallback
                    except ImportError:
                        generate_with_fallback = None

                if generate_with_fallback is None:
                    answer = "*OpenAI provider module not installed. Falling back to Ollama.*"
                    placeholder.markdown(answer)
                else:
                    # Get selected model from session state (set by sidebar)
                    openai_model = st.session_state.get("openai_model", "gpt-4o-mini")
                    placeholder.markdown("*Generating response via OpenAI...*")
                    result = generate_with_fallback(
                        primary="openai",
                        messages=recent,
                        system=system_prompt,
                        model=openai_model,
                        fallback="ollama",
                        fallback_model=st.session_state.get("ollama_model"),
                        max_tokens=2048,
                        temperature=0.3,
                    )
                    if result.error and not result.text:
                        # Both providers failed -- show BOTH errors clearly
                        # so the user sees the real OpenAI problem, not just
                        # the masked Ollama fallback failure.
                        _openai_err = result.fallback_reason or result.error
                        answer = f"*LLM call failed.*\n\n"
                        answer += f"**OpenAI error:** {_openai_err}\n\n"
                        if result.fell_back:
                            answer += f"**Ollama fallback also failed:** {result.error}\n\n"
                        # Actionable hint for the most common error (429 quota)
                        if "429" in _openai_err or "quota" in _openai_err.lower() or "RateLimit" in _openai_err:
                            answer += (
                                "**This is a billing issue, not a code issue.** Your "
                                "OpenAI account has no usable credit. Add credits at "
                                "platform.openai.com/settings/organization/billing -- "
                                "then click Test connection again. "
                                "Or switch Provider to Ollama (free) in the sidebar.\n\n"
                            )
                        answer += "---\n\n*Showing top retrieved sources instead:*\n\n"
                        for i, r in enumerate(results[:5], 1):
                            answer += f"**[{i}] {r.get('title', 'Untitled')}**\n"
                            answer += short(r.get("text") or "", 300) + "\n\n"
                        placeholder.markdown(answer)
                    else:
                        answer = result.text
                        # Show fallback warning if it happened (with REAL reason)
                        if result.fell_back:
                            placeholder.warning(
                                f"OpenAI failed, used Ollama as backup. "
                                f"OpenAI said: {result.fallback_reason}"
                            )
                            st.markdown(answer)
                        else:
                            # Show cost for this query (small, easy to glance at)
                            placeholder.markdown(answer)
                            if result.cost_usd > 0:
                                try:
                                    from backend.llm.cost_tracker import format_usd, get_today_cost
                                except ImportError:
                                    from backend.llm.cost_tracker import format_usd, get_today_cost
                                st.caption(
                                    f"Query: {format_usd(result.cost_usd)} "
                                    f"({result.tokens_in} in + {result.tokens_out} out tokens) "
                                    f"| Today: {format_usd(get_today_cost())}"
                                )
            else:
                # EXISTING: Ollama streaming path (unchanged)
                try:
                    for chunk in llm.stream_chat(
                        messages=recent, system=system_prompt,
                        max_tokens=2048, temperature=0.3,
                    ):
                        full += chunk
                        placeholder.markdown(full + " :black_small_square:")
                    placeholder.markdown(full)
                    answer = full
                except Exception as e:
                    answer = f"*LLM call failed: {e}*\n\n"
                    for i, r in enumerate(results[:5], 1):
                        answer += f"**[{i}] {r.get('title', 'Untitled')}**\n"
                        answer += short(r.get("text") or "", 300) + "\n\n"
                    placeholder.markdown(answer)

        if results:
            with st.expander(f"{len(results)} sources used"):
                for i, src in enumerate(results, 1):
                    title_s = src.get("title") or "Untitled"
                    section = src.get("section") or "?"
                    ps = src.get("page_start")
                    pe = src.get("page_end")
                    st.markdown(f"**[{i}] {title_s}**")
                    st.caption(f"Section: {section}  |  Pages: {ps}-{pe}")
                    st.text(short(src.get("text") or "", 500))

        if web_results:
            with st.expander(f"{len(web_results)} web sources used"):
                for i, w in enumerate(web_results, 1):
                    title_w = w.get("title") or "Untitled"
                    authors = w.get("authors") or []
                    venue = w.get("venue") or w.get("source") or ""
                    year = w.get("year")
                    cc = w.get("citation_count")
                    cc_str = f"  |  {cc} citations" if isinstance(cc, int) and cc > 0 else ""
                    url = w.get("url") or ""
                    st.markdown(f"**[W{i}] {title_w}**")
                    if authors:
                        st.caption(
                            f"{', '.join(authors[:4])}"
                            f"{' et al.' if len(authors) > 4 else ''}  |  "
                            f"{venue}{', ' + str(year) if year else ''}{cc_str}"
                        )
                    if url:
                        st.markdown(f"[{url}]({url})")
                    st.text(short(w.get("abstract") or "", 500))

        # ------- Code execution (optional) -------
        code_runs = []  # persisted into the turn for history rendering
        if use_code and answer:
            blocks = extract_python_blocks(answer)
            if blocks:
                with st.expander(
                    f"Running {len(blocks)} code block"
                    f"{'s' if len(blocks) > 1 else ''} ...",
                    expanded=True,
                ):
                    for bi, code in enumerate(blocks, 1):
                        st.markdown(f"**Block {bi}**")
                        st.code(code, language="python")
                        run_status = st.status(f"Executing block {bi} ...", expanded=False)
                        try:
                            result = run_sandbox_code(code, timeout_sec=float(code_timeout))
                        except Exception as exc:
                            result = {
                                "ok": False,
                                "error": f"executor crashed: {exc}",
                                "stdout": "",
                                "plots": [],
                                "elapsed_sec": 0.0,
                            }

                        if result.get("ok"):
                            run_status.update(
                                label=f"Block {bi}: ran in {result.get('elapsed_sec', 0):.2f}s",
                                state="complete",
                            )
                        else:
                            run_status.update(
                                label=f"Block {bi}: {result.get('error', 'failed')}",
                                state="error",
                            )

                        if result.get("stdout"):
                            st.text(result["stdout"])
                        plots = result.get("plots") or []
                        for pi, b64 in enumerate(plots, 1):
                            try:
                                import base64 as _b64
                                img_bytes = _b64.b64decode(b64)
                                st.image(img_bytes, caption=f"Block {bi} -- plot {pi}")
                            except Exception:
                                pass
                        if not result.get("ok") and result.get("traceback"):
                            with st.expander("Traceback"):
                                st.code(result["traceback"], language="text")

                        code_runs.append({
                            "block_index": bi,
                            "code": code,
                            "ok": result.get("ok", False),
                            "error": result.get("error"),
                            "stdout": result.get("stdout", "")[:5000],
                            "n_plots": len(plots),
                            "elapsed_sec": result.get("elapsed_sec", 0.0),
                        })

        # Persist the assistant turn -- combine local + web sources so
        # they all reappear when the conversation is reloaded.
        combined_for_history = list(results) if results else []
        if web_results:
            # Mark web ones so the renderer can distinguish them
            for w in web_results:
                combined_for_history.append({**w, "_kind": "web"})
        # Stash code-run results in a parallel structure on the same sources list
        if code_runs:
            for cr in code_runs:
                combined_for_history.append({**cr, "_kind": "code_run"})
        mem.append_turn(current_sid, "assistant", answer, sources=combined_for_history)


# (Footer removed in v3 -- was clutter)
