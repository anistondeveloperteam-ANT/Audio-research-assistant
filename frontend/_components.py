"""
_components.py  --  AudioLab AI v5 components

v5 changes from v4:
  - All HTML returned by helpers is SINGLE-LINE (no embedded newlines,
    no indented multi-line strings). This is the REAL fix for the
    "</div> visible as text" bug -- Streamlit's markdown was
    misinterpreting indented HTML as a code block.
  - render_header simplified: one st.markdown with one flat HTML string,
    no st.columns trick needed.
"""

from __future__ import annotations

import html as _html
from typing import Optional

import streamlit as st

try:
    from . import _theme as T
    from ._logo import logo_with_wordmark_html, logo_svg, logo_header_html
except ImportError:
    import _theme as T
    from _logo import logo_with_wordmark_html, logo_svg, logo_header_html


def _e(s: Optional[str]) -> str:
    if s is None:
        return ""
    return _html.escape(str(s))


def render_header(status_pill: Optional[tuple] = None) -> None:
    """Top header in the white main area. SINGLE-LINE HTML.

    v7: uses logo_header_html which is bigger (36px logo, 1.5rem wordmark)
    with a sky-blue accent underline. Has .alab-brand class so CSS can
    animate the waveform bars on hover.
    """
    pill_html = ""
    if status_pill:
        label, kind = status_pill
        pill_html = render_pill_html(label, kind)

    brand_html = logo_header_html()
    header_html = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:0.5rem 0 1rem 0;margin-bottom:1.25rem;'
        f'border-bottom:1px solid {T.COLOR_BORDER};">'
        f'<div>{brand_html}</div>'
        f'<div style="text-align:right;">{pill_html}</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Logo + wordmark for navy sidebar. SINGLE-LINE HTML."""
    logo_html = logo_with_wordmark_html(22, on_dark=True, show_tagline=True)
    html = (
        f'<div style="padding:0.25rem 0 0.875rem 0;'
        f'border-bottom:1px solid {T.COLOR_SIDEBAR_BORDER};'
        f'margin-bottom:0.5rem;">{logo_html}</div>'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_pill_html(label: str, kind: str = "neutral") -> str:
    kind = kind if kind in ("success", "warning", "info", "neutral") else "neutral"
    return f'<span class="alab-pill alab-pill-{kind}">{_e(label)}</span>'


def show_pill(label: str, kind: str = "neutral") -> None:
    st.markdown(render_pill_html(label, kind), unsafe_allow_html=True)


def sidebar_section(title: str) -> None:
    st.sidebar.markdown(
        f'<div class="alab-side-section">{_e(title)}</div>',
        unsafe_allow_html=True,
    )


def render_evidence_card(
    number: int,
    title: str,
    section: Optional[str] = None,
    pages: Optional[str] = None,
    text_preview: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    meta_parts = []
    if section:
        meta_parts.append(f"Section: {_e(section)}")
    if pages:
        meta_parts.append(f"Pages {_e(pages)}")
    if badge:
        meta_parts.append(f'<span style="color:{T.COLOR_ACCENT};">{_e(badge)}</span>')
    meta_str = "  &bull;  ".join(meta_parts)

    text_html = ""
    if text_preview:
        preview = _e(text_preview[:300] + ("..." if len(text_preview) > 300 else ""))
        text_html = f'<div class="alab-evidence-text">{preview}</div>'

    html = (
        f'<div class="alab-evidence">'
        f'<div><span class="alab-evidence-num">{number}</span>'
        f'<span class="alab-evidence-title">{_e(title)}</span></div>'
        f'<div class="alab-evidence-meta">{meta_str}</div>'
        f'{text_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state() -> None:
    logo_html = logo_svg(72, color=T.COLOR_ACCENT)
    html = (
        f'<div style="text-align:center;padding:{T.SPACE_2XL} {T.SPACE_MD};'
        f'color:{T.COLOR_TEXT_MUTED};">'
        f'<div style="margin-bottom:{T.SPACE_MD};">{logo_html}</div>'
        f'<div style="font-family:{T.FONT_HEADING};font-size:{T.SIZE_2XL};'
        f'color:{T.COLOR_TEXT};margin-bottom:{T.SPACE_SM};">'
        f'Welcome to {T.BRAND_NAME}</div>'
        f'<div style="max-width:560px;margin:0 auto;line-height:{T.LH_RELAXED};'
        f'font-size:{T.SIZE_BASE};color:{T.COLOR_TEXT_MUTED};">'
        f'{T.BRAND_TAGLINE}. Ask a question about your library of papers, '
        f'run a simulation, or explore the latest work on arXiv and '
        f'Semantic Scholar.</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def configure_page() -> None:
    st.set_page_config(
        page_title=T.BRAND_NAME,
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                f"### {T.BRAND_NAME}\n\n"
                f"{T.BRAND_TAGLINE}\n\n"
                f"Version: {T.BRAND_VERSION}"
            ),
        },
    )
