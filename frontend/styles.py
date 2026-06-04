"""
_styles.py  --  AudioLab AI v3 CSS payload

v3 changes from v2:
  - COMPLETE sidebar widget coverage: file_uploader, expander, text inputs,
    textareas, select, multiselect, slider, button, checkbox -- all dark-themed
  - Inline code in sidebar no longer renders as white box (was bug in v2)
  - File uploader drop zone is dark navy (was white in v2)
  - Placeholder text is light enough to read on dark
  - All inputs have proper focus rings in sky-blue
"""

import streamlit as st
try:
    from . import theme as T
except ImportError:
    import theme as T


_FONT_IMPORT = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
"""


def _build_css() -> str:
    return f"""
<style>
{_FONT_IMPORT}

/* ============================================================
   GLOBAL TYPOGRAPHY
   ============================================================ */
html, body, [class*="css"], .stApp, .stMarkdown, .stText {{
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_BASE};
    line-height: {T.LH_NORMAL};
    color: {T.COLOR_TEXT};
}}

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
    font-family: {T.FONT_HEADING};
    font-weight: 600;
    color: {T.COLOR_TEXT};
    letter-spacing: {T.LS_TIGHT};
    line-height: {T.LH_TIGHT};
}}
h1, .stMarkdown h1 {{ font-size: {T.SIZE_3XL}; margin: {T.SPACE_MD} 0 {T.SPACE_SM}; }}
h2, .stMarkdown h2 {{ font-size: {T.SIZE_2XL}; margin: {T.SPACE_MD} 0 {T.SPACE_SM}; }}
h3, .stMarkdown h3 {{ font-size: {T.SIZE_XL};  margin: {T.SPACE_SM} 0 {T.SPACE_XS}; }}

p, .stMarkdown p, label {{
    color: {T.COLOR_TEXT};
    line-height: {T.LH_NORMAL};
    margin-bottom: {T.SPACE_SM};
}}

code, .stCode, pre, .stMarkdown code {{
    font-family: {T.FONT_MONO} !important;
    font-size: 0.8rem;
}}
.stMarkdown pre, .stCode pre {{
    background: {T.COLOR_CODE_BG} !important;
    color: {T.COLOR_CODE_TEXT} !important;
    border: 1px solid {T.COLOR_CODE_BORDER};
    border-radius: {T.RADIUS_MD};
    padding: {T.SPACE_MD} !important;
}}
.stMarkdown p code, .stMarkdown li code, .stMarkdown td code {{
    background: {T.COLOR_BG_HOVER};
    color: {T.COLOR_ACCENT_HOVER};
    padding: 1px 5px;
    border-radius: {T.RADIUS_SM};
    border: 1px solid {T.COLOR_BORDER};
    font-size: 0.85em;
}}

a, .stMarkdown a {{
    color: {T.COLOR_ACCENT} !important;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.12s ease;
}}
a:hover, .stMarkdown a:hover {{
    color: {T.COLOR_ACCENT_HOVER} !important;
    border-bottom-color: {T.COLOR_ACCENT};
}}

/* ============================================================
   PAGE FRAME (white)
   ============================================================ */
.stApp {{
    background: {T.COLOR_BG};
}}

.main .block-container {{
    max-width: 960px;
    padding-top: {T.SPACE_LG};
    padding-bottom: {T.SPACE_2XL};
    padding-left: {T.SPACE_XL};
    padding-right: {T.SPACE_XL};
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

header[data-testid="stHeader"] {{
    background: transparent;
    height: 0;
}}

/* ============================================================
   SIDEBAR FRAME (dark navy)
   ============================================================ */
section[data-testid="stSidebar"] {{
    background: {T.COLOR_SIDEBAR_BG} !important;
    border-right: 1px solid {T.COLOR_SIDEBAR_BORDER};
}}
section[data-testid="stSidebar"] > div:first-child {{
    padding-top: {T.SPACE_LG};
    padding-left: {T.SPACE_MD};
    padding-right: {T.SPACE_MD};
}}

/* All visible text in sidebar gets light color (defensive baseline) */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] em {{
    color: {T.COLOR_SIDEBAR_TEXT};
}}

/* Sidebar default headings -> styled like .alab-side-section */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    font-family: {T.FONT_BODY} !important;
    font-size: {T.SIZE_XS} !important;
    font-weight: 600 !important;
    color: {T.COLOR_SIDEBAR_TEXT_DIM} !important;
    text-transform: uppercase !important;
    letter-spacing: {T.LS_WIDE} !important;
    margin-top: {T.SPACE_LG} !important;
    margin-bottom: {T.SPACE_SM} !important;
    padding-bottom: {T.SPACE_XS} !important;
    border-bottom: 1px solid {T.COLOR_SIDEBAR_BORDER} !important;
}}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr {{
    border-color: {T.COLOR_SIDEBAR_BORDER};
    border-top: 1px solid {T.COLOR_SIDEBAR_BORDER};
    margin: {T.SPACE_MD} 0;
}}

/* ============================================================
   SIDEBAR INLINE CODE -- FIX: was white box in v2
   ============================================================ */
section[data-testid="stSidebar"] code,
section[data-testid="stSidebar"] .stMarkdown code,
section[data-testid="stSidebar"] p code {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    color: {T.COLOR_ACCENT} !important;
    border: 1px solid {T.COLOR_SIDEBAR_BORDER} !important;
    padding: 1px 5px;
    border-radius: {T.RADIUS_SM};
    font-family: {T.FONT_MONO};
    font-size: 0.85em;
}}

/* ============================================================
   SIDEBAR TEXT INPUTS, TEXTAREAS  -- FIX: was unreadable in v2
   ============================================================ */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="search"],
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] textarea {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    color: {T.COLOR_SIDEBAR_TEXT} !important;
    border: 1px solid {T.COLOR_SIDEBAR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-size: {T.SIZE_SM};
    font-family: {T.FONT_BODY};
}}

/* Placeholder text -- visible on dark */
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {{
    color: {T.COLOR_SIDEBAR_TEXT_DIM} !important;
    opacity: 1;
}}

section[data-testid="stSidebar"] .stTextInput input:focus,
section[data-testid="stSidebar"] .stTextArea textarea:focus,
section[data-testid="stSidebar"] textarea:focus,
section[data-testid="stSidebar"] input:focus {{
    border-color: {T.COLOR_ACCENT} !important;
    box-shadow: 0 0 0 1px {T.COLOR_ACCENT};
    outline: none;
}}

/* ============================================================
   SIDEBAR SELECT / DROPDOWN
   ============================================================ */
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] [data-baseweb="select"] {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    color: {T.COLOR_SIDEBAR_TEXT} !important;
    border: 1px solid {T.COLOR_SIDEBAR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: {T.COLOR_SIDEBAR_TEXT} !important;
}}

/* Select dropdown arrow */
section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
    fill: {T.COLOR_SIDEBAR_TEXT_MUTED} !important;
}}

/* ============================================================
   SIDEBAR FILE UPLOADER  -- v4: stronger upload button, hover state
   ============================================================ */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: transparent;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    border: 1px dashed {T.COLOR_SIDEBAR_BORDER} !important;
    color: {T.COLOR_SIDEBAR_TEXT_MUTED} !important;
    border-radius: {T.RADIUS_MD} !important;
    transition: border-color 0.15s ease, background 0.15s ease;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {{
    border-color: {T.COLOR_ACCENT} !important;
    border-style: dashed !important;
    background: {T.COLOR_SIDEBAR_BG} !important;
}}
/* Body text inside dropzone (the "Drag and drop file here" caption) */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {{
    color: {T.COLOR_SIDEBAR_TEXT_MUTED} !important;
}}
/* The internal "Browse files" button -- v4 makes it INTERACTIVE-LOOKING */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {{
    background: {T.COLOR_ACCENT} !important;
    color: white !important;
    border: 1px solid {T.COLOR_ACCENT} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
    box-shadow: 0 1px 3px rgba(14, 165, 233, 0.3) !important;
    transition: all 0.15s ease;
    opacity: 1 !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button:hover {{
    background: {T.COLOR_ACCENT_HOVER} !important;
    border-color: {T.COLOR_ACCENT_HOVER} !important;
    color: white !important;
    box-shadow: 0 2px 6px rgba(14, 165, 233, 0.45) !important;
    transform: translateY(-1px);
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:active,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button:active {{
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(14, 165, 233, 0.3) !important;
}}
/* Make sure the button text inside is also bright */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button * {{
    color: white !important;
}}
/* Already-uploaded file pill */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    color: {T.COLOR_SIDEBAR_TEXT} !important;
    border-radius: {T.RADIUS_MD};
    padding: {T.SPACE_SM};
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {{
    color: {T.COLOR_SIDEBAR_TEXT} !important;
}}

/* ============================================================
   SIDEBAR BUTTONS
   ============================================================ */
section[data-testid="stSidebar"] .stButton button {{
    background: transparent !important;
    color: {T.COLOR_SIDEBAR_TEXT} !important;
    border: 1px solid {T.COLOR_SIDEBAR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_SM};
    font-weight: 500;
    padding: 0.35rem 0.85rem;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
}}
section[data-testid="stSidebar"] .stButton button:hover {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    border-color: {T.COLOR_ACCENT} !important;
    color: {T.COLOR_ACCENT} !important;
    transform: translateY(-1px);
}}
section[data-testid="stSidebar"] .stButton button:active {{
    transform: translateY(0);
}}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background: linear-gradient(135deg, {T.COLOR_ACCENT} 0%, {T.COLOR_ACCENT_HOVER} 100%) !important;
    color: white !important;
    border: 1px solid {T.COLOR_ACCENT} !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 1px 3px rgba(14, 165, 233, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    transition: all 0.2s ease !important;
}}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
    background: linear-gradient(135deg, {T.COLOR_ACCENT_HOVER} 0%, {T.COLOR_ACCENT_ACTIVE} 100%) !important;
    border-color: {T.COLOR_ACCENT_HOVER} !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.45),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-1px);
}}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:active {{
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(14, 165, 233, 0.4) !important;
}}

/* ============================================================
   SIDEBAR EXPANDERS  -- FIX: long-term memory + upload paper panels
   ============================================================ */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    border: none !important;
    background: transparent !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] .streamlit-expanderHeader {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    color: {T.COLOR_SIDEBAR_TEXT} !important;
    border: 1px solid {T.COLOR_SIDEBAR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_SM} !important;
    font-weight: 500 !important;
    padding: {T.SPACE_SM} {T.SPACE_MD} !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {{
    border-color: {T.COLOR_ACCENT} !important;
    color: {T.COLOR_ACCENT} !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {{
    color: inherit !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
    background: {T.COLOR_SIDEBAR_BG} !important;
    border: 1px solid {T.COLOR_SIDEBAR_BORDER};
    border-top: none;
    border-bottom-left-radius: {T.RADIUS_MD};
    border-bottom-right-radius: {T.RADIUS_MD};
    padding: {T.SPACE_MD};
    color: {T.COLOR_SIDEBAR_TEXT};
}}

/* ============================================================
   SIDEBAR CHECKBOXES, RADIOS, SLIDERS
   ============================================================ */
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label {{
    color: {T.COLOR_SIDEBAR_TEXT};
    font-size: {T.SIZE_SM};
}}
section[data-testid="stSidebar"] .stSlider * {{
    color: {T.COLOR_SIDEBAR_TEXT};
}}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {{
    background: {T.COLOR_ACCENT} !important;
}}
section[data-testid="stSidebar"] .stSlider [role="slider"] {{
    background: {T.COLOR_ACCENT} !important;
    border-color: {T.COLOR_ACCENT} !important;
}}

/* Sidebar alerts (info/success boxes) -- darken */
section[data-testid="stSidebar"] .stAlert {{
    background: {T.COLOR_SIDEBAR_BG_ELEV} !important;
    color: {T.COLOR_SIDEBAR_TEXT} !important;
    border-left: 3px solid {T.COLOR_ACCENT};
    border-radius: {T.RADIUS_MD};
}}
section[data-testid="stSidebar"] .stAlert * {{
    color: {T.COLOR_SIDEBAR_TEXT} !important;
}}

/* Sidebar "Available" success styling specifically */
section[data-testid="stSidebar"] .stAlert[data-baseweb*="success"] {{
    border-left-color: {T.COLOR_SUCCESS};
}}

/* ============================================================
   MAIN AREA INPUTS / BUTTONS
   ============================================================ */
.main .stTextInput input, .main .stTextArea textarea, .stChatInput input {{
    background: {T.COLOR_BG_INPUT} !important;
    color: {T.COLOR_TEXT} !important;
    border: 1px solid {T.COLOR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_BASE};
}}
.main .stTextInput input:focus,
.main .stTextArea textarea:focus,
.stChatInput input:focus {{
    border-color: {T.COLOR_ACCENT} !important;
    box-shadow: 0 0 0 1px {T.COLOR_ACCENT};
    outline: none;
}}

.main .stButton button, .main .stDownloadButton button {{
    background: {T.COLOR_BG} !important;
    color: {T.COLOR_TEXT} !important;
    border: 1px solid {T.COLOR_BORDER_STRONG} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_SM};
    font-weight: 500;
    padding: 0.35rem 0.85rem;
    box-shadow: {T.SHADOW_SM};
    transition: all 0.12s ease;
}}
.main .stButton button:hover, .main .stDownloadButton button:hover {{
    border-color: {T.COLOR_ACCENT} !important;
    color: {T.COLOR_ACCENT} !important;
}}

.main .stButton button[kind="primary"] {{
    background: {T.COLOR_ACCENT} !important;
    color: white !important;
    border-color: {T.COLOR_ACCENT} !important;
}}
.main .stButton button[kind="primary"]:hover {{
    background: {T.COLOR_ACCENT_HOVER} !important;
    border-color: {T.COLOR_ACCENT_HOVER} !important;
    color: white !important;
}}

.main .stSelectbox > div > div {{
    background: {T.COLOR_BG_INPUT} !important;
    border: 1px solid {T.COLOR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
    color: {T.COLOR_TEXT};
}}

/* ============================================================
   CHAT MESSAGES (compact)
   ============================================================ */
/* ============================================================
   CHAT MESSAGES (v14: assistant stands out, user fades back)
   ============================================================
   Visual goal: the eye should land on the ASSISTANT bubble (the
   answer), not the USER bubble (the question they already know).
*/

[data-testid="stChatMessage"] {{
    padding: {T.SPACE_MD} !important;
    border-radius: {T.RADIUS_LG};
    margin-bottom: {T.SPACE_SM};
    font-size: {T.SIZE_BASE};
    /* v14: gentle fade-in so responses feel alive when they render */
    animation: alab-msg-fade-in 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}}

@keyframes alab-msg-fade-in {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to   {{ opacity: 1; transform: translateY(0);   }}
}}

/* USER message: faded back, neutral grey, thin left edge -------- */
[data-testid="stChatMessage"]:has([aria-label*="user" i]),
[data-testid="stChatMessage"]:has([data-testid*="user" i]),
.user-message {{
    background: {T.COLOR_BG_ALT} !important;
    border: 1px solid {T.COLOR_BORDER} !important;
    border-left: 2px solid {T.COLOR_BORDER_STRONG} !important;
}}

/* ASSISTANT message: this is the ANSWER. Make it pop -------- */
[data-testid="stChatMessage"]:has([aria-label*="assistant" i]),
[data-testid="stChatMessage"]:has([data-testid*="assistant" i]),
.assistant-message {{
    background: linear-gradient(135deg,
        {T.COLOR_ACCENT_BG} 0%,
        #FFFFFF 100%) !important;
    border: 1px solid {T.COLOR_ACCENT_BORDER} !important;
    border-left: 3px solid {T.COLOR_ACCENT} !important;
    box-shadow:
        0 1px 2px rgba(14, 165, 233, 0.06),
        0 2px 8px rgba(14, 165, 233, 0.08) !important;
    position: relative;
}}

/* Subtle hover: shadow grows slightly */
[data-testid="stChatMessage"]:has([aria-label*="assistant" i]):hover,
[data-testid="stChatMessage"]:has([data-testid*="assistant" i]):hover {{
    box-shadow:
        0 1px 2px rgba(14, 165, 233, 0.08),
        0 4px 14px rgba(14, 165, 233, 0.12) !important;
    transition: box-shadow 0.2s ease;
}}

/* ============================================================
   CHAT INPUT BAR  -- v5 interactive polish
   ============================================================ */
[data-testid="stChatInput"] {{
    background: {T.COLOR_BG} !important;
    border-top: 1px solid {T.COLOR_BORDER};
    padding: 1rem 1.5rem !important;
}}

/* The actual input area */
[data-testid="stChatInput"] > div {{
    background: {T.COLOR_BG_ALT} !important;
    border: 1.5px solid {T.COLOR_BORDER} !important;
    border-radius: 12px !important;
    padding: 4px 8px !important;
    box-shadow: {T.SHADOW_SM} !important;
    transition: all 0.2s ease;
}}

[data-testid="stChatInput"] > div:hover {{
    border-color: {T.COLOR_BORDER_STRONG} !important;
    box-shadow: {T.SHADOW_MD} !important;
}}

[data-testid="stChatInput"] > div:focus-within {{
    border-color: {T.COLOR_ACCENT} !important;
    box-shadow: 0 0 0 3px {T.COLOR_ACCENT_BG}, {T.SHADOW_MD} !important;
    background: {T.COLOR_BG} !important;
}}

/* The textarea inside */
[data-testid="stChatInput"] textarea, .stChatInput textarea {{
    background: transparent !important;
    color: {T.COLOR_TEXT} !important;
    border: none !important;
    font-size: 0.95rem !important;
    font-family: {T.FONT_BODY};
    padding: 0.55rem 0.6rem !important;
    box-shadow: none !important;
    outline: none !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: {T.COLOR_TEXT_DIM} !important;
    opacity: 1 !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    box-shadow: none !important;
    outline: none !important;
}}

/* The send button (arrow) */
[data-testid="stChatInput"] button {{
    background: {T.COLOR_ACCENT} !important;
    color: white !important;
    border: 1px solid {T.COLOR_ACCENT} !important;
    border-radius: 8px !important;
    transition: all 0.15s ease;
}}
[data-testid="stChatInput"] button:hover {{
    background: {T.COLOR_ACCENT_HOVER} !important;
    border-color: {T.COLOR_ACCENT_HOVER} !important;
    transform: scale(1.05);
}}
[data-testid="stChatInput"] button:active {{
    transform: scale(0.95);
}}
[data-testid="stChatInput"] button svg {{
    fill: white !important;
}}

/* ============================================================
   MAIN AREA EXPANDERS ("8 sources used")
   ============================================================ */
.main .streamlit-expanderHeader,
.main [data-testid="stExpander"] summary {{
    background: {T.COLOR_BG_ALT} !important;
    color: {T.COLOR_TEXT_MUTED} !important;
    border: 1px solid {T.COLOR_BORDER} !important;
    border-radius: {T.RADIUS_MD} !important;
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_SM};
    font-weight: 500;
    padding: {T.SPACE_SM} {T.SPACE_MD};
}}
.main .streamlit-expanderHeader:hover,
.main [data-testid="stExpander"] summary:hover {{
    border-color: {T.COLOR_ACCENT} !important;
    color: {T.COLOR_TEXT} !important;
}}
.main [data-testid="stExpander"] {{
    border: none !important;
}}
.main [data-testid="stExpanderDetails"] {{
    background: {T.COLOR_BG_ALT};
    border: 1px solid {T.COLOR_BORDER};
    border-top: none;
    border-bottom-left-radius: {T.RADIUS_MD};
    border-bottom-right-radius: {T.RADIUS_MD};
    padding: {T.SPACE_MD};
}}

/* ============================================================
   ALERTS (main area)
   ============================================================ */
.main .stAlert[data-baseweb*="success"] {{
    background: {T.COLOR_SUCCESS_BG} !important;
    border-left: 3px solid {T.COLOR_SUCCESS};
    color: {T.COLOR_TEXT};
}}
.main .stAlert[data-baseweb*="warning"] {{
    background: {T.COLOR_WARNING_BG} !important;
    border-left: 3px solid {T.COLOR_WARNING};
    color: {T.COLOR_TEXT};
}}
.main .stAlert[data-baseweb*="error"] {{
    background: {T.COLOR_ERROR_BG} !important;
    border-left: 3px solid {T.COLOR_ERROR};
    color: {T.COLOR_TEXT};
}}
.main .stAlert[data-baseweb*="info"] {{
    background: {T.COLOR_ACCENT_BG} !important;
    border-left: 3px solid {T.COLOR_ACCENT};
    color: {T.COLOR_TEXT};
}}

.main hr {{
    border: none;
    border-top: 1px solid {T.COLOR_BORDER};
    margin: {T.SPACE_MD} 0;
}}

/* ============================================================
   SIDEBAR ALWAYS OPEN -- BUTTON REMOVED  (v19, reverted from v18)
   ============================================================
   Decision history:
     v16  forced always-open + hid button  -> user: "works perfectly"
     v18  added native collapse back       -> user: collapse kept
          resetting when switching conversations, found it annoying
     v19  REVERT to v16 behavior: always open, button completely
          removed. An always-open sidebar can NEVER reset, which
          also solves the "sidebar automatically resets" complaint.

   Pure CSS, no JavaScript (the lesson from v9-v15). We override the
   3 properties Streamlit uses to collapse the sidebar (transform,
   min-width, max-width) so it physically cannot slide closed, and
   we hide both the collapse and expand buttons.
*/

/* Force sidebar ALWAYS visible at fixed width; disable collapse animation */
section[data-testid="stSidebar"] {{
    transform: translateX(0) !important;
    min-width: 244px !important;
    max-width: 244px !important;
    visibility: visible !important;
    opacity: 1 !important;
    transition: none !important;
}}

/* Hide the "<<" collapse button completely */
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="header"],
button[data-testid="baseButton-headerNoPadding"] {{
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}}

/* Hide the ">>" expand button completely (never needed -- always open) */
[data-testid="stExpandSidebarButton"] {{
    display: none !important;
}}

/* ============================================================ */

/* ============================================================
   AUDIOLAB CUSTOM CLASSES
   ============================================================ */

/* v7: Bigger branded header with hover animations
   --------------------------------------------------------- */

/* The wrapping .alab-brand div is the trigger for child animations */
.alab-brand {{
    cursor: default;
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* Subtle scale up when hovering the brand block */
.alab-brand:hover .alab-brand-logo {{
    filter: drop-shadow(0 2px 4px rgba(14, 165, 233, 0.25));
}}

/* The accent underline GROWS when you hover the brand */
.alab-brand:hover .alab-brand-underline {{
    width: 100% !important;
}}

/* Waveform bars: each bar gets a wave animation on hover, staggered */
.alab-brand:hover .alab-wave-bar {{
    animation: alab-wave 1.2s ease-in-out infinite;
    transform-origin: center;
}}
.alab-brand:hover .alab-wave-bar-1 {{ animation-delay: 0.00s; }}
.alab-brand:hover .alab-wave-bar-2 {{ animation-delay: 0.05s; }}
.alab-brand:hover .alab-wave-bar-3 {{ animation-delay: 0.10s; }}
.alab-brand:hover .alab-wave-bar-4 {{ animation-delay: 0.15s; }}
.alab-brand:hover .alab-wave-bar-5 {{ animation-delay: 0.20s; }}
.alab-brand:hover .alab-wave-bar-6 {{ animation-delay: 0.15s; }}
.alab-brand:hover .alab-wave-bar-7 {{ animation-delay: 0.10s; }}
.alab-brand:hover .alab-wave-bar-8 {{ animation-delay: 0.05s; }}
.alab-brand:hover .alab-wave-bar-9 {{ animation-delay: 0.00s; }}

@keyframes alab-wave {{
    0%, 100% {{ transform: scaleY(1); opacity: 1; }}
    50%      {{ transform: scaleY(1.15); opacity: 0.85; }}
}}

/* Base bar style (no animation when not hovered) */
.alab-wave-bar {{
    transition: opacity 0.2s ease;
}}

/* --------------------------------------------------------- */

.alab-header-meta {{
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_XS};
    color: {T.COLOR_TEXT_DIM};
    letter-spacing: {T.LS_WIDE};
    text-transform: uppercase;
}}

.alab-pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: {T.RADIUS_PILL};
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: {T.LS_WIDE};
    text-transform: uppercase;
    font-family: {T.FONT_BODY};
    border: 1px solid;
    vertical-align: middle;
}}
.alab-pill-success {{
    background: {T.COLOR_SUCCESS_BG};
    color: {T.COLOR_SUCCESS};
    border-color: {T.COLOR_SUCCESS_BORDER};
}}
.alab-pill-warning {{
    background: {T.COLOR_WARNING_BG};
    color: {T.COLOR_WARNING};
    border-color: {T.COLOR_WARNING_BORDER};
}}
.alab-pill-info {{
    background: {T.COLOR_ACCENT_BG};
    color: {T.COLOR_ACCENT_HOVER};
    border-color: {T.COLOR_ACCENT_BORDER};
}}
.alab-pill-neutral {{
    background: {T.COLOR_BG_ALT};
    color: {T.COLOR_TEXT_MUTED};
    border-color: {T.COLOR_BORDER};
}}

.alab-evidence {{
    background: {T.COLOR_CITATION_BG};
    border: 1px solid {T.COLOR_BORDER};
    border-left: 3px solid {T.COLOR_CITATION_BORDER};
    border-radius: {T.RADIUS_MD};
    padding: {T.SPACE_MD};
    margin-bottom: {T.SPACE_SM};
}}
.alab-evidence-num {{
    display: inline-block;
    background: {T.COLOR_ACCENT};
    color: white;
    padding: 1px 7px;
    border-radius: {T.RADIUS_SM};
    font-family: {T.FONT_MONO};
    font-size: {T.SIZE_XS};
    font-weight: 600;
    margin-right: {T.SPACE_SM};
}}
.alab-evidence-title {{
    font-family: {T.FONT_HEADING};
    font-size: {T.SIZE_LG};
    font-weight: 600;
    color: {T.COLOR_TEXT};
    margin: {T.SPACE_XS} 0;
}}
.alab-evidence-meta {{
    font-size: {T.SIZE_XS};
    color: {T.COLOR_TEXT_DIM};
    margin-bottom: {T.SPACE_SM};
}}
.alab-evidence-text {{
    font-size: {T.SIZE_SM};
    color: {T.COLOR_TEXT_MUTED};
    line-height: {T.LH_NORMAL};
}}

.alab-side-section {{
    font-family: {T.FONT_BODY};
    font-size: {T.SIZE_XS};
    font-weight: 600;
    color: {T.COLOR_SIDEBAR_TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: {T.LS_WIDE};
    margin-top: {T.SPACE_LG};
    margin-bottom: {T.SPACE_SM};
    padding-bottom: {T.SPACE_XS};
    border-bottom: 1px solid {T.COLOR_SIDEBAR_BORDER};
}}

/* Scrollbars */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}
::-webkit-scrollbar-track {{
    background: {T.COLOR_BG};
}}
::-webkit-scrollbar-thumb {{
    background: {T.COLOR_BORDER_STRONG};
    border-radius: {T.RADIUS_PILL};
    border: 2px solid {T.COLOR_BG};
}}
::-webkit-scrollbar-thumb:hover {{
    background: {T.COLOR_TEXT_DIM};
}}
section[data-testid="stSidebar"] ::-webkit-scrollbar-track {{
    background: {T.COLOR_SIDEBAR_BG};
}}
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
    background: {T.COLOR_SIDEBAR_BORDER};
    border: 2px solid {T.COLOR_SIDEBAR_BG};
}}

/* ============================================================
   v17 POLISH  --  tighter spacing + smooth interactions
   ============================================================ */

/* Tighten vertical gaps in the sidebar so it scrolls less */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.55rem !important;
}}
section[data-testid="stSidebar"] hr {{
    margin: 0.6rem 0 !important;
}}

/* Smooth transitions on all interactive widgets (subtle, professional) */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
section[data-testid="stSidebar"] .stButton button,
.main .stButton button {{
    transition: border-color 0.15s ease, background-color 0.15s ease,
                box-shadow 0.15s ease, color 0.15s ease !important;
}}

/* Selectbox: refined focus ring in accent color */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]:focus-within {{
    border-color: {T.COLOR_ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.18) !important;
}}

/* Buttons: gentle lift on hover for tactile feel */
.main .stButton button:hover,
section[data-testid="stSidebar"] .stButton button:hover {{
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.18) !important;
    transform: translateY(-1px);
}}
.main .stButton button:active,
section[data-testid="stSidebar"] .stButton button:active {{
    transform: translateY(0);
}}

/* Modern heading weight + letter-spacing for Space Grotesk */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}

/* Sidebar section labels: refined uppercase micro-label look */
section[data-testid="stSidebar"] .alab-sidebar-section {{
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    opacity: 0.65;
}}

/* Chat input: slightly larger, more inviting */
.main [data-testid="stChatInput"] textarea {{
    font-size: {T.SIZE_BASE} !important;
    font-family: {T.FONT_BODY} !important;
}}
.main [data-testid="stChatInput"]:focus-within {{
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.15) !important;
}}

/* ============================================================
   TOAST POPUPS  --  readable, never clipped  (v21)
   ============================================================
   v20 fixed colour (dark bg, white text) but the box was still
   clipped at the top edge and long messages overflowed.
   v21: push the toast container WELL below the top edge, let the
   box grow to fit its text (auto height, no overflow clip), and
   keep messages short (handled in chat_ui.py).
*/
[data-testid="stToast"] {{
    background: {T.COLOR_SIDEBAR_BG} !important;
    color: #ffffff !important;
    border: 1px solid {T.COLOR_ACCENT} !important;
    border-left: 4px solid {T.COLOR_ACCENT} !important;
    border-radius: {T.RADIUS_MD} !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45) !important;
    padding: 1rem 1.2rem !important;
    font-family: {T.FONT_BODY} !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    line-height: 1.45 !important;
    min-width: 300px !important;
    max-width: 420px !important;
    /* let the box grow to fit the text, never clip it */
    height: auto !important;
    min-height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    white-space: normal !important;
}}
/* Every text node inside the toast: white, fully visible, no clip */
[data-testid="stToast"] *,
[data-testid="stToast"] div,
[data-testid="stToast"] span,
[data-testid="stToast"] p {{
    color: #ffffff !important;
    opacity: 1 !important;
    overflow: visible !important;
    white-space: normal !important;
    text-overflow: clip !important;
    max-height: none !important;
}}
/* Push the whole toast container DOWN from the top edge so the box
   is never sliced off. 4.5rem clears the browser chrome + header. */
[data-testid="stToastContainer"] {{
    top: 4.5rem !important;
    right: 1.5rem !important;
    overflow: visible !important;
}}

</style>
"""


def inject_global_styles() -> None:
    """Inject the AudioLab AI v3 CSS into the page."""
    st.markdown(_build_css(), unsafe_allow_html=True)
