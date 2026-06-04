"""
_theme.py  --  AudioLab AI v2 design tokens

Palette: white main area + navy sidebar + sky-blue accents
Density: compact (GitHub-like)
Personality: clean professional dashboard, with serif headings
             for academic warmth.

Color naming follows Tailwind conventions for clarity.
"""

# ----------------------------------------------------------------------
# Colors -- mixed light+dark palette
# ----------------------------------------------------------------------

# Main area (light)
COLOR_BG = "#FFFFFF"              # main page background -- clean white
COLOR_BG_ALT = "#F8FAFC"          # very light grey (slate-50) for subtle cards
COLOR_BG_HOVER = "#F1F5F9"        # slate-100, used for hover states
COLOR_BG_INPUT = "#FFFFFF"        # input backgrounds stay white

# Sidebar (dark navy, GitHub/Slack-style)
COLOR_SIDEBAR_BG = "#0F172A"           # slate-900, deep navy
COLOR_SIDEBAR_BG_ELEV = "#1E293B"      # slate-800, hover/active items
COLOR_SIDEBAR_TEXT = "#F1F5F9"         # slate-100, primary
COLOR_SIDEBAR_TEXT_MUTED = "#94A3B8"   # slate-400, secondary
COLOR_SIDEBAR_TEXT_DIM = "#64748B"     # slate-500, section labels
COLOR_SIDEBAR_BORDER = "#334155"       # slate-700, dividers

# Text on white
COLOR_TEXT = "#0F172A"            # slate-900, body
COLOR_TEXT_MUTED = "#475569"      # slate-600, secondary
COLOR_TEXT_DIM = "#94A3B8"        # slate-400, captions

# Borders (on white)
COLOR_BORDER = "#E2E8F0"          # slate-200, light border
COLOR_BORDER_STRONG = "#CBD5E1"   # slate-300, emphasis

# Accent (sky blue, Tailwind 500)
COLOR_ACCENT = "#0EA5E9"          # primary brand accent
COLOR_ACCENT_HOVER = "#0284C7"    # sky-600, hover state
COLOR_ACCENT_ACTIVE = "#0369A1"   # sky-700, active state
COLOR_ACCENT_BG = "#F0F9FF"       # sky-50, subtle accent background
COLOR_ACCENT_BORDER = "#7DD3FC"   # sky-300, soft border with accent tone

# Semantic
COLOR_SUCCESS = "#16A34A"         # green-600
COLOR_SUCCESS_BG = "#F0FDF4"
COLOR_SUCCESS_BORDER = "#86EFAC"

COLOR_WARNING = "#CA8A04"         # yellow-600
COLOR_WARNING_BG = "#FEFCE8"
COLOR_WARNING_BORDER = "#FDE047"

COLOR_ERROR = "#DC2626"           # red-600
COLOR_ERROR_BG = "#FEF2F2"
COLOR_ERROR_BORDER = "#FECACA"

# Code blocks
COLOR_CODE_BG = "#F8FAFC"         # slate-50 on white
COLOR_CODE_TEXT = "#1E293B"       # slate-800
COLOR_CODE_BORDER = "#E2E8F0"

# User/Assistant chat bubbles (subtle)
COLOR_USER_BUBBLE_BG = "#F0F9FF"          # sky-50 -- user gets faint accent
COLOR_USER_BUBBLE_BORDER = "#BAE6FD"      # sky-200
COLOR_ASSISTANT_BUBBLE_BG = "#F8FAFC"     # slate-50 -- assistant neutral
COLOR_ASSISTANT_BUBBLE_BORDER = "#E2E8F0" # slate-200

# Citation cards
COLOR_CITATION_BG = "#F8FAFC"
COLOR_CITATION_BORDER = "#0EA5E9"  # sky-500 accent on left edge

# ----------------------------------------------------------------------
# Typography
# ----------------------------------------------------------------------

FONT_BODY = (
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    'Roboto, "Helvetica Neue", Arial, sans-serif'
)
FONT_HEADING = (
    '"Space Grotesk", "Inter", -apple-system, BlinkMacSystemFont, '
    '"Segoe UI", sans-serif'
)
FONT_MONO = (
    '"JetBrains Mono", "Fira Code", "SF Mono", Menlo, '
    'Monaco, Consolas, monospace'
)

# Sizes -- COMPACT density (smaller than v1)
SIZE_XS = "0.7rem"        # ~11px for tiny captions, pills
SIZE_SM = "0.8125rem"     # 13px secondary
SIZE_BASE = "0.875rem"    # 14px body (GitHub uses ~14px)
SIZE_LG = "1rem"          # 16px lead paragraphs
SIZE_XL = "1.125rem"      # 18px small headings
SIZE_2XL = "1.375rem"     # 22px section headings
SIZE_3XL = "1.75rem"      # 28px page title

LH_TIGHT = "1.2"
LH_NORMAL = "1.5"
LH_RELAXED = "1.65"

LS_TIGHT = "-0.01em"
LS_NORMAL = "0"
LS_WIDE = "0.05em"

# ----------------------------------------------------------------------
# Spacing -- COMPACT
# ----------------------------------------------------------------------

SPACE_XS = "0.2rem"       # 3px
SPACE_SM = "0.375rem"     # 6px
SPACE_BASE = "0.5rem"     # 8px
SPACE_MD = "0.75rem"      # 12px
SPACE_LG = "1rem"         # 16px
SPACE_XL = "1.5rem"       # 24px
SPACE_2XL = "2rem"        # 32px

# ----------------------------------------------------------------------
# Radii -- modern but not playful
# ----------------------------------------------------------------------

RADIUS_SM = "4px"
RADIUS_MD = "6px"
RADIUS_LG = "8px"
RADIUS_PILL = "999px"

# ----------------------------------------------------------------------
# Shadows -- subtle, professional
# ----------------------------------------------------------------------

SHADOW_SM = "0 1px 2px 0 rgba(15, 23, 42, 0.05)"
SHADOW_MD = "0 1px 3px 0 rgba(15, 23, 42, 0.08), 0 1px 2px 0 rgba(15, 23, 42, 0.04)"
SHADOW_LG = "0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 2px 4px -1px rgba(15, 23, 42, 0.04)"


# ----------------------------------------------------------------------
# Branding constants
# ----------------------------------------------------------------------

BRAND_NAME = "AudioLab AI"
BRAND_TAGLINE = "Research companion for audio signal processing"
BRAND_VERSION = "Phase 2 — Batch 12B"
