"""
_logo.py  --  AudioLab AI v8 inline SVG logo

v8 changes from v7:
  - logo_header_html: bigger (48px logo + 1.875rem wordmark)
                      now includes "Audio Research Companion" tagline
                      beneath the wordmark, visible by default
  - logo_with_wordmark_html unchanged (still used for sidebar)
  - ALL HTML stays SINGLE-LINE (no </div> bug)
"""

try:
    from . import _theme as T
except ImportError:
    import _theme as T


def logo_svg(size_px: int = 28, color: str = None, dim_bar: bool = True,
             css_class: str = "") -> str:
    """Inline SVG mark. Single-line HTML."""
    color = color or T.COLOR_ACCENT
    crossbar_opacity = "0.5" if dim_bar else "0.7"
    cls_attr = f' class="{css_class}"' if css_class else ""
    return (
        f'<svg{cls_attr} width="{size_px}" height="{size_px}" viewBox="0 0 64 64" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;">'
        f'<g fill="{color}">'
        f'<rect class="alab-wave-bar alab-wave-bar-1" x="6"  y="28" width="3" height="12" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-2" x="12" y="22" width="3" height="20" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-3" x="18" y="14" width="3" height="32" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-4" x="24" y="8"  width="3" height="44" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-5" x="30" y="4"  width="3" height="50" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-6" x="36" y="8"  width="3" height="44" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-7" x="42" y="14" width="3" height="32" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-8" x="48" y="22" width="3" height="20" rx="1.5"/>'
        f'<rect class="alab-wave-bar alab-wave-bar-9" x="54" y="28" width="3" height="12" rx="1.5"/>'
        f'</g>'
        f'<rect x="14" y="42" width="36" height="2" rx="1" '
        f'fill="{color}" opacity="{crossbar_opacity}"/>'
        f'</svg>'
    )


def logo_with_wordmark_html(
    size_px: int = 24,
    on_dark: bool = False,
    show_tagline: bool = True,
) -> str:
    """Compact logo + wordmark side-by-side. Used in SIDEBAR. Single-line HTML."""
    if on_dark:
        text_color = T.COLOR_SIDEBAR_TEXT
        sub_color = T.COLOR_SIDEBAR_TEXT_DIM
        logo_color = T.COLOR_ACCENT
    else:
        text_color = T.COLOR_TEXT
        sub_color = T.COLOR_TEXT_DIM
        logo_color = T.COLOR_ACCENT

    tagline_html = ""
    if show_tagline:
        body_safe = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
        tagline_html = (
            f"<span style=\"font-family:'Inter',{body_safe};"
            f'font-size:0.65rem;color:{sub_color};'
            f'letter-spacing:0.08em;text-transform:uppercase;'
            f'font-weight:500;margin-top:1px;">'
            f'Research Companion</span>'
        )

    heading_safe = "'Inter', system-ui, -apple-system, sans-serif"
    return (
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'{logo_svg(size_px, color=logo_color)}'
        f'<div style="display:flex;flex-direction:column;line-height:1.1;">'
        f"<span style=\"font-family:'Space Grotesk',{heading_safe};"
        f'font-size:1.15rem;font-weight:600;'
        f'color:{text_color};letter-spacing:-0.01em;">'
        f'{T.BRAND_NAME}</span>'
        f'{tagline_html}'
        f'</div>'
        f'</div>'
    )


def logo_header_html() -> str:
    """v8 BIGGER: 48px logo + 1.875rem wordmark + ALWAYS-VISIBLE tagline.

    Layout:
        [48px waveform]   AudioLab AI
                         AUDIO RESEARCH COMPANION   (tagline, always visible)
                         [thin sky-blue gradient bar]
    """
    heading_safe = "'Inter', system-ui, -apple-system, sans-serif"
    body_safe = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    return (
        f'<div class="alab-brand" '
        f'style="display:flex;align-items:center;gap:16px;">'
        # Logo on left: BIGGER (48px)
        f'{logo_svg(48, color=T.COLOR_ACCENT, css_class="alab-brand-logo")}'
        # Wordmark + tagline stack on right
        f'<div style="display:flex;flex-direction:column;line-height:1.15;">'
        # Brand name: BIGGER (1.875rem)
        f"<span style=\"font-family:'Space Grotesk',{heading_safe};"
        f'font-size:1.875rem;font-weight:600;'
        f'color:{T.COLOR_TEXT};letter-spacing:-0.015em;'
        f'line-height:1.1;">'
        f'{T.BRAND_NAME}</span>'
        # Tagline: NEW in v8, always visible
        f"<span style=\"font-family:'Inter',{body_safe};"
        f'font-size:0.72rem;font-weight:500;'
        f'color:{T.COLOR_TEXT_DIM};letter-spacing:0.12em;'
        f'text-transform:uppercase;margin-top:4px;">'
        f'Audio Research Companion</span>'
        # Sky-blue accent underline
        f'<div class="alab-brand-underline" '
        f'style="height:2px;width:48px;'
        f'background:linear-gradient(90deg,{T.COLOR_ACCENT} 0%,'
        f'{T.COLOR_ACCENT_HOVER} 100%);'
        f'border-radius:2px;margin-top:8px;'
        f'transition:width 0.3s cubic-bezier(0.4,0,0.2,1);"></div>'
        f'</div>'
        f'</div>'
    )
