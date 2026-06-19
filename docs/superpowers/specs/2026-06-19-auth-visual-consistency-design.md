# Auth Visual Consistency & Polish — Design Spec

- **Date:** 2026-06-19
- **Status:** Approved (design); implementation plan pending
- **Scope:** `webapp/static/{login.html, reset.html, auth.css, auth-bg.js}` — paint only
- **Goal:** Make the whole auth experience visually polished and **consistent**. No new features, no
  flow/behavior changes, backend untouched.

## Problem / Context
The auth experience is split across two **independent design systems** that have visibly drifted:

| | `login.html` (sign in / up / forgot) | `reset.html` + `auth.css` + `auth-bg.js` |
|---|---|---|
| Tokens | `--bg0`, `--card-bg`, `--text`/`--muted`, colored `--ok`/`--err` | `--bg-0`, `--glass-bg`, `--text-strong`/`-mid`/`-faint`, **monochrome** |
| Card | 372px · radius 22 · blur 22 | 408px · radius **26** · blur **40** |
| Inputs | 52px · radius 13 · float via top/size | radius **15** · float via translate/scale |
| Submit | radius 13 · green "done" · opacity hover | radius 15 · stays white · brightness+arrow hover |
| Theme toggle | 40px **circle**, top 16 | 42px **rounded-square**, top 22 |
| Messages | **colored** error/success | monochrome |
| Extras | vignette, autofill hardening, sr-only, DOB calendar | password-strength bar |

Side-by-side they look like two different apps. Login also has internal rough edges: the calendar
hardcodes `#fff` instead of tokens, `prefers-reduced-motion` covers only `.card`/spinner, the message
box pops in with no transition, and `.msg.ok` ignores the `--ok` token.

## Decision
**Canonicalize on login's look.** Login's refined system becomes the single visual standard; bring
`reset.html`/`auth.css` up to match it; fix login's internal nits. Keep `login.html` structurally
stable (still self-contained); rewrite `auth.css` to login's spec; leave `auth-bg.js` unchanged.

(Rejected alternatives: a shared-stylesheet refactor — cleanest but highest risk/effort; light
harmonization only — too shallow to be truly consistent.)

## The single visual standard (login's spec, applied everywhere)
- **Tokens:** `--bg0/--bg1`, `--card-bg/--card-brd`, `--text/--muted`, `--input-bg`, `--ring`,
  `--ok/--err` (dark + light) per `login.html`. `auth.css` adopts the same names + values.
- **Card:** 372px (max `calc(100vw - 32px)`), padding `30/28/24`, radius **22**, `blur(22px) saturate(1.25)`,
  shadow `0 35px 80px -25px rgba(0,0,0,.62)` + `inset 0 1px 0 rgba(255,255,255,.10)`, `.authing` ring glow.
- **Inputs:** height 52, padding `18/14/6`, radius **13**, focus `border-color:var(--ring)` +
  `0 0 0 3px color-mix(var(--ring) 16%)`, floating label via `top`/`font-size`, `-webkit-autofill`
  hardening. Peek toggle: 34px, `top:9px`, radius 9.
- **Submit:** height 50, radius **13**, `background:var(--text)`/`color:var(--bg0)`, hover `opacity:.93`
  **+ arrow `translateX(3px)`** (uniform on both), active `scale(.99)`, spinner (login colors),
  **done → `--ok` green + check** (reset gains this).
- **Secondary button** (`.google`): `--input-bg` bg, `--card-brd` border, hover border `--ring`.
- **Theme toggle:** 40px **circle**, `top/right:16px`, `--card-bg` bg, `--card-brd` border, `blur(14px)`,
  hover border `--ring`. Reset's `.top-bar/.icon-btn` restyled to this.
- **Messages:** radius 11, padding `10/12`; `.err` = `--err` text on `color-mix(--err 10%)` + border
  `color-mix(--err 30%)`; `.ok` = **`--ok`** text on `--input-bg`. Short **opacity fade-in** on show.
- **Type scale:** Inter + `font-feature-settings:"cv05","ss01"` + body `letter-spacing:-.006em`;
  H1 18px/650, sub 13px, inputs 15px, labels float to 11px.
- **Background:** sober static radial gradient (done) + login's **`.vignette`** added to reset.
- **Motion:** unify timings; **complete `prefers-reduced-motion`** (inputs, labels, calendar, messages,
  buttons) in both files.

## Per-file changes

### `login.html` — internal polish only (look preserved)
- **Tokenize the calendar's pure white without changing its appearance:** add `--cal-sel-bg`/`--cal-sel-fg`
  (dark `#fff`/`#0a0a0b`, light inverted), use in `.cal-cell.sel`; replace `.cal-title`/`.cal-cell`
  hardcoded `#fff` and the asymmetric `.cal-cell.out` color with tokens. The pure-white calendar look
  stays **identical** — only routed through tokens.
- `.msg` opacity fade-in; `.msg.ok` uses `--ok`.
- Replace the 2-element reduced-motion block with a comprehensive one.
- Submit: add the arrow `translateX(3px)` hover nudge.
- Calendar bg: use a token instead of inline `color-mix(--bg1 …)` (same render).

### `auth.css` — rewrite to the canonical spec
- Replace `:root`/`html.light` token blocks with login's token set (names + values).
- Re-point every rule (`.card`, inputs/labels, `.btn-primary`, `.icon-btn`, `.msg`, `.bar`, `.foot`,
  `.hint`, `.stage`) to the canonical values; switch label float to login's mechanism; add autofill
  hardening; add `.vignette` rule.
- **Strength bar** (`.bar`/`#bar`): restyle to tokens — track `--input-bg`+`--card-brd`, fill graded
  weak→strong `--err`→`--muted`→`--ok`. Kept (reset-only). Not added to login (would be a new feature).

### `reset.html` — minimal markup to match
- Add `<div class="vignette"></div>`; ensure the theme-toggle markup matches login's circle button;
  confirm the `.msg` success/error classes the JS sets map to the new `.ok/.err` styles. No logic changes.

### `auth-bg.js` — unchanged
Already reduced to theme-toggle persistence + card tilt.

## Explicitly preserved
The pure-white calendar appearance; all auth functionality and flows (sign in/up/forgot, reset, Google,
validation, DOB calendar year→month→day, password peek/strength, theme persistence, card tilt); all
backend routes and `webapp/server.py`. Paint-only.

## Out of scope
No new features (no strength bar on login, no new flows/fields), no workspace recolor, no
token-architecture refactor of login into a shared stylesheet.

## Verification
1. `python run.py` with `ENABLE_AUTH=true`; open `/login` and a `/reset?token=…` link.
   - **Parity** in dark + light: card, inputs, submit, circle theme toggle, messages, fonts, vignette.
   - **States:** focus ring, floating labels, password peek, colored error + success, submit
     loading→done(green check), DOB calendar (pure-white selection intact, year→month→day), graded
     strength bar.
   - Theme toggle on each screen → instant cohesive recolor, no flash.
   - `prefers-reduced-motion` → motion stops on inputs/labels/calendar/messages/buttons on both screens.
2. Node compile-check on `login.html` inline script; `pyflakes backend webapp` + `python -m pytest -q`
   (598 passing) as a safety net (frontend-only change).

## Risk / rollback
Front-end/paint-only; no routes/contracts touched. `login.html` keeps its structure. Rollback =
`git restore` the touched files.
