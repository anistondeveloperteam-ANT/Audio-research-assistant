# UI Re-skin Prompt — "Research Assistant" (change BACKGROUND + COLORS only)

> Copy everything below the line into your design tool (Claude / Stitch / v0 / etc.). It captures the
> whole product so the redesign keeps **every feature, layout, and interaction identical** and changes
> **only the background and the color palette**.

---

## 0) YOUR TASK — read this first (HARD CONSTRAINT)

You are re-skinning an existing web app called **Research Assistant**. **Change ONLY the background and
the color scheme.** Do **NOT** change anything else: not the layout, the 3-column structure, spacing,
component sizes, border-radii, fonts, font-sizes, animations/timings, copy/labels, icons, or any
click / hover / keyboard / drag behavior. Every button, panel, badge, drawer, modal, and interaction
must stay exactly where and how it is today — you are repainting the walls, not moving them.

**You may touch only:**
1. The CSS color custom-properties (palette tokens) for **both** themes in `webapp/static/styles.css`
   (`:root` = dark, `html.light` = light).
2. The `#gradient` radial-gradient colors (`--grad-a`, `--grad-b`, `--bg-0`).
3. The animated background network's material **colors/opacities** in `webapp/static/app-bg.js`
   (`applyTheme3D()` / `start2D()`) — these are JS hex literals, **not** CSS vars, so you must edit
   them there. (The auth pages read `--net-*` CSS vars instead; see §3.)
4. A few **hard-coded color** spots: the code/console terminal backgrounds, the syntax-accent color,
   the console log status colors, and the single amber "library warning" accent.

**You may NOT touch:** any layout/structure/behavior — see §2.

Deliver: the rewritten `:root` + `html.light` token blocks, the updated `#gradient` colors, the updated
`app-bg.js` material colors, and the recolored terminal/accent spots. Keep both **dark (default)** and
**light** themes fully working (the app flips between them with the `html.light` class).

---

## 1) WHAT THE APP IS

A self-hosted AI research assistant. A dark, glassmorphic, single-page workspace: a left **sidebar**
(conversations + library + model + account), a center **chat thread** with a composer, and a right
**sources drawer**. It answers questions with cited evidence, shows a live "reasoning" timeline, and —
for coding tasks — an autonomous **code-agent** timeline with an in-card IDE + sandbox console. There
is a separate **login/auth** screen. Two themes: **dark** (default) and **light**, toggled by a button
(persisted in `localStorage`).

---

## 2) DESIGN SYSTEM TO PRESERVE EXACTLY (do NOT change any of this)

- **Shell:** a 3-column flex layout — `.app` (flex row, `gap:18px`, `padding:18px`, full viewport) →
  `.sidebar` (266px, collapses to 74px) · `.center` (flex:1) · `.drawer` (340px, collapses to 0).
  Two fixed full-viewport background layers sit behind everything: `#gradient` (animated radial
  gradient) and `#three` (the animated network canvas).
- **Glassmorphism:** translucent panels with `backdrop-filter: blur(10–26px) saturate(135–140%)`,
  thin 1px borders, large soft drop-shadows, and a 1px inset top highlight. **Keep the blur amounts,
  the layering, the shadow geometry, and the inset-highlight** — only their *tint/color* may change.
- **Radii:** cards/panels 20–24px, buttons 10–14px, pills 999px; chat bubbles use asymmetric radii.
  **Keep all radius values.**
- **Spacing scale, paddings, gaps, margins:** keep exactly.
- **Fonts:** **Inter** for everything, **JetBrains Mono** for code/console/numbers. Keep families,
  sizes, weights, line-heights.
- **Animations / @keyframes / timings:** `drift` (34s background), card pop-in, shimmer, the 3D
  spinning "thinking" cube, the reasoning orb orbit, hover tilt, flash/glow, etc. — keep all timings
  and motion; only colors may change. Respect `prefers-reduced-motion` exactly as today.
- **Animated 3D network (app-bg.js):** keep the geometry/topology (≈130 nodes, ≈210 nodes variant,
  k=3 nearest-neighbor edges, 28 travelling pulses, camera fov 62 / z 120, fog, mouse parallax) and
  the Canvas-2D fallback. **Only the node/line/pulse COLORS and opacities change.**
- **Theme mechanism:** the `html.light` class toggle, the `localStorage 'ara-theme'` persistence, the
  pre-paint inline `<head>` script (avoids flash), the `app-bg.js` MutationObserver that re-themes the
  canvas, and the sun/moon icon swap — keep all of it.
- **Every component, label, icon, and interaction in §4–§5, and all responsive breakpoints
  (1080 / 820 / 620px).**

---

## 3) THE CURRENT PALETTE (what you are replacing)

**Dark (`:root`, default):**
- Page `--bg-0: #070708`; drifting radial gradient `--grad-a: #141416`, `--grad-b: #0a0a0b`.
- Text `--text-strong: #fff` → `--text-mid: rgba(255,255,255,.60)` → `--text-faint: rgba(255,255,255,.38)`.
- Glass films `--glass-bg: rgba(255,255,255,.035)` / `.08` / `.12`; borders `--glass-brd: rgba(255,255,255,.12)` / `.22`.
- Panels `--panel-bg / --content-bg: rgba(16,16,20,.46–.50)`; menus `--menu-bg: rgba(18,18,20,.92)`.
- **Solid-button accent is INVERTED:** `--btn-bg: #fff`, `--btn-text: #0a0a0b` (white button, near-black text).
- `--user-bubble: rgba(44,44,52,.44)`; `--danger: #ef4444`; large soft shadow `0 24px 70px -24px rgba(0,0,0,.85)`.
- Network background (app-bg.js): white additive-blended dots + faint lines + travelling pulses
  (opacity ≈ .68 / .14 / .75).

**Light (`html.light`):**
- `--bg-0: #fff`; gradient `--grad-a: #ededee`, `--grad-b: #fbfbfc`.
- Text near-black `#0a0a0b` → `rgba(10,10,11,.58)` → `.40`.
- Glass becomes white-on-white `rgba(255,255,255,.42/.72)` + dark tint `rgba(10,10,11,.06)`.
- Solid button inverts to **black** (`--btn-bg: #0a0a0b`, `--btn-text: #fff`); `--danger: #dc2626`.
- Network re-colors to soft greys at lower opacity.

**Hard-coded color spots (not CSS vars):**
- Code/console "terminal" blocks: `#0d0d12` (`.code`, `.mdcode`, `.answer-body pre`), `#0b0b0e`
  (`.console`); in light mode forced to solid dark `#0f1117`.
- macOS traffic-light dots `#ff5f56` / `#febc2e` / `#28c840`; syntax accent `#82aaff`.
- Console log statuses (`.ok` green / `.warn` amber / `.stage` blue); the **only** non-monochrome UI
  accent is the amber "incomplete library" warning `rgba(234,179,8,*)` / `#d4a017`.

> The current aesthetic is **monochrome glass on a near-black animated connectome.** You are free to
> introduce a real color identity (e.g. a brand accent, a tinted background, a colored gradient/
> network) — just produce a cohesive **dark + light** pair with good contrast (WCAG AA text), keep the
> "frosted glass over an animated background" feeling, and only change colors/background.

### NEW PALETTE — specify your target (fill in or let the tool propose)
Provide a value for each token in **both** themes (or say "propose a cohesive scheme"):
`--bg-0, --grad-a, --grad-b, --text-strong, --text-mid, --text-faint, --glass-bg, --glass-bg-2,
--glass-bg-3, --panel-bg, --content-bg, --menu-bg, --glass-brd, --glass-brd-2, --btn-bg, --btn-text,
--user-bubble, --danger, --shadow`; plus the network dot/line/pulse colors+opacity (dark & light); the
terminal background(s); the syntax accent; and the warning amber.

---

## 4) EVERY SCREEN, COMPONENT & INTERACTION (keep all behavior; recolor only)

### 4.1 Login / auth screen (`login.html`, `reset.html`)
- A centered ~372px **glass card** over an **animated "waves" canvas** background + a vignette overlay;
  a round 40px **theme-toggle** button (top-right, sun/moon swap).
- Card: brand row (magnifier logomark + **"Research Assistant"**), a **subtitle that swaps per mode**
  ("Sign in to continue." / "Create your account." / "Reset your password."), and a `<form>`.
- **Modes** (one card, no reload): **Sign in** (Email-or-username + Password), **Sign up** (Username +
  Email + extras + Password), **Forgot password** (Email-or-username), and **Continue with Google**.
  Floating labels; a success/error **message box**; mode-switch links at the bottom; password submit.
- `reset.html` is the password-reset landing page (own auth.css + a node-network + wireframe torus
  background). **Re-skin both** to the same new palette/background, keep the flow.
- **Change:** card glass tint, background animation colors, accents, message-box colors. **Keep:**
  card size, fields, modes, validation, labels, the animated-background motion.

### 4.2 Workspace shell + left sidebar
- **Sidebar head:** logomark (32px magnifier) + **"Research Assistant"** title + **collapse button**
  (chevron). Click collapse → toggles `.collapsed` (266px↔74px, hides labels, centers icons, rotates
  chevron, persists `localStorage 'ara-sidebar'`). On ≤820px a hamburger `#menuToggle` does the same.
- **New chat** (solid accent button, + icon) → creates a session, selects it, focuses composer.
  Keyboard **Ctrl/Cmd+K** = new chat.
- **Add papers** (glass button) → opens the upload modal (hidden when local library disabled).
- **History list:** UPPERCASE date-group labels **Today / Yesterday / Previous 7 days / Earlier**, then
  conversation rows (chat-dot + title). **Hover** a row → reveals **Rename** (pencil) + **Delete**
  (trash, turns danger-red) actions. Click a row → selects it (`.active` = accent left-border +
  stronger bg). Empty state: "No conversations yet…".
- **Footer:** **Library pill** ("**N** papers indexed" + caret → opens Library modal); **Model picker**
  (`#modelBtn` shows model name + vendor → opens a `.model-menu` listbox; each option shows a check on
  the current one, dimmed if unavailable; selecting toasts "Model switched to …"); **Account row**
  (avatar initials + name) with **theme toggle** (sun/moon) and **logout** (shown when authenticated).
- **Top bar (center):** hamburger (mobile), conversation title, and a **"Sources"** toggle (book icon)
  that opens the right drawer for the latest answer.

### 4.3 Chat thread + answer rendering (center)
- Thread = centered 820px column of exchanges. **User message:** right-aligned bubble (`--user-bubble`)
  with a hover **Edit** pencil and (if multiple) a **question-version switcher** (‹ i/total ›).
- **Assistant answer** stacks three blocks: a live **"thinking" cube** ("Thinking…/Regenerating…"); a
  **reasoning timeline** (`.reason`) — an orb/counter header ("Reasoning…" → "Reasoned for X.Xs",
  click to collapse when done), steps with spinner→check nodes, "reason chips" like *"Found N relevant
  sources"* (click → opens sources), and a raw monospace thinking stream; then the **answer card**.
- **Answer card:** a **grade badge** (🟢 *From your library* `badge-lib` / 🟡 *Library + web*
  `badge-mix` / 🔵 *From the web* `badge-web`), a **speed badge** ("X.Xs · model") or **"From memory ·
  NN%"** cache badge, the rendered **markdown** body, inline **citation chips** `[n]`
  (`.chip-pdf`/`.chip-web`; hover → popover with title/section/pages/snippet; click → open URL or focus
  the source card), an **"unverified"** superscript for ungrounded bits, an optional **low-confidence**
  footer, and an **action bar** (Copy / Regenerate / Delete + an **answer-version switcher** ‹ i/total ›).
- **Markdown code** renders as a `.mdcode` card: traffic-light dots + language label + **Copy** button +
  syntax-highlighted block (dark terminal).
- **Change:** bubble/card/badge tints, chip colors, the grade-badge colors, the terminal & syntax
  colors. **Keep:** all of the above structure, the citation popover behavior, version switching, the
  card hover-tilt, and the markdown rendering.

### 4.4 Composer (bottom)
- A glass pill: **textarea** ("Ask anything — or give it a coding task…"), a **Fast / Deep** segmented
  toggle (active = `.on`, persisted `'ara-mode'`), and a **Send** button (turns into a red **Stop**
  while streaming; disabled when empty). **Enter** sends, **Shift+Enter** newline. A **"Jump to latest"**
  pill appears when scrolled up. **Welcome/empty state:** hero magnifier + a heading ("What do your
  papers say?" / "What would you like to research?") + a 2×2 grid of **example** prompt buttons.
- **Change:** pill/segment/send colors, the Stop-state red, example-card tints. **Keep:** the layout,
  the toggle, the send/stop behavior, the helper text, the welcome grid.

### 4.5 Code-agent timeline (coding tasks)
- A collapsible **`.agent` panel**: header (star icon + title "Agent — writing & verifying code" →
  "solved & verified" / "best attempt", an "Attempt N" pill, chevron) + a **steps list** (spinner →
  check / red-X / pending nodes; labels like "Read the requirements", "Wrote N correctness tests",
  "Built a reference oracle", "Verifying against N held-out checks", "Running in the sandbox").
- One **in-card IDE block** (`.code`: window dots + "solution.py" + "python" badge + Copy; highlighted
  code) and one **sandbox console** (`.console`: dots + "SANDBOX OUTPUT" + stdout/stderr) — both are
  **replaced in place** each attempt (single live result). A **code footer** shows a "Verified" /
  "Best attempt" status tag + a **"Run again"** button.
- **Change:** terminal backgrounds, status-tag colors (verified green / partial amber), node colors,
  syntax/console colors. **Keep:** the timeline, the live-replace behavior, the IDE/console layout.

### 4.6 Sources drawer (right)
- Slides in (340px) when "Sources" / a citation / a reason-chip is clicked. Header: book icon +
  "Sources" + a **count pill** + close-X. An **answer-nav** (‹ question · i/total ›) appears when more
  than one answer has sources. Body = **source cards**: a numbered badge (`.pdf` = solid inverted,
  `.web` = outlined), a **type chip** ("Paper" / "Web" / "GitHub" / "PDF"), title, meta (section · pp.
  N–M, or published date / path:line / p.N), a clamped snippet with **Show more/less**, and an **Open
  source** link. A card with a URL opens it on click (unless selecting text or clicking the link).
  Citing `[n]` **flashes** the matching card. **Change:** drawer/card tints, the num-badge & type-chip
  colors, the flash-glow color. **Keep:** open/close, nav, expand, focus-flash.

### 4.7 Library + upload modals
- **Add-papers modal:** title + a **dropzone** ("Drop PDFs here, or browse", "Up to 50 MB per file"),
  a per-file list (icon, name/size, **progress bar**, status "Queued… / Indexing… / Indexed ✓ / Already
  indexed / Failed"), a streaming **upload log** (ok/warn/stage lines), and a footer summary + **Done**.
  Drag-over highlights the dropzone; closing mid-ingest confirms a cancel.
- **Library modal:** title + "N papers · embedded and searchable", a list of rows (icon, name, "N
  chunks", delete trash) with an **"incomplete"** state (amber "not embedded" + a "⚠ N half-done" banner
  + "Remove half-done"). Empty/loading/error states.
- **Change:** modal glass, progress-bar fill, status colors (success green / error red / amber warning),
  log line colors. **Keep:** the dropzone, the upload flow, the lists, the confirms.

### 4.8 Version tree + per-message actions
- **Question-version switcher** (‹ i/total › above a user bubble) and **answer-version switcher** (‹
  i/total › in the action bar) — appear only when >1 version; arrows disable at the ends. **Edit** a
  question (pencil → inline textarea with Cancel / **Save & resend**, Enter saves, Esc cancels) creates
  a new question version; **Regenerate** creates a new answer version; **Delete** removes the exchange
  (with confirm). All blocked (with a toast) while streaming. **Change:** switcher/toolbar icon colors,
  the editing-ring color. **Keep:** all version logic and the toolbar.

---

## 5) GLOBAL STATES (recolor, don't restructure)
Empty / welcome · thinking (pre-stream cube) · reasoning live vs done-collapsed · streaming (answer with
a "▍" caret, **red Stop** button) · stopped · error toast · finalized (action bar + badges) ·
low-confidence note · cached badge · agent step running/done/fail/pending · agent verified vs best-
attempt · drawer open/closed · upload queued/indexing/done/error · library normal/incomplete · sidebar
expanded/collapsed · **dark vs light theme** · reduced-motion. A shared **toast** appears bottom-center.

---

## 6) OUTPUT FORMAT (what to hand back)
1. The rewritten **`:root`** (dark) and **`html.light`** (light) CSS custom-property blocks for
   `styles.css` — every color token, both themes.
2. The updated **`#gradient`** radial-gradient colors.
3. The updated **`app-bg.js`** `applyTheme3D()` (and `start2D()`) node/line/pulse **colors + opacities**
   for dark and light (these are JS hex literals — recolor them there).
4. The recolored hard-coded spots: terminal backgrounds (`.code`/`.console`/`.mdcode`/`pre`, dark +
   light), syntax accent, console log statuses, and the amber library-warning color.
5. A short note confirming **no layout/structure/behavior was changed** — only background + colors.

**Do not output any other file changes.** Keep both themes accessible (AA contrast) and the
"frosted glass over an animated background" character intact.
