# Auth Visual Consistency & Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the login and reset auth screens look like one premium, consistent design by canonicalizing on `login.html`'s visual system, and fix login's internal polish nits — paint only.

**Architecture:** `login.html` is the single visual standard (self-contained, inline `<style>`/`<script>`). `reset.html` is restyled by rewriting its shared stylesheet `auth.css` to login's spec, plus two presentational markup/JS tweaks in `reset.html`. `auth-bg.js` (reset's theme-toggle + card tilt) is untouched. No backend, routes, or flows change.

**Tech Stack:** Vanilla HTML/CSS/JS, no build step. CSS custom properties for theming (`html.light` flips dark→light). Inter font. Verification via the existing Python test suite + `pyflakes` + a Node syntax check on the inline JS + a manual visual pass.

## Global Constraints
- **Paint only.** No new features, no new fields/flows, no backend or route changes. (From the spec.)
- **Preserve the pure-white calendar appearance** in `login.html` exactly — the tokenization must render identically (dark = pure `#fff` text + `#fff`/`#0a0a0b` selection). (User requested pure white in a prior turn.)
- **No new dependencies.** (Project rule.)
- **`auth-bg.js` is not modified.**
- After every change run: `python -m pytest -q` (expect 598 passed, 3 skipped) and `pyflakes backend webapp` (expect clean). These are a safety net — this change touches no Python, so they must stay green, not improve.
- Frontend has no build step: edit files in `webapp/static/` and reload.
- `auth.css` is loaded **only** by `reset.html` (login is self-contained; `index.html` uses `styles.css`). So `auth.css` edits affect only the reset page.

---

## File Structure
- **`webapp/static/login.html`** — self-contained login (sign in / up / forgot) + DOB calendar. *Modify the inline `<style>` only* (tokens + calendar + messages + submit + reduced-motion). No JS/markup change.
- **`webapp/static/auth.css`** — the reset page's stylesheet. *Full rewrite* to login's canonical spec.
- **`webapp/static/reset.html`** — reset page markup + inline JS. *Two small edits*: add the vignette layer; presentational wiring for colored success + graded strength bar.
- **`webapp/static/auth-bg.js`** — unchanged.

---

## Task 0: Establish a clean baseline (commit current uncommitted UI work)

The working tree already contains the approved-but-uncommitted sober-background and DOB-calendar work across these files. Commit it first so the polish edits land as clean, correctly-attributed increments. (If you prefer to split/relabel these yourself, do so before Task 1 — they cannot be cleanly split per-file because `login.html` contains both prior changes.)

**Files:**
- Modify (commit as-is): `webapp/server.py`, `webapp/static/index.html`, `webapp/static/styles.css`, `webapp/static/auth.css`, `webapp/static/login.html`, `webapp/static/reset.html`; Deleted: `webapp/static/app-bg.js`.

- [ ] **Step 1: Confirm what's pending**

Run: `git -C . status --short`
Expected: ` M webapp/server.py`, `D  webapp/static/app-bg.js`, ` M webapp/static/auth.css`, ` M webapp/static/index.html`, ` M webapp/static/login.html`, ` M webapp/static/reset.html`, ` M webapp/static/styles.css` (and nothing else uncommitted besides docs).

- [ ] **Step 2: Run the safety net before committing**

Run: `python -m pytest -q`
Expected: `598 passed, 3 skipped`.
Run: `pyflakes backend webapp`
Expected: no output (clean).

- [ ] **Step 3: Commit the baseline**

```bash
git add webapp/server.py webapp/static/index.html webapp/static/styles.css \
        webapp/static/auth.css webapp/static/login.html webapp/static/reset.html \
        webapp/static/app-bg.js
git commit -m "feat(ui): sober static backgrounds + custom date-of-birth calendar

Replace the animated 3D/canvas backgrounds with a calm static gradient on
login, chat, and reset; delete the unused app-bg.js network engine; add a
custom premium DOB calendar (year -> month -> day, pure-white selection)
to the login signup form."
```

- [ ] **Step 4: Verify a clean tree**

Run: `git -C . status --short`
Expected: empty (no modified tracked files).

---

## Task 1: Polish `login.html` internally (look preserved)

**Files:**
- Modify: `webapp/static/login.html` (inline `<style>` only)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: the canonical message style (`.msg.ok` = green tint, `.msg.err` = red tint, `.msg.show` fade), the canonical reduced-motion rule, and the calendar-token pattern — Task 2 mirrors these visual rules in `auth.css`.

All edits are exact string replacements in the `<style>` block. The calendar edits **must render identically** — they only route the existing hardcoded `#fff`/`#0a0a0b` through new tokens.

- [ ] **Step 1: Add calendar + selection tokens to the dark `:root` block**

Find (lines ~18-19):
```css
    --ok:#22c55e; --err:#fb7185;
    color-scheme:dark;
```
Replace with:
```css
    --ok:#22c55e; --err:#fb7185;
    --cal-ink:#fff; --cal-out:rgba(255,255,255,.42); --cal-sel-bg:#fff; --cal-sel-fg:#0a0a0b;
    --cal-bg:color-mix(in srgb,var(--bg1) 93%,transparent);
    color-scheme:dark;
```

- [ ] **Step 2: Add the light-mode calendar token flips**

Find (lines ~26-27):
```css
    --ok:#16a34a; --err:#e11d48;
    color-scheme:light;
```
Replace with:
```css
    --ok:#16a34a; --err:#e11d48;
    --cal-ink:var(--text); --cal-out:var(--muted); --cal-sel-bg:#0a0a0b; --cal-sel-fg:#fff;
    color-scheme:light;
```
(`--cal-bg` is inherited from `:root`; it references `--bg1`, which already flips per theme.)

- [ ] **Step 3: Point the calendar background at the token**

Find:
```css
    background:color-mix(in srgb,var(--bg1) 93%,transparent);
```
Replace with:
```css
    background:var(--cal-bg);
```

- [ ] **Step 4: Tokenize the calendar title color and drop its light override**

Find:
```css
  .cal-title{flex:1;text-align:left;background:none;border:0;cursor:pointer;color:#fff;
```
Replace with:
```css
  .cal-title{flex:1;text-align:left;background:none;border:0;cursor:pointer;color:var(--cal-ink);
```
Then find and delete this line entirely:
```css
  html.light .cal-title{color:var(--text)}
```

- [ ] **Step 5: Tokenize the day-cell colors and drop the light overrides**

Find:
```css
  .cal-cell{height:38px;border:0;background:transparent;color:#fff;cursor:pointer;border-radius:10px;
```
Replace with:
```css
  .cal-cell{height:38px;border:0;background:transparent;color:var(--cal-ink);cursor:pointer;border-radius:10px;
```
Find:
```css
  .cal-cell.out{color:rgba(255,255,255,.42)}
```
Replace with:
```css
  .cal-cell.out{color:var(--cal-out)}
```
Find:
```css
  .cal-cell.sel{background:#fff;color:#0a0a0b;font-weight:600}
```
Replace with:
```css
  .cal-cell.sel{background:var(--cal-sel-bg);color:var(--cal-sel-fg);font-weight:600}
```
Then find and delete these three light-override lines entirely:
```css
  html.light .cal-cell{color:var(--text)}
  html.light .cal-cell.out{color:var(--muted)}
  html.light .cal-cell.sel{background:#0a0a0b;color:#fff}
```

- [ ] **Step 6: Make `.msg.ok` a green tint (symmetric with `.msg.err`)**

Find:
```css
  .msg.ok{color:var(--text);background:var(--input-bg);border-color:var(--card-brd)}
```
Replace with:
```css
  .msg.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 10%,transparent);border-color:color-mix(in srgb,var(--ok) 30%,transparent)}
```

- [ ] **Step 7: Add a fade-in to messages**

Find:
```css
  .msg.show{display:block}
```
Replace with:
```css
  .msg.show{display:block;animation:msgIn .2s ease}
```
Then add the keyframes immediately after the `.msg a{...}` line (`.msg a{color:var(--text);font-weight:600}`):
```css
  @keyframes msgIn{from{opacity:0;transform:translateY(-2px)}to{opacity:1;transform:none}}
```

- [ ] **Step 8: Add the submit-arrow hover nudge**

Find:
```css
  .s-label{display:flex;align-items:center;gap:8px}
```
Replace with:
```css
  .s-label{display:flex;align-items:center;gap:8px}
  .s-label svg{transition:transform .2s ease}
  .submit:hover .s-label svg{transform:translateX(3px)}
```

- [ ] **Step 9: Make reduced-motion comprehensive**

Find:
```css
  @media (prefers-reduced-motion:reduce){.card{transition:none}.s-spin{animation:none}}
```
Replace with:
```css
  @media (prefers-reduced-motion:reduce){*{transition:none !important;animation:none !important}}
```

- [ ] **Step 10: Verify the inline JS still parses**

Run:
```bash
node -e "const fs=require('fs');const h=fs.readFileSync('webapp/static/login.html','utf8');const re=/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;let m,ok=true;while((m=re.exec(h))){try{new Function(m[1]);}catch(e){ok=false;console.log('ERR',e.message);}}console.log(ok?'JS OK':'JS FAIL');"
```
Expected: `JS OK` (no JS was changed; this confirms the `<style>` edits didn't corrupt the file).

- [ ] **Step 11: Run the safety net**

Run: `python -m pytest -q` → expect `598 passed, 3 skipped`.
Run: `pyflakes backend webapp` → expect clean.

- [ ] **Step 12: Visual check (manual)**

Run: `python run.py` (with `ENABLE_AUTH=true`); open `http://localhost:8600/login`.
Confirm, in **both** dark and light (toggle top-right):
- Open the signup form → **Date of birth** → the calendar selection is still **pure white** in dark (white pill, near-black number), inverted in light. Year→month→day flow unchanged. (Tokenization must look identical to before.)
- Trigger a success message (e.g. forgot-password flow) → it's now **green-tinted** and **fades in**; an error message is red-tinted and fades.
- Hover the **Sign in** button → the arrow nudges right ~3px.
- With OS "reduce motion" on → calendar/message/spinner animations and hovers don't animate.

- [ ] **Step 13: Commit**

```bash
git add webapp/static/login.html
git commit -m "style(login): tokenize calendar colors, polish messages + motion

Route the calendar's pure-white text/selection through --cal-* tokens
(identical render), make .msg.ok a green tint symmetric with .msg.err,
fade messages in, nudge the submit arrow on hover, and make
prefers-reduced-motion comprehensive."
```

---

## Task 2: Bring the reset screen to the canonical look

**Files:**
- Modify (full rewrite): `webapp/static/auth.css`
- Modify: `webapp/static/reset.html` (add vignette div; presentational wiring for colored success + graded strength bar)

**Interfaces:**
- Consumes: the canonical visual rules established in Task 1 (`.msg` colored+fade, reduced-motion, card/input/submit/toggle spec from `login.html`). This task reproduces them for reset's class names (`.btn-primary`, `.icon-btn`, `.bar`, etc.).
- Produces: a reset page visually identical to login. Terminal task; nothing depends on it.

- [ ] **Step 1: Replace `auth.css` with the canonical stylesheet**

Overwrite `webapp/static/auth.css` with exactly this content:
```css
/* Premium auth design system — used by reset.html. Matches login.html's look.
   Monochrome glass; tokens flip between dark (default) and light (html.light). */
:root{
  --bg0:#08090c; --bg1:#13151b;
  --card-bg:rgba(18,20,26,.55); --card-brd:rgba(255,255,255,.12);
  --text:#f3f4f6; --muted:#9aa0aa; --input-bg:rgba(255,255,255,.05);
  --ring:rgba(255,255,255,.65);
  --ok:#22c55e; --err:#fb7185;
  color-scheme:dark;
}
html.light{
  --bg0:#eef1f6; --bg1:#dde2ec;
  --card-bg:rgba(255,255,255,.62); --card-brd:rgba(15,20,30,.10);
  --text:#14181e; --muted:#5b626d; --input-bg:rgba(10,15,25,.04);
  --ring:rgba(20,30,45,.45);
  --ok:#16a34a; --err:#e11d48;
  color-scheme:light;
}

*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;}
body{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  font-feature-settings:"cv05","ss01"; letter-spacing:-.006em;
  background:var(--bg0); color:var(--text);
  overflow:hidden; -webkit-font-smoothing:antialiased;
  transition:background .6s, color .6s;
}

/* ---- sober static background + vignette (matches login) ---- */
#gradient{position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(130% 130% at 25% 12%,var(--bg1),var(--bg0) 58%);}
.vignette{position:fixed;inset:0;z-index:1;pointer-events:none;
  background:radial-gradient(120% 120% at 50% 38%,transparent 46%,rgba(0,0,0,.34) 100%);}
html.light .vignette{background:radial-gradient(120% 120% at 50% 38%,transparent 52%,rgba(20,30,45,.10) 100%);}

/* ---- theme toggle (40px circle, matches login) ---- */
.top-bar{position:fixed;top:16px;right:16px;z-index:20;display:flex;gap:10px;align-items:center;}
.icon-btn{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;cursor:pointer;
  background:var(--card-bg);border:1px solid var(--card-brd);color:var(--text);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);transition:border-color .2s;}
.icon-btn:hover{border-color:var(--ring);}
.icon-btn svg{width:18px;height:18px;}
.moon{display:none;} .light .moon{display:block;} .light .sun{display:none;}

/* ---- stage / card (matches login) ---- */
.stage{position:relative;z-index:10;height:100%;display:grid;place-items:center;perspective:1400px;padding:24px;}
.card{
  position:relative;width:372px;max-width:calc(100vw - 32px);padding:30px 28px 24px;border-radius:22px;
  background:var(--card-bg);border:1px solid var(--card-brd);
  backdrop-filter:blur(22px) saturate(1.25);-webkit-backdrop-filter:blur(22px) saturate(1.25);
  box-shadow:0 35px 80px -25px rgba(0,0,0,.62),inset 0 1px 0 rgba(255,255,255,.10);
  transform-style:preserve-3d;transition:box-shadow .35s ease;will-change:transform;
}
.card.authing{box-shadow:0 35px 80px -25px rgba(0,0,0,.62),inset 0 1px 0 rgba(255,255,255,.10),0 0 0 1px var(--ring),0 0 40px -6px var(--ring);}

.brand-row{display:flex;align-items:center;gap:11px;}
.mark{width:34px;height:34px;border-radius:11px;flex:none;border:1px solid var(--card-brd);
  display:grid;place-items:center;background:var(--input-bg);box-shadow:inset 0 1px 0 rgba(255,255,255,.10);}
.mark svg{width:18px;height:18px;color:var(--text);}
h1{font-size:18px;font-weight:650;letter-spacing:-.022em;line-height:1.1;}
.tag{margin:9px 0 0;font-size:13px;color:var(--muted);font-weight:400;line-height:1.5;}

form{margin-top:20px;display:flex;flex-direction:column;gap:12px;}

.field{position:relative;}
.field input{
  width:100%;height:52px;padding:18px 14px 6px;font-size:15px;font-family:inherit;
  color:var(--text);background:var(--input-bg);border:1px solid var(--card-brd);
  border-radius:13px;outline:none;transition:border-color .18s,box-shadow .18s;
}
.field input:focus{border-color:var(--ring);box-shadow:0 0 0 3px color-mix(in srgb,var(--ring) 16%,transparent);}
.field label{position:absolute;left:14px;top:16px;font-size:15px;color:var(--muted);
  pointer-events:none;transition:top .15s ease,font-size .15s ease,letter-spacing .15s ease;}
.field input:focus + label,
.field input:not(:placeholder-shown) + label,
.field input:-webkit-autofill + label{top:7px;font-size:11px;letter-spacing:.03em;}
.field input:-webkit-autofill,
.field input:-webkit-autofill:hover,
.field input:-webkit-autofill:focus{
  -webkit-text-fill-color:var(--text);
  -webkit-box-shadow:0 0 0 1000px var(--input-bg) inset;
  caret-color:var(--text);
  transition:background-color 9999s ease 0s,border-color .18s,box-shadow .18s;
}

.has-peek input{padding-right:46px;}
.peek{position:absolute;right:8px;top:9px;width:34px;height:34px;border:none;background:transparent;
  cursor:pointer;color:var(--muted);display:grid;place-items:center;border-radius:9px;transition:color .2s;}
.peek:hover{color:var(--text);}
.peek svg{width:18px;height:18px;}
.peek .eye-off{display:none;}
.peek.on .eye{display:none;}
.peek.on .eye-off{display:block;}

/* ---- primary button (matches login) ---- */
.btn-primary{
  position:relative;margin-top:4px;height:50px;border:none;border-radius:13px;cursor:pointer;
  background:var(--text);color:var(--bg0);font-family:inherit;font-size:15px;font-weight:600;
  letter-spacing:.01em;display:flex;align-items:center;justify-content:center;gap:8px;
  transition:transform .1s,opacity .2s,background .25s;overflow:hidden;
}
.btn-primary:hover{opacity:.93;}
.btn-primary:active{transform:scale(.99);}
.btn-primary svg{width:17px;height:17px;transition:transform .2s ease;}
.btn-primary:hover svg.arrow{transform:translateX(3px);}
.btn-primary .spinner,.btn-primary .check{display:none;}
.btn-primary.loading .label,.btn-primary.loading svg.arrow{display:none;}
.btn-primary.loading .spinner{display:block;}
.btn-primary.done{background:var(--ok);color:#fff;}
.btn-primary.done .label,.btn-primary.done svg.arrow,.btn-primary.done .spinner{display:none;}
.btn-primary.done .check{display:block;}
.btn-primary[disabled]{cursor:default;}
.spinner{width:20px;height:20px;border-radius:50%;
  border:2.5px solid color-mix(in srgb,var(--bg0) 22%,transparent);border-top-color:var(--bg0);
  animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}

/* ---- messages (colored, fade in; matches login) ---- */
.msg{margin-top:4px;font-size:13px;line-height:1.45;padding:10px 12px;border-radius:11px;
  display:none;border:1px solid transparent;}
.msg.show{display:block;animation:msgIn .2s ease;}
.msg.err{color:var(--err);background:color-mix(in srgb,var(--err) 10%,transparent);border-color:color-mix(in srgb,var(--err) 30%,transparent);}
.msg.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 10%,transparent);border-color:color-mix(in srgb,var(--ok) 30%,transparent);}
.msg a{color:var(--text);font-weight:600;}
@keyframes msgIn{from{opacity:0;transform:translateY(-2px);}to{opacity:1;transform:none;}}

/* ---- password-strength bar (graded weak->strong) ---- */
.bar{height:5px;border-radius:4px;background:var(--input-bg);overflow:hidden;margin-top:2px;border:1px solid var(--card-brd);}
.bar i{display:block;height:100%;width:0;border-radius:4px;background:var(--text);transition:width .25s ease,background .25s ease;}
.bar[data-s="1"] i,.bar[data-s="2"] i{background:var(--err);}
.bar[data-s="3"] i{background:var(--muted);}
.bar[data-s="4"] i{background:var(--ok);}

.foot{margin-top:16px;text-align:center;font-size:13px;color:var(--muted);}
.foot a{color:var(--text);font-weight:600;text-decoration:none;}
.foot a:hover{text-decoration:underline;}

.hint{position:fixed;bottom:20px;left:0;right:0;text-align:center;z-index:10;font-size:11.5px;color:var(--muted);letter-spacing:.02em;}

.hidden{display:none !important;}

@media (prefers-reduced-motion:reduce){
  *{transition:none !important;animation:none !important;}
}
```

- [ ] **Step 2: Add the vignette layer to `reset.html`**

Find:
```html
  <div id="gradient"></div>
```
Replace with:
```html
  <div id="gradient"></div>
  <div class="vignette"></div>
```

- [ ] **Step 3: Wire reset's success message to the green `.ok` style**

Find:
```javascript
function showMsg(html, ok) { msg.innerHTML = html; msg.className = "msg show" + (ok ? "" : " err"); }
```
Replace with:
```javascript
function showMsg(html, ok) { msg.innerHTML = html; msg.className = "msg show" + (ok ? " ok" : " err"); }
```
(Presentational only — same call sites, success now renders green like login.)

- [ ] **Step 4: Wire the strength bar's graded color**

Find:
```javascript
pw.addEventListener("input", () => {
  const v = pw.value; let s = 0;
  if (v.length >= 6) s++; if (v.length >= 10) s++;
  if (/[A-Z]/.test(v) && /[a-z]/.test(v)) s++; if (/[0-9\W]/.test(v)) s++;
  bar.style.width = ([0, 25, 50, 80, 100][s] || 0) + "%";
});
```
Replace with:
```javascript
pw.addEventListener("input", () => {
  const v = pw.value; let s = 0;
  if (v.length >= 6) s++; if (v.length >= 10) s++;
  if (/[A-Z]/.test(v) && /[a-z]/.test(v)) s++; if (/[0-9\W]/.test(v)) s++;
  bar.style.width = ([0, 25, 50, 80, 100][s] || 0) + "%";
  bar.parentElement.dataset.s = v ? s : "";
});
```
(Presentational only — `bar` is the `<i id="bar">`; its parent is the `.bar` track, whose `data-s` selects the fill color in CSS.)

- [ ] **Step 5: Run the safety net**

Run: `python -m pytest -q` → expect `598 passed, 3 skipped`.
Run: `pyflakes backend webapp` → expect clean.

- [ ] **Step 6: Visual check (manual) — reset matches login**

Run: `python run.py` (with `ENABLE_AUTH=true`). Open a reset page directly, e.g. `http://localhost:8600/reset?token=test` (the form renders even with an invalid token; it shows a token message but all styling is visible).
Confirm in **both** dark and light:
- Card size/radius/blur, fonts, the **vignette**, and the **40px circle** theme toggle (top-right 16px) match `/login` side-by-side.
- Inputs: focus shows the ring + colored border; floating labels animate; password peek toggles.
- Type in **New password** → the strength bar fills and is **red → grey → green** as it strengthens.
- Submit with mismatched/short passwords → **red** error message that fades in. A valid submit shows the spinner then the **green "done"** check.
- Toggle theme → instant cohesive recolor, no flash.
- With OS "reduce motion" on → no transitions/animations on either screen.

- [ ] **Step 7: Commit**

```bash
git add webapp/static/auth.css webapp/static/reset.html
git commit -m "style(reset): unify auth.css + reset onto login's visual system

Rewrite auth.css to login's tokens and component specs (card, inputs,
circle theme toggle, primary button, vignette), add colored/fading
messages and a graded password-strength bar, and drop the dead
input[type=date] rules and per-child translateZ. reset.html gains the
vignette layer and presentational wiring for the green success state and
graded bar. No flow/behavior changes."
```

---

## Task 3: Whole-feature verification (cross-screen parity gate)

**Files:** none modified — this is the QA gate that confirms the deliverable.

- [ ] **Step 1: Final safety net**

Run: `python -m pytest -q` → `598 passed, 3 skipped`.
Run: `pyflakes backend webapp` → clean.
Run the Node JS check from Task 1 Step 10 → `JS OK`.

- [ ] **Step 2: Side-by-side parity pass**

Run: `python run.py` (`ENABLE_AUTH=true`). Open `/login` and `/reset?token=test` in two tabs. In **both dark and light**, confirm they read as the same design: card, fonts, inputs + focus ring, primary button (radius, inverted fill, arrow nudge, green done), circle theme toggle, vignette, and message styling all match. The login DOB calendar still shows a **pure-white** selection.

- [ ] **Step 3: Confirm git state**

Run: `git -C . status --short`
Expected: empty (Tasks 0–2 committed; nothing left uncommitted).

---

## Self-Review

**Spec coverage** (spec → task):
- Canonical tokens/card/inputs/submit/secondary/toggle/messages/type/background/motion → Task 2 (auth.css) + Task 1 (login message/motion/submit).
- login: tokenize calendar pure-white → Task 1 Steps 1-5 (render-identical). `.msg.ok` uses `--ok` + fade → Steps 6-7. Comprehensive reduced-motion → Step 9. Submit arrow nudge → Step 8. Calendar bg token → Step 3.
- auth.css rewrite (tokens, components, autofill, vignette, strength bar graded) → Task 2 Step 1. Dead `input[type=date]` rules + `.card::before` + per-child `translateZ` removed (not present in the new file). 
- reset.html: vignette layer + class mapping (success `.ok`, graded bar) → Task 2 Steps 2-4.
- auth-bg.js unchanged → not touched in any task. ✓
- Preserve pure-white calendar → Global Constraints + Task 1 render-identical edits + Task 1 Step 12 / Task 3 Step 2 checks. ✓
- Out of scope (no strength bar on login, no new flows, no workspace recolor, no shared-stylesheet refactor) → respected; none added. ✓

**Placeholder scan:** none — every step shows exact find/replace or full file content and exact commands.

**Type/selector consistency:** message classes `.msg.show/.err/.ok` identical in login (Task 1) and auth.css (Task 2); `msgIn`/`spin` keyframes defined in each file that uses them; `.bar[data-s="N"]` (Task 2 Step 1 CSS) matches `dataset.s = …` (Step 4 JS); `svg.arrow` styling (Task 2) matches reset's existing `<svg class="arrow">`; login's arrow nudge targets `.s-label svg` which is login's actual arrow markup.

**Note (flagged for the reviewer):** Task 2 Steps 3-4 make two **presentational** one-line JS edits in `reset.html` (success → `.ok` class; strength → `data-s`). These add no flow/behavior — they only realize the spec's "reset gains colored states" and "graded strength bar." If you want strictly zero JS edits in reset, drop Step 4 (bar stays single-color `--text`) and Step 3 (success stays neutral rather than green); the screens still match structurally.

---

## Execution Handoff
Two execution options:
1. **Subagent-Driven (recommended)** — a fresh subagent per task with review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
