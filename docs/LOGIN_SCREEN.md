# Login Screen — Complete Reference

Everything about the **sign-in screen** of the Research Assistant: what the user sees, every
feature and field, the client-side behaviour, the backend it talks to, the data model, security,
configuration, and how to customize it.

---

## 1. At a glance

A single, self-contained **monochrome-glass** auth screen that handles three modes in one card:

| Mode | What it does |
|------|--------------|
| **Sign in** | Log in with **email *or* username** + password |
| **Sign up** | Create an account: username, email, date of birth, password, confirm |
| **Forgot** | Request a password-reset link using your identifier + date of birth |

Plus **"Continue with Google"** (when configured), a **dark/light theme toggle**, and an animated
**3D background** (drifting gradient + a 2D node-network + a Three.js wireframe torus-knot) with a
subtle **3D card tilt**.

The screen is served at **`/login`**. When authentication is enabled and you're not signed in,
visiting **`/`** redirects here automatically.

---

## 2. Files involved

### Front end (`webapp/static/`)
| File | Role |
|------|------|
| **`login.html`** | The page: markup for all three modes + the inline auth-wiring script. |
| **`auth.css`** | Shared monochrome-glass design system (login **and** reset). Theme tokens flip on `html.light`. |
| **`auth-bg.js`** | Shared visual engine: 2D node-network canvas, Three.js torus-knot, card tilt, theme toggle (persisted). |
| **`reset.html`** | The password-reset page (`/reset?token=…`), same look; strength bar + confirm. |

### Back end (`webapp/`, `backend/auth/`)
| File | Role |
|------|------|
| **`server.py`** | FastAPI routes: `/api/me`, `/api/login`, `/api/signup`, `/api/logout`, `/api/forgot-password`, `/api/reset-password`, `/auth/google/*`, and the `/login` / `/reset` page routes. |
| **`auth.py`** (`webapp.auth`) | Auth on/off flag, signup flag, session-secret handling, current-user helper, `LOCAL_USER`. |
| **`backend/auth/users.py`** | SQLite user store: create/verify users, password hashing (PBKDF2), DOB/email validation, reset tokens. |
| **`google_oauth`**, **`mailer`** | Google OAuth helper + outbound email for reset links. |

> Related setup guide: [`docs/GOOGLE_SIGNIN.md`](GOOGLE_SIGNIN.md) for enabling "Continue with Google".

---

## 3. The three modes (one card, no page reloads)

Every field/section carries a `data-modes="…"` attribute listing the modes it belongs to.
`setMode(m)` toggles the `.hidden` class on each, retitles the button/footer, and focuses the first
field. No navigation — it's instant.

| Element | Visible in modes |
|--------|------------------|
| Email-or-username field (`#userid`) | `login`, `forgot` |
| Username (`#username`) | `signup` |
| Email (`#email`) | `signup` |
| Date of birth (`#dob`) | `signup`, `forgot` |
| Password (`#password`) | `login`, `signup` |
| Confirm password (`#confirm`) | `signup` |
| "Forgot?" link | `login` |
| Social / Google block | `login`, `signup` (only if Google is configured) |

Mode-dependent copy:

- **Button:** "Sign in" → "Create account" → "Send reset link".
- **Google button:** "Continue with Google" (login) / "Sign up with Google" (signup).
- **Footer:** "New here? **Sign up**" ⇄ "Already have an account? **Sign in**" ⇄ "**← Back to sign in**".
- The footer hides entirely in login mode when sign-ups are disabled (`signup=false`).

On load the page calls **`GET /api/me`** to learn `signup` (is sign-up allowed) and `google` (is
Google configured), then starts in **login** mode.

---

## 4. Fields & validation

Validation runs **client-side first** (instant feedback) and is **re-enforced server-side** (the
real gate). Client regexes:

```js
USER_RE  = /^[A-Za-z0-9._@-]{3,64}$/     // username: 3–64 chars, letters digits . _ - @
EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/   // basic email shape
```

### Sign in
- Email **or** username + password — both required.
- The backend resolves an email **or** a username to the account (`resolve_user`).

### Sign up
All fields required, then:
- **Username** must match `USER_RE` (3–64 chars: letters, digits, `.` `_` `-` `@`).
- **Email** must match `EMAIL_RE`.
- **Date of birth** must be a real, non-future date; the user must be **≥ 13 years old**
  (computed by `ageFromDob`).
- **Password** ≥ **6** characters.
- **Confirm** must equal the password.

### Forgot
- Identifier (email/username) + date of birth — both required.
- The DOB acts as a **knowledge factor**: it must match what's on file (see §7).

---

## 5. Feature details

### 5.1 Floating-label glass inputs
Each field is an `<input>` with a `<label>` that floats up/shrinks on focus or when filled
(`:not(:placeholder-shown)`). The inputs are translucent glass (`--input-bg`) with a focus ring.

### 5.2 Password "eye" toggle (peek)
Password and Confirm fields have a `.peek` button. Clicking it flips the input between
`type="password"` and `type="text"`, swaps the eye / eye-off icon, and updates the `aria-label`
("Show password" ⇄ "Hide password"). It's `tabindex="-1"` so it doesn't interrupt tab order.

### 5.3 Date-of-birth picker
The DOB input is **`type="text"` at rest** so its floating label matches the other fields, then
becomes a **native date picker** on focus and reverts to text on blur if left empty:

```js
fields.dob.setAttribute("max", new Date().toISOString().slice(0,10));  // no future dates
fields.dob.addEventListener("focus", () => fields.dob.type = "date");
fields.dob.addEventListener("blur",  () => { if (!fields.dob.value) fields.dob.type = "text"; });
```

The native calendar is themed to match dark/light (`color-scheme`).

### 5.4 Email-or-username login
You can sign in with **either** your email **or** your username. The server's `resolve_user()`
maps either form to the canonical account.

### 5.5 Forgot password (with DOB second factor)
1. Submit identifier + DOB to `POST /api/forgot-password`.
2. The server **never reveals whether the account exists** (anti-enumeration): it always returns a
   generic "If an account matches, a reset link has been sent."
3. If the account exists **and** the DOB matches, it issues a one-time reset token and emails the
   `/reset?token=…` link (valid for a limited window — the UI states **~30 minutes**).
4. In dev/local setups the response may include `reset_url` directly, which the page renders as a
   clickable "Reset your password →" link.

The reset page (`/reset`) validates the token, shows a **password-strength bar**, requires the new
password ≥ 6 chars and confirmation, calls `POST /api/reset-password`, then sends you back to
`/login`.

### 5.6 "Continue with Google"
Shown only when Google OAuth is configured (`google:true` from `/api/me`). Clicking it navigates to
`/auth/google/login`, which redirects to Google with a CSRF **state** token. On callback the server
verifies the state, exchanges the code, requires a **verified** email, then **finds or creates** the
account by email and signs you in. See [`docs/GOOGLE_SIGNIN.md`](GOOGLE_SIGNIN.md).

### 5.7 Theme toggle (dark / light)
Top-right button toggles the `light` class on `<html>` and persists the choice in
`localStorage["ara-theme"]` (default **dark**). Every color is a CSS variable that flips between the
`:root` (dark) and `html.light` token sets — including the background network colors.

### 5.8 Animated 3D background
Three stacked layers behind the card (all `pointer-events:none`):
1. **`#gradient`** — a CSS radial-gradient that slowly drifts (26s loop).
2. **`#network`** — a 2D `<canvas>` node-network: ~64 nodes drift and connect with lines, with
   **travelling pulses** along edges and **mouse parallax**.
3. **`#three`** — a **Three.js wireframe torus-knot** that rotates continuously and leans toward the
   cursor.

The **card itself tilts in 3D** (`rotateX/rotateY`) following the mouse, and glows
(`.card.authing`) while a request is in flight. If WebGL/Three.js is unavailable it degrades
gracefully (gradient + 2D network remain). **`prefers-reduced-motion`** disables the 2D animation
and the tilt.

### 5.9 Submit-button states
One button animates through the whole request:
- **idle** → label + arrow,
- **`.loading`** → spinner (label/arrow hidden),
- **`.done`** → check mark, then `location.href = "/"` after ~750 ms.

### 5.10 Inline error messages
A `.msg` box shows validation and server errors. URL `?error=…` values (from failed Google
redirects / rate limits) are mapped to friendly text:

| `?error=` | Message |
|-----------|---------|
| `google` | "Google sign-in didn't complete. Please try again." |
| `google_email` | "Your Google account email isn't verified." |
| `google_off` | "Google sign-in isn't configured on this server." |
| `rate` | "Too many attempts — please wait a minute." |

### 5.11 Accessibility & UX niceties
- `aria-label`s on the theme and peek buttons; `autocomplete` hints on every field
  (`username` / `email` / `bday` / `current-password` / `new-password`).
- `autofocus` on the first field; focus moves to the right field on mode switch.
- `prefers-reduced-motion` honored for the background/tilt.
- `novalidate` form — validation is handled in JS for consistent messaging.

---

## 6. Backend API contract

All requests/responses are JSON unless noted. All write endpoints are **rate-limited per client**.

### `GET /api/me`
Tells the page how to render.
- Auth **off**: `{ "auth": false, "user_id": "local" }`
- Auth **on**: `{ "auth": true, "user_id": <id|null>, "signup": <bool>, "google": <bool> }`

### `POST /api/login`  *(rate limit: 10)*
Body: `{ "user_id": "<email or username>", "password": "…" }`
- Resolves email **or** username, verifies the password.
- **200** `{ "ok": true, "user_id": "<id>" }` and sets the session cookie.
- **401** `{ "ok": false, "error": "Invalid email/username or password." }`

### `POST /api/signup`  *(rate limit: 5)*
Body: `{ "user_id": "<username>", "password": "…", "email": "…", "date_of_birth": "YYYY-MM-DD" }`
- **403** if sign-ups are disabled.
- **400** if email or DOB missing, or `create_user` rejects (bad username/email/DOB, password < 6,
  duplicate username/email).
- **200** `{ "ok": true, "user_id": "<id>" }` and signs you in.

### `POST /api/forgot-password`  *(rate limit: 5)*
Body: `{ "identifier": "<email or username>", "date_of_birth": "YYYY-MM-DD" }`
- Always **200** with a generic message (no account enumeration).
- When the account exists **and** the DOB matches, also issues a token and emails the reset link;
  the response may include `reset_url` in local/dev setups.

### `POST /api/reset-password`  *(rate limit: 10)*
Body: `{ "token": "…", "password": "…" }`
- **400** if password < 6 (checked **before** consuming the token), or the token is invalid/expired.
- **200** `{ "ok": true, "user_id": "<id>" }`, then clears the session so you sign in fresh.

### `POST /api/logout`
Clears the session → `{ "ok": true }`.

### `GET /auth/google/login`  *(rate limit: 10)*
Redirects to Google with a CSRF `state`. If not configured → `/login?error=google_off`.

### `GET /auth/google/callback?code=…&state=…`
Verifies `state`, exchanges the code, requires a **verified** email, finds/creates the account by
email, signs in, and redirects to `/`. Failures redirect to `/login?error=google` or
`?error=google_email`.

### Page routes
- `GET /login` → serves `login.html`.
- `GET /reset` → serves `reset.html`.
- `GET /` → redirects to `/login` when auth is enabled and there's no session.

---

## 7. Data model & security

- **Password hashing:** `PBKDF2-HMAC-SHA256`, **200,000 rounds**, a fresh **16-byte random salt**
  per user. Stored as `pbkdf2_sha256$<rounds>$<salt_hex>$<hash_hex>` — **never plaintext**.
- **User store:** a small **SQLite** database (`data/auth.db`) with `user_id`, password hash, email,
  date of birth, admin flag.
- **Server-side validation** (in `create_user`, mirrors the client):
  username `^[A-Za-z0-9._@-]{3,64}$`, valid email, valid DOB, password ≥ 6, unique username + email.
  `valid_dob` requires an ISO `YYYY-MM-DD` date that isn't in the future and is within a plausible
  range.
- **Sessions:** a signed cookie. Set **`AUTH_SECRET_KEY`** in `.env` for stable sessions — without
  it a temporary key is used and sessions reset on restart (a warning is logged).
- **Forgot-password hardening:** generic response (no enumeration), timing equalized between the
  found/not-found branches, plus the **DOB knowledge factor**.
- **Google OAuth:** CSRF `state` parameter + **`email_verified`** check; the auto-created account's
  random password is never used (Google users sign in via Google or reset it).
- **Rate limiting** on every auth action (login 10, signup 5, forgot 5, reset 10, google 10 per
  window) to blunt brute-force/abuse.

---

## 8. Configuration (environment variables)

Set these in **`.env`** (never commit secrets):

| Variable | Purpose |
|----------|---------|
| `ENABLE_AUTH` | `true` turns the login system on. **Default `false`** → the app is open and owned by a single `local` user (no login screen). |
| `ENABLE_SIGNUP` | Allow self-service account creation. When off, the "Sign up" footer/section is hidden and `/api/signup` returns 403. |
| `AUTH_SECRET_KEY` | Signs the session cookie. **Required for production** (stable sessions). Generate: `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Enable "Continue with Google". See [`docs/GOOGLE_SIGNIN.md`](GOOGLE_SIGNIN.md). |
| `PUBLIC_BASE_URL` | Pins the base URL used for the OAuth callback and reset links (recommended behind a proxy / in production). |
| SMTP / mailer settings | Used to send password-reset emails (`mailer.send_email`). |

### Auth on/off behaviour
- **`ENABLE_AUTH=false` (default):** open mode — everything belongs to a single user `"local"`; the
  login screen isn't enforced.
- **`ENABLE_AUTH=true`:** `/` redirects to `/login` until you're signed in. On your **first** login,
  any pre-auth (`local`) conversations are adopted into your account (single-user self-hosted
  convenience; multi-user deployments are untouched).

---

## 9. End-to-end flows

**Sign in**
1. `/login` loads → `GET /api/me` decides whether Sign-up / Google appear.
2. Enter email-or-username + password → `POST /api/login`.
3. On success the button shows a check, then redirects to `/`.

**Sign up**
1. Footer "Sign up" → fields become username / email / DOB / password / confirm.
2. Client validates (regexes, age ≥ 13, password ≥ 6, match) → `POST /api/signup`.
3. Account created + signed in → redirect to `/`.

**Forgot → reset**
1. "Forgot?" → enter identifier + DOB → `POST /api/forgot-password`.
2. Use the emailed (or shown) `/reset?token=…` link → set a new password → `POST /api/reset-password`.
3. Session cleared → sign in with the new password.

**Google**
1. "Continue with Google" → `/auth/google/login` → Google consent.
2. `/auth/google/callback` verifies + finds/creates the account → signed in → `/`.

---

## 10. Customizing

| Want to change… | Edit |
|------------------|------|
| Colors / glass / shadows / radii | CSS variables at the top of **`auth.css`** (`:root` for dark, `html.light` for light). |
| Card width, padding, blur | `.card` in `auth.css`. |
| Labels / button text / footer copy | `setMode()` in the inline script of **`login.html`**. |
| Validation rules (client) | `USER_RE`, `EMAIL_RE`, `ageFromDob`, and the `submit` handler in `login.html`. |
| Validation rules (server) | `create_user` / `valid_dob` / `set_password` in **`backend/auth/users.py`**. |
| Background density / speed | `NODE_COUNT`, pulse logic, and the torus-knot rotation in **`auth-bg.js`**. |
| Password min length | `len < 6` checks in both `login.html` (client) and `backend/auth/users.py` (server). |
| Minimum signup age | the `age < 13` check in `login.html`. |

---

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Login screen never appears; app is just open | `ENABLE_AUTH` is `false` (default). Set it to `true` and restart. |
| "Sessions reset on restart" / logged out after restart | `AUTH_SECRET_KEY` not set — add it to `.env`. |
| No "Sign up" option | `ENABLE_SIGNUP` is off (or `signup:false` from `/api/me`). |
| No Google button | `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` not set, or `ENABLE_AUTH=false`. See `docs/GOOGLE_SIGNIN.md`. |
| `?error=google_off` | Same as above — server doesn't see both Google env vars. |
| Google `redirect_uri_mismatch` | Registered redirect URI ≠ what the app sent. Match scheme/host/port/path exactly; set `PUBLIC_BASE_URL`. |
| "Too many attempts" | Rate limit hit — wait a minute. |
| Reset link "invalid or expired" | Link older than its window (~30 min) or already used — request a new one. |
| Background is static (not animating) | `prefers-reduced-motion` is enabled in the OS/browser — the auth background intentionally calms down. |
