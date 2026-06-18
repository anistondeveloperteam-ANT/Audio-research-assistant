# Enabling "Sign in with Google"

The OAuth flow is **already built** into the app (`/auth/google/login` → Google → `/auth/google/callback`),
and the **"Continue with Google"** button on `/login` appears automatically once it's configured. So this
is a **one-time setup**, not a code change. It takes ~5 minutes.

When a user signs in with Google, the app finds their account by email — or **creates one automatically**
(no password needed; they always sign in via Google). This works even if `ENABLE_SIGNUP=false`.

---

## 0. Prerequisites (in your `.env`)

Google sign-in only runs when the login system is on:

```ini
ENABLE_AUTH=true
# Sign cookies — generate once:  python -c "import secrets; print(secrets.token_hex(32))"
AUTH_SECRET_KEY=<paste the generated 64-char value>
```

> Never commit `.env`. Edit `.env` (not `.env.example`).

---

## 1. Create a Google Cloud project

1. Go to <https://console.cloud.google.com/> and sign in.
2. Top bar → **project picker** → **New Project** → name it (e.g. *Research Assistant*) → **Create**.
3. Make sure that project is selected.

## 2. Configure the OAuth consent screen

1. Left menu → **APIs & Services → OAuth consent screen**.
2. **User type: External** → **Create**.
3. Fill the required fields: **App name**, **User support email**, **Developer contact email** → **Save and continue**.
4. **Scopes**: you can leave defaults (the app only requests `openid`, `email`, `profile` — all non‑sensitive)
   → **Save and continue**.
5. **Test users**: while the app is in **Testing** status, only listed users can sign in.
   **Add the Google email(s)** you'll test with → **Save and continue**.

## 3. Create the OAuth client ID

1. Left menu → **APIs & Services → Credentials**.
2. **+ Create Credentials → OAuth client ID**.
3. **Application type: Web application**. Name it (e.g. *Research Assistant Web*).
4. Under **Authorized redirect URIs**, click **+ Add URI** and add **exactly**:

   ```
   http://localhost:8600/auth/google/callback
   ```

   For a deployed server, also add your real URL, e.g.:

   ```
   https://research.example.com/auth/google/callback
   ```

   *(Authorized JavaScript origins are optional for this server-side flow and can be left empty.)*
5. **Create**. A dialog shows your **Client ID** and **Client secret** — copy both.

## 4. Put the credentials in `.env`

```ini
GOOGLE_CLIENT_ID=1234567890-abcdefg.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxx

# Recommended: pin the base URL so the redirect URI is always exactly what Google expects.
PUBLIC_BASE_URL=http://localhost:8600
```

## 5. Restart and test

```bash
python run.py --web         # serves http://localhost:8600
```

1. Open **http://localhost:8600/login** (use the **same host** you registered — `localhost`, not `127.0.0.1`).
2. **"Continue with Google"** is now visible → click it.
3. Pick your Google account / consent → you're redirected back **signed in** to `/`.

---

## Production notes

- **Use `PUBLIC_BASE_URL`** (e.g. `https://research.example.com`). The redirect URI the app sends is
  `PUBLIC_BASE_URL + /auth/google/callback`; it must match a registered URI **character-for-character**.
- **HTTPS**: register the `https://…` callback and run behind TLS. If you're behind a reverse proxy,
  setting `PUBLIC_BASE_URL` avoids the proxy reporting `http` internally and breaking the match.
- **Go live for everyone**: on the **OAuth consent screen**, click **Publish app** (move from *Testing* to
  *In production*). With only `openid/email/profile` scopes, Google verification is generally **not required**.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Button doesn't appear on `/login` | `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` not set, or `ENABLE_AUTH=false`. Restart after editing `.env`. |
| `…?error=google_off` | Same — the server doesn't see both Google env vars. |
| Google shows **redirect_uri_mismatch** | The registered URI ≠ what the app sent. Match the scheme/host/port/path exactly (incl. `localhost` vs `127.0.0.1`). Set `PUBLIC_BASE_URL`. |
| **Access blocked / app not verified** | Consent screen is in *Testing* — add your email under **Test users**, or **Publish app**. |
| `…?error=google_email` | The Google account's email isn't verified. Use a verified Google account. |
| `…?error=rate` | Too many Google attempts in a minute — wait and retry. |

**Env summary**

| Variable | Purpose |
|---|---|
| `ENABLE_AUTH=true` | Turns the login system on (required). |
| `AUTH_SECRET_KEY` | Signs session cookies (required when auth is on). |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Enable Google sign-in. |
| `PUBLIC_BASE_URL` | Pins the OAuth/reset URLs (recommended). |
