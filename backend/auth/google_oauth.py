"""
Google "Sign in with Google" — OAuth 2.0 authorization-code flow.

Optional: enabled only when GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET are set in .env.
Uses `requests` (already a dependency) — no heavy OAuth library.

Set up (one time):
  1. https://console.cloud.google.com/apis/credentials -> Create OAuth client ID -> Web app
  2. Authorized redirect URI:  http://localhost:8600/auth/google/callback
     (and your public URL's /auth/google/callback for production)
  3. Put the client id/secret in .env:  GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
"""
from __future__ import annotations

import os
import urllib.parse
from typing import Any, Dict

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _client_id() -> str:
    return (os.getenv("GOOGLE_CLIENT_ID") or "").strip()


def _client_secret() -> str:
    return (os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()


def enabled() -> bool:
    return bool(_client_id() and _client_secret())


def authorize_url(redirect_uri: str, state: str) -> str:
    params = {
        "client_id": _client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    """Exchange the auth code for an access token, then fetch the user's profile.
    Returns the userinfo dict (email, email_verified, name, picture, sub). Raises on error."""
    import requests  # lazy: only needed when a user actually signs in with Google

    tok = requests.post(TOKEN_URL, data={
        "code": code,
        "client_id": _client_id(),
        "client_secret": _client_secret(),
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=15)
    tok.raise_for_status()
    access_token = tok.json().get("access_token")
    if not access_token:
        raise ValueError("Google did not return an access token")
    info = requests.get(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
    info.raise_for_status()
    return info.json()
