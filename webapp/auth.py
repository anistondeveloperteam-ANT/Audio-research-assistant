"""
Web-layer authentication helpers for the FastAPI app.

When ENABLE_AUTH=true, members sign in with a user_id + password (accounts created
by an admin via `python -m backend.auth.users add <id>`). The signed session cookie
then carries the user; every conversation is stored and listed per user.

When ENABLE_AUTH=false (default), the app is open and everything is owned by a single
shared "local" user — preserving the original single-user behaviour.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Request

ROOT = Path(__file__).resolve().parents[1]

# Paths that never require a login (the login/reset pages, auth endpoints, static assets).
PUBLIC_PATHS = {"/", "/login", "/reset", "/api/login", "/api/logout", "/api/me",
                "/api/signup", "/api/forgot-password", "/api/reset-password",
                "/auth/google/login", "/auth/google/callback", "/favicon.ico"}

LOCAL_USER = "local"


def _flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


def auth_enabled() -> bool:
    return _flag("ENABLE_AUTH")


def signup_enabled() -> bool:
    """Optional self-service registration. Off by default (admin creates accounts)."""
    return _flag("ENABLE_SIGNUP")


def reset_return_link() -> bool:
    """When true (single-user self-hosted, no SMTP), the forgot-password endpoint
    returns the reset link to the page so it works without an email server. Keep OFF
    in any multi-user/production deployment (use SMTP instead)."""
    return _flag("AUTH_RESET_RETURN_LINK")


def session_secret() -> str:
    """Secret that signs the session cookie. Set AUTH_SECRET_KEY for stable sessions."""
    secret = (os.getenv("AUTH_SECRET_KEY") or "").strip()
    if not secret:
        # Random per-process fallback: secure, but logins drop on restart. Warn loudly.
        secret = os.urandom(32).hex()
        print("[auth] WARNING: AUTH_SECRET_KEY is not set — using a temporary key; "
              "sessions reset on restart. Set AUTH_SECRET_KEY in .env for production.")
    return secret


def session_max_age():
    """Cookie lifetime in seconds, or None for a session cookie (cleared when the
    browser closes — the default, so each visit requires signing in)."""
    raw = (os.getenv("SESSION_MAX_AGE") or "").strip()
    if not raw:
        return None
    try:
        secs = int(raw)
        return secs if secs > 0 else None
    except ValueError:
        return None


def current_user(request: Request) -> str:
    """The signed-in user id, or the shared 'local' user when auth is disabled."""
    if not auth_enabled():
        return LOCAL_USER
    return (request.session.get("user_id") or "").strip() or LOCAL_USER


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static")
