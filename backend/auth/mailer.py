"""
Optional SMTP email sender for password-reset links.

If SMTP_HOST is set in .env, password-reset emails are sent for real. If not, this
returns False and the caller falls back to logging the link (or returning it when
AUTH_RESET_RETURN_LINK is on for a single-user self-hosted setup) — so the feature
works with or without an email server.

Env: SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_TLS (true).
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def _flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


def email_configured() -> bool:
    return bool((os.getenv("SMTP_HOST") or "").strip())


def send_email(to_addr: str, subject: str, body: str) -> bool:
    """Send a plain-text email via SMTP. Returns True on success, False if SMTP is
    not configured or sending fails."""
    host = (os.getenv("SMTP_HOST") or "").strip()
    to_addr = (to_addr or "").strip()
    if not host or not to_addr:
        return False
    port = int(os.getenv("SMTP_PORT", "587") or "587")
    user = (os.getenv("SMTP_USER") or "").strip()
    password = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS") or ""
    sender = (os.getenv("SMTP_FROM") or user or "no-reply@localhost").strip()
    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=15) as srv:
                if user:
                    srv.login(user, password)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as srv:
                if _flag("SMTP_TLS", "true"):
                    srv.starttls(context=ctx)
                if user:
                    srv.login(user, password)
                srv.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001 - never let email errors break the request
        print(f"[auth] password-reset email send failed: {exc}")
        return False
