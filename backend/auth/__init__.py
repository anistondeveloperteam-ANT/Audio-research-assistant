"""User authentication: a small SQLite-backed user store with salted PBKDF2
password hashing, plus an admin CLI to manage accounts.

    python -m backend.auth.users add  alice
    python -m backend.auth.users list
    python -m backend.auth.users passwd alice
    python -m backend.auth.users delete alice

Import the functions from ``backend.auth.users`` (e.g.
``from backend.auth.users import verify_user``).
"""
