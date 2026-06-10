"""Tests for email storage + the password-reset token flow (temp DB, no network)."""
import time

import pytest

from backend.auth import users


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(users, "AUTH_DB", tmp_path / "auth.db")
    return users


def test_email_stored_normalized_and_resolvable(db):
    db.create_user("alice", "secret1", email="Alice@Example.COM")
    assert db.get_email("alice") == "alice@example.com"
    assert db.resolve_user("alice@example.com") == "alice"   # by email
    assert db.resolve_user("alice") == "alice"               # by user id
    assert db.resolve_user("nobody") is None


def test_duplicate_email_is_rejected(db):
    db.create_user("alice", "secret1", email="x@y.com")
    with pytest.raises(ValueError):
        db.create_user("bobby", "secret1", email="X@Y.com")  # case-insensitive dup


def test_invalid_email_is_rejected(db):
    with pytest.raises(ValueError):
        db.create_user("carol", "secret1", email="not-an-email")


def test_reset_token_is_single_use(db):
    db.create_user("alice", "secret1")
    token = db.create_reset_token("alice")
    assert db.consume_reset_token(token) == "alice"
    assert db.consume_reset_token(token) is None             # replay blocked
    assert db.consume_reset_token("garbage") is None


def test_reset_token_expires(db, monkeypatch):
    db.create_user("alice", "secret1")
    token = db.create_reset_token("alice")
    monkeypatch.setattr(time, "time", lambda: 10 ** 12)      # far future
    assert db.consume_reset_token(token) is None


def test_issuing_a_new_token_invalidates_the_old_one(db):
    db.create_user("alice", "secret1")
    old = db.create_reset_token("alice")
    new = db.create_reset_token("alice")
    assert db.consume_reset_token(old) is None               # superseded
    assert db.consume_reset_token(new) == "alice"


def test_reset_then_login_with_new_password(db):
    db.create_user("alice", "oldpass1")
    token = db.create_reset_token("alice")
    uid = db.consume_reset_token(token)
    db.set_password(uid, "brandnew9")
    assert db.verify_user("alice", "brandnew9") is True
    assert db.verify_user("alice", "oldpass1") is False
