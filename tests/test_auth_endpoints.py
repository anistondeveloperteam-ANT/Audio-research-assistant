"""Security regression tests for the auth/reset HTTP endpoints (TestClient, temp DB)."""
import urllib.parse as up

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTH", "true")
    monkeypatch.setenv("ENABLE_SIGNUP", "true")
    monkeypatch.setenv("AUTH_RESET_RETURN_LINK", "true")
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 64)
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "false")
    from backend.auth import users
    monkeypatch.setattr(users, "AUTH_DB", tmp_path / "auth.db")
    import webapp.server as server
    monkeypatch.setattr(server, "_is_loopback", lambda r: True)   # TestClient host isn't 127.0.0.1
    server._RATE_BUCKETS.clear()
    return TestClient(server.app), server, users


def test_forgot_never_leaks_link_in_multi_user(client):
    # CRITICAL regression: with >1 account, the reset link must never be returned to the
    # caller (else knowing a username = account takeover).
    c, server, users = client
    users.create_user("alice", "pw12345")
    users.create_user("bobby", "pw12345")
    server._RATE_BUCKETS.clear()
    r = c.post("/api/forgot-password", json={"identifier": "alice"}).json()
    assert "reset_url" not in r


def test_forgot_returns_link_only_for_single_user_loopback(client):
    c, server, users = client
    users.create_user("solo", "pw12345")
    server._RATE_BUCKETS.clear()
    r = c.post("/api/forgot-password", json={"identifier": "solo"}).json()
    assert "reset_url" in r        # single user + loopback self-hosted escape hatch


def test_forgot_is_rate_limited(client):
    c, server, users = client
    users.create_user("rluser", "pw12345")
    server._RATE_BUCKETS.clear()
    codes = [c.post("/api/forgot-password", json={"identifier": "rluser"}).status_code
             for _ in range(7)]
    assert 429 in codes


def test_login_is_rate_limited(client):
    c, server, users = client
    users.create_user("victim", "pw12345")
    server._RATE_BUCKETS.clear()
    codes = [c.post("/api/login", json={"user_id": "victim", "password": "wrong"}).status_code
             for _ in range(12)]
    assert 429 in codes


def test_short_password_does_not_burn_the_token(client):
    c, server, users = client
    users.create_user("duser", "oldpw12")
    server._RATE_BUCKETS.clear()
    ru = c.post("/api/forgot-password", json={"identifier": "duser"}).json()["reset_url"]
    token = up.parse_qs(up.urlparse(ru).query)["token"][0]
    assert c.post("/api/reset-password", json={"token": token, "password": "ab"}).status_code == 400
    # token survived the bad attempt:
    assert c.post("/api/reset-password", json={"token": token, "password": "goodpw9"}).status_code == 200


def test_login_accepts_email_or_username(client):
    c, server, users = client
    users.create_user("dave", "pw12345", email="dave@x.com")
    server._RATE_BUCKETS.clear()
    assert c.post("/api/login", json={"user_id": "dave@x.com", "password": "pw12345"}).status_code == 200
    server._RATE_BUCKETS.clear()
    assert c.post("/api/login", json={"user_id": "dave", "password": "pw12345"}).status_code == 200
    server._RATE_BUCKETS.clear()
    assert c.post("/api/login", json={"user_id": "dave@x.com", "password": "wrong"}).status_code == 401


def test_signup_creates_user_with_dob(client):
    c, server, users = client
    server._RATE_BUCKETS.clear()
    r = c.post("/api/signup", json={"user_id": "newbie", "password": "pw12345",
                                    "email": "newbie@x.com", "date_of_birth": "1995-06-20"})
    assert r.status_code == 200 and r.json().get("ok")
    assert users.get_dob("newbie") == "1995-06-20"
    assert users.get_email("newbie") == "newbie@x.com"


def test_signup_requires_email_and_dob(client):
    c, server, users = client
    server._RATE_BUCKETS.clear()
    assert c.post("/api/signup", json={"user_id": "noemail", "password": "pw12345",
                                       "date_of_birth": "1990-01-01"}).status_code == 400
    server._RATE_BUCKETS.clear()
    assert c.post("/api/signup", json={"user_id": "nodob", "password": "pw12345",
                                       "email": "n@x.com"}).status_code == 400


def test_forgot_requires_matching_dob(client):
    c, server, users = client
    users.create_user("dobby", "pw12345", email="d@x.com", date_of_birth="1990-04-04")
    server._RATE_BUCKETS.clear()
    # wrong DOB -> generic response, no reset link issued
    assert "reset_url" not in c.post(
        "/api/forgot-password", json={"identifier": "dobby", "date_of_birth": "2000-01-01"}).json()
    server._RATE_BUCKETS.clear()
    # correct DOB -> link returned (single-user loopback escape hatch)
    assert "reset_url" in c.post(
        "/api/forgot-password", json={"identifier": "dobby", "date_of_birth": "1990-04-04"}).json()


def test_forgot_skips_dob_check_for_accounts_without_one(client):
    # Legacy/admin-created accounts (no DOB on file) still reset without a DOB match.
    c, server, users = client
    users.create_user("olduser", "pw12345")
    server._RATE_BUCKETS.clear()
    assert "reset_url" in c.post(
        "/api/forgot-password", json={"identifier": "olduser", "date_of_birth": ""}).json()


def test_signup_rejects_bad_dob(client):
    c, server, users = client
    server._RATE_BUCKETS.clear()
    r = c.post("/api/signup", json={"user_id": "baddob", "password": "pw12345",
                                    "email": "b@x.com", "date_of_birth": "2999-01-01"})
    assert r.status_code == 400


def test_forgot_unknown_user_is_generic(client):
    c, server, users = client
    users.create_user("real", "pw12345")
    server._RATE_BUCKETS.clear()
    r = c.post("/api/forgot-password", json={"identifier": "ghost"}).json()
    assert r.get("ok") is True and "reset_url" not in r
