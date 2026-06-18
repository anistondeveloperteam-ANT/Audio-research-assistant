"""Tests for user authentication: password hashing, the user store, and
per-user conversation isolation. No network or external services."""
import pytest

from backend.auth import users
from backend.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the user store at a throwaway SQLite file."""
    monkeypatch.setattr(users, "AUTH_DB", tmp_path / "auth.db")
    return users


# ---- password hashing ------------------------------------------------
def test_hash_roundtrip():
    h = users.hash_password("secret123")
    assert h.startswith("pbkdf2_sha256$")
    assert users.check_password("secret123", h)
    assert not users.check_password("wrong", h)


def test_hash_is_salted():
    # Same password hashes differently each time (random salt).
    assert users.hash_password("samepw") != users.hash_password("samepw")


def test_check_password_rejects_garbage():
    assert not users.check_password("x", "not-a-valid-hash")


# ---- user store ------------------------------------------------------
def test_create_and_verify(store):
    store.create_user("alice", "hunter22")
    assert store.verify_user("alice", "hunter22")
    assert not store.verify_user("alice", "nope")
    assert not store.verify_user("ghost", "whatever")   # unknown user
    assert store.user_exists("alice")
    assert store.count_users() == 1


def test_duplicate_and_validation(store):
    store.create_user("bob", "secret1")
    with pytest.raises(ValueError):
        store.create_user("bob", "secret2")             # duplicate id
    with pytest.raises(ValueError):
        store.create_user("ab", "secret1")              # id too short
    with pytest.raises(ValueError):
        store.create_user("okid", "123")                # password too short


def test_set_password_and_delete(store):
    store.create_user("carol", "secret1")
    store.set_password("carol", "newpass1")
    assert store.verify_user("carol", "newpass1")
    assert not store.verify_user("carol", "secret1")
    assert store.delete_user("carol")
    assert not store.user_exists("carol")
    with pytest.raises(ValueError):
        store.set_password("carol", "whatever1")        # no such user


# ---- date of birth ---------------------------------------------------
def test_valid_dob():
    assert users.valid_dob("1990-05-14")
    assert not users.valid_dob("")               # empty
    assert not users.valid_dob("14-05-1990")     # wrong format
    assert not users.valid_dob("1990-13-01")     # not a real calendar date
    assert not users.valid_dob("3000-01-01")     # in the future
    assert not users.valid_dob("1800-01-01")     # implausibly old


def test_create_user_stores_dob(store):
    store.create_user("dana", "secret1", email="dana@x.com", date_of_birth="1992-03-08")
    assert store.get_dob("dana") == "1992-03-08"
    assert store.get_email("dana") == "dana@x.com"
    assert store.get_dob("ghost") is None        # unknown user
    store.create_user("noddob", "secret1")       # dob is optional at the store level
    assert store.get_dob("noddob") is None


def test_create_user_rejects_bad_dob(store):
    with pytest.raises(ValueError):
        store.create_user("eric", "secret1", date_of_birth="not-a-date")
    with pytest.raises(ValueError):
        store.create_user("erin", "secret1", date_of_birth="2999-01-01")   # future


# ---- per-user conversation isolation --------------------------------
def test_sessions_are_per_user(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db")
    a = mem.create_session(user_id="alice")
    b = mem.create_session(user_id="bob")
    assert {s["id"] for s in mem.list_sessions(user_id="alice")} == {a}
    assert {s["id"] for s in mem.list_sessions(user_id="bob")} == {b}
    # list_sessions() with no user returns everything (admin/back-compat).
    assert {s["id"] for s in mem.list_sessions()} == {a, b}
    assert mem.session_owner(a) == "alice"
    assert mem.session_owner("missing") is None


def test_reassign_sessions_folds_local_into_user(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db")
    s1 = mem.create_session(user_id="local")
    s2 = mem.create_session(user_id="local")
    keep = mem.create_session(user_id="anjan")
    moved = mem.reassign_sessions("local", "anjan")
    assert moved == 2
    assert {s["id"] for s in mem.list_sessions(user_id="anjan")} == {s1, s2, keep}
    assert mem.list_sessions(user_id="local") == []
    # guards: nothing left under 'local', and same-user is a no-op
    assert mem.reassign_sessions("local", "anjan") == 0
    assert mem.reassign_sessions("anjan", "anjan") == 0
