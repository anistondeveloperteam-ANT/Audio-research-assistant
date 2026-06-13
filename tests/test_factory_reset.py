"""Factory-reset file wipes — temp dir, no Oracle, no real DBs."""
from backend.maintenance import factory_reset as fr


def _make_data(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / "extracted").mkdir()
    (tmp_path / "external_cache").mkdir()
    (tmp_path / "papers" / "a.pdf").write_bytes(b"%PDF-1.4 x")
    (tmp_path / "papers" / "b.pdf").write_bytes(b"%PDF-1.4 y")
    (tmp_path / "conversations.db").write_text("c")
    (tmp_path / "conversations.db-wal").write_text("w")
    (tmp_path / "memory.db").write_text("m")
    (tmp_path / "llm_costs.db").write_text("$")
    (tmp_path / "auth.db").write_text("u")
    (tmp_path / "memory.db.bak-1781163228").write_text("old")
    (tmp_path / "extracted" / "incremental_index_manifest.json").write_text("{}")
    (tmp_path / "external_cache" / "hit.json").write_text("{}")
    (tmp_path / "evaluation_questions.json").write_text("[]")          # committed config — keep
    (tmp_path / "llm_eval_questions.json").write_text("[]")            # committed config — keep


def test_full_wipe_removes_everything_but_keeps_config(tmp_path):
    _make_data(tmp_path)
    fr.run(tmp_path, oracle=False)        # oracle=False so the test needs no DB connection

    # user/runtime data gone
    assert not (tmp_path / "conversations.db").exists()
    assert not (tmp_path / "conversations.db-wal").exists()   # sidecar removed too
    assert not (tmp_path / "memory.db").exists()
    assert not (tmp_path / "llm_costs.db").exists()
    assert not (tmp_path / "auth.db").exists()
    assert not (tmp_path / "memory.db.bak-1781163228").exists()
    assert list((tmp_path / "papers").glob("*.pdf")) == []
    assert list((tmp_path / "extracted").iterdir()) == []     # manifest cleared
    assert list((tmp_path / "external_cache").iterdir()) == []
    # cache directories themselves survive (the app re-uses them)
    assert (tmp_path / "extracted").is_dir() and (tmp_path / "papers").is_dir()
    # committed config is preserved
    assert (tmp_path / "evaluation_questions.json").exists()
    assert (tmp_path / "llm_eval_questions.json").exists()


def test_keep_flags_spare_categories(tmp_path):
    _make_data(tmp_path)
    fr.run(tmp_path, oracle=False, auth=False, pdfs=False)
    assert (tmp_path / "auth.db").exists()                    # auth spared
    assert (tmp_path / "papers" / "a.pdf").exists()           # pdfs spared
    assert not (tmp_path / "conversations.db").exists()       # chats still wiped
    assert not (tmp_path / "llm_costs.db").exists()           # costs still wiped


def test_wipe_backups_only(tmp_path):
    _make_data(tmp_path)
    removed = fr.wipe_backups(tmp_path)
    assert any("bak" in line for line in removed)
    assert not (tmp_path / "memory.db.bak-1781163228").exists()
    assert (tmp_path / "memory.db").exists()                  # only .bak removed here
