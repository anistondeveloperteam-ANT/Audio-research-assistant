"""Library degrades fast + cleanly when the database is offline (e.g. the Oracle Docker container
isn't running). The connect is bounded by tcp_connect_timeout, and library_stats reports the on-disk
PDF count with papers=None so the UI can show 'database offline' instead of stalling. No real DB."""
import oracledb

import webapp.ingest as ingest


def test_connect_passes_tcp_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(oracledb, "connect", lambda **kw: captured.update(kw) or "CONN")
    assert ingest._connect() == "CONN"
    assert captured.get("tcp_connect_timeout", 0) > 0      # bounded, so offline fails fast


def test_connect_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("ORACLE_CONNECT_TIMEOUT", "7")
    assert ingest._db_connect_timeout() == 7.0
    monkeypatch.setenv("ORACLE_CONNECT_TIMEOUT", "not-a-number")
    assert ingest._db_connect_timeout() == 3.0             # falls back to the safe default


def test_library_stats_offline_is_graceful(monkeypatch, tmp_path):
    """DB unreachable -> still report PDFs on disk, papers=None (drives the 'database offline' UI)."""
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 x")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 y")

    def boom(**kw):
        raise RuntimeError("DB offline")

    monkeypatch.setattr(oracledb, "connect", boom)
    out = ingest.library_stats()
    assert out["pdfs"] == 2                                 # files on disk are still counted
    assert out["papers"] is None and out["vectors"] is None  # offline -> unknown, not a crash


def test_list_papers_offline_returns_empty(monkeypatch):
    def boom(**kw):
        raise RuntimeError("DB offline")

    monkeypatch.setattr(oracledb, "connect", boom)
    assert ingest.list_papers() == []                      # never raises -> UI shows empty/offline
