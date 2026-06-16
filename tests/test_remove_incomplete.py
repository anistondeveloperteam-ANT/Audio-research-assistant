"""Remove HALF-DONE papers (parsed but not embedded) so they can be re-uploaded. Fully offline —
the Oracle connection is faked, so no DB is touched."""
import backend.ingestion.ingest_papers as ip


class _LOB:
    def __init__(self, v):
        self._v = v

    def read(self):
        return self._v


class FakeCursor:
    def __init__(self, incomplete_rows):
        self._incomplete = list(incomplete_rows)
        self.executed = []
        self._last = ""

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._last = sql

    def fetchall(self):
        if "embedding IS NULL" in self._last:          # the find_incomplete_papers SELECT
            return list(self._incomplete)
        return []

    def close(self):
        pass


class FakeConn:
    def __init__(self, cur):
        self._cur = cur
        self.committed = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True

    def close(self):
        pass


def test_find_incomplete_papers_parses_rows_and_lob():
    cur = FakeCursor([(1, "a.pdf"), (2, _LOB("b.pdf"))])
    out = ip.find_incomplete_papers(cur)
    assert out == [(1, "a.pdf"), (2, "b.pdf")]
    assert "embedding IS NULL" in cur.executed[-1][0]
    assert "NOT EXISTS" in cur.executed[-1][0]          # also catches zero-chunk papers


def test_purge_paper_deletes_links_chunks_and_paper():
    cur = FakeCursor([])
    ip.purge_paper(cur, 7)
    sqls = " ".join(s for s, _ in cur.executed)
    assert "DELETE FROM chunk_concepts" in sqls
    assert "DELETE FROM chunks WHERE paper_id" in sqls
    assert "DELETE FROM papers WHERE id" in sqls


def test_remove_incomplete_deletes_rows_and_files(monkeypatch, tmp_path):
    cur = FakeCursor([(1, "a.pdf"), (2, "b.pdf")])
    conn = FakeConn(cur)
    monkeypatch.setattr(ip, "connect", lambda: conn)
    monkeypatch.setattr(ip, "PAPERS_DIR", tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.pdf").write_bytes(b"y")

    removed = ip.remove_incomplete_papers(delete_files=True)

    assert removed == ["a.pdf", "b.pdf"]
    assert conn.committed is True
    assert not (tmp_path / "a.pdf").exists() and not (tmp_path / "b.pdf").exists()
    # 2 papers x 3 deletes each = 6 DELETEs after the initial SELECT
    assert sum(1 for s, _ in cur.executed if s.strip().startswith("DELETE")) == 6


def test_remove_incomplete_keeps_files_when_not_requested(monkeypatch, tmp_path):
    cur = FakeCursor([(1, "a.pdf")])
    monkeypatch.setattr(ip, "connect", lambda: FakeConn(cur))
    monkeypatch.setattr(ip, "PAPERS_DIR", tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"x")

    removed = ip.remove_incomplete_papers(delete_files=False)

    assert removed == ["a.pdf"]
    assert (tmp_path / "a.pdf").exists()                # file kept; next ingest re-processes it


def test_remove_incomplete_none(monkeypatch):
    monkeypatch.setattr(ip, "connect", lambda: FakeConn(FakeCursor([])))
    assert ip.remove_incomplete_papers() == []
