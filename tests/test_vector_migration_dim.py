"""vector_migration self-heals a stale-dimension embedding_vec column.

After switching the embedding model (768-d Gemini -> 1024-d bge), the existing `embedding_vec`
column is still VECTOR(768), so every insert fails with ORA-51803. The migration must detect the
mismatch (from user_tab_columns.char_length) and DROP + recreate the column at the new dimension,
then re-migrate from the CLOBs. These tests use a fake cursor — no Oracle needed.
"""
import json

import backend.database.vector_migration as vm


class FakeCursor:
    """Records executed SQL and answers the metadata/SELECTs the migration needs."""
    def __init__(self, *, vec_col_dim=None, clob_rows=None, first_embedding=None):
        self.sql = []
        self._vec_col_dim = vec_col_dim          # None => column absent
        self._clob_rows = clob_rows or []        # [(id, json_str), ...] pending migration
        self._first_embedding = first_embedding  # json_str for detect_embedding_dim
        self._fetch = None
        self.updates = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        low = " ".join(sql.lower().split())
        if "fetch first 1 rows only" in low:                       # detect_embedding_dim probe
            self._fetch = (self._first_embedding,) if self._first_embedding is not None else None
        elif "count(*) from user_tab_columns" in low:              # column_exists
            self._fetch = (1 if self._vec_col_dim is not None else 0,)
        elif "data_type, char_length from user_tab_columns" in low:  # column_vector_dim
            self._fetch = ("VECTOR", self._vec_col_dim) if self._vec_col_dim is not None else None
        elif "drop column embedding_vec" in low:
            self._vec_col_dim = None                               # column gone after drop
        elif low.startswith("alter table chunks add embedding_vec"):
            self._vec_col_dim = _dim_from_add(sql)                 # now exists at the new dim
        elif "and embedding_vec is null" in low:                   # migrate_clob_to_vector SELECT
            self._fetch_rows = list(self._clob_rows)
        elif "update chunks set embedding_vec" in low:
            self.updates.append(params["id"])

    def fetchone(self):
        return self._fetch

    def fetchall(self):
        return getattr(self, "_fetch_rows", [])

    def close(self):
        pass


def _dim_from_add(sql):
    import re
    m = re.search(r"vector\((\d+)", sql.lower())
    return int(m.group(1)) if m else None


class FakeConn:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1

    def close(self):
        pass


# ---- column_vector_dim reads the dimension from the metadata ----
def test_column_vector_dim_reads_dimension():
    cur = FakeCursor(vec_col_dim=768)
    assert vm.column_vector_dim(cur, "chunks", "embedding_vec") == 768


def test_column_vector_dim_none_when_absent():
    cur = FakeCursor(vec_col_dim=None)
    assert vm.column_vector_dim(cur, "chunks", "embedding_vec") is None


# ---- ensure_embedding_vec_column: create / recreate / leave-alone ----
def test_creates_column_when_missing():
    cur, conn = FakeCursor(vec_col_dim=None), FakeConn()
    assert vm.ensure_embedding_vec_column(cur, conn, 1024) == "created"
    joined = " ".join(cur.sql).lower()
    assert "add embedding_vec vector(1024, float32)" in joined
    assert "drop column" not in joined


def test_recreates_column_on_dimension_mismatch():
    cur, conn = FakeCursor(vec_col_dim=768), FakeConn()        # stale 768-d column, new 1024-d data
    assert vm.ensure_embedding_vec_column(cur, conn, 1024) == "recreated"
    joined = " ".join(cur.sql).lower()
    assert "drop column embedding_vec" in joined               # old column dropped
    assert "add embedding_vec vector(1024, float32)" in joined  # recreated at the new dim


def test_leaves_column_when_dimension_matches():
    cur, conn = FakeCursor(vec_col_dim=1024), FakeConn()
    assert vm.ensure_embedding_vec_column(cur, conn, 1024) == "ok"
    joined = " ".join(cur.sql).lower()
    assert "drop column" not in joined
    assert "add embedding_vec" not in joined


# ---- detect_embedding_dim from a stored CLOB vector ----
def test_detect_embedding_dim_from_stored_vector():
    cur = FakeCursor(first_embedding=json.dumps([0.0] * 1024))
    assert vm.detect_embedding_dim(cur, default=768) == 1024     # data wins over a stale default


def test_detect_embedding_dim_falls_back_when_empty():
    cur = FakeCursor(first_embedding=None)
    assert vm.detect_embedding_dim(cur, default=1024) == 1024


# ---- end-to-end on the fake: a 768 column + 1024-d CLOBs -> recreate then migrate all ----
def test_migrate_after_recreate_writes_all_rows():
    rows = [(i, json.dumps([0.1] * 1024)) for i in range(3)]
    cur, conn = FakeCursor(vec_col_dim=768, clob_rows=rows), FakeConn()
    vm.ensure_embedding_vec_column(cur, conn, 1024)             # 768 -> recreated at 1024
    updated, skipped = vm.migrate_clob_to_vector(cur, conn, 1024)
    assert (updated, skipped) == (3, 0)
    assert cur.updates == [0, 1, 2]                            # every chunk migrated


def test_migrate_skips_wrong_dimension_rows():
    rows = [(1, json.dumps([0.1] * 768)), (2, json.dumps([0.2] * 1024))]
    cur, conn = FakeCursor(vec_col_dim=1024, clob_rows=rows), FakeConn()
    updated, skipped = vm.migrate_clob_to_vector(cur, conn, 1024)
    assert updated == 1 and skipped == 1                       # only the 1024-d row migrates
    assert cur.updates == [2]
