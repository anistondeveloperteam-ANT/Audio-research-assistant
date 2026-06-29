"""Optional semantic chunking (SEMANTIC_CHUNKER): topic-coherent prose boundaries via Chonkie +
Model2Vec, with a fail-safe fallback to the legacy sentence-packer. The real model is never loaded
here — the chunker is mocked, so the suite stays fast and fully offline."""
import backend.ingestion.semantic_chunker as sc
import backend.ingestion.document_chunker as dc


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakeChunker:
    """Stand-in for Chonkie's SemanticChunker: splits on '||' so tests need no model."""
    def __call__(self, text):
        return [_FakeChunk(s) for s in text.split("||")]


# ---- enable flag ----
def test_semantic_chunker_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIC_CHUNKER", raising=False)
    assert sc.semantic_chunker_enabled() is False
    monkeypatch.setenv("SEMANTIC_CHUNKER", "true")
    assert sc.semantic_chunker_enabled() is True


# ---- semantic_split contract ----
def test_semantic_split_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SEMANTIC_CHUNKER", raising=False)
    assert sc.semantic_split("some real text here") is None          # off -> caller uses legacy


def test_semantic_split_none_when_chunker_unavailable(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CHUNKER", "true")
    monkeypatch.setattr(sc, "_get_chunker", lambda: None)            # chonkie/model missing
    assert sc.semantic_split("some real text") is None


def test_semantic_split_uses_chunker_when_enabled(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CHUNKER", "true")
    monkeypatch.setattr(sc, "_get_chunker", lambda: _FakeChunker())
    assert sc.semantic_split("alpha || beta || gamma") == ["alpha", "beta", "gamma"]


def test_semantic_split_empty_result_returns_none(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CHUNKER", "true")

    class _Empty:
        def __call__(self, text):
            return []

    monkeypatch.setattr(sc, "_get_chunker", lambda: _Empty())
    assert sc.semantic_split("text") is None                         # no segments -> fall back


def test_semantic_split_failsafe_on_error(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CHUNKER", "true")

    class _Boom:
        def __call__(self, text):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(sc, "_get_chunker", lambda: _Boom())
    assert sc.semantic_split("text") is None                         # never raises into ingest


# ---- config knobs ----
def test_threshold_and_chunk_size_defaults_and_guards(monkeypatch):
    monkeypatch.delenv("SEMANTIC_CHUNKER_THRESHOLD", raising=False)
    monkeypatch.delenv("SEMANTIC_CHUNK_SIZE", raising=False)
    assert sc._threshold() == 0.75
    assert sc._chunk_size() == 512
    monkeypatch.setenv("SEMANTIC_CHUNKER_THRESHOLD", "2.0")          # clamped into 0-1
    assert sc._threshold() == 1.0
    monkeypatch.setenv("SEMANTIC_CHUNK_SIZE", "notanint")            # bad value -> default
    assert sc._chunk_size() == 512


# ---- integration with the document chunker ----
def test_chunk_text_uses_semantic_segments_when_enabled(monkeypatch):
    seg1 = "This is the first coherent topic that is comfortably over the minimum chunk length. " * 2
    seg2 = "This is a second, semantically distinct topic that also clears the minimum length easily. " * 2
    monkeypatch.setattr(dc, "semantic_split", lambda text: [seg1, seg2])
    chunks = dc.chunk_text("ignored body", section="Methods", page_start=2, page_end=3, parser="pymupdf")
    assert len(chunks) == 2
    assert chunks[0]["text"].startswith("This is the first coherent topic")
    assert chunks[0]["section"] == "Methods" and chunks[0]["page_start"] == 2   # schema preserved


def test_chunk_text_drops_tiny_semantic_segments(monkeypatch):
    monkeypatch.setattr(dc, "semantic_split", lambda text: ["tiny", "x" * 200])
    chunks = dc.chunk_text("ignored", section="X")
    assert len(chunks) == 1 and len(chunks[0]["text"]) >= dc.MIN_CHUNK_CHARS   # 'tiny' dropped


def test_chunk_text_falls_back_to_legacy_when_semantic_none(monkeypatch):
    monkeypatch.setattr(dc, "semantic_split", lambda text: None)               # semantic off/failed
    text = ("First sentence is clearly long enough to survive. Second sentence also has plenty of "
            "characters to be kept around. Third sentence rounds out the legacy packed chunk nicely.")
    chunks = dc.chunk_text(text, section="Intro")
    assert len(chunks) >= 1 and chunks[0]["section"] == "Intro"                # legacy path ran
