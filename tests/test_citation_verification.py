"""
CITATION-VERIFICATION GATE — before any citation reaches the user it must (1) reference a source that
EXISTS and (2) genuinely SUPPORT the claim it's attached to. Fabricated references (a provably-bogus
DOI/arXiv-id) and misattributed citations (a source — external OR corpus — that doesn't support the
claim) are removed; a legitimately-unindexed source is advisory (kept); lookups are deterministic,
cached, bounded, and fail-open.

Proves the spec's (a)-(g): (a) bogus id -> fabricated -> removed; (b) valid id -> exists; (c) unindexed
-> unresolvable -> kept; (d) unsupported source -> misattributed -> removed (external + corpus);
(e) general-knowledge claim ends with NO citation rather than an irrelevant corpus one; (f) lookups
deterministic + cached + fail-open on transient errors; (g) a real supporting source keeps its citation.
Plus a 4-answer demo. All lookups + the support judge are mocked — no network.
"""
import pytest

import backend.external_search.biblio_lookup as biblio
from backend.answering import citation_verifier as cv


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("CITATION_VERIFICATION", "true")     # conftest disables it suite-wide


def test_has_fabricated_flags_only_fabricated_removals():
    # A FABRICATED (provably-bogus) cited source means a hallucinated reference -> answer is NOT verified.
    assert cv.has_fabricated([(3, "fabricated")]) is True
    assert cv.has_fabricated([(2, "misattributed"), (5, "fabricated")]) is True
    # Misattributed-only (source exists, content sound, wrong citation dropped) does NOT flip verified.
    assert cv.has_fabricated([(2, "misattributed")]) is False
    assert cv.has_fabricated([]) is False
    assert cv.has_fabricated(None) is False


class _Provider:
    """Fake LLM support-judge: returns a fixed {"supported": [[claim_no, source_no], ...]} verdict."""
    is_available = True
    model = "fake"

    def __init__(self, supported_json: str):
        self._json = supported_json

    def stream_chat(self, messages, system="", **k):
        return [self._json]


def _src(n, **kw):
    s = {"n": n, "source_type": "web", "url": "", "title": f"S{n}", "text": f"text {n}"}
    s.update(kw)
    return s


_ARXIV = "https://arxiv.org/abs/2401.12345"


# ---- identifier extraction ----
def test_extract_identifier_doi_and_arxiv():
    assert cv.extract_identifier({"url": _ARXIV}) == ("arxiv", "2401.12345")
    assert cv.extract_identifier({"url": "https://doi.org/10.1145/3292500.3330701"}) == \
        ("doi", "10.1145/3292500.3330701")
    assert cv.extract_identifier({"doi": "10.1000/xyz123"})[0] == "doi"
    assert cv.extract_identifier({"url": "https://example.com/page", "source_type": "web"}) == (None, None)
    assert cv.extract_identifier({"source_type": "local_pdf", "url": ""}) == (None, None)


# ---- (a) a provably-bogus id is detected as fabricated and removed ----
def test_bogus_arxiv_id_is_fabricated_and_removed(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: False)   # definitive not-found
    answer = "Beamforming improves SNR [1]."
    clean, removed = cv.verify_citations(None, answer=answer, sources=[_src(1, url=_ARXIV)])
    assert removed == [(1, "fabricated")]
    assert "[1]" not in clean and clean.strip() == "Beamforming improves SNR."


# ---- (b) a real source with a valid id passes existence ----
def test_valid_id_passes_existence(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "MVDR is a beamformer [1]."
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'),
                                         answer=answer, sources=[_src(1, url=_ARXIV)])
    assert removed == [] and clean == answer


# ---- (c) a legitimately-unindexed source is 'unresolvable' (advisory) — NOT removed ----
def test_unindexed_source_is_advisory_not_removed(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: None)    # transient / can't tell
    # source WITH an id but unresolvable, and a book WITHOUT any id (existence implicit) — both kept
    answer = "Claim one [1]. Claim two [2]."
    sources = [_src(1, url=_ARXIV), _src(2, source_type="web", url="https://example.com/book")]
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1], [2, 2]]}'),
                                         answer=answer, sources=sources)
    assert removed == [] and clean == answer


def test_verify_existence_classifies(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: False)
    monkeypatch.setattr(biblio, "doi_exists", lambda _d: None)
    sources = [_src(1, url=_ARXIV),                                   # bogus arxiv -> not_found
               _src(2, url="https://doi.org/10.1234/x"),             # doi transient -> unresolvable
               _src(3, source_type="web", url="https://example.com")]  # no id -> exists
    out = cv.verify_existence(sources, {1, 2, 3})
    assert out == {1: "not_found", 2: "unresolvable", 3: "exists"}


# ---- (d) a source that does NOT support the claim is misattributed and removed (external + corpus) ----
@pytest.mark.parametrize("stype,url", [("web", "https://example.com/x"), ("local_pdf", "")])
def test_unsupported_source_is_misattributed_and_removed(stype, url):
    answer = "The capital of France is Paris [1]."
    sources = [_src(1, source_type=stype, url=url, title="Audio DSP paper",
                    text="This paper is about microphone array beamforming, unrelated to geography.")]
    clean, removed = cv.verify_citations(_Provider('{"supported": []}'), answer=answer, sources=sources)
    assert removed == [(1, "misattributed")]
    assert "[1]" not in clean and clean.strip() == "The capital of France is Paris."


# ---- (e) a general-knowledge claim ends with NO citation rather than an irrelevant corpus one ----
def test_general_knowledge_claim_loses_irrelevant_corpus_citation():
    answer = "Water boils at 100 degrees Celsius at sea level [1]."
    corpus = [_src(1, source_type="local_pdf", url="", title="Speech enhancement survey",
                   text="A survey of deep-learning speech enhancement methods.")]
    clean, removed = cv.verify_citations(_Provider('{"supported": []}'), answer=answer, sources=corpus)
    assert removed == [(1, "misattributed")]
    assert "[" not in clean                                    # the fact now stands with no citation


# ---- (f) deterministic + cached + fail-open ----
def test_biblio_lookup_is_cached_and_deterministic(monkeypatch):
    store = {}
    monkeypatch.setattr(biblio, "cache_get", lambda k, ttl=0: store.get(k))
    monkeypatch.setattr(biblio, "cache_set", lambda k, v: store.__setitem__(k, v))
    calls = []
    monkeypatch.setattr(biblio, "_status_get", lambda url, **k: (calls.append(url), 200)[1])
    assert biblio.doi_exists("10.1234/abc") is True
    assert biblio.doi_exists("10.1234/abc") is True            # second call served from cache
    assert len(calls) == 1                                     # network hit exactly once (deterministic)


def test_fail_open_on_transient_lookup_error(monkeypatch):
    def _boom(_id):
        raise RuntimeError("network down")
    monkeypatch.setattr(biblio, "arxiv_id_exists", _boom)
    answer = "Claim [1]."
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'),
                                         answer=answer, sources=[_src(1, url=_ARXIV)])
    assert removed == [] and clean == answer                   # transient -> unresolvable -> kept


def test_fail_open_when_support_judge_raises(monkeypatch):
    class _Raise:
        is_available = True
        model = "x"
        def stream_chat(self, *a, **k):
            raise RuntimeError("provider down")
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "Claim [1]."
    clean, removed = cv.verify_citations(_Raise(), answer=answer, sources=[_src(1, url=_ARXIV)])
    assert removed == [] and clean == answer                   # support fail-open -> keep


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("CITATION_VERIFICATION", "false")
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: False)
    answer = "Claim [1]."
    clean, removed = cv.verify_citations(None, answer=answer, sources=[_src(1, url=_ARXIV)])
    assert removed == [] and clean == answer


# ---- regression: code/math literals are never parsed or rewritten as citations ----
def test_code_intact_when_a_citation_is_removed(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "Claim A [1].\n```\nM = [[1, 0], [0, 1]]\n```\nClaim B [1]."
    sources = [_src(1, url=_ARXIV, text="supports only claim A")]
    # judge: claim 1 (A) supported, claim 2 (B) not -> drop [1] from B's sentence only; code untouched.
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'), answer=answer, sources=sources)
    assert removed == [(1, "misattributed")]
    assert "[[1, 0], [0, 1]]" in clean              # matrix literal intact (not seen as citations)
    assert "Claim A [1]." in clean                  # supported citation kept
    assert "Claim B." in clean and "Claim B [1]" not in clean


def test_array_index_and_inline_code_not_seen_as_citations(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "Use arr[0] and `coef = [1, 2]` here [1]."
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'),
                                         answer=answer, sources=[_src(1, url=_ARXIV, text="t")])
    assert removed == [] and clean == answer        # arr[0] / inline [1,2] untouched; real [1] kept


# ---- regression: identifier extraction is HOST-anchored (no false removal of real web sources) ----
def test_identifier_extraction_is_host_anchored():
    for url in ("https://example.com/?ref=arxiv.org/abs/2401.12345",
                "https://example.com/r?u=https://doi.org/10.1234/x",
                "https://mydoi.org/10.1234/x", "https://sub.arxiv.org/abs/2401.12345",
                "https://doi.org.evil.com/10.1234/x"):
        assert cv.extract_identifier({"url": url}) == (None, None), url
    assert cv.extract_identifier({"url": "https://arxiv.org/abs/2401.12345"}) == ("arxiv", "2401.12345")
    assert cv.extract_identifier({"url": "https://doi.org/10.1234/abc"}) == ("doi", "10.1234/abc")


def test_real_web_source_with_doi_like_url_is_not_removed(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: False)   # would 404 if naively extracted
    answer = "Transformers improved ASR accuracy [1]."
    src = _src(1, source_type="web", url="https://scholar.example.com/?cite=arxiv.org/abs/2401.99999",
               text="Transformers improved ASR accuracy.")
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'), answer=answer, sources=[src])
    assert removed == [] and clean == answer        # host isn't arxiv.org -> existence implicit -> kept


# ---- regression: a source with no excerpt text is kept (can't fairly judge from a title) ----
def test_empty_excerpt_source_is_kept(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "Some claim [1]."
    clean, removed = cv.verify_citations(_Provider('{"supported": []}'),
                                         answer=answer, sources=[_src(1, url=_ARXIV, title="A paper", text="")])
    assert removed == [] and clean == answer


# ---- regression: grouped citation drops only the bad member ----
def test_grouped_citation_drops_only_unsupported_member(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "Claim [1, 2]."
    sources = [_src(1, url=_ARXIV, text="supports"),
               _src(2, url="https://arxiv.org/abs/2402.54321", text="off-topic")]
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'), answer=answer, sources=sources)
    assert removed == [(2, "misattributed")]
    assert "[1]" in clean and "[1, 2]" not in clean and "[2]" not in clean


# ---- (g) a claim with a real supporting source keeps its correct citation ----
def test_real_supporting_source_is_kept(monkeypatch):
    monkeypatch.setattr(biblio, "arxiv_id_exists", lambda _id: True)
    answer = "WPE reduces reverberation [1]."
    sources = [_src(1, url=_ARXIV, title="WPE dereverberation",
                    text="Weighted prediction error (WPE) reduces late reverberation in speech.")]
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1]]}'), answer=answer, sources=sources)
    assert removed == [] and clean == answer


# ---- DEMO: one answer mixing fabricated / misattributed-corpus / general-knowledge / genuinely-sourced ----
def test_demo_mixed_answer(monkeypatch):
    # source 1: bogus arxiv (fabricated); source 2: corpus, unrelated (misattributed); source 3: real arxiv
    # that supports its claim (kept). A 4th claim is general knowledge with no citation (stays clean).
    monkeypatch.setattr(biblio, "arxiv_id_exists",
                        lambda _id: {"2401.00001": False, "2401.99999": True}.get(_id))
    answer = ("Claim A about method X [1]. Claim B about geography [2]. "
              "Claim C about WPE dereverberation [3]. Claim D is common knowledge.")
    sources = [
        _src(1, url="https://arxiv.org/abs/2401.00001", title="X", text="about method X"),
        _src(2, source_type="local_pdf", url="", title="Audio survey", text="speech enhancement survey"),
        _src(3, url="https://arxiv.org/abs/2401.99999", title="WPE", text="WPE dereverberation works"),
    ]
    # support judge: only claim C <- source 3 genuinely supports; A's source exists-checks out of band.
    clean, removed = cv.verify_citations(_Provider('{"supported": [[1, 1], [3, 3]]}'),
                                         answer=answer, sources=sources)
    reasons = dict(removed)
    assert reasons.get(1) == "fabricated"        # bogus id removed regardless of support
    assert reasons.get(2) == "misattributed"     # corpus paper didn't support the geography claim
    assert "[3]" in clean                         # the genuinely-sourced claim keeps its citation
    assert "[1]" not in clean and "[2]" not in clean
    assert clean.rstrip().endswith("common knowledge.")   # the un-cited fact is untouched
