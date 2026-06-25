"""COMPUTE-IN-CODE SOURCE OF TRUTH for any quantitative answer.

The system evaluates the answer's OWN shown arithmetic DETERMINISTICALLY (no LLM, no eval()) and makes
that computed value authoritative: a model-asserted number that differs is OVERRIDDEN, a dropped-factor
final number is corrected to what the formula actually yields, every shown equality is made literally
true, all restatements of the result agree, and unit/convention/magnitude problems are flagged. General
by construction — proven here across storage, finance, physics, geometry and statistics, none privileged
(computing-and-comparing catches every error class without enumerating them).

Proves the spec's (a)-(g); topic-agnostic. Deterministic + offline (the live-path case mocks the model).
"""
import webapp.chat_logic as cl
from backend.answering.arithmetic_check import (
    _find_calc_equalities,
    safe_eval,
    verify_calculation,
)
from backend.memory.store import MemoryStore


def _all_equalities_true(text: str) -> bool:
    """Every shown 'EXPR = NUM' in `text` is literally true (stated == computed at the shown precision)."""
    return all(abs(round(e.computed, e.decimals) - e.stated) <= 1e-9 for e in _find_calc_equalities(text))


# ======================================================================================
# safe_eval — a correct, SAFE evaluator (computes arithmetic; rejects anything that isn't).
# ======================================================================================
def test_safe_eval_computes_and_rejects_non_arithmetic():
    assert safe_eval("44100 * 16 * 2 * 180 / 8") == 31752000
    assert safe_eval("(3 + 4) * 2") == 14
    assert safe_eval("2 ^ 10") == 1024
    assert safe_eval("10 / 0") is None              # never 'fix' a result to infinity
    assert safe_eval("rm -rf /") is None            # not arithmetic -> no value
    assert safe_eval("__import__('os').system('x')") is None


# ======================================================================================
# (a) the computed value is the source of truth; a differing ASSERTED number is overridden.
# ======================================================================================
def test_a_model_asserted_number_is_overridden_by_the_computed_value():
    r = verify_calculation("Area = 12 * 8 = 100 sq m.")      # model asserts 100; 12*8 = 96
    assert "12 * 8 = 96" in r.fixed_text and "100" not in r.fixed_text
    assert ("100", "96") in r.corrections                    # the slip was overridden, not trusted


# ======================================================================================
# (b) a shown FORMULA is evaluated; a dropped-factor final number is corrected to its actual value.
# ======================================================================================
def test_b_dropped_factor_in_a_formula_is_corrected():
    # storage: bits / 8 -> bytes; the model kept the pre-/8 value as the result (a dropped factor).
    r = verify_calculation("Size = 44100 x 16 x 2 x 180 / 8 = 254016000 bytes.")
    assert "= 31752000 bytes" in r.fixed_text                # corrected to the formula's actual value
    assert "254016000" not in r.fixed_text


# ======================================================================================
# (c) every shown arithmetic equality is literally TRUE after the check — across domains.
# ======================================================================================
def test_c_every_shown_equality_is_true_after_check():
    for draft in (
        "Interest = 2000 * 5 / 100 = 150 dollars.",          # finance  -> 100
        "Force = 1200 * 9 = 12000 newtons.",                 # physics  -> 10800
        "Mean = (4 + 8 + 6) / 3 = 9.",                       # stats    -> 6
        "Volume = 2 * 3 * 4 = 30 cubic cm.",                 # geometry -> 24
    ):
        r = verify_calculation(draft)
        assert _all_equalities_true(r.fixed_text), r.fixed_text


# ======================================================================================
# (d) every stated INSTANCE of the result agrees (summary / body / headline).
# ======================================================================================
def test_d_a_result_shown_two_ways_agrees_everywhere():
    # the result is computed twice in the WORK with the same slip; each equality is independently
    # corrected to the one computed value, so no contradictory SHOWN figures remain.
    r = verify_calculation("Area = 12 * 8 = 100 sq m. Check: 8 * 12 = 100 sq m.")
    assert r.fixed_text.count("96") == 2 and "100" not in r.fixed_text


def test_d_correcting_one_equality_never_disturbs_a_correct_one():
    # SAFETY (adversarial): fixing one slip must never rewrite another equation's figures — and never
    # turn a literally-true equality ('1 + 4 = 5') false.
    r = verify_calculation("First 2 * 2 = 5 kg. Second 1 + 4 = 5 kg.")
    assert "2 * 2 = 4 kg" in r.fixed_text and "1 + 4 = 5 kg" in r.fixed_text


def test_d_unrelated_same_unit_quantity_is_NOT_touched():
    # SAFETY (adversarial): a correct equality next to a DIFFERENT quantity that shares its unit is left
    # ALONE — the engine only rewrites the RHS of an equality it proved wrong, never free prose.
    for s in (
        "Floor 8 * 8 = 64 sq m. The answer is 40 sq m for the patio.",
        "The weight is 2 * 2 = 4 kg. Separately, the box holds 5 kg of sand.",
    ):
        assert verify_calculation(s).fixed_text == s, s


# ======================================================================================
# (e) unit/convention consistency + magnitude sanity.
# ======================================================================================
def test_e_mixed_binary_decimal_convention_is_flagged_not_verified():
    ok = verify_calculation("5 KiB = 5 * 1024 = 5120 bytes.")
    assert ok.convention_ok and ok.verified
    mixed = verify_calculation("One step uses 4 * 1024 = 4096; another uses 4 * 1000 = 4000.")
    assert not mixed.convention_ok and not mixed.verified     # decimal vs binary mixed -> not verified


def test_e_dropped_factor_restores_the_correct_magnitude():
    # a dropped /1000 leaves the stated value 1000x too large; the computed value restores the magnitude.
    r = verify_calculation("Rate = 8000000 / 1000 = 8000000 kb per s.")
    assert "= 8000 kb" in r.fixed_text and "8000000 kb" not in r.fixed_text


# ======================================================================================
# (f) the computation RUNS ON THE LIVE calculation path and is not skipped.
# ======================================================================================
class _Trace:
    def set(self, **k):
        return self

    def end(self):
        pass


class _P:
    is_available = True
    model = "fake"

    def __init__(self, draft):
        self.draft = draft

    def stream_chat(self, messages, system="", **k):
        if "answer-quality judge" in (system or "").lower():     # the reasoning verify (dependent)
            return ['{"ok": true, "score": 95}']
        return [self.draft]

    def unavailable_message(self):
        return "n/a"


def test_f_runs_on_the_live_reasoning_path_overriding_a_model_slip(monkeypatch, tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, "area of a 12 by 8 plot?")
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    draft = "The plot area is 12 * 8 = 100 sq m."                          # planted slip (96, not 100)
    monkeypatch.setattr(cl, "get_provider", lambda: _P(draft))
    events = list(cl._reasoning_fallback("area of a 12 by 8 plot?", mem, sid, qid["turn_id"],
                                         qid["node_id"], "u", False, None, None, _Trace()))
    done = [e for e in events if e["type"] == "done"][0]
    assert "12 * 8 = 96" in done["answer"] and "100" not in done["answer"]   # computed in code + overridden


# ======================================================================================
# (g) a CORRECT calculation passes through unchanged (no false corrections).
# ======================================================================================
def test_g_correct_calculation_is_unchanged():
    good = ("Bits = 44100 x 16 x 2 x 180 = 254016000. Bytes = 254016000 / 8 = 31752000 bytes "
            "= about 30 MB.")
    r = verify_calculation(good)
    assert r.fixed_text == good and not r.corrections and r.verified


def test_g_innocent_non_arithmetic_text_is_never_touched():
    # version numbers, ranges, percentages, |a-b|, comma-numbers, model names — none are arithmetic slips.
    for s in (
        "GPT-4 scored 8 / 10 = 80% on the benchmark.",
        "The gap 3 - 5 = 2 points narrowed.",
        "Sections 3.2 * 4 = 12 are out of scope.",         # '3.2' is a section number
        "Revenue rose 1000 * 2 = 2,000 over the period.",  # comma-grouped result is correct
        "Use Python 3 and label v2 here.",
    ):
        assert verify_calculation(s).fixed_text == s, s


def test_g_expr_equals_expr_is_never_corrupted():
    # SAFETY (adversarial): when the RHS is itself an expression ('a = b + c', a factorisation
    # 'a = b * c = d'), the line must be left ALONE — overriding the RHS's first operand would
    # fabricate a false equality.
    for s in (
        "Then 10 * 2 = 5 * 4 holds.",
        "Compare 2 * 2 = 1 + 3 here.",
        "Cross-check: 15 * 4 = 30 * 2 = 60.",   # a valid factorization, in prose
        "Split 100 / 4 = 5 * 5 each.",
        "So 20 + 5 = 30 − 5 here.",        # Unicode minus (U+2212) — common in LaTeX/Word/LLM output
        "Mul 10 * 2 = 5 ⋅ 4 done.",        # dot operator (LaTeX \\cdot)
        "Sum 2 * 2 = 1 ＋ 3 ok.",           # fullwidth plus
    ):
        assert verify_calculation(s).fixed_text == s, s


def test_g_leading_sign_and_markdown_bullets_are_never_corrupted():
    # SAFETY (adversarial): a leading '-' from a markdown bullet or a signed first operand must not be
    # read as a unary minus that injects a negative result.
    for s in (
        "- 2 * 2 = 5 items remaining.",       # markdown bullet, not "-2 * 2 = -4"
        "- 3 * 4 = 10 units done.",
        "Change: -5 * 3 = 20 dollars.",       # signed first operand -> ambiguous, leave alone
        "Net effect -4 * 2 = 5 overall.",
    ):
        assert verify_calculation(s).fixed_text == s, s
