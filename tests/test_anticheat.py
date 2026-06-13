"""Static anti-reward-hacking scanner: one test per rule, plus clean code that must NOT flag."""
from backend.agent import anticheat
from backend.agent.anticheat import scan_for_cheating


# ---- the rules each fire on the pattern they target -------------------
def test_flags_function_that_ignores_its_arguments():
    code = "def black_scholes(S, K, T, r, sigma):\n    return 10.4506\n"
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("ignores its arguments" in r for r in rep.reasons)


def test_flags_hardcoded_value_equal_to_test_expected():
    tests = "def test_call():\n    assert price(100, 100, 1, 0.05, 0.2) == 10.4506\n"
    code = "def price(s, k, t, r, sig):\n    x = s + k\n    return 10.4506\n"
    rep = scan_for_cheating(code, tests_code=tests)
    assert rep.flagged
    assert any("expected result" in r for r in rep.reasons)


def test_flags_enumerating_specific_inputs():
    code = (
        "def solve(x):\n"
        "    if x == 7919:\n"
        "        return 999.5\n"
        "    elif x == 6361:\n"
        "        return 123.4\n"
        "    return 0.0\n"
    )
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("enumerates specific inputs" in r for r in rep.reasons)


def test_flags_referencing_the_test_functions():
    code = "def f(x):\n    if 'test_foo' in dir():\n        return 1\n    return test_foo()\n"
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("test function" in r for r in rep.reasons)


def test_flags_reading_a_file_to_peek():
    code = "def f(x):\n    data = open('tests.py').read()\n    return len(data)\n"
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("open(" in r or "peeking" in r for r in rep.reasons)


def test_flags_harness_tampering_sys_modules():
    code = "import sys\ndef f(x):\n    sys.modules['x'] = None\n    return x\n"
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("sys.modules" in r for r in rep.reasons)


def test_flags_sleep_timing_trick():
    code = "import time\ndef f(x):\n    time.sleep(0.001)\n    return x\n"
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("sleep" in r for r in rep.reasons)


def test_flags_overriding_print():
    code = "def f(x):\n    return x\nprint = lambda *a, **k: None\n"
    rep = scan_for_cheating(code, tests_code="")
    assert rep.flagged
    assert any("print" in r for r in rep.reasons)


# ---- clean, genuine code must NOT be flagged (precision) --------------
def test_clean_blackscholes_not_flagged():
    tests = ("def test_call():\n    assert abs(bs(100, 100, 1, 0.05, 0.2, 'call') - 10.4506) < 1e-3\n")
    code = (
        "import math\n"
        "def _ncdf(x):\n"
        "    return 0.5 * (1 + math.erf(x / math.sqrt(2)))\n"
        "def bs(S, K, T, r, sigma, kind='call'):\n"
        "    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))\n"
        "    d2 = d1 - sigma * math.sqrt(T)\n"
        "    if kind == 'call':\n"
        "        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)\n"
        "    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)\n"
    )
    rep = scan_for_cheating(code, tests_code=tests)
    assert not rep.flagged, rep.reasons


def test_no_arg_constant_function_not_flagged_as_hardcode():
    # A constant-valued task legitimately returns a constant from a NO-ARG function; only a
    # function that takes inputs and returns the expected literal is hardcoding.
    tests = "def test_g():\n    assert gravity() == 9.81\n"
    assert not scan_for_cheating("def gravity():\n    return 9.81\n", tests_code=tests).flagged
    # but a function WITH params returning that same literal IS flagged
    assert scan_for_cheating("def g(x):\n    return 9.81\n", tests_code=tests).flagged


def test_clean_recursion_with_trivial_base_cases_not_flagged():
    # fib has two `== 0/1 -> return 0/1` branches: trivial constants must not trip the rule.
    code = ("def fib(n):\n"
            "    if n == 0:\n        return 0\n"
            "    if n == 1:\n        return 1\n"
            "    return fib(n - 1) + fib(n - 2)\n")
    rep = scan_for_cheating(code, tests_code="")
    assert not rep.flagged, rep.reasons


def test_unparseable_code_is_not_a_cheat():
    rep = scan_for_cheating("def f(:\n  return", tests_code="")
    assert rep.flagged is False and rep.reasons == []


def test_anticheat_enabled_env_toggle(monkeypatch):
    monkeypatch.setenv("AGENT_ANTICHEAT_SCAN", "false")
    assert anticheat.anticheat_enabled() is False
    monkeypatch.setenv("AGENT_ANTICHEAT_SCAN", "true")
    assert anticheat.anticheat_enabled() is True
