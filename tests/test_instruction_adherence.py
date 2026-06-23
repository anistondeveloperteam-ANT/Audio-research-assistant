"""The solution must follow the request's EXPLICIT instructions — print the EXACT named output
quantities and use the SPECIFIED method — not a plausible substitute. Universal across tasks/domains.

Proves:
  (a) when a request names specific output quantities, code that prints DIFFERENT quantities is caught
      as incomplete (completeness gate); code that prints the named ones is verified;
  (b) when a request specifies a METHOD, code using a different method produces different values and
      FAILS the named-method reference; code using the specified method is verified;
  (c)+(d) the prompts that drive generation / requirements / reference / checks carry the adherence
      rules so the named outputs and named method are required and enforced.

Deterministic: no network, no Docker, no real LLM.
"""
import types

from backend.agent import loop


def _res(stdout, ok=True):
    return types.SimpleNamespace(ok=ok, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


# ======================================================================================
# Prompt content: named outputs + specified method are required end-to-end.
# ======================================================================================
def test_prompts_require_named_outputs_and_specified_method():
    gen = loop._GEN_SYSTEM.lower()
    assert "exactly the named outputs" in gen and "specified method" in gen and "do not substitute" in gen
    req = loop._REQ_SYSTEM.lower()
    assert "named method" in req and "must-use" in req
    ref = loop._REFERENCE_SYSTEM.lower()
    assert "must use that" in ref and "spec is authoritative" in ref          # named method authoritative
    deliv = loop._DELIVERABLES_SYSTEM.lower()
    assert "exact stated meaning" in deliv and "not a related-but-different" in deliv
    defn = loop._DEFINITION_SYSTEM.lower()
    assert "specifies a method" in defn and "different method" in defn


def _base_env(monkeypatch, **over):
    base = {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
            "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
            "AGENT_DEFINITION_GATE": "false", "AGENT_DELIVERY_GATES": "false",
            "AGENT_ROOT_CAUSE_DIAGNOSIS": "false", "AGENT_ANTICHEAT_SCAN": "false",
            "AGENT_MASKING_SCAN": "false", "AGENT_TEST_CRITIC": "false", "AGENT_PARALLEL_N": "1",
            "AGENT_VERIFY_SEEDS": "1", "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "1",
            "AGENT_STALL_LIMIT": "1"}
    base.update(over)
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


# ======================================================================================
# (a) NAMED OUTPUTS — the completeness gate enforces the exact quantity the request names.
# ======================================================================================
class _OutP:
    is_available = True
    name, model = "openai", "test"

    def __init__(self, gen):
        self.gen = gen

    def stream_chat(self, messages, system="", **k):
        if system == loop._REQ_SYSTEM:
            return ["- compute() and PRINT the sum of coefficients"]
        if system == loop._DELIVERABLES_SYSTEM:
            return ["sum of coefficients"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_basic():\n    assert compute() is not None\n"]
        if system == loop._GEN_SYSTEM:
            return [self.gen]
        if system == loop._DRIVER_SYSTEM:
            return ["print(compute())"]
        return [""]


class _OutRunner:
    def __call__(self, code, **k):
        if "ModuleType('_sol')" in code:                         # visible test harness
            return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")
        if "RIGHT_OUT" in code:                                  # demo capture of the right solution
            return _res("sum of coefficients = 5.0\n")
        return _res("peak = 5.0\n")                              # WRONG_OUT prints a quantity never asked


def test_wrong_output_quantity_is_flagged_incomplete(monkeypatch):
    _base_env(monkeypatch, AGENT_DELIVERY_GATES="true")
    gen = ("def compute():\n    return 5.0  # WRONG_OUT\n"
           "if __name__ == '__main__':\n    print('peak =', compute())\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _OutP(gen))
    monkeypatch.setattr(loop, "run_python_auto", _OutRunner())
    res = loop.run_agent("compute and print the sum of coefficients", use_search=False)
    assert res.verification != "verified"                       # the named output is absent -> incomplete


def test_correct_output_quantity_is_verified(monkeypatch):
    _base_env(monkeypatch, AGENT_DELIVERY_GATES="true")
    gen = ("def compute():\n    return 5.0  # RIGHT_OUT\n"
           "if __name__ == '__main__':\n    print('sum of coefficients =', compute())\n")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _OutP(gen))
    monkeypatch.setattr(loop, "run_python_auto", _OutRunner())
    res = loop.run_agent("compute and print the sum of coefficients", use_search=False)
    assert res.verification == "verified"                       # prints exactly the named quantity


# ======================================================================================
# (b) NAMED METHOD — the reference uses the specified method, so a different-method candidate's values
# diverge and fail; the specified-method candidate matches and is verified.
# ======================================================================================
class _MethP:
    is_available = True
    name, model = "openai", "test"

    def __init__(self, gen):
        self.gen = gen

    def stream_chat(self, messages, system="", **k):
        if system == loop._REFERENCE_SYSTEM:
            return ["def filter_signal(x):\n    return x  # USES_CONV: filter the signal via convolution\n"]
        if system == loop._REQ_SYSTEM:
            return ["- filter_signal(x): apply the filter using CONVOLUTION; return the filtered signal"]
        if system == loop._TESTS_SYSTEM:
            return ["def test_method():\n    import numpy as np\n"
                    "    assert filter_signal(np.arange(5.0)) is not None\n"]
        if system == loop._GEN_SYSTEM:
            return [self.gen]
        return [""]


class _MethRunner:
    """The reference uses convolution (USES_CONV). A candidate that substitutes a causal filter
    (USES_CAUSAL) produces different values -> the named-method test FAILS. USES_CAUSAL only ever
    appears when the candidate is the wrong method (the reference is always convolution)."""
    def __call__(self, code, **k):
        if "held-out runner (seeded)" in code:
            return _res("TESTS_PASSED 1/1\n")
        if "ModuleType('_sol')" in code:
            ok = "USES_CAUSAL" not in code
            return _res(f"TEST test_method {'PASS' if ok else 'FAIL'}\nTESTS_PASSED {1 if ok else 0}/1\n")
        return _res("TESTS_PASSED 0/0\n")


def test_wrong_method_fails_via_named_method_reference(monkeypatch):
    _base_env(monkeypatch, AGENT_REFERENCE_TESTS="true", AGENT_TEST_VALIDATION="true")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _MethP(
        "def filter_signal(x):\n    return x  # USES_CAUSAL: causal filter the signal (wrong method)\n"))
    monkeypatch.setattr(loop, "run_python_auto", _MethRunner())
    res = loop.run_agent("apply the filter using convolution; return the filtered signal", use_search=False)
    assert res.verification != "verified"                       # different method -> values diverge -> fails


def test_specified_method_is_verified(monkeypatch):
    _base_env(monkeypatch, AGENT_REFERENCE_TESTS="true", AGENT_TEST_VALIDATION="true")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: _MethP(
        "def filter_signal(x):\n    return x  # USES_CONV: filter the signal via convolution\n"))
    monkeypatch.setattr(loop, "run_python_auto", _MethRunner())
    res = loop.run_agent("apply the filter using convolution; return the filtered signal", use_search=False)
    assert res.verification == "verified"                       # uses the specified method -> matches
