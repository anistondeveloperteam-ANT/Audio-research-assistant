"""Root-cause diagnosis in the refine loop: on a failed check a SEPARATE DIAGNOSIS agent must DIAGNOSE
the mechanism (map the symptom to the likely code cause) and emit a structured FIX DIRECTIVE the
generation agent then applies — instead of the single agent mutating blindly — and it must never make
a conserved quantity pass by MASKING (forced renormalisation / clamping).

Proves:
  (a) a failing conservation/invariant check is diagnosed to the UPDATE / INTEGRATION RULE and the
      corrected (e.g. symplectic) scheme is accepted — not a tolerance tweak;
  (b) a wrong-sign / wrong-direction result is diagnosed to the GOVERNING EQUATION and corrected;
  (c) a check that passes only via forced renormalisation is detected as MASKING and rejected;
  (d) on a true stall the loop stops honestly as "partial" with the diagnosed reason;
  (e) the diagnosis agent is a structured critic (FAILING CHECK / MECHANISM / FIX DIRECTIVE) that runs
      ONCE per failed round — bounded concurrency, never per candidate;
plus real-Python physics demos showing a naive scheme fails an invariant while the corrected scheme
reaches genuine conservation (and a masked scheme only looks conserved).

Deterministic: no network, no Docker, no real LLM.
"""
import types

import numpy as np

from backend.agent import loop
from backend.agent.anticheat import scan_for_cheating


# ======================================================================================
# Masking detector (anticheat): forced renormalisation / clamping inside an evolution loop.
# ======================================================================================
_RENORM_IN_LOOP = (
    "import numpy as np\n"
    "def evolve(steps):\n"
    "    psi = np.array([1.0, 0.0])\n"
    "    H = np.array([[1.1, 0.0], [0.0, 0.9]])\n"
    "    for _ in range(steps):\n"
    "        psi = H @ psi\n"
    "        psi = psi / np.linalg.norm(psi)   # forces norm == 1 every step\n"
    "    return psi\n")

_RENORM_ONCE = (
    "import numpy as np\n"
    "def evolve(steps):\n"
    "    psi = np.array([1.0, 0.0])\n"
    "    H = np.array([[0.0, 1.0], [-1.0, 0.0]])\n"
    "    for _ in range(steps):\n"
    "        psi = psi + 0.01 * (H @ psi)\n"
    "    return psi / np.linalg.norm(psi)       # one-shot normalisation, OUTSIDE the loop\n")

_CONSERVE_TASK = "evolve the wavefunction in time so the probability norm is conserved"


def test_masking_flags_renormalization_in_loop(monkeypatch):
    monkeypatch.setenv("AGENT_MASKING_SCAN", "true")
    rep = scan_for_cheating(_RENORM_IN_LOOP, "", _CONSERVE_TASK)
    assert rep.flagged and any("renormalis" in r for r in rep.reasons)


def test_masking_ignores_one_shot_normalization(monkeypatch):
    monkeypatch.setenv("AGENT_MASKING_SCAN", "true")
    rep = scan_for_cheating(_RENORM_ONCE, "", _CONSERVE_TASK)
    assert not rep.flagged


def test_masking_ignores_non_conservation_task(monkeypatch):
    # Power iteration legitimately renormalises every step; it is NOT a conservation task, so the
    # same in-loop normalisation must NOT be flagged (no false positive).
    monkeypatch.setenv("AGENT_MASKING_SCAN", "true")
    rep = scan_for_cheating(_RENORM_IN_LOOP, "", "compute the dominant eigenvector by power iteration")
    assert not rep.flagged


def test_masking_flags_clip_in_loop(monkeypatch):
    monkeypatch.setenv("AGENT_MASKING_SCAN", "true")
    code = ("import numpy as np\n"
            "def evolve(steps):\n"
            "    u = np.ones(5)\n"
            "    for _ in range(steps):\n"
            "        u = 1.2 * u\n"
            "        u = np.clip(u, 0.0, 1.0)   # clamps the state every step\n"
            "    return u\n")
    rep = scan_for_cheating(code, "", "simulate the diffusion; the total mass must be conserved")
    assert rep.flagged and any("clamp" in r for r in rep.reasons)


def test_masking_scan_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AGENT_MASKING_SCAN", "false")
    rep = scan_for_cheating(_RENORM_IN_LOOP, "", _CONSERVE_TASK)
    assert not rep.flagged


def test_masking_flags_renormalization_for_population_task(monkeypatch):
    # Biology/ecology (NOT physics): total population must be conserved; renormalising every step masks it.
    monkeypatch.setenv("AGENT_MASKING_SCAN", "true")
    code = ("import numpy as np\n"
            "def simulate(steps):\n"
            "    pop = np.array([0.6, 0.3, 0.1])\n"
            "    for _ in range(steps):\n"
            "        pop = pop * np.array([1.1, 1.0, 0.9])\n"
            "        pop = pop / pop.sum()        # forces the total to 1 every step\n"
            "    return pop\n")
    rep = scan_for_cheating(code, "", "simulate an SIR epidemic; the total population must be conserved")
    assert rep.flagged and any("renormalis" in r for r in rep.reasons)


def test_masking_flags_clip_for_signal_task(monkeypatch):
    # Signal/audio (NOT physics): signal power should emerge from a correct filter, not be clamped.
    monkeypatch.setenv("AGENT_MASKING_SCAN", "true")
    code = ("import numpy as np\n"
            "def filt(x, steps):\n"
            "    y = np.array(x, dtype=float)\n"
            "    for _ in range(steps):\n"
            "        y = 1.3 * y\n"
            "        y = np.clip(y, -1.0, 1.0)    # clamps the signal every step\n"
            "    return y\n")
    rep = scan_for_cheating(code, "", "filter the signal so its power is preserved (parseval)")
    assert rep.flagged and any("clamp" in r for r in rep.reasons)


# ======================================================================================
# Diagnosis prompt + helper.
# ======================================================================================
def test_diagnose_prompt_is_a_structured_critic_separate_from_generation():
    s = loop._DIAGNOSE_SYSTEM.lower()
    assert "diagnosis agent" in s                              # a distinct role...
    assert "do not write the final code" in s                 # ...that directs but does NOT author code
    assert "root-cause" in s
    assert "domain-independent" in s                           # the mapping is explicitly cross-domain
    # the three STRUCTURED parts the generation agent consumes:
    assert "failing check:" in s and "mechanism:" in s and "fix directive:" in s
    assert "update" in s and "iteration" in s and "rule" in s  # conservation -> update/iteration rule
    assert "wrong sign" in s and "governing equation" in s     # sign -> equation/rule
    assert "instability" in s                                  # blow-up -> stability/step
    assert "masking" in s                                      # forced enforcement rejected
    assert "different mechanism" in s                          # vary if the same check keeps failing
    assert "population" in s and "money" in s and "signal power" in s   # cross-domain quantities named


def test_diagnose_failure_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("AGENT_ROOT_CAUSE_DIAGNOSIS", "false")
    assert loop._diagnose_failure(object(), "t", "r", "code", "symptom", ["test_x"]) == ""


def test_diagnose_failure_passes_symptom_and_asks_for_a_different_mechanism(monkeypatch):
    monkeypatch.setenv("AGENT_ROOT_CAUSE_DIAGNOSIS", "true")
    seen = {}

    class P:
        is_available, name, model = True, "openai", "t"

        def stream_chat(self, messages, system="", **k):
            seen["system"], seen["user"] = system, messages[-1]["content"]
            return ["ROOT CAUSE: the velocity update has the wrong sign; flip it."]

    out = loop._diagnose_failure(P(), "task", "reqs", "def f():\n    pass",
                                 "TEST test_energy FAIL\nenergy keeps growing", ["test_energy"],
                                 prev_diagnosis="ROOT CAUSE: an earlier guess", repeated=True)
    assert out.startswith("ROOT CAUSE")
    assert seen["system"] == loop._DIAGNOSE_SYSTEM
    assert "test_energy" in seen["user"] and "DIFFERENT" in seen["user"]


# ======================================================================================
# Real-Python physics demos: a naive scheme fails an invariant; the corrected scheme conserves;
# a masked scheme only LOOKS conserved.
# ======================================================================================
def test_demo_energy_euler_drifts_symplectic_conserves():
    """SHO x'' = -x. Energy E = 0.5(v^2 + x^2) must stay ~constant."""
    def euler(steps, dt=0.05):
        x, v, E = 1.0, 0.0, []
        for _ in range(steps):
            x, v = x + dt * v, v - dt * x                 # explicit Euler (non-symplectic)
            E.append(0.5 * (v * v + x * x))
        return np.array(E)

    def symplectic(steps, dt=0.05):
        x, v, E = 1.0, 0.0, []
        for _ in range(steps):
            v = v - dt * x                                # symplectic (semi-implicit) Euler
            x = x + dt * v
            E.append(0.5 * (v * v + x * x))
        return np.array(E)

    drift = np.abs(euler(600) - 0.5).max()
    bounded = np.abs(symplectic(600) - 0.5).max()
    assert drift > 0.3            # naive scheme FAILS a "energy conserved to 0.1" invariant
    assert bounded < 0.05         # corrected scheme: energy genuinely bounded
    assert bounded < drift / 5    # the mechanism, not the tolerance, was the fix


def test_demo_advection_sign_controls_direction():
    """Upwind advection: the sign of the flux term decides which way the packet moves."""
    def advect(steps, sign, C=0.4):
        u = np.zeros(120)
        u[50:60] = 1.0
        for _ in range(steps):
            u = u - sign * C * (u - np.roll(u, 1))        # sign=+1 -> moves +x (right)
        return u

    com = lambda v: float((np.arange(v.size) * v).sum() / v.sum())
    right, wrong = advect(30, +1), advect(30, -1)
    assert com(right) > 60        # correct sign: the packet moves right (started ~54.5)
    assert com(wrong) < 49        # wrong sign: it moves the wrong way -> fails "moves right"


def test_demo_masking_makes_a_wrong_scheme_look_conserved():
    """A non-unitary step makes the norm grow; renormalising each step forces norm==1 (a masked,
    false 'conservation') while the real dynamics are wrong."""
    def evolve(steps, mask):
        psi = np.array([1.0, 0.0])
        H = np.array([[1.1, 0.0], [0.0, 1.1]])            # non-unitary: norm grows
        norms = []
        for _ in range(steps):
            psi = H @ psi
            if mask:
                psi = psi / np.linalg.norm(psi)           # MASK: force norm == 1
            norms.append(float(np.linalg.norm(psi)))
        return np.array(norms)

    masked, real = evolve(20, True), evolve(20, False)
    assert np.allclose(masked, 1.0)   # masked norm trivially "conserved" -> a FALSE pass
    assert real.max() > 5.0           # the real dynamics are non-unitary (norm blows up)


# ======================================================================================
# Real-Python CROSS-DOMAIN demos: the SAME failure shapes in non-physics domains. A naive rule fails
# the domain's invariant; the corrected rule conserves it (genuinely, not masked).
# ======================================================================================
def test_demo_biology_sir_total_population_conserved():
    """BIOLOGY/epidemiology. SIR is a closed system: S+I+R must stay constant. A naive update adds new
    infections to I without subtracting them from S (the wrong update RULE) -> the total drifts; the
    corrected update removes the same flux from S -> the total is conserved."""
    N = 1000.0

    def step_naive(S, I, R, beta=0.3, gamma=0.1, dt=0.1):
        newinf = beta * S * I / N
        return S, I + dt * (newinf - gamma * I), R + dt * (gamma * I)   # BUG: S not decremented

    def step_correct(S, I, R, beta=0.3, gamma=0.1, dt=0.1):
        newinf = beta * S * I / N
        return S - dt * newinf, I + dt * (newinf - gamma * I), R + dt * (gamma * I)

    def worst_drift(step, n=200):
        S, I, R, worst = 990.0, 10.0, 0.0, 0.0
        for _ in range(n):
            S, I, R = step(S, I, R)
            worst = max(worst, abs((S + I + R) - N))
        return worst

    assert worst_drift(step_naive) > 1.0        # naive update RULE: total population drifts (fails)
    assert worst_drift(step_correct) < 1e-6     # corrected rule: total conserved (emergent, not forced)


def test_demo_signal_energy_orthonormal_preserves_naive_loses():
    """SIGNAL/AUDIO. Parseval: an ORTHONORMAL transform preserves signal energy (sum of squares). A
    non-orthonormal transform changes the energy; the orthonormal one preserves it -- a domain
    invariant, no physics involved."""
    rng = np.random.RandomState(0)
    x = rng.randn(8)
    energy = float(np.sum(x * x))
    naive = float(np.sum((rng.randn(8, 8) @ x) ** 2))            # non-orthonormal: energy NOT preserved
    q, _ = np.linalg.qr(rng.randn(8, 8))
    correct = float(np.sum((q @ x) ** 2))                        # orthonormal: Parseval holds
    assert abs(naive - energy) > 1.0                            # naive: energy invariant FAILS
    assert abs(correct - energy) < 1e-9                         # corrected: energy genuinely preserved


# ======================================================================================
# Integration: the loop diagnoses the mechanism, the corrected scheme is accepted, masking is
# rejected, and a true stall is reported honestly with the diagnosed reason.
# ======================================================================================
def _res(stdout):
    return types.SimpleNamespace(ok=True, exit_code=0, stdout=stdout, stderr="", duration=0.1,
                                 error="", summary="ok")


class _DProvider:
    """Routes by system prompt. Returns the BAD scheme first; once the injected ROOT-CAUSE diagnosis
    reaches the solver prompt, returns the GOOD (corrected) scheme. `good=""` never recovers."""
    is_available = True
    name, model = "openai", "test"

    def __init__(self, requirements, tests, definition, bad, good, diagnosis):
        self.requirements, self.tests, self.definition = requirements, tests, definition
        self.bad, self.good, self.diagnosis = bad, good, diagnosis
        self.calls = []

    def stream_chat(self, messages, system="", **k):
        user = messages[-1]["content"] if messages else ""
        self.calls.append((system, user))
        if system == loop._REQ_SYSTEM:
            return [self.requirements]
        if system == loop._TESTS_SYSTEM:
            return [self.tests]
        if system == loop._DEFINITION_SYSTEM:
            return [self.definition]
        if system == loop._DIAGNOSE_SYSTEM:
            return [self.diagnosis]
        if system == loop._GEN_SYSTEM:
            return [self.good if ("ROOT CAUSE" in user and self.good) else self.bad]
        return [""]


def _heldout_runner():
    """Visible tests pass for any candidate; the held-out invariant passes only for GOOD_SCHEME."""
    def run(code, **k):
        if "held-out runner (seeded)" in code:
            ok = "GOOD_SCHEME" in code
            return _res(f"TEST test_definition_conserved {'PASS' if ok else 'FAIL'}\n"
                        f"TESTS_PASSED {1 if ok else 0}/1\n")
        return _res("TEST test_basic PASS\nTESTS_PASSED 1/1\n")
    return run


def _diag_env(monkeypatch, *, anticheat="false", masking="false"):
    for k, v in {"AGENT_REFERENCE_TESTS": "false", "AGENT_TEST_VALIDATION": "false",
                 "AGENT_NONUNIQUE_VALIDATION": "false", "AGENT_HIDDEN_TESTS": "false",
                 "AGENT_DEFINITION_GATE": "true", "AGENT_DELIVERY_GATES": "false",
                 "AGENT_ROOT_CAUSE_DIAGNOSIS": "true", "AGENT_ANTICHEAT_SCAN": anticheat,
                 "AGENT_MASKING_SCAN": masking, "AGENT_PARALLEL_N": "1", "AGENT_VERIFY_SEEDS": "1",
                 "AUTO_REVIEW": "false", "AGENT_MAX_ATTEMPTS": "4", "AGENT_STALL_LIMIT": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AGENT_MODEL_STRONG", raising=False)
    monkeypatch.setattr("backend.answering.task_classifier.infer_task_type",
                        lambda t: "numeric_algorithm")
    monkeypatch.setattr(loop, "docker_available", lambda: True)


def test_conservation_failure_targets_integrator_and_is_fixed(monkeypatch):
    _diag_env(monkeypatch)
    prov = _DProvider(
        requirements="- simulate_orbit(steps): evolve a 2-D orbit; the total energy must be conserved",
        tests="def test_basic():\n    assert simulate_orbit(5) is not None\n",
        definition="def test_definition_conserved():\n    assert True\n",
        bad="def simulate_orbit(steps):\n    return 'euler'  # BAD_SCHEME: explicit Euler, energy drifts\n",
        good="def simulate_orbit(steps):\n    return 'verlet'  # GOOD_SCHEME: symplectic, energy conserved\n",
        diagnosis=("ROOT CAUSE: the integration scheme is explicit Euler, which is non-conservative, "
                   "so the energy invariant drifts; switch to a symplectic velocity-Verlet update."))
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _heldout_runner())

    events = []
    res = loop.run_agent("implement simulate_orbit(steps) conserving energy", use_search=False,
                         on_event=events.append)

    assert res.verification == "verified"
    assert "GOOD_SCHEME" in res.best_code                      # the corrected SCHEME was accepted
    assert loop._DIAGNOSE_SYSTEM in [s for s, _u in prov.calls]  # it diagnosed before rewriting
    diag = [e for e in events if e.get("type") == "diagnosis"]
    assert diag and "symplectic" in diag[0]["message"].lower()
    # the diagnosis (the mechanism, not a looser tolerance) reached the solver's next prompt
    gen_users = [u for s, u in prov.calls if s == loop._GEN_SYSTEM]
    assert any("ROOT CAUSE" in u and "symplectic" in u.lower() for u in gen_users)


def test_wrong_direction_is_diagnosed_and_equation_corrected(monkeypatch):
    _diag_env(monkeypatch)
    prov = _DProvider(
        requirements="- advect(steps): a packet with positive velocity must move in +x",
        tests="def test_basic():\n    assert advect(5) is not None\n",
        definition="def test_definition_conserved():\n    assert True\n",
        bad="def advect(steps):\n    return 'left'   # BAD_SCHEME: wrong sign, packet moves the wrong way\n",
        good="def advect(steps):\n    return 'right'  # GOOD_SCHEME: corrected flux sign\n",
        diagnosis=("ROOT CAUSE: the flux term has the WRONG SIGN in the governing equation, so the "
                   "packet moves the wrong direction; flip the sign of the advection term."))
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _heldout_runner())

    events = []
    res = loop.run_agent("implement advect(steps); a positive-velocity packet must move right",
                         use_search=False, on_event=events.append)

    assert res.verification == "verified"
    assert "GOOD_SCHEME" in res.best_code
    diag = [e for e in events if e.get("type") == "diagnosis"]
    assert diag and "sign" in diag[0]["message"].lower()


def test_masking_solution_is_rejected(monkeypatch):
    _diag_env(monkeypatch, anticheat="true", masking="true")
    masking_sol = (
        "import numpy as np\n"
        "def evolve_state(steps):\n"
        "    psi = np.array([1.0, 0.0])\n"
        "    H = np.array([[1.1, 0.0], [0.0, 0.9]])\n"
        "    for _ in range(steps):\n"
        "        psi = H @ psi\n"
        "        psi = psi / np.linalg.norm(psi)   # MASK: force the norm to 1 every step\n"
        "    return psi\n")
    prov = _DProvider(
        requirements="- evolve_state(steps): evolve the wavefunction; the norm must be conserved",
        tests="def test_basic():\n    assert evolve_state(3) is not None\n",
        definition="def test_definition_conserved():\n    assert True\n",
        bad=masking_sol, good="",
        diagnosis="ROOT CAUSE: masking — the norm is forced by renormalising each step; use a unitary step.")
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _heldout_runner())

    res = loop.run_agent("evolve_state(steps) — evolve the wavefunction conserving the norm",
                         use_search=False)

    assert res.verification != "verified"                      # masked pass is never accepted
    assert any("renormalis" in r for r in res.cheat_flags)     # rejected specifically AS masking


def test_true_stall_reports_partial_with_diagnosis(monkeypatch):
    _diag_env(monkeypatch)
    prov = _DProvider(
        requirements="- simulate_orbit(steps): evolve a 2-D orbit; the total energy must be conserved",
        tests="def test_basic():\n    assert simulate_orbit(5) is not None\n",
        definition="def test_definition_conserved():\n    assert True\n",
        bad="def simulate_orbit(steps):\n    return 'euler'  # BAD_SCHEME, never fixed\n",
        good="",  # never recovers -> the held-out invariant keeps failing -> stall
        diagnosis=("ROOT CAUSE: the explicit-Euler integrator is non-conservative; switch to a "
                   "symplectic scheme."))
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _heldout_runner())

    events = []
    res = loop.run_agent("implement simulate_orbit(steps) conserving energy", use_search=False,
                         on_event=events.append)

    assert res.verification == "partial"                       # honest, not a fake "verified"
    assert "root cause" in (res.answer or "").lower()          # the partial states the diagnosed reason
    assert [e for e in events if e.get("type") == "diagnosis"]


def test_population_conservation_failure_is_fixed_across_domain(monkeypatch):
    # The SAME loop on a BIOLOGY task: a conserved total population drifts -> the diagnosis targets the
    # UPDATE RULE (not a tolerance) and the corrected rule is accepted. Proves the loop is domain-agnostic.
    _diag_env(monkeypatch)
    prov = _DProvider(
        requirements="- simulate_sir(steps): evolve an SIR model; the total population S+I+R is conserved",
        tests="def test_basic():\n    assert simulate_sir(5) is not None\n",
        definition="def test_definition_conserved():\n    assert True\n",
        bad="def simulate_sir(steps):\n    return 'leaky'  # BAD_SCHEME: S not decremented, total drifts\n",
        good="def simulate_sir(steps):\n    return 'balanced'  # GOOD_SCHEME: flux removed from S\n",
        diagnosis=("ROOT CAUSE: the update rule adds new infections to I without subtracting them from "
                   "S, so the total population is not conserved; remove the same flux from the S update."))
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _heldout_runner())

    events = []
    res = loop.run_agent("implement simulate_sir(steps) conserving the total population",
                         use_search=False, on_event=events.append)

    assert res.verification == "verified"
    assert "GOOD_SCHEME" in res.best_code                       # the corrected UPDATE RULE was accepted
    diag = [e for e in events if e.get("type") == "diagnosis"]
    assert diag and "update rule" in diag[0]["message"].lower()


def test_diagnosis_is_bounded_one_step_per_failed_round(monkeypatch):
    # ROLES STAY SEPARATE + BOUNDED CONCURRENCY: the diagnosis agent is ONE extra step per FAILED
    # round, never per candidate. With best-of-N fan-out (AGENT_PARALLEL_N=2) a failed round generates
    # 2 candidates but triggers exactly ONE diagnosis; the verifying round breaks before diagnosing.
    # So no extra unbounded agents are spawned regardless of the candidate count, and the diagnosed
    # FIX DIRECTIVE is what reaches the generation agent's next prompt.
    _diag_env(monkeypatch)
    monkeypatch.setenv("AGENT_PARALLEL_N", "2")
    prov = _DProvider(
        requirements="- simulate_orbit(steps): evolve a 2-D orbit; the total energy must be conserved",
        tests="def test_basic():\n    assert simulate_orbit(5) is not None\n",
        definition="def test_definition_conserved():\n    assert True\n",
        bad="def simulate_orbit(steps):\n    return 'euler'  # BAD_SCHEME: explicit Euler, energy drifts\n",
        good="def simulate_orbit(steps):\n    return 'verlet'  # GOOD_SCHEME: symplectic, energy conserved\n",
        diagnosis=("ROOT CAUSE: the explicit-Euler integrator is non-conservative; switch to a "
                   "symplectic velocity-Verlet update."))
    monkeypatch.setattr(loop, "get_provider", lambda *a, **k: prov)
    monkeypatch.setattr(loop, "run_python_auto", _heldout_runner())

    res = loop.run_agent("implement simulate_orbit(steps) conserving energy", use_search=False)

    assert res.verification == "verified"
    systems = [s for s, _u in prov.calls]
    diag_calls = systems.count(loop._DIAGNOSE_SYSTEM)
    gen_calls = systems.count(loop._GEN_SYSTEM)
    assert diag_calls == 1                      # exactly ONE diagnosis for the one failed round
    assert gen_calls >= 3                        # best-of-2 fan-out ran (2 in the failed round + the next)
    assert diag_calls < gen_calls                # diagnosis is per-ROUND, NOT per-candidate (bounded)
    # the structured FIX DIRECTIVE is the explicit hand-off the generation agent receives:
    gen_users = [u for s, u in prov.calls if s == loop._GEN_SYSTEM]
    assert any("FIX DIRECTIVE" in u for u in gen_users)
