"""
Static anti-reward-hacking scan for agent-generated solution code.

When code is "accepted only if the tests pass", a weak model can game the tests instead of
solving the task — hardcoding the expected outputs, special-casing the visible inputs, faking
functions that ignore their arguments, or tampering with the test harness. The tests then pass
while the code is wrong, producing a FALSE "verified" stamp.

This module statically inspects the SOLUTION source (AST + a few regexes) BEFORE its passing
result is trusted, and flags those patterns. A flag means "do not trust this pass — regenerate",
never "refuse to run" (the sandbox security gate is separate, in `hooks.py`). Stdlib only.

It is the cheap, high-precision first line of defence; the held-out hidden tests + invariant
checks (run on random inputs in the sandbox) are the second, behavioural line in `loop.py`.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

# Trivial literals that are NOT evidence of hardcoding — everyone legitimately uses these
# (base cases, indices, flags), so they must never trip the "hardcoded output" rules.
_TRIVIAL = {0, 1, 2, 3, -1, 10, 100, 1000, 0.0, 1.0, 0.5, 2.0, None}

_HARNESS_RE = re.compile(r"\b(__run_all_tests|unittest|pytest)\b")

# Tasks where a CONSERVED quantity must EMERGE from the dynamics — masking rules fire ONLY here, so
# legitimate per-step normalisation in other tasks (e.g. power iteration) is never flagged. The terms
# are DOMAIN-INDEPENDENT (physics, biology/ecology, finance, signal/audio, chemistry); deliberately we
# do NOT key on bare "power"/"iteration" so power-iteration tasks (which legitimately renormalise each
# step) are not mistaken for masking.
_CONSERVE_RE = re.compile(
    r"conserv|preserv|invariant|\benergy\b|evolv|integrat|propagat|simulat|dynamic|time[-\s]?step|"
    r"schr[oö]dinger|hamiltonian|symplectic|unitar|momentum|wavefunction|trajector|orbit|relax|"
    r"diffus|advect|parseval|"                                            # physics / PDE / signal-energy
    r"populat|abundance|epidemic|infect|suscept|predator|prey|specie|ecolog|"   # biology / ecology
    r"portfolio|wealth|capital|budget|\bmoney\b|\bcash\b|"                # finance
    r"\bsignal\b|waveform|"                                               # signal / audio
    r"concentrat|reaction|"                                               # chemistry
    r"probabilit|\bmass\b|\bnorm\b|total\s+(population|probability|count|mass)",   # general conserved
    re.I)


@dataclass
class CheatReport:
    flagged: bool
    reasons: List[str] = field(default_factory=list)


def anticheat_enabled() -> bool:
    """Live read so tests/runtime can toggle it (AGENT_ANTICHEAT_SCAN, default on)."""
    return os.getenv("AGENT_ANTICHEAT_SCAN", "true").strip().lower() not in ("0", "false", "no", "off")


def masking_scan_enabled() -> bool:
    """Live read (AGENT_MASKING_SCAN, default on): also flag MASKING — code that FORCES a conserved
    quantity to pass (renormalising the evolving state every step, or clamping it inside the loop)
    instead of letting it emerge from correct dynamics. Fires only on conservation/evolution tasks.
    Off skips just the masking rules; the test-gaming rules still run."""
    return os.getenv("AGENT_MASKING_SCAN", "true").strip().lower() not in ("0", "false", "no", "off")


def _is_nontrivial_const(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value not in _TRIVIAL
    if isinstance(value, str):
        return len(value) >= 3            # short strings are usually labels/keys, not answers
    return False


def _test_literals(tests_code: str) -> Set:
    """Non-trivial literal constants that appear in the tests (their expected values/inputs)."""
    lits: Set = set()
    try:
        tree = ast.parse(tests_code or "")
    except SyntaxError:
        return lits
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and _is_nontrivial_const(node.value):
            lits.add(node.value)
    return lits


def _called_names(tests_code: str) -> Set[str]:
    """Function names the tests call directly = the solution's core functions to scrutinise."""
    names: Set[str] = set()
    try:
        tree = ast.parse(tests_code or "")
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if not node.func.id.startswith("test_"):
                names.add(node.func.id)
    return names


def _ignores_arguments(fn: ast.FunctionDef) -> Optional[str]:
    """Rule (fake work): a function that takes parameters, uses NONE of them, and returns a
    constant — it pretends to compute while ignoring its inputs."""
    params = [a.arg for a in (list(getattr(fn.args, "posonlyargs", [])) + list(fn.args.args)
                              + list(fn.args.kwonlyargs)) if a.arg not in ("self", "cls")]
    if not params:
        return None
    used: Set[str] = set()
    returns_const = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            returns_const = True
    if returns_const and not (set(params) & used):
        return f"function '{fn.name}' ignores its arguments and returns a constant (fake work)"
    return None


def _enumerates_inputs(tree: ast.AST) -> Optional[str]:
    """Rule (test-case enumeration): >=2 `if arg == <const>: return <const>` branches where a
    compared or returned constant is non-trivial — i.e. special-casing specific inputs. Trivial
    base cases (n == 0/1) never count, so real recursion isn't flagged."""
    count = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        cmp = node.test
        if not any(isinstance(op, ast.Eq) for op in cmp.ops):
            continue
        cmp_nontrivial = any(isinstance(c, ast.Constant) and _is_nontrivial_const(c.value)
                             for c in cmp.comparators)
        ret_nontrivial = any(
            isinstance(b, ast.Return) and isinstance(b.value, ast.Constant)
            and _is_nontrivial_const(b.value.value)
            for b in ast.walk(node) if isinstance(b, ast.Return))
        ret_any_const = any(
            isinstance(b, ast.Return) and isinstance(b.value, ast.Constant)
            for b in ast.walk(node) if isinstance(b, ast.Return))
        if ret_any_const and (cmp_nontrivial or ret_nontrivial):
            count += 1
    if count >= 2:
        return f"enumerates specific inputs with {count} hardcoded if/return branches"
    return None


def _returns_test_literal(tree: ast.AST, tlits: Set) -> Optional[str]:
    """Rule (hardcoded expected output): a function that TAKES inputs returns a constant equal to
    one of the tests' non-trivial expected values. Restricted to functions WITH parameters so a
    legitimate constant-valued task (a no-arg function returning a constant) is never flagged."""
    if not tlits:
        return None
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        params = [a.arg for a in (list(getattr(fn.args, "posonlyargs", [])) + list(fn.args.args)
                                  + list(fn.args.kwonlyargs)) if a.arg not in ("self", "cls")]
        if not params:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
                v = node.value.value
                if not isinstance(v, bool) and v in tlits:
                    return f"hardcodes a value equal to a test's expected result ({v!r})"
    return None


def _references_harness_or_tests(tree: ast.AST, code: str) -> Optional[str]:
    """Rule (test/harness awareness): the solution names the test functions, reads files
    (peeking the test file), or imports a test framework."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("test_"):
            return f"references the test function '{node.id}'"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            return "calls open(...) — possible test-file peeking"
        if isinstance(node, ast.Attribute) and node.attr in ("read_text", "read_bytes"):
            return "reads a file — possible test-file peeking"
        if isinstance(node, ast.Name) and node.id == "__file__":
            return "references __file__ — possible test-file peeking"
    if _HARNESS_RE.search(code or ""):
        return "references the test harness (__run_all_tests / unittest / pytest)"
    return None


def _harness_tampering(tree: ast.AST, code: str) -> Optional[str]:
    """Rule (harness/state exploits): touching sys.modules/builtins, sleeping (timing trick),
    or monkeypatching time / print / the runner."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "modules"
                and isinstance(node.value, ast.Name) and node.value.id == "sys"):
            return "manipulates sys.modules"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sleep":
            return "calls sleep() (timing trick)"
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr in ("time", "perf_counter"):
                    return "monkeypatches the clock (time.*)"
                if isinstance(t, ast.Name) and t.id in ("print", "__run_all_tests"):
                    return f"overrides '{t.id}' (harness tampering)"
    if re.search(r"\bimport\s+builtins\b", code or ""):
        return "imports builtins (harness tampering)"
    return None


def _forced_in_loop(tree: ast.AST) -> Optional[str]:
    """Inside a loop: renormalising the state (dividing a value by its own norm / sum / abs) or
    clamping it with clip() every step — forcing a conserved quantity to pass instead of letting it
    emerge from the dynamics. Only the IN-LOOP form is flagged, so a single end-of-run normalisation
    is never mistaken for masking."""
    def _has_norm_call(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                if name in ("norm", "sum", "abs"):
                    return True
        return False
    for loop in (n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))):
        for node in ast.walk(loop):
            denom = None
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
                denom = node.value
            elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                  and isinstance(node.value.op, ast.Div)):
                denom = node.value.right
            if denom is not None and _has_norm_call(denom):
                return ("renormalises the state inside the evolution loop — a conserved norm/"
                        "probability is forced, not emergent (masking)")
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                if name == "clip":
                    return ("clamps the state with clip() inside the evolution loop — masking "
                            "instability instead of fixing the scheme")
    return None


def _masking_reasons(tree: ast.AST, task: str) -> List[str]:
    """Masking = forcing a conserved quantity to pass. Conservative: only on conservation/evolution
    tasks (so legitimate per-step normalisation elsewhere is never flagged); fail-open."""
    if not masking_scan_enabled() or not _CONSERVE_RE.search(task or ""):
        return []
    r = _forced_in_loop(tree)
    return [r] if r else []


def scan_for_cheating(solution_code: str, tests_code: str = "", task: str = "") -> CheatReport:
    """Inspect the solution source for reward-hacking patterns. Returns a CheatReport; an
    unparseable solution is NOT a cheat (it will simply fail the tests normally)."""
    code = solution_code or ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return CheatReport(False, [])

    tlits = _test_literals(tests_code)
    reasons: List[str] = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        r = _ignores_arguments(fn)
        if r:
            reasons.append(r)
    for check in (
        _enumerates_inputs(tree),
        _returns_test_literal(tree, tlits),
        _references_harness_or_tests(tree, code),
        _harness_tampering(tree, code),
    ):
        if check:
            reasons.append(check)
    reasons.extend(_masking_reasons(tree, task))

    seen: Set[str] = set()
    uniq = [r for r in reasons if not (r in seen or seen.add(r))]
    return CheatReport(bool(uniq), uniq)
