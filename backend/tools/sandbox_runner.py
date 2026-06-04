"""
_sandbox_runner.py  --  Batch 11 sandboxed runtime

This file runs INSIDE a subprocess spawned by code_executor.run_code().
It reads source code from stdin, executes it inside a restricted
namespace, captures stdout + any matplotlib plots produced, and
writes a JSON envelope to stdout that the parent process reads.

It is deliberately NOT importable as a normal module from the chat
UI. It only runs as: `python _sandbox_runner.py`.

If you find yourself reading this file because something went wrong:
  - The parent process is code_executor.run_code()
  - This child process writes its result as one line of JSON at the
    end of its stdout (after any user prints).
  - Any uncaught exception is caught and reported in the JSON.
"""

from __future__ import annotations

# Force matplotlib to NOT spawn a GUI window
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import io
import json
import sys
import time
import base64
import traceback
from contextlib import redirect_stdout, redirect_stderr


# Allowlist of imports the sandbox will tolerate. The user's code can
# only import names in this set; AST analysis (in the parent) screens
# this BEFORE we even run, and the safety-shim __import__ below blocks
# anything that slips through.
ALLOWED_IMPORTS = frozenset({
    "math",
    "statistics",
    "random",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "operator",
    "collections",
    "re",
    "json",
    "string",
    "time",
    "datetime",
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.fft",
    "scipy.linalg",
    "scipy.stats",
    "scipy.special",
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.io",
    "scipy.io.wavfile",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.figure",
    "matplotlib.colors",
    "matplotlib.cm",
    "pandas",
    "sympy",
    # Batch 12: pre-built DSP helpers
    "dsp_toolkit",
    # Common submodules of allowed packages -- prefix matching handles
    # nested ones like numpy.linalg, scipy.signal.windows, etc.
})


def _is_allowed_import(name: str) -> bool:
    """Return True if `name` (or its top-level package) is allowed."""
    if name in ALLOWED_IMPORTS:
        return True
    top = name.split(".")[0]
    if top in ALLOWED_IMPORTS:
        return True
    # Permit subpackages of allowed top packages
    for allowed in ALLOWED_IMPORTS:
        if name.startswith(allowed + "."):
            return True
    return False


# The original __import__ -- we replace builtins.__import__ with a
# version that consults the allowlist first.
_real_import = __import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if not _is_allowed_import(name):
        raise ImportError(
            f"Import of {name!r} is not allowed in the sandbox. "
            f"Allowed: numpy, scipy, matplotlib, pandas, math, "
            f"statistics, and similar."
        )
    return _real_import(name, globals, locals, fromlist, level)


# Builtins we *remove* from the user's namespace. These are the dangerous
# ones. (Most damage comes from `open`, `exec`, `__import__`, and the
# few that read attributes of arbitrary objects.)
DISALLOWED_BUILTINS = (
    "open",
    "exec",
    "eval",
    "compile",
    "exit",
    "quit",
    "input",
    "breakpoint",
    "help",
    "license",
    "copyright",
    "credits",
)


def _build_sandbox_globals():
    """Construct the globals dict the user's code will execute against."""
    import builtins as _b
    safe_builtins = {}
    for name in dir(_b):
        if name.startswith("_"):
            continue
        if name in DISALLOWED_BUILTINS:
            continue
        safe_builtins[name] = getattr(_b, name)
    # Keep a small set of dunder names the user might reasonably need
    safe_builtins["__name__"] = "__sandbox__"
    safe_builtins["__import__"] = _safe_import
    globals_dict = {"__builtins__": safe_builtins}
    # Pre-inject dsp_toolkit public API so the LLM doesn't need explicit
    # imports. The LLM still CAN write `from dsp_toolkit import ...` if
    # it wants -- both work.
    try:
        import dsp_toolkit as _dsp  # sibling module; this runs as a standalone script
        # Inject every name in PUBLIC_API plus the module itself
        injected = []
        for name in getattr(_dsp, "PUBLIC_API", ()):
            if hasattr(_dsp, name):
                globals_dict[name] = getattr(_dsp, name)
                injected.append(name)
        # Also inject common aliases (so `from dsp_toolkit import X` and
        # bare `mvdr_beamform(...)` both work)
        for alias in (
            "mvdr_beamform", "mvdr", "delay_sum", "das", "lcmv",
            "music", "srp_phat", "steering_vector",
            "covariance", "sample_cov", "simulate_signals",
            "simulate_array", "plot_beampattern", "plot_spectrum",
            "simulate_room", "pesq", "stoi", "broadband_das",
        ):
            if hasattr(_dsp, alias):
                globals_dict[alias] = getattr(_dsp, alias)
        globals_dict["dsp_toolkit"] = _dsp
    except Exception:
        # If dsp_toolkit isn't importable for any reason, skip injection;
        # user code can still `import dsp_toolkit` itself.
        pass
    return globals_dict


# ----------------------------------------------------------------------
# Plot capture
# ----------------------------------------------------------------------

def _capture_plots():
    """Return a list of base64-encoded PNGs for every open matplotlib
    figure, then close them. Empty list if matplotlib wasn't used."""
    out = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return out
    fignums = plt.get_fignums()
    for num in fignums:
        try:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            out.append(base64.b64encode(buf.read()).decode("ascii"))
            plt.close(fig)
        except Exception:
            # don't let a broken figure abort the rest of the capture
            pass
    return out


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

def main():
    # 1. Read source from stdin
    source = sys.stdin.read()
    if not source.strip():
        sys.stdout.write("\n__SANDBOX_RESULT__")
        sys.stdout.write(json.dumps({
            "ok": True,
            "error": None,
            "error_type": None,
            "traceback": None,
            "stdout": "",
            "plots": [],
            "elapsed_sec": 0.0,
        }))
        sys.stdout.flush()
        return

    # 2. Truncate output if it gets enormous
    MAX_STDOUT_CHARS = 10_000

    # 3. Execute with stdout capture
    captured = io.StringIO()
    captured_err = io.StringIO()
    start = time.time()
    error = None
    error_type = None
    tb_text = None

    sandbox_globals = _build_sandbox_globals()

    try:
        with redirect_stdout(captured), redirect_stderr(captured_err):
            exec(compile(source, "<user_code>", "exec"), sandbox_globals, sandbox_globals)
    except SystemExit as e:
        # User code called sys.exit(); not catastrophic, just report
        error = f"sys.exit({e.code})"
        error_type = "SystemExit"
    except KeyboardInterrupt:
        error = "KeyboardInterrupt"
        error_type = "KeyboardInterrupt"
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"
        error_type = type(e).__name__
        tb_text = traceback.format_exc()

    elapsed = time.time() - start

    plots = _capture_plots()

    stdout_text = captured.getvalue()
    stderr_text = captured_err.getvalue()
    if len(stdout_text) > MAX_STDOUT_CHARS:
        stdout_text = (
            stdout_text[:MAX_STDOUT_CHARS]
            + f"\n... (output truncated at {MAX_STDOUT_CHARS} chars)"
        )
    if stderr_text and stderr_text.strip():
        # Append stderr at the end of stdout, marked clearly
        stdout_text += "\n[stderr]\n" + stderr_text

    envelope = {
        "ok": error is None,
        "error": error,
        "error_type": error_type,
        "traceback": tb_text,
        "stdout": stdout_text,
        "plots": plots,
        "elapsed_sec": round(elapsed, 3),
    }
    # Single line of JSON at the very end (the parent reads only this)
    sys.stdout.write("\n__SANDBOX_RESULT__")
    sys.stdout.write(json.dumps(envelope))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
