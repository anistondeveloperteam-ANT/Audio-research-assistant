import os
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Reduce noisy third-party logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore")

def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

@contextmanager
def suppress_output():
    """
    Hide noisy backend output in user mode.
    In debug mode, show everything.
    """
    if DEBUG_MODE:
        yield
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "backend_silent.log", "a", encoding="utf-8") as f:
            with redirect_stdout(f), redirect_stderr(f):
                warnings.filterwarnings("ignore")
                yield