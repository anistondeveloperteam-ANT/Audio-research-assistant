# Sandbox image for running AI-generated Python.
#
# The throwaway containers run with --network none, so anything the generated
# code imports must already be in the image. This bakes in the common scientific
# stack (numpy/scipy/DSP/ML/plots/audio) so simulations actually run and verify.
#
# Built ONCE automatically on first use (see code_runner.ensure_sandbox_image);
# containers then run network-off with the usual CPU/mem/pids/timeout caps.
FROM python:3.11-slim

# System libs a couple of scientific wheels need at runtime (audio I/O).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Headless matplotlib (no display in the sandbox) and no .pyc clutter.
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The scientific stack most simulations / DSP / audio / ML code needs.
RUN pip install --no-cache-dir \
      numpy \
      scipy \
      pandas \
      matplotlib \
      scikit-learn \
      sympy \
      soundfile

# Defense-in-depth: generated code runs as an unprivileged user.
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /home/sandbox
