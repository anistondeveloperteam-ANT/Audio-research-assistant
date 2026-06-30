# Research Assistant — web app image.
# Runs the FastAPI server with uvicorn. Oracle 23ai and the chat-model API stay EXTERNAL
# (as they should) — see DEPLOY.md (Path D). The code agent needs the host Docker socket
# mounted at runtime (-v /var/run/docker.sock:/var/run/docker.sock); without it everything
# works except the write-and-run-code agent. GPU is optional (add --gpus all at run time).
#
# Build:  docker build -t research-assistant .
# Run:    docker run -d -p 8600:8600 --env-file .env -v "$(pwd)/data:/app/data" research-assistant

FROM python:3.11-slim

# Faster, cleaner Python in a container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs PyMuPDF / numpy stacks may need at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so the layer caches across code changes.
# (Heavy: torch + transformers + sentence-transformers. Expect a multi-GB image.)
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code (the .dockerignore keeps tests, docs, data, .env, caches, and dev tools OUT).
COPY . .

# Run as a non-root user; data/ is a mounted volume at runtime.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8600

# bind 0.0.0.0 so the port is reachable from outside the container.
CMD ["python", "-m", "uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8600"]
