# Three stages built from one file:
#   base    - Python 3.11 + pinned dependencies, shared by the two images below
#   dev     - development container: adds git; the project folder is mounted at
#             run time (see docker-compose.yml), so code changes need no rebuild
#   runtime - production image: code copied in, no git; built by default
#             (docker build .) because it is the last stage

# ---------------------------------------------------------------------------
# Python 3.11 matches the version the code and tests are run on
FROM python:3.11-slim AS base

WORKDIR /app

# Show logs live in Docker/CloudWatch; LOG_LEVEL and LOG_FORMAT (text or json)
# are read by src/utils/logging_config.py
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=text \
    GIT_PYTHON_REFRESH=quiet

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the core (non-AWS) dependencies at the exact tested versions
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# Run as an unprivileged user instead of root
RUN useradd --create-home --uid 1000 appuser

# ---------------------------------------------------------------------------
FROM base AS dev

# git lets model lineage read the real commit and detect uncommitted changes in
# the mounted project folder. The settings stop Linux git from reporting every
# file of a Windows checkout as modified:
#   safe.directory - the mounted folder is owned by a different user
#   core.autocrlf  - the Windows checkout uses CRLF line endings
#   core.fileMode  - Windows mounts make every file look executable
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app \
    && git config --system core.autocrlf true \
    && git config --system core.fileMode false

USER appuser

# Interactive shell by default; docker-compose.yml defines test/train/mlflow commands
CMD ["bash"]

# ---------------------------------------------------------------------------
FROM base AS runtime

# Copy project files (.dockerignore keeps venv, .git, secrets and local runs out)
COPY --chown=appuser:appuser . .

# Writable locations for MLflow runs, processed splits and saved models; /app
# itself must be writable too, since MLflow creates mlflow.db there
RUN mkdir -p mlruns models data/processed \
    && chown appuser:appuser /app \
    && chown -R appuser:appuser mlruns models data/processed

# The image has no .git folder, so model lineage reads the commit from these
# (docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) ...)
ARG GIT_COMMIT=""
ARG GIT_BRANCH=""
ENV GIT_COMMIT=${GIT_COMMIT} \
    GIT_BRANCH=${GIT_BRANCH}
# Standard labels so registries and `docker inspect` show the source commit
LABEL org.opencontainers.image.revision=${GIT_COMMIT} \
      org.opencontainers.image.source="https://github.com/anugrah-sandiego/AAI-540-Project-Group-5"

USER appuser

# Expose port for MLflow UI (if needed)
EXPOSE 5000

# Default command
CMD ["python", "-m", "src.pipelines.training_pipeline"]
