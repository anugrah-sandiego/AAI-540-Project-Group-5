# Four stages built from one file:
#   base    - Python 3.11 + pinned dependencies, shared by dev and runtime
#   dev     - development container: adds git; the project folder is mounted at
#             run time (see docker-compose.yml), so code changes need no rebuild
#   aws     - aws-tools: the SageMaker SDK, kept apart because it needs older
#             package versions than the training code
#   ui      - Streamlit dashboard; calls the prediction API, so no ML libraries
#   runtime - production image for local runs and SageMaker (train/serve); built
#             by default (docker build .) because it is the last stage

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
FROM python:3.11-slim AS aws

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Code is mounted at run time (see the aws-tools service in docker-compose.yml)
COPY requirements-aws.txt ./
RUN pip install -r requirements-aws.txt && useradd --create-home --uid 1000 appuser
USER appuser

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS ui

WORKDIR /app
ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1     STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dashboard only: scores come from the API, so no ML libraries are installed.
# Code and the test split are mounted at run time (see the ui service in docker-compose.yml)
COPY requirements-ui.txt constraints.txt ./
RUN pip install -r requirements-ui.txt -c constraints.txt && useradd --create-home --uid 1000 appuser
USER appuser

# ---------------------------------------------------------------------------
FROM base AS runtime

# SageMaker starts custom images as `train` or `serve`
COPY docker/train docker/serve /usr/local/bin/
RUN chmod 755 /usr/local/bin/train /usr/local/bin/serve

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

# SageMaker mounts /opt/ml owned by root, so images pushed for SageMaker are built
# with RUN_AS=root (scripts/deploy.sh); local and production runs stay unprivileged
ARG RUN_AS=appuser
USER ${RUN_AS}

# Prediction service port (SageMaker contract)
EXPOSE 8080

# Default command: train locally; SageMaker overrides it with `train` or `serve`
CMD ["python", "-m", "src.pipelines.training_pipeline"]
