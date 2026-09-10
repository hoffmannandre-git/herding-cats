# syntax=docker/dockerfile:1.7

# ---- base ----------------------------------------------------------------
FROM python:3.13-slim AS base

# Reasonable locale / tz defaults. Override at runtime via -e if needed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random

# System packages: curl is needed for healthchecks and `pull-model` style
# scripts; tini gives us a clean PID 1 (so SIGTERM from `docker stop`
# reaches uvicorn and it can shut down gracefully).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/*

# ---- deps ----------------------------------------------------------------
# Install the package and the full test+server extras into the system
# interpreter. Using the `test` extra gives us pytest + ruff + fastapi +
# uvicorn in one shot, so the same image can run both the API and the
# test pipeline.
WORKDIR /opt/herding-cats
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . && pip install ".[test]"

# ---- runtime -------------------------------------------------------------
# Non-root user — uvicorn doesn't need root.
RUN groupadd --system --gid 1001 herding_cats \
 && useradd  --system --uid 1001 --gid herding_cats --create-home --shell /bin/bash herding_cats

# Mount point for the host `./data` directory.
RUN mkdir -p /data && chown -R herding_cats:herding_cats /data
# `HERDING_CATS_*` is the canonical prefix. `LOCALCREW_*` is accepted as a
# legacy fallback by `herding_cats.helpers.env.get_env`; we still set the
# most common ones here so older `.env` files keep working.
ENV HERDING_CATS_DATA_DIR=/data \
    HERDING_CATS_OLLAMA_URL=http://ollama:11434 \
    HERDING_CATS_MODEL=llama3.1:8b \
    HERDING_CATS_HOST=0.0.0.0 \
    HERDING_CATS_PORT=18000

USER herding_cats
WORKDIR /home/herding_cats

EXPOSE 18000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:18000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["herding-cats-api"]
