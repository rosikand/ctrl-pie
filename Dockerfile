# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:22-alpine
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${NODE_IMAGE} AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend/ ./
RUN npm run build


FROM ${PYTHON_IMAGE} AS backend-build

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}"
WORKDIR /build/backend
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/src/ ./src/

# The control-plane container never performs local GPU inference. Installing
# explicit CPU wheels prevents pip from pulling multi-gigabyte CUDA runtimes.
# Build tools and Linux input headers are needed only to compile evdev.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.10.0+cpu" \
      "torchvision==0.25.0+cpu" \
    && python -m pip install . \
    && python -m pip check


FROM ${PYTHON_IMAGE} AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/ctrl-pi
ENV XDG_CACHE_HOME=/var/lib/ctrl-pi/cache
ENV HF_HOME=/var/lib/ctrl-pi/cache/huggingface
ENV FRONTEND_DIST_DIR=/app/frontend/dist
ENV RECORDING_STAGING_DIR=/var/lib/ctrl-pi/recordings
ENV CTRL_PI_LISTEN_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ctrl-pi \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/ctrl-pi --shell /usr/sbin/nologin ctrl-pi \
    && install -d -o ctrl-pi -g ctrl-pi /var/lib/ctrl-pi /var/lib/ctrl-pi/cache /var/lib/ctrl-pi/recordings

COPY --from=backend-build /opt/venv /opt/venv
COPY --from=frontend-build /build/frontend/dist/ /app/frontend/dist/
WORKDIR /app
COPY alembic.ini ./alembic.ini
COPY backend/alembic/ ./backend/alembic/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/ctrl-pi-entrypoint

USER ctrl-pi
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import json,os,urllib.request; port=os.environ.get('CTRL_PI_LISTEN_PORT','8000'); response=urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=2); assert response.status == 200 and json.load(response).get('status') == 'ok'"]

ENTRYPOINT ["ctrl-pi-entrypoint"]
CMD ["uvicorn", "ctrl_pi.main:app", "--host", "0.0.0.0", "--workers", "1"]


# The all-CAN hardware image is an explicit opt-in target. Its named
# `i2rt-source` build context must be an operator-controlled local checkout;
# nothing in this stage clones, fetches, or resolves a moving upstream branch.
FROM runtime AS yam-cell-deps

USER root

ARG CTRL_PI_I2RT_DEPENDENCY_COMMIT

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=i2rt-source / /tmp/i2rt-source/
COPY docker/i2rt-worker-requirements.txt /tmp/i2rt-worker-requirements.txt

# i2rt's full project metadata conflicts with ctrl-pi's policy stack over the
# rerun-sdk major version. The supervised worker imports only the YAM factory
# path, so install its small, pinned runtime closure and prove that the exact
# operator source can import that path without installing or copying the source.
RUN if ! printf '%s\n' "${CTRL_PI_I2RT_DEPENDENCY_COMMIT}" \
       | grep -Eq '^[0-9a-f]{40}$'; then \
      echo "CTRL_PI_I2RT_DEPENDENCY_COMMIT must be an exact lowercase 40-hex commit" >&2; \
      exit 2; \
    fi \
    && test "$(git -C /tmp/i2rt-source rev-parse --show-toplevel)" = /tmp/i2rt-source \
    && test "$(git -C /tmp/i2rt-source rev-parse --verify 'HEAD^{commit}')" = "${CTRL_PI_I2RT_DEPENDENCY_COMMIT}" \
    && git -C /tmp/i2rt-source status \
       --porcelain=v1 --ignored --untracked-files=all \
       > /tmp/i2rt-source-status \
    && test ! -s /tmp/i2rt-source-status \
    && python -m pip install -r /tmp/i2rt-worker-requirements.txt \
    && PYTHONPATH=/tmp/i2rt-source python -c \
       'import i2rt.robots.get_robot; import i2rt.robots.utils' \
    && python -m pip check \
    && rm -rf /tmp/i2rt-source /tmp/i2rt-worker-requirements.txt \
       /tmp/i2rt-source-status


FROM runtime AS yam-cell-runtime

USER root

ARG CTRL_PI_I2RT_DEPENDENCY_COMMIT

ENV CTRL_PI_I2RT_DEPENDENCY_COMMIT=${CTRL_PI_I2RT_DEPENDENCY_COMMIT}

LABEL org.ctrl-pi.i2rt-dependency-commit=${CTRL_PI_I2RT_DEPENDENCY_COMMIT}

# Runtime preflight uses local, read-only Git identity/status inspection only.
# Git is not used to contact a remote or alter the mounted checkout.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=yam-cell-deps /opt/venv /opt/venv

USER ctrl-pi
