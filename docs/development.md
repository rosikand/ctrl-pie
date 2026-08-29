---
title: "Development"
description: "Run ctrl-π from source, seed mock data, and execute the required quality gates."
icon: "code-2"
---

Source development runs FastAPI and Vite as two processes. It uses the same
service boundaries as production, but leaves `FRONTEND_DIST_DIR` unset so Vite
owns frontend assets and hot reload. Mock mode is the default and requires no
YAM hardware or Modal account.

For the component and state model, read [Architecture](/architecture). For a
single production container, use [Docker deployment](/docker-deployment).

## Prerequisites

- Python 3.11 and `venv`
- Node.js 22 and npm
- PostgreSQL 14+ for database-backed flows
- FFmpeg with the `libx264` encoder for recording
- `make` and `curl` for the documented gates

Linux is the production target. macOS development is supported against the
mock driver, mock camera, Stub compute, and fake-Hub test boundaries. SocketCAN
and USB device integration are Linux concerns.

Confirm the recording encoder before debugging an episode failure:

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264
```

## Install from a checkout

Run these commands from the repository root:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

On Linux without a local CUDA runtime, install the CPU Torch wheels first. This
prevents LeRobot's dependency resolution from downloading multi-gigabyte CUDA
wheels into the control-plane environment:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' \
  'torchvision==0.25.0+cpu'
```

Install the backend and its test dependencies, then install the locked
frontend tree:

```bash
python -m pip install -e './backend[dev]'
npm --prefix frontend ci
```

The backend package pins LeRobot 0.4.4 and Modal 1.5.4. Do not upgrade either
independently and assume the runtime or teardown contracts remain compatible.
The synchronous `ctrl_pi.trainer.Client` is part of this same editable Python
package; there is no separate published client package in V1.

## Configure services

Create the gitignored environment file:

```bash
cp .env.example .env
```

Mock arms and Stub inference work with external credentials blank. A complete
Record, Datasets, Training, or persisted Inference workflow needs PostgreSQL.
Hub browsing/upload needs `HF_TOKEN` and `HF_NAMESPACE`. Real Modal inference
also needs the Modal API and Proxy Token pairs described in
[Configuration and credentials](/configuration).

### Apply migrations

Alembic deliberately reads `DATABASE_URL` from the process environment; it
does not parse `.env`. Export the same URL you placed there, then upgrade from
the repository root:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini upgrade head
```

Use a disposable database for migration experiments. Do not point tests or
manual downgrade work at production data. The production Docker entrypoint
runs only `upgrade head` and only when `DATABASE_URL` is nonblank.

## Run the development servers

Start FastAPI in the first terminal:

```bash
. .venv/bin/activate
.venv/bin/uvicorn ctrl_pi.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Start Vite in a second terminal:

```bash
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api`
and `/ws` to port 8000. FastAPI exposes interactive OpenAPI documentation at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Do not add `--workers` or run several Uvicorn processes. `RigLease`, live
telemetry, recording subprocesses, inference tasks, and action queues are
process-local and deliberately require one owner. ctrl-π has no authentication,
so bind either development server to a LAN address only on a trusted,
firewalled network.

To exercise production static serving without Docker, build once and point
FastAPI at the result:

```bash
npm run build
FRONTEND_DIST_DIR=frontend/dist \
  .venv/bin/uvicorn ctrl_pi.main:app --host 127.0.0.1 --port 8000
```

Open port 8000 in this mode. Root and known React deep links (including a
dotted repository slug at `/datasets/<repo>`) return the Vite index, while
`/api`, `/ws`, missing assets, and missing media retain their API/404 behavior.

## Seed browsable mock data

With a migrated `DATABASE_URL`, run:

```bash
make seed
```

The command idempotently creates the two mock robot rows and two deterministic
training runs. Existing robot rows are changed only when they carry ctrl-π's
seed ownership marker. If `HF_TOKEN` and `HF_NAMESPACE` are configured, seed
also records a small real MP4 and uploads a private LeRobot dataset to the
fixed `HF_NAMESPACE/ctrl-pi-yam-sample` target.

The uploader refuses a pre-existing unrelated target; it never uses
`exist_ok=True` to overwrite one. Without Hub credentials, database seeding
succeeds and the external upload is clearly skipped. Without `DATABASE_URL`,
the command exits nonzero after reporting that database seeding was skipped.

`Makefile` defaults to `.venv/bin/python`. To use a different interpreter
explicitly, override `PYTHON`:

```bash
make seed PYTHON=python3.11
```

## Tests and milestone gates

The normal backend suite uses local SQLite fixtures and fake provider/Hub
boundaries; it must not contact external services:

```bash
.venv/bin/python -m pytest -q backend/tests
```

Run one focused module while iterating:

```bash
.venv/bin/python -m pytest -q backend/tests/test_recordings.py
```

Build the exact frontend production bundle:

```bash
npm run build
```

Check Python import/bytecode errors and whitespace before handoff:

```bash
.venv/bin/python -m compileall -q backend/src/ctrl_pi backend/tests
git diff --check
```

Validate the Mintlify navigation, frontmatter, internal routes, anchors, and
assets without installing a documentation dependency:

```bash
npm run docs:check
```

Preview or run Mintlify's strict build validation with the pinned CLI:

```bash
npm run docs:dev
npm run docs:validate
```

Mintlify is configured as a monorepo project rooted at `/docs`; its required
configuration file is `docs/docs.json`.

Every milestone also requires Alembic against a real PostgreSQL database and
both development servers to start cleanly. From Milestone 11 onward, the
end-to-end gate is:

```bash
make smoke
```

That default intentionally uses the configured real Hugging Face account. It
creates, verifies, and deletes a uniquely named private dataset; it does not
use Modal. Review [Full mock smoke gate](/smoke-test) and grant the token's
delete permission before running it.

For an offline development run with the deterministic filesystem Hub:

```bash
PYTHONPATH=backend/src \
  .venv/bin/python -m ctrl_pi.smoke --fake-hub
```

This still records five seconds of camera output, performs real LeRobot
conversion, executes exactly 100 Stub-policy actions, and verifies every local
resource is released.

## Database changes

Model declarations live in `backend/src/ctrl_pi/models.py`; migrations live
under `backend/alembic/versions/`. Append a revision rather than editing an
already shipped migration. Test an upgrade against PostgreSQL and review the
generated SQL before committing. Large binary artifacts and live telemetry do
not belong in a migration or ORM column.

## Debugging lifecycle failures

- A recording start failure usually means FFmpeg/libx264 is missing, the rig
  is leased, or the selected arms have the wrong roles.
- A database API returning 503 usually means `DATABASE_URL` is absent or the
  migrations have not been applied.
- Hub browsing requires both the explicit namespace and a token that belongs
  to that user or organization.
- Real inference requires the HF token, Modal lifecycle credentials, and the
  distinct Modal Proxy Token pair. Settings reports presence/readiness without
  returning values.
- Stop Uvicorn with `Ctrl-C` and let lifespan teardown finish. If a Modal App
  cannot be proven stopped, use [Modal operations](/modal-operations)
  rather than deleting resources by a name prefix.

Never print `.env`, raw provider exceptions, request headers, presigned URLs,
or local Hub cache paths when diagnosing a failure.

## Related guides

- [Architecture](/architecture)
- [Configuration and credentials](/configuration)
- [Record and teleoperate](/recording)
- [YAM driver interface](/yam-driver)
- [Inference](/inference)
- [Trainer API](/trainer-api)
- [Docker deployment](/docker-deployment)
