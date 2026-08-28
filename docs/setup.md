# Setup and configuration

ctrl-π runs on a machine connected to the YAM arms and uses services in your
own accounts. PostgreSQL stores small control-plane state, Hugging Face stores
datasets and model artifacts, and Modal is used only for real inference.
Mock mode can exercise the entire local workflow without hardware or Modal.

For a packaged installation, start with [Docker deployment](docker-deployment.md).
For editable source installation and two development servers, use
[Development](development.md).

## Configuration file

Copy the tracked template to the gitignored `.env` file:

```bash
cp .env.example .env
```

Edit it before running migrations or any real integration. A blank secret is
treated as missing and appears in the Settings first-run checklist. Do not
commit `.env`, paste it into an issue, bake it into an image, or prefix a Vite
variable with a backend secret.

### Environment reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Full control-plane use | PostgreSQL 14+ URL. Blank starts first-run mode, but database-backed APIs return 503. |
| `HF_TOKEN` | Hub browsing/upload and real model deployment | Backend-only Hugging Face token. |
| `HF_NAMESPACE` | Hub browsing/upload and real model deployment | Exact user or organization that ctrl-π is allowed to enumerate and mutate. |
| `MODAL_TOKEN_ID` | Real Modal lifecycle, unless a local Modal profile is mounted | Modal API credential ID. Must be paired with `MODAL_TOKEN_SECRET`. |
| `MODAL_TOKEN_SECRET` | Real Modal lifecycle, unless a local Modal profile is mounted | Modal API credential secret. |
| `MODAL_PROXY_TOKEN_ID` | Real LeRobot endpoint traffic | Distinct Modal Proxy Token ID beginning `wk-`. |
| `MODAL_PROXY_TOKEN_SECRET` | Real LeRobot endpoint traffic | Distinct Modal Proxy Token secret beginning `ws-`. |
| `CTRL_PI_MOCK_MODE` | No; defaults to `true` | Selects mock arms plus Stub compute when true, and the real YAM driver plus Modal compute when false. Hardware mode never falls back to the mock driver. |
| `YAM_CAN_INTERFACE` | Hardware mode | Exact SocketCAN interface for the follower, for example `can0`. |
| `YAM_LEADER_PORT` | Hardware mode | Exact GELLO leader serial device; prefer a stable `/dev/serial/by-id/...` path. |
| `YAM_MUJOCO_XML_PATH` | Hardware mode | Explicit YAM MuJoCo XML used for follower gravity compensation; ctrl-π never guesses a developer checkout path. |
| `YAM_GRIPPER_TYPE` | Hardware mode | Must be `crank_4310` in V1. The pinned plugin's linear-gripper calibration path is not usable with LeRobot 0.4.4, so linear values fail before device access. |
| `YAM_LEADER_CALIBRATION_ID` | Hardware mode | Stable LeRobot calibration ID; defaults to `yam-leader`. |
| `YAM_LEADER_CALIBRATION_DIR` | Hardware mode | Directory containing `<YAM_LEADER_CALIBRATION_ID>.json`; startup is noninteractive and will not create it. |
| `RECORDING_STAGING_DIR` | No | Local MP4/JSONL workspace; defaults to `.ctrl-pi/recordings`. |
| `RECORDING_FPS` | No | Initial/fallback recording rate, 1–60; defaults to 20. The PostgreSQL-backed Settings value takes precedence after it is saved. |
| `FRONTEND_DIST_DIR` | Production static serving only | Built Vite directory containing `index.html`; unset in source development. Docker sets `/app/frontend/dist`. |
| `CTRL_PI_BIND_ADDRESS` | Docker Compose only | Host address published by Compose; defaults to `127.0.0.1`. |
| `CTRL_PI_PORT` | Docker Compose only | Host port published by Compose; defaults to 8000. |

Non-secret UI settings live in PostgreSQL: recording FPS, default runtime,
default Modal GPU label, and a deployment timeout from 1 to 30 minutes.
`HF_NAMESPACE` is shown read-only from the environment. The Settings API never
accepts or returns a database URL or Hub/Modal token.

## PostgreSQL and Supabase

ctrl-π depends only on standard PostgreSQL and Alembic. A local PostgreSQL 14+
server and a hosted provider such as Supabase use the same schema.

Create an empty database and a role allowed to connect and run DDL migrations,
then set a URL such as:

```dotenv
DATABASE_URL="postgresql://ctrl_pi:URL_ENCODED_PASSWORD@db.example:5432/ctrl_pi?sslmode=require"
```

`postgres://`, `postgresql://`, and `postgresql+psycopg://` prefixes are
accepted and normalized to the Psycopg 3 driver. Percent-encode credentials
that contain URL-special characters. Use TLS when the database is not local.

For Supabase, copy a PostgreSQL connection string from the project's database
connection panel. Use a direct connection or session-mode pooler that permits
schema migrations; transaction-only pooling is not needed. ctrl-π does not use
the Supabase browser SDK, REST endpoint, anon key, or service-role key. Never
put any of those—or `DATABASE_URL`—in the frontend environment.

Alembic reads `DATABASE_URL` from the process environment rather than loading
`.env` itself. From the repository root, export the same URL and migrate:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini upgrade head
```

Docker passes the Compose environment to its entrypoint and performs this
migration automatically before Uvicorn starts. The application does not call
`create_all` in production.

After startup, `GET /api/settings/status` runs a small `SELECT 1` connection
check. A blank URL leaves the server available for the setup checklist while
database-backed tabs return a clear 503.

## Hugging Face Hub

Set the namespace explicitly; ctrl-π never guesses from the token:

```dotenv
HF_NAMESPACE=my-user-or-org
HF_TOKEN=hf_replace_with_a_backend_token
```

Use a fine-grained token with access to private artifacts in that exact
namespace. Read access is sufficient for dataset/model discovery and private
video playback. Recording upload and `make seed` require repository creation
and write access. The default real `make smoke` gate additionally creates and
deletes one uniquely named private dataset after verifying its ownership
marker; grant delete permission only when you intend to run that gate.

Dataset discovery post-filters to the exact namespace and the `LeRobot` tag.
Model discovery also post-filters to the exact namespace. Cards and metadata
are downloaded with an explicit token at an immutable revision. Private video
is streamed through a same-origin backend route, so neither the token nor a
local Hub cache path appears in the browser.

New recording uploads default to private. The upload API accepts only a slug,
prepends `HF_NAMESPACE`, and refuses an existing unrelated repository. A
server-owned `.ctrl-pi.json` marker permits a safe retry after a partial
upload; client metadata alone never proves ownership.

The seed command uses the fixed private target
`HF_NAMESPACE/ctrl-pi-yam-sample`. It safely refuses a pre-existing unrelated
repository instead of overwriting it.

## Modal

No Modal credentials are needed in mock mode. Real compute requires a Modal
account plus two distinct credential pairs.

### API credentials

`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` authorize App creation, inspection,
and teardown. Set both variables or neither. When neither is present, the
pinned Modal SDK may use the active profile created by `modal setup`. A Docker
container can use that profile only when the host's `.modal.toml` is mounted
read-only as described in [Docker deployment](docker-deployment.md).

### Proxy credentials

Real inference web endpoints use Modal's provider-native proxy authentication.
Create a Modal Proxy Token for the backend and configure its distinct `wk-`
ID and `ws-` secret:

```dotenv
MODAL_PROXY_TOKEN_ID=wk_replace_with_proxy_id
MODAL_PROXY_TOKEN_SECRET=ws_replace_with_proxy_secret
```

Modal API credentials cannot substitute for Proxy Tokens. These values are
sent only as `Modal-Key` and `Modal-Secret` headers from the backend to a
validated HTTPS `.modal.run` endpoint. They are never persisted or returned to
the browser.

Real LeRobot deployment also needs `HF_TOKEN`, because the exact model commit
is downloaded once while building the deployment image. The serving container
then runs with Hub and Transformers offline mode enforced. Real OpenPI loading
is explicitly unavailable in V1; selecting OpenPI is deterministic only in
mock mode.

Each deployment uses one container at most, no warm pool, an idle scaledown no
longer than 60 seconds, and a hard timeout no longer than 30 minutes. Stop must
verify zero provider tasks. If ordinary teardown is interrupted, follow
[Modal operator cleanup](modal-operations.md). Real GPU work costs money;
always stop a test deployment and verify teardown before leaving it.

## Mock mode and real YAM configuration

With `CTRL_PI_MOCK_MODE=true`, the backend supplies two process-local arms,
`yam-leader` and `yam-follower`, a synthetic camera, and credential-free Stub
compute/runtime behavior. Recording still uses real FFmpeg, writes a real MP4,
and converts through LeRobot. Hub operations remain real unless a test or the
offline smoke command explicitly injects the filesystem fake.

With `CTRL_PI_MOCK_MODE=false`, ctrl-π constructs the real driver backed by the
pinned `lerobot-robot-yam==0.1.1`,
`lerobot-teleoperator-yam-gello==0.1.1`, and `yam-common==0.1.1` packages. The
follower uses SocketCAN and the leader uses a calibrated Dynamixel serial bus.
Configure every `YAM_*` value above before connecting hardware. The leader
calibration file must already exist; backend startup never opens an interactive
calibration prompt.

The process remains available when hardware configuration, a plugin import, or
a device connection fails. In that state `/api/arms` returns the two stable arm
IDs as disconnected and Settings reports a sanitized, actionable reason.
Motion calls fail before vendor mutation, and hardware mode never substitutes a
`MockYAMDriver`. Correct the configuration or device problem and restart the
single backend process.

Only the real follower accepts commands. Joint and gripper jogs are checked
against both ctrl-π step bounds and the pinned YAM joint limits; Cartesian jog
is explicitly unsupported. Before the first position command the adapter exits
zero-torque mode into the plugin's position-control mode. A command failure
latches the driver unavailable and attempts zero torque rather than silently
continuing.

Validate package versions, configuration, model assets, calibration, and
device/interface visibility without opening either bus:

```bash
make yam-probe
```

Only after that preflight passes, secure the arm, clear the workspace, have an
operator at the emergency stop, and opt into a connected sample:

```bash
PYTHONPATH=backend/src \
  .venv/bin/python -m ctrl_pi.yam_probe --connect
```

The connected probe never jogs or sends an application action, but it is not
electrically read-only: the leader plugin configures its serial bus and the
follower plugin starts its gravity-compensation/control thread. It prints only
safe IDs, roles, connection state, and bus state, then always attempts the
bounded shutdown path, including follower safe mode and both device closes.
A stuck vendor I/O call can prevent cleanup and requires operator intervention.
This repository has
no YAM hardware attached in CI, so fake-vendor tests do not replace the
Ubuntu-box validation checklist in [YAM driver interface](yam-driver.md).
Container device visibility is covered by
[Docker deployment](docker-deployment.md).

## Network exposure

ctrl-π has no V1 authentication. The Compose default binds to
`127.0.0.1`; change it only for a trusted, firewalled LAN. Source Uvicorn and
Vite must receive the same treatment. Do not expose port 8000 or 5173 directly
to the public Internet, and do not place a public reverse proxy in front of
the application without adding an authentication boundary outside ctrl-π.

## Verify setup

After starting both services or the production container, check:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/settings/status
```

The health endpoint proves that FastAPI is reachable. The Settings response
reports sanitized connection/readiness state; it never echoes credentials.
For a complete deterministic integration check, use the
[full mock smoke gate](smoke-test.md).
