---
title: "Configuration and credentials"
description: "Configure PostgreSQL, Hugging Face, Modal, recording storage, and YAM environment values."
icon: "key-round"
---

ctrl-π reads process configuration from the environment and a gitignored
`.env` file. PostgreSQL stores non-secret control-plane metadata and
preferences. Browser code never receives a database URL, Hub token, or Modal
credential. Settings does display the selected non-secret YAM paths, but all
device enumeration and access stays behind backend services.

Create the local file from the tracked template:

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

<Warning>
  Do not commit `.env`, paste it into an issue, include it in run metadata,
  prefix a backend secret with `VITE_`, or bake it into a container image.
</Warning>

## Environment reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | All persisted workflows | PostgreSQL 14+ connection URL. A blank value keeps first-run mode available while database-backed APIs return 503. |
| `HF_TOKEN` | Hub browsing/upload, real policy deployment, and managed-training artifacts | Backend-only Hugging Face token for the exact configured namespace. |
| `HF_NAMESPACE` | Hub browsing/upload, real policy deployment, and managed-training artifacts | Exact user or organization ctrl-π may enumerate and mutate. It is never guessed from the token. |
| `MODAL_TOKEN_ID` | Real inference or managed-training lifecycle unless a profile is used | Modal API credential ID. Must be paired with `MODAL_TOKEN_SECRET`. |
| `MODAL_TOKEN_SECRET` | Real inference or managed-training lifecycle unless a profile is used | Modal API credential secret. |
| `MODAL_CONFIG_PATH` | Optional | Modal profile path; defaults to `~/.modal.toml`. |
| `MODAL_PROFILE` | Optional | Explicit Modal profile name. |
| `MODAL_PROXY_TOKEN_ID` | Real inference endpoint traffic | Distinct Modal Proxy Token ID beginning `wk-`. |
| `MODAL_PROXY_TOKEN_SECRET` | Real inference endpoint traffic | Distinct Modal Proxy Token secret beginning `ws-`. |
| `CTRL_PI_MOCK_MODE` | Optional | Defaults to `true`. `false` selects real YAM plus Modal and never falls back to mock implementations. |
| `CTRL_PI_I2RT_CHECKOUT` | All-CAN Docker build/runtime | Absolute host path to the operator-controlled local i2rt checkout. Compose mounts it `/opt/i2rt:ro`; ctrl-π never fetches it. |
| `CTRL_PI_I2RT_DEPENDENCY_COMMIT` | All-CAN Docker build/runtime | Exact lowercase 40-character commit used to build hardware dependencies. The configured and mounted commit must match. |
| `YAM_CAN_INTERFACE` | Legacy hardware bootstrap only | Ephemeral SocketCAN interface for the retained one-follower adapter. New cell rows persist adapter serial instead. |
| `YAM_LEADER_PORT` | Legacy hardware bootstrap only | GELLO leader serial device; prefer `/dev/serial/by-id/...`. |
| `YAM_MUJOCO_XML_PATH` | Legacy hardware bootstrap only | Explicit YAM MuJoCo XML for gravity compensation and forward kinematics. |
| `YAM_GRIPPER_TYPE` | Legacy hardware bootstrap only | `crank_4310` for the retained GELLO topology. |
| `YAM_LEADER_CALIBRATION_ID` | Legacy hardware bootstrap only | Existing LeRobot calibration ID; defaults to `yam-leader`. |
| `YAM_LEADER_CALIBRATION_DIR` | Legacy hardware bootstrap only | Directory containing `<calibration-id>.json`. Startup never calibrates interactively. |
| `RECORDING_STAGING_DIR` | Optional | Local pre-upload MP4, JSONL, and manifest root; defaults to `.ctrl-pi/recordings`. |
| `RECORDING_FPS` | Optional | Initial recording rate from 1 to 60; defaults to 20. A saved Settings value takes precedence. |
| `FRONTEND_DIST_DIR` | Production static serving | Built Vite directory. Docker sets `/app/frontend/dist`; source development leaves it unset. |
| `CTRL_PI_LISTEN_PORT` | Production Docker | Uvicorn's internal port; defaults to 8000. Use 8010 for host-network hardware mode when 8000 is occupied. |
| `CTRL_PI_BIND_ADDRESS` | Docker Compose | Published host address; defaults to `127.0.0.1`. |
| `CTRL_PI_PORT` | Bridge-mode Docker Compose | Published host port; defaults to 8000 and is separate from the internal listen port. |

## PostgreSQL

ctrl-π depends only on standard PostgreSQL and Alembic. A local PostgreSQL 14+
server and hosted providers such as Supabase use the same schema.

```dotenv
DATABASE_URL="postgresql://ctrl_pi:URL_ENCODED_PASSWORD@db.example:5432/ctrl_pi?sslmode=require"
```

`postgres://`, `postgresql://`, and `postgresql+psycopg://` URLs are accepted
and normalized to Psycopg 3. Percent-encode URL-special characters in
credentials and require TLS for a non-local database.

For Supabase, use a direct connection or session-mode pooler that permits DDL
migrations. ctrl-π does not need the browser SDK, REST URL, anon key, or
service-role key.

Alembic intentionally does not parse `.env`. Export the same URL when applying
migrations from source:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini upgrade head
```

The Docker entrypoint applies `upgrade head` automatically when the URL is
nonblank.

PostgreSQL also stores one non-secret physical YAM cell and its normalized arm
rows. SocketCAN rows contain stable adapter serials, never current `canN`.
Tokens, database credentials, and Modal profile paths are never stored there.

## Hugging Face

```dotenv
HF_NAMESPACE=my-user-or-org
HF_TOKEN=hf_replace_with_a_backend_token
```

Use a fine-grained token scoped to the exact namespace:

- Read access supports dataset/model discovery and private video playback.
- Write and repository-creation access supports recording upload,
  `make seed`, and managed-training output repositories/checkpoints.
- The real `make smoke` gate also deletes the uniquely named private dataset
  that it creates. Grant delete permission only when you intend to run it.

The readiness check calls `whoami` with the explicit token and accepts only the
authenticated user or one of its organizations as `HF_NAMESPACE`. Dataset and
model results are post-filtered to that exact owner. Uploads default to private
and cannot choose another namespace from the browser.

## Modal API credentials

Modal App creation, inspection, and verified stop require one complete
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` pair. Set both or neither. If neither is
present, the pinned Modal SDK can read a profile created by `modal setup`.

`MODAL_PROFILE` chooses a named profile. Without it, ctrl-π selects exactly one
profile marked active, then falls back to the literal `default` profile. A
single unmarked profile with another name is not selected implicitly.

For Docker, mount the profile read-only at `/home/ctrl-pi/.modal.toml`; see
[Docker deployment](/docker-deployment).

## Modal Proxy Tokens

Real policy endpoints use provider-native proxy authentication and require a
second, distinct pair:

```dotenv
MODAL_PROXY_TOKEN_ID=wk_replace_with_proxy_id
MODAL_PROXY_TOKEN_SECRET=ws_replace_with_proxy_secret
```

API credentials beginning `ak-`/`as-` cannot substitute for proxy credentials
beginning `wk-`/`ws-`. The backend sends the latter only as `Modal-Key` and
`Modal-Secret` headers to a validated HTTPS `.modal.run` endpoint. Neither pair
is stored in PostgreSQL or returned to the browser.

Real LeRobot inference deployment also needs `HF_TOKEN` while building the Modal image.
The selected model commit is downloaded once as a build secret; serving then
enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Managed training does not use Proxy Tokens. Its trusted Modal wrapper receives
`HF_TOKEN` through a Modal Secret to snapshot exact inputs and publish only to
the pre-created, marker-verified output repository; the fixed SmolVLA/LeRobot child
process runs with that token removed. Launch requests and stored jobs never
contain either credential pair.

## Local recording storage

`RECORDING_STAGING_DIR` is durable until upload is proven. A finalized episode
contains MP4, synchronized JSONL, and a manifest; PostgreSQL stores only its
relative key and summary. Back up this directory along with PostgreSQL.

Normal `docker compose down` preserves the `ctrl-pi-data` volume.
`docker compose down --volumes` permanently removes unuploaded recordings.
ctrl-π deletes raw staging only after the Hub revision is verified and the
durable `uploaded` database transition commits.

## YAM cell configuration and consent

In hardware mode, use **Settings → YAM Cell** to discover stable adapter
identities, assign logical roles/pairs/groups/sides, run passive preflight, and
persist one physical cell. All-CAN setup stores its i2rt mount path and exact
commit in the cell; image dependency marker, configured commit, and mounted
checkout must match. Changing that commit requires rebuilding the hardware
image, not merely editing configuration.

The `YAM_*` environment values above are a legacy one-pair bootstrap. They do
not describe an all-CAN cell, do not authorize startup to open hardware, and a
saved physical row takes precedence.

Automatic connection is disabled for a new setup. It requires a separate
motion-risk acknowledgment before ctrl-π may reconnect selected arms on a
later boot or hot-plug. Explicit **Connect** requires its own acknowledgment;
any selected `linear_4310` or `crank_4310` follower also requires a distinct
acknowledgement that calibration moves its jaws.
See [YAM setup](/yam-setup) before enabling either path.

## Network boundary

V1.1 has no login or API key. Keep `CTRL_PI_BIND_ADDRESS=127.0.0.1` unless the
host is on a trusted, firewalled LAN. Do not expose ports 8000 or 5173 directly
to the Internet or add a public reverse proxy without an authentication layer
outside ctrl-π.

The browser and typed `CtrlPiClient` use this same unauthenticated REST
boundary. The SDK accepts no PostgreSQL, Hugging Face, Modal, or hardware
credentials; keep those in the backend environment. See
[REST and Python SDK](/python-sdk).

## Related guides

- [First-time setup](/setup)
- [YAM setup](/yam-setup)
- [Inference](/inference)
- [Docker deployment](/docker-deployment)
