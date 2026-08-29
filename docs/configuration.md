---
title: "Configuration and credentials"
description: "Configure PostgreSQL, Hugging Face, Modal, recording storage, and YAM environment values."
icon: "key-round"
---

ctrl-π reads process configuration from the environment and a gitignored
`.env` file. PostgreSQL stores only non-secret preferences. Browser code never
receives a database URL, Hub token, Modal credential, or hardware path.

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
| `HF_TOKEN` | Hub browsing/upload and real policy deployment | Backend-only Hugging Face token for the exact configured namespace. |
| `HF_NAMESPACE` | Hub browsing/upload and real policy deployment | Exact user or organization ctrl-π may enumerate and mutate. It is never guessed from the token. |
| `MODAL_TOKEN_ID` | Real Modal lifecycle unless a profile is used | Modal API credential ID. Must be paired with `MODAL_TOKEN_SECRET`. |
| `MODAL_TOKEN_SECRET` | Real Modal lifecycle unless a profile is used | Modal API credential secret. |
| `MODAL_CONFIG_PATH` | Optional | Modal profile path; defaults to `~/.modal.toml`. |
| `MODAL_PROFILE` | Optional | Explicit Modal profile name. |
| `MODAL_PROXY_TOKEN_ID` | Real inference endpoint traffic | Distinct Modal Proxy Token ID beginning `wk-`. |
| `MODAL_PROXY_TOKEN_SECRET` | Real inference endpoint traffic | Distinct Modal Proxy Token secret beginning `ws-`. |
| `CTRL_PI_MOCK_MODE` | Optional | Defaults to `true`. `false` selects real YAM plus Modal and never falls back to mock implementations. |
| `YAM_CAN_INTERFACE` | Hardware mode | SocketCAN interface for the follower, such as `can0`. |
| `YAM_LEADER_PORT` | Hardware mode | GELLO leader serial device; prefer `/dev/serial/by-id/...`. |
| `YAM_MUJOCO_XML_PATH` | Hardware mode | Explicit YAM MuJoCo XML for gravity compensation and forward kinematics. |
| `YAM_GRIPPER_TYPE` | Hardware mode | Must be `crank_4310` in V1. |
| `YAM_LEADER_CALIBRATION_ID` | Hardware mode | Existing LeRobot calibration ID; defaults to `yam-leader`. |
| `YAM_LEADER_CALIBRATION_DIR` | Hardware mode | Directory containing `<calibration-id>.json`. Startup never calibrates interactively. |
| `RECORDING_STAGING_DIR` | Optional | Local pre-upload MP4, JSONL, and manifest root; defaults to `.ctrl-pi/recordings`. |
| `RECORDING_FPS` | Optional | Initial recording rate from 1 to 60; defaults to 20. A saved Settings value takes precedence. |
| `FRONTEND_DIST_DIR` | Production static serving | Built Vite directory. Docker sets `/app/frontend/dist`; source development leaves it unset. |
| `CTRL_PI_BIND_ADDRESS` | Docker Compose | Published host address; defaults to `127.0.0.1`. |
| `CTRL_PI_PORT` | Docker Compose | Published host port; defaults to 8000. |

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

## Hugging Face

```dotenv
HF_NAMESPACE=my-user-or-org
HF_TOKEN=hf_replace_with_a_backend_token
```

Use a fine-grained token scoped to the exact namespace:

- Read access supports dataset/model discovery and private video playback.
- Write and repository-creation access supports recording upload and
  `make seed`.
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

Real LeRobot deployment also needs `HF_TOKEN` while building the Modal image.
The selected model commit is downloaded once as a build secret; serving then
enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Local recording storage

`RECORDING_STAGING_DIR` is durable until upload is proven. A finalized episode
contains MP4, synchronized JSONL, and a manifest; PostgreSQL stores only its
relative key and summary. Back up this directory along with PostgreSQL.

Normal `docker compose down` preserves the `ctrl-pi-data` volume.
`docker compose down --volumes` permanently removes unuploaded recordings.
ctrl-π deletes raw staging only after the Hub revision is verified and the
durable `uploaded` database transition commits.

## Network boundary

V1 has no login or API key. Keep `CTRL_PI_BIND_ADDRESS=127.0.0.1` unless the
host is on a trusted, firewalled LAN. Do not expose ports 8000 or 5173 directly
to the Internet or add a public reverse proxy without an authentication layer
outside ctrl-π.

## Related guides

- [First-time setup](/setup)
- [YAM setup](/yam-setup)
- [Inference](/inference)
- [Docker deployment](/docker-deployment)
