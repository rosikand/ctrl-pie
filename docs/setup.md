---
title: "Detailed setup reference"
description: "Configure production services, YAM-cell hardware mode, networking, and readiness."
icon: "list-checks"
---

ctrl-π runs on the machine connected to the YAM cell and uses services in
accounts you own. PostgreSQL stores small control-plane state, Hugging Face
stores datasets/models, and Modal runs real inference and managed training.
Mock mode exercises the local product loop without YAM or Modal hardware.

Start with [Docker deployment](/docker-deployment) for a packaged install or
[Development](/development) for two editable source servers.

## Environment

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

Never commit `.env`, paste it into an issue, bake it into an image, or expose a
backend secret as a `VITE_` variable.

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Persisted workflows | PostgreSQL 14+ URL. Blank keeps first-run status available while DB APIs return 503. |
| `HF_TOKEN` / `HF_NAMESPACE` | Hub workflows | Backend-only token and exact user/organization namespace. ctrl-π never guesses the owner. |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Real inference/training | Complete Modal lifecycle credential pair, unless a reviewed local profile is mounted. |
| `MODAL_PROXY_TOKEN_ID` / `MODAL_PROXY_TOKEN_SECRET` | Real inference | Distinct `wk-`/`ws-` endpoint Proxy Token pair. |
| `CTRL_PI_MOCK_MODE` | Optional | Defaults `true`; `false` selects hardware/real compute and never falls back to mocks. |
| `CTRL_PI_I2RT_CHECKOUT` | All-CAN Docker | Absolute host path to an operator-owned local i2rt checkout. |
| `CTRL_PI_I2RT_DEPENDENCY_COMMIT` | All-CAN Docker | Exact full commit used to bake dependencies. It must match cell config and runtime mount. |
| `YAM_CAN_INTERFACE`, `YAM_LEADER_PORT`, `YAM_MUJOCO_XML_PATH`, `YAM_GRIPPER_TYPE`, `YAM_LEADER_CALIBRATION_*` | Legacy only | Retained one-GELLO + one-CAN `crank_4310` bootstrap. Not the V1.2 all-CAN model. |
| `RECORDING_STAGING_DIR` | Optional | Durable pre-upload episode root; defaults `.ctrl-pi/recordings`. |
| `RECORDING_FPS` | Optional | Initial/fallback rate, 1–60; defaults 20. Saved Settings value wins. |
| `CTRL_PI_LISTEN_PORT` | Docker | Uvicorn/container port; defaults 8000. Host-network hardware normally uses 8010. |
| `CTRL_PI_BIND_ADDRESS`, `CTRL_PI_PORT` | Bridge-mode Compose | Published host address/port; defaults `127.0.0.1:8000`. |

Full details are in [Configuration and credentials](/configuration).

## PostgreSQL

Use standard PostgreSQL 14+ or a hosted provider such as Supabase. For a
non-local database, use TLS and percent-encode URL-special password characters:

```dotenv
DATABASE_URL="postgresql://ctrl_pi:URL_ENCODED_PASSWORD@db.example:5432/ctrl_pi?sslmode=require"
```

Use a direct connection or session pooler that permits DDL. ctrl-π does not
use Supabase browser keys. Docker applies `alembic upgrade head` before
Uvicorn when the URL is nonblank. From source, Alembic reads the exported
environment directly:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini upgrade head
```

PostgreSQL stores one normalized `yam_cells` row plus typed `yam_cell_arms`.
It never stores runtime `canN`, service credentials, or live telemetry. The
legacy `yam_setups` row remains readable for compatibility.

### Creating the leader calibration artifact

This section applies only to the retained serial-GELLO compatibility adapter.
If its selected leader has no calibration JSON, create it
outside the ctrl-π web application and backend. Stop ctrl-π and every other serial owner first,
secure/support the rig, and keep a physical operator at the emergency stop.

```bash
.venv/bin/lerobot-calibrate \
  --teleop.type=yam_leader \
  --teleop.port=/dev/serial/by-id/REPLACE_WITH_LEADER \
  --teleop.id=yam-leader \
  --teleop.calibration_dir=/absolute/path/to/leader-calibration
```

The interactive command opens/configures the leader and writes the reviewed
artifact. It does not connect, calibrate, or
validate the follower. Restart ctrl-π, select the same stable path/ID/root,
then rerun legacy passive discovery and structural preflight before Connect.

These commands have not been executed on a physical YAM in this development
environment. File readiness does not prove directions, offsets, limits,
model fidelity, bus behavior, or E-stop response. This calibration procedure
does not apply to a CAN `yam_teaching_handle`, whose detection-only range check
and external encoder maintenance boundary are documented separately.

## Hugging Face

```dotenv
HF_NAMESPACE=my-user-or-org
HF_TOKEN=hf_backend_token
```

Use a fine-grained token for that exact namespace. Read supports private
dataset/model browsing; recording, seed, and managed-training artifacts need
write/create. The real smoke gate creates and ownership-safely deletes one
private dataset, so grant delete only when running that gate. Tokens never
reach browser code.

## Modal

Set both lifecycle variables or neither. When absent, the pinned SDK may use
an explicitly selected local profile. Inference additionally requires the
distinct Proxy Token pair; API credentials cannot substitute for endpoint
credentials. Managed training uses lifecycle credentials but not Proxy Tokens.

Real compute costs money. Deployment/job outcomes are not terminal until
provider stop and zero running tasks are verified. Follow
[Modal operations](/modal-operations) if cleanup becomes uncertain.

## YAM cell first-time setup

Build/start the [all-CAN Docker override](/docker-deployment#primary-all-can-yam-cell-deployment)
with a clean, exact local i2rt checkout mounted `/opt/i2rt:ro`. The public
upstream does not advertise the field-tested commit; ctrl-π never pulls or
selects latest. Changing the commit requires a hardware-image rebuild.

Before starting hardware mode, prove no saved setup has unattended
auto-restore enabled. Then use **Settings → YAM Cell**:

1. **Discover** bounded stable USB-CAN identities and current runtime
   interfaces. No device is opened.
2. Assign each arm's logical ID/name, role, pair/group/side, transport,
   durable identity, and end effector. `canN` is never durable identity.
3. Enter `/opt/i2rt` plus the exact dependency commit; add follower frame-map
   or soft-limit artifacts only if they actually exist for the current mount.
4. **Preflight** topology, stable identity resolution, link state, exact
   read-only source, and artifact structure. It never imports a motion-capable
   object, pings/enables motors, calibrates a gripper, or changes a link.
5. **Save** the cell. It is applied in place without opening hardware.
6. Optionally run the separately acknowledged read-only teaching-handle range
   check. It detects stuck input but never zeroes an encoder.
7. Clear/support the selected arm and place an operator at independent
   power/E-stop controls before **Connect**. General motion acknowledgement is
   required; a selected `linear_4310` or `crank_4310` follower needs a second
   acknowledgement because jaw calibration moves.

An absent frame map means intentional 1:1 mapping. An absent soft-limit file
means `NO SASH GUARD`, not a default. Never reuse historical limits from a
different/failed mount.

Connected followers can be energized/holding and resist manual movement.
Explicit disconnect revokes writes, safe-idles/closes, and reaps the worker.
Support the arm as control releases. Uncertain teardown keeps the bus blocked
and requires operator review.

Automatic restoration is disabled initially. Enabling it requires separate
unattended-motion consent and the 4310-gripper calibration acknowledgement
where relevant. Missing hardware is passively rechecked; a runtime fault is
latched rather than blindly retried.

The canonical API is `/api/yam/cell` with `discover`, `preflight`, `connect`,
`disconnect`, and `handle-check` actions. `/api/yam/setup` remains a V1.1
compatibility route. See [YAM cell setup](/yam-setup) and
[YAM driver interface](/yam-driver).

## Mock versus hardware

Mock mode exposes four deterministic arms in two independent declared pairs,
synthetic camera frames, and Stub inference. It exercises recording, Hub
conversion, resource-scoped leases, and sync lifecycle without devices. Mock
save/reset never mutates the persisted physical cell.

Hardware mode exposes configured arms even when disconnected/missing so status
remains actionable. Missing config/device/source never falls back to a mock.
Followers remain the only inference targets. The supervised all-CAN adapter
rejects one-shot jog in V1.2; mock and retained legacy adapters keep bounded
jog compatibility. Teleop start is observation-only and unsynchronized;
enabling the approximately three-second correction is a separate acknowledged
motion action.

Do not use the legacy `make yam-probe` documentation to qualify an all-CAN
cell. Use passive Settings/API preflight and then the exact ordered
[H0–H7 field acceptance](/yam-field-test).

## Networking

ctrl-π has no login or RBAC. Bind bridge mode to loopback by default. In
host-network hardware mode Uvicorn listens directly on
`CTRL_PI_LISTEN_PORT`—use 8010 on the reference box because another service
owns 8000. Restrict that port with the host firewall/trusted LAN.

Do not expose ports 8000, 8010, or 5173 directly to the public Internet. The
Python SDK is a peer client of the same unauthenticated API, not a privileged
path.

## Verify setup

For an ordinary default-port deployment:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/settings/status
curl --fail http://127.0.0.1:8000/api/yam/cell
```

Use 8010 for the documented field hardware deployment. Status responses are
sanitized and never echo service credentials, vendor payloads, or raw CAN.
Complete the [full mock smoke gate](/smoke-test) before physical discovery.
