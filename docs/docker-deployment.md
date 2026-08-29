---
title: "Docker deployment"
description: "Run mock, all-CAN YAM-cell, or legacy GELLO deployments with explicit isolation."
icon: "container"
---

ctrl-π ships one non-root production service. A Node build stage compiles the
Vite application; one Uvicorn worker then serves the static application and
FastAPI control plane. The worker owns process-local rig leases, recordings,
and inference cleanup. Do not scale it horizontally.

PostgreSQL, Hugging Face, and Modal remain external services in accounts you
control. Compose does not install a database, bake credentials into the image,
mount the Docker socket, use `privileged: true`, or grant `NET_ADMIN`.

Use Docker Engine with current Compose/Buildx. The supplied hardware overrides
use Compose's `!reset` tag and require Compose 2.24 or newer.

## Mock and ordinary bridge mode

```bash
cp .env.example .env
${EDITOR:-vi} .env
docker compose build
docker compose up -d --wait
curl --fail http://127.0.0.1:8000/api/health
```

Leave `CTRL_PI_MOCK_MODE=true`. The four-arm mock cell, synthetic camera, and
Stub workload need no YAM device or Modal account. `DATABASE_URL` must be
reachable from inside the container; container-local `localhost` does not
name the Ubuntu host.

Bridge mode has two distinct ports:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CTRL_PI_LISTEN_PORT` | `8000` | Uvicorn's internal/container port. |
| `CTRL_PI_PORT` | `8000` | Host port published by Compose. |
| `CTRL_PI_BIND_ADDRESS` | `127.0.0.1` | Host address published by Compose. |

For example, this maps host loopback 8010 to internal 8000:

```bash
CTRL_PI_PORT=8010 docker compose up -d --wait
```

The host-network hardware overrides do not publish a port. There,
`CTRL_PI_LISTEN_PORT=8010` makes Uvicorn itself listen on host port 8010 so it
can coexist with another service on 8000.

ctrl-π has no authentication. Keep it on loopback or a trusted, firewalled
LAN; never expose it directly to the public Internet.

Useful mock checks are:

```bash
docker compose ps
docker compose logs -f app
docker compose exec app python -m ctrl_pi.smoke --fake-hub
```

## Data, credentials, and shutdown

The named `ctrl-pi-data` volume is mounted at `/var/lib/ctrl-pi`. It contains
unuploaded recording staging and caches. PostgreSQL and Hugging Face are the
durable control-plane/artifact stores, but a finalized recording remains local
until its upload is verified. `docker compose down` preserves the volume;
`docker compose down --volumes` destroys it.

The image runs as UID/GID 10001 and drops every Linux capability. Hardware
overrides add only `NET_RAW`. They retain `no-new-privileges`, make the image
root filesystem read-only, and provide a bounded `/tmp` tmpfs. The data volume
remains writable. If you replace it with a bind mount, grant UID 10001 access.

Use Compose for shutdown so Uvicorn receives SIGTERM and the 90-second grace
period:

```bash
docker compose stop
docker compose down
```

The backend stops new writes before recordings, inference, workers, and the
driver. A YAM worker that cannot prove safe-idle/close/reap remains an explicit
latched error; do not interpret container exit alone as a safe arm.

## Primary all-CAN YAM-cell deployment

The primary V1.2 hardware path supports configurable SocketCAN YAM leaders
with `yam_teaching_handle` and followers with `linear_4310` or `crank_4310`
grippers. It does not assume four arms, particular serials, or fixed `canN`
names.

Use [docker-compose.yam-cell.yml](https://github.com/rosikand/ctrl-pie/blob/v1.2-yam-cell/docker-compose.yam-cell.yml).
The override:

- uses host networking so existing host SocketCAN links are visible;
- publishes no Compose port and defaults Uvicorn to 8010;
- mounts `/sys:ro` for bounded USB-adapter serial discovery;
- mounts one operator-owned local i2rt checkout at `/opt/i2rt:ro`;
- adds `NET_RAW` for existing CAN raw sockets;
- passes no serial device and does not mount `/dev`;
- has no `NET_ADMIN`, so ctrl-π cannot bring a link up/down or change bitrate;
  and
- never uses privileged mode.

The `:ro` bind is a safety gate, not just a permissions suggestion. Runtime
preflight accepts only the filesystem's kernel `ST_RDONLY` mount flag as
read-only proof. Host `chmod`, UID ownership, and `os.access`-style effective
permission checks do not substitute for that flag.

These access assumptions are not claimed as universally proven. H1 must
confirm that UID 10001, `NET_RAW`, read-only sysfs, and host networking are
sufficient on the target Ubuntu kernel. Do not broaden access during the
session; return any needed change for review.

### Supply an exact local i2rt checkout

The field-tested commit is
`a32cfa660151d75319751be69cfa580a1f8b344c`, but public upstream does not
advertise that commit or the field branch. ctrl-π therefore never clones it,
fetches it, or silently chooses latest. Point Compose at the exact
operator-controlled local checkout and the commit you have independently
reviewed:

```bash
export CTRL_PI_I2RT_CHECKOUT=/absolute/path/to/i2rt
export CTRL_PI_I2RT_DEPENDENCY_COMMIT=a32cfa660151d75319751be69cfa580a1f8b344c

test "$(git -C "$CTRL_PI_I2RT_CHECKOUT" rev-parse --show-toplevel)" = \
  "$(realpath "$CTRL_PI_I2RT_CHECKOUT")"
test "$(git -C "$CTRL_PI_I2RT_CHECKOUT" rev-parse --verify 'HEAD^{commit}')" = \
  "$CTRL_PI_I2RT_DEPENDENCY_COMMIT"
test -z "$(git -C "$CTRL_PI_I2RT_CHECKOUT" status \
  --porcelain=v1 --ignored --untracked-files=all)"
```

If any command fails, stop. Do not run `git fetch`, switch branches, reset,
clean, or modify the host i2rt environment as a ctrl-π setup step. Obtain the
exact source from its owner. The final status command rejects modified,
staged, untracked, and ignored content; an ignored file can still shadow an
import from the pinned tree.

The `yam-cell-runtime` image target installs dependency declarations from that
local clean checkout, then uninstalls the packaged i2rt source. The checked-in
`docker/i2rt-build-constraints.txt` is passed through `PIP_CONSTRAINT`, so
pip's isolated `ruckig` build honors i2rt's required
`scikit-build-core<0.10` ceiling. The image retains the
build identity in the immutable image environment and OCI label as
`CTRL_PI_I2RT_DEPENDENCY_COMMIT`. At runtime, the supervised worker can import
i2rt only from `/opt/i2rt:ro`. That bind must expose the kernel `ST_RDONLY`
mount flag inside the container; chmod or an effective-permission scan is not
accepted as proof. Application preflight requires the image marker, configured
cell commit, and mounted checkout HEAD to match. Neither build nor preflight
consults a remote.

Changing the configured or mounted i2rt commit requires rebuilding this
hardware target with the same new `CTRL_PI_I2RT_DEPENDENCY_COMMIT`. Merely
editing the cell row or changing the read-only mount must fail preflight; it
must never reuse dependencies baked from a different revision.

### Build and start without connecting

Set the host-network listen port and render the merged configuration. Use the
required identity-verifying Make target; do not replace it with an ad hoc
`docker build` or a build from a remote i2rt URL:

```bash
export CTRL_PI_LISTEN_PORT=8010
docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml config -q
make docker-yam-cell
docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml \
  up -d --wait --no-build
curl --fail http://127.0.0.1:8010/api/health
```

Starting hardware mode does not grant new motion consent. Before startup,
however, prove that neither `yam_cells` nor the retained `yam_setups` row has
unattended auto-restore enabled. The exact read-only audit and explicit revoke
commands are in [H1 of the field handoff](https://github.com/rosikand/ctrl-pie/blob/v1.2-yam-cell/V1_2_FIELD_TEST_HANDOFF.md#h1--passive-real-cell-discovery).
Run them through the ordinary non-hardware one-shot image, rerun the read-only
audit, and require false/absent proof. Do not start the hardware application on
an ambiguous result.

In Settings → YAM Cell, enter `/opt/i2rt` and the same exact commit. Discover
and preflight are passive: they inspect bounded sysfs, link state, the local
checkout, and configured artifacts without opening a CAN device, importing a
motion-capable YAM object, pinging/enabling a motor, calibrating a gripper, or
changing a link.

The separate teaching-handle range test opens an explicitly acknowledged,
read-only CAN encoder diagnostic. Connect is a different motion boundary.
Connecting a follower can enable/hold the arm, and calibrating a `linear_4310`
or `crank_4310` follower moves the jaws.

The supervised all-CAN adapter deliberately rejects one-shot jog in V1.2.
Validate physical action only through the ordered sync/inference gates below.
Bounded jog remains available to mock mode and the retained legacy adapter.

Follow the ordered [YAM cell field acceptance](/yam-field-test) procedure.

## Retained legacy GELLO deployment

[docker-compose.legacy-gello.yml](https://github.com/rosikand/ctrl-pie/blob/v1.2-yam-cell/docker-compose.legacy-gello.yml)
retains the V1.1 topology: one SocketCAN follower with `crank_4310` and one
serial GELLO leader. Do not combine this override with the all-CAN override.

Select exactly one stable serial device and the two read-only artifact roots:

```bash
export YAM_LEADER_DEVICE=/dev/serial/by-id/your-gello-leader
export YAM_LEADER_GID="$(stat -Lc '%g' "$YAM_LEADER_DEVICE")"
export YAM_CAN_INTERFACE=can0
export YAM_MODEL_DIR=/absolute/path/to/model-assets
export YAM_MODEL_XML=yam_crank_4310_d405.xml
export YAM_CALIBRATION_DIR=/absolute/path/to/leader-calibration
export YAM_LEADER_CALIBRATION_ID=yam-leader
export CTRL_PI_LISTEN_PORT=8010

docker compose -f docker-compose.yml -f docker-compose.legacy-gello.yml config -q
docker compose -f docker-compose.yml -f docker-compose.legacy-gello.yml \
  up -d --wait
```

The container sees that one device as `/dev/yam-leader`; it does not receive
all serial devices or all of `/dev`. The override retains `NET_RAW`, drops all
other capabilities, and leaves CAN link administration on the host.

Legacy `YAM_CAN_INTERFACE` is an intentional compatibility bootstrap. New
YAM-cell rows persist a USB-CAN adapter serial and resolve current `canN` at
runtime instead.

## Host SocketCAN ownership

The Ubuntu host or its existing reviewed service owns bitrate and link state.
ctrl-π's passive path does not run `ip link set`, Lux `yam_bringup.sh`, or a
motor ping. Inspect existing state without mutation:

```bash
ip -details -statistics link show type can
```

If a required link is down or misconfigured, stop ctrl-π and follow the host's
approved CAN procedure. Do not grant the container `NET_ADMIN` as a shortcut.

## Modal cleanup

If ordinary inference or managed-training cleanup cannot converge, use the
ownership-verified emergency command. It is independent of PostgreSQL startup:

```bash
docker compose run --rm app python -m ctrl_pi.modal_panic
```

This can stop paid resources and requires fresh operator approval. Review
[Modal operations](/modal-operations) before running it.
