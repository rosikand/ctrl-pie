---
title: "YAM cell setup"
description: "Discover stable identities, configure a YAM cell, preflight passively, and connect selected arms explicitly."
icon: "cable"
---

V1.2 configures one primary YAM-specific cell containing up to 16 typed arms.
It supports multiple CAN-connected YAM leaders/followers and retains the V1.1
serial-GELLO topology through a compatibility adapter. The deterministic mock
cell has two independent leader/follower pairs; product logic is not hard-coded
to four arms, those pair names, adapter serials, or Linux `canN` values.

<Warning>
  No physical YAM was available in the V1.2 development environment. Mock and
  fake-worker tests validate software boundaries, not controller behavior,
  directions, offsets, gravity compensation, gripper calibration, limits,
  E-stop response, container permissions, or safe disconnect on the real rig.
</Warning>

## Cell and arm identity

Each cell arm records:

- stable `logical_id` and display name;
- `leader` or `follower` role;
- optional `pair_id`, `group_id`, and `side`;
- `socketcan` or retained `serial` transport;
- durable physical identity;
- end effector (`yam_teaching_handle`, `linear_4310`, `crank_4310`, or legacy
  `gello` as allowed by role/transport);
- follower-only frame-map and soft-limit paths; and
- optional legacy model/calibration fields.

For SocketCAN, durable identity is the USB-CAN adapter serial. Discovery maps
that serial to the current kernel interface. `can0`, `can1`, and similar names
are telemetry only and are never saved as identity. Missing, duplicate, or
colliding identities fail closed; ctrl-π does not substitute another arm.

A declared pair contains exactly one leader and one follower and has consistent
side/group metadata. Pair ports, when supplied, must be unique. Pair identity
prevents a left leader from accidentally controlling a right follower.

## Passive discovery and preflight

Settings → YAM Cell follows this order:

1. **Discover hardware** — enumerate bounded sysfs SocketCAN devices and
   stable legacy serial candidates.
2. **Assign arms** — enter logical role, pair/group/side, stable identity, and
   end-effector type.
3. **Preflight** — validate topology, identity resolution, link state, exact
   local i2rt source, and configured artifacts.
4. **Save cell** — persist normalized cell/arm rows without opening hardware.
5. **Connect selected arms** — a separate acknowledged active operation.

Discovery/preflight may inspect bounded files, sysfs, link state, and Git
identity. It must not:

- open a CAN/serial device or construct an i2rt/vendor YAM object;
- ping, enable, zero, or command a motor;
- calibrate a gripper;
- bring a CAN link up/down or change bitrate; or
- start gravity compensation or a control loop.

The Ubuntu host owns working CAN link configuration. A down/missing link is a
readiness result, not permission for ctrl-π to mutate it.

## Exact local i2rt source

All-CAN workers load only an operator-mounted read-only checkout. Enter its
container path (normally `/opt/i2rt`) and one full lowercase 40-character
commit in the cell. Preflight accepts read-only status only when the mounted
filesystem reports the kernel `ST_RDONLY` flag; host `chmod`, ownership, or an
effective-permission check is not sufficient. Docker's `/opt/i2rt:ro` bind
supplies that flag. Preflight also rejects dirty, missing, or different source.
The hardware Docker image records the commit whose dependencies it contains;
all identities must match.

ctrl-π never fetches i2rt, resolves a branch name, or chooses latest. In
particular, the earlier field-tested commit is not advertised by public
upstream; see [Docker deployment](/docker-deployment) for the local-source
build contract.

## Frame maps and soft limits

Follower commands use this order:

1. apply the six-joint frame map;
2. apply the six-joint soft-limit clamp; and
3. preserve the separately normalized gripper command.

A frame map has exactly six `sign` and `offset` entries. Signs are `+1` or
`-1`; a `+1` entry requires zero offset so the mapping remains involutive. An
absent frame-map path means intentional 1:1 mapping and is valid.

Soft limits have six nullable lower/upper entries; `null` means unbounded. An
absent artifact does not create a default. Preflight and Arms display:

```text
NO SASH GUARD
```

The reference cell has no current soft-limit artifacts. Historical limits
belonged to a failed/different mount and must not be restored. Open-table
development may proceed under the reviewed operator policy; in-hood operation
remains blocked until limits are measured for the current mount.

## Teaching-handle range health

Adapter/link connectivity does not prove a CAN `yam_teaching_handle` is ready.
After passive preflight, run the separate range test for one leader. It requires
an acknowledgement because it opens an active, read-only CAN diagnostic. Move
the trigger released → squeezed → released; ctrl-π reports reachability,
observed minimum/maximum, and healthy/stuck status independently of arm
connectivity.

The test never zeros an encoder. A pinned value such as `1.000 .. 1.000` is an
unhealthy handle. Re-zero remains a separately approved external i2rt encoder
manager procedure with the trigger mechanically released; there is no UI,
REST, or SDK zero action in V1.2.

## Explicit connect and disconnect

Connect can target selected logical arm IDs. It requires a fresh general
hardware-motion acknowledgement. Selecting any `linear_4310` or `crank_4310`
follower also requires a distinct acknowledgement with this meaning:

```text
Connecting this follower will enable the arm controller and calibrate its
4310 gripper. The jaws will move. Clear the jaws and arm workspace.
```

The driver re-resolves stable identity immediately before opening the worker.
It refuses a second writer to the same bus. A connected follower is visibly
energized/holding and may resist manual motion. Never force it by hand.

Disconnect first revokes writes, then requests safe-idle/device close and
joins/reaps the worker. If bounded cleanup is uncertain, the bus remains
blocked and status remains an error; do not claim the arm is limp merely
because the request returned or a container stopped. Support the arm when
torque/control releases.

Automatic restoration is off for a new cell. Enabling it requires separate
unattended-motion consent and the 4310-gripper calibration acknowledgement
where applicable. Missing hardware is passively rechecked, but a runtime or
vendor error latches for manual review and is not blindly retried.

## Pair teleop boundary

Start Teleop acquires one declared pair and reads both arms. It starts with
sync disabled, exposes live joint deltas, and performs zero follower writes.
After inspecting those deltas and clearing the workspace, **Enable Sync** is a
second freshly acknowledged motion action. It slowly interpolates the follower
toward the latest mapped leader target over approximately three seconds, then
begins tracking. **Disable Sync** stops writes while pair telemetry remains.

This lifecycle is required; merely selecting the correct pair is not enough.
See [Recording and teleoperation](/recording) and the ordered
[field acceptance](/yam-field-test).

## REST and Python SDK

The canonical routes are:

| Operation | Route |
| --- | --- |
| Status | `GET /api/yam/cell` |
| Passive discovery | `POST /api/yam/cell/discover` |
| Passive preflight | `POST /api/yam/cell/preflight` |
| Save | `PUT /api/yam/cell` |
| Connect selected/all | `POST /api/yam/cell/connect` |
| Disconnect selected/all | `POST /api/yam/cell/disconnect` |
| Handle range | `POST /api/yam/cell/handle-check` |
| Reset | `DELETE /api/yam/cell` |

The typed client exposes `get_yam_cell()`, `discover_yam_cell()`,
`preflight_yam_cell()`, `save_yam_cell()`, `connect_yam_arms()`,
`disconnect_yam_arms()`, and `check_yam_handle()`. The V1.1
`/api/yam/setup` and setup client methods remain compatibility aliases; new
multi-arm integrations should use the cell names.

Both UI and SDK call the same backend checks. Neither exposes raw CAN/motor
commands or bypasses motion acknowledgements.

## Mock behavior

The default mock cell has two declared pairs and four stable logical arms:

- `yam-leader` / `yam-follower` (right pair);
- `yam-leader-left` / `yam-follower-left` (left pair).

It supplies fake stable adapter identities, shuffled-runtime-map test seams,
healthy/unhealthy handle fixtures, per-arm holding state, and calibrated-4310
metadata. Mock mode does not request physical motion acknowledgements. Mock
save/reset never overwrites or deletes the persisted physical cell. Mock
success is not discovery, calibration, or physical validation.

## Legacy topology

The retained adapter continues to support one stable
`/dev/serial/by-id/...` GELLO leader with one SocketCAN `crank_4310` follower
and existing LeRobot model/calibration artifacts. It remains available through
the old setup model and [separate Docker override](/docker-deployment#retained-legacy-gello-deployment).
Do not use it to describe or connect the four-CAN reference cell.
