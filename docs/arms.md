---
title: "Robots"
description: "Read live YAM telemetry and diagnostics, with mock/legacy bounded jog compatibility."
icon: "bot"
---

Robots is the live hardware view. It streams point-in-time snapshots over
`/ws/arms`; joint state, pose, gripper values, bus health, and loop timing are
never persisted. The earlier `/arms` URL now redirects to `/robots`.

## Select an arm

**Robots** lists every configured cell arm with its role, pairing, connection,
and bus. Opening a row shows that arm's connection summary and joint state
first; pose, gripper or teaching handle, pair alignment, loop diagnostics,
identity, and manual jog are collapsed under **Details**. Mock mode exposes
two declared pairs: `yam-leader` / `yam-follower` and `yam-leader-left` /
`yam-follower-left`. Hardware logical IDs come from the saved cell and remain
stable even if USB enumeration changes or startup fails.

The connection strip shows:

- driver connection state;
- transport plus durable adapter identity and current runtime CAN interface;
- leader/follower role plus pair/group/side;
- connected control state and explicit energized/holding flags;
- end-effector and teaching-handle health; and
- the timestamp of the latest live snapshot.

Manual controls are enabled only when the selected driver reports that jog is
supported. The supervised all-CAN adapter rejects one-shot jog even while
telemetry is live and the arm is connected.

## Telemetry panels

| Panel | Values |
| --- | --- |
| Joint state | Six named positions in radians/degrees, velocity, effort, and optional motor temperature |
| End-effector pose | XYZ in millimetres plus roll, pitch, and yaw; real mode derives this from the configured MuJoCo model |
| Gripper | Normalized position where `0=closed` and `1=open`, derived closed state, velocity, and optional force |
| Loop diagnostics | Observed i2rt/control-loop source, frequency, cycle time, jitter, dropped cycles, and available bus counters |

An em dash means the active adapter cannot supply that field. The earlier
field-tested i2rt path observed follower loops around 263–275 Hz and a leader
around 419–426 Hz at `bilateral_kp=0.0`. Those values are reference
observations, not product health thresholds. Inspect degradation/instability
in context rather than declaring failure from one fixed number.

Follower cards show `NO SASH GUARD` when no soft-limit artifact is configured.
That is a truthful warning, not a synthesized limit.

## Manual jog

The UI offers three command groups:

- **Joint** — choose a named joint and a 1°, 5°, or 10° increment.
- **Cartesian** — translate 5 mm or rotate 5° per click.
- **Gripper** — open or close by 10% normalized travel.

The API rejects unknown axes, non-finite values, disconnected arms, lease
conflicts, and out-of-bounds deltas. Schema maxima are 0.25 radians for a joint,
0.05 metres for translation, 0.15 radians for rotation, and 0.10 for gripper
travel.

<Warning>
  The supervised all-CAN adapter rejects every one-shot jog in V1.2. Mock mode
  retains bounded jogs, and the legacy hardware adapter retains follower joint
  and gripper jog compatibility; Cartesian and leader jogs still fail there.
  A connected follower may be energized and holding—never force it by hand.
</Warning>

## Control ownership

Inference leases one follower; teleop leases its declared leader and follower.
Mock/legacy jog also leases one follower. Disjoint pairs can remain independent,
while overlapping resources fail immediately with HTTP 409. Commands are not
queued. Run exactly one Uvicorn worker; the resource lease is not a distributed
lock. The hardware adapter additionally refuses a second local writer on one
resolved bus.

## Connection recovery

The frontend reconnects its telemetry WebSocket and labels the state as
**Connecting**, **Telemetry live**, **Reconnecting**, or **Offline**. In real
mode, a driver sample or worker failure latches motion unavailable rather than
serving a stale snapshot as healthy.

A saved physical cell reconnects after boot or hot-plug only when the
operator explicitly enabled automatic connection and acknowledged its motion
risk and, where applicable, jaw-calibration motion. While prerequisites are
missing, the backend repeats passive preflight and makes one serialized
connection attempt when the selected arms become ready. It does not respawn a
latched worker/runtime error. After making the rig safe and correcting the
fault, use explicit **Connect** in Settings.

For real-device mapping and failure semantics, see
[YAM driver interface](/yam-driver). For a disconnected real rig, begin with
[YAM setup](/yam-setup) and [Troubleshooting](/troubleshooting).
