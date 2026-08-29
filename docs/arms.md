---
title: "Arms"
description: "Read live YAM telemetry, diagnostics, and issue bounded manual jog commands."
icon: "bot"
---

The Arms page is the live hardware view. It streams point-in-time snapshots
over `/ws/arms`; joint state, pose, gripper values, bus health, and loop timing
are never persisted.

## Select an arm

Use **Active arm** to switch between the configured leader and follower. In
mock mode the standard identities are `yam-leader` and `yam-follower`.
Hardware mode exposes those same stable IDs even when startup fails, making a
disconnect visible without changing database relationships.

The connection strip shows:

- driver connection state;
- device bus/interface state and configured bitrate;
- leader or follower role; and
- the timestamp of the latest live snapshot.

Manual controls are enabled only while WebSocket telemetry is live and the
selected arm reports connected.

## Telemetry panels

| Panel | Values |
| --- | --- |
| Joint state | Six named positions in radians/degrees, velocity, effort, and optional motor temperature |
| End-effector pose | XYZ in millimetres plus roll, pitch, and yaw; real mode derives this from the configured MuJoCo model |
| Gripper | Normalized position where `0=closed` and `1=open`, derived closed state, velocity, and optional force |
| Loop diagnostics | ctrl-π sampling frequency, cycle time, jitter, dropped cycles, and available bus counters |

An em dash means the active driver cannot supply that field. The pinned real
leader has no effort measurement; temperatures may be absent; real gripper
force and CAN error counters are unavailable. Driver timing describes ctrl-π's
sampling loop, not motor firmware timing.

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
  The real V1 adapter accepts only follower joint and gripper jogs. Cartesian
  jog and leader commands fail explicitly. Software bounds do not replace
  firmware limits, physical guards, or an emergency stop.
</Warning>

## Control ownership

Manual jog, teleoperation, and inference share one process-local `RigLease`.
A conflicting command fails immediately with HTTP 409; commands are not queued
behind another owner. Run exactly one Uvicorn worker, since the lease is not a
distributed lock or CAN arbiter.

## Connection recovery

The frontend reconnects its telemetry WebSocket and labels the state as
**Connecting**, **Telemetry live**, **Reconnecting**, or **Offline**. In real
mode, a driver sample or worker failure latches motion unavailable rather than
serving a stale snapshot as healthy. Correct the host or device issue and
restart the backend; V1 does not automatically reconnect hardware.

For real-device mapping and failure semantics, see
[YAM driver interface](/yam-driver). For a disconnected real rig, begin with
[YAM setup](/yam-setup) and [Troubleshooting](/troubleshooting).
