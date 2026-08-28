# YAM driver interface

`YAMDriver` is ctrl-π's synchronous boundary around arm hardware. HTTP
handlers, teleoperation, recording, and inference use this interface; vendor
CAN and serial objects do not cross it. `MockYAMDriver` remains the complete,
default implementation for development. With `CTRL_PI_MOCK_MODE=false`, the
application instead selects `RealYAMDriver`; hardware mode never falls back to
mock arms after a configuration, import, calibration, or connection failure.

The authoritative types are in
[`drivers/yam.py`](../backend/src/ctrl_pi/drivers/yam.py). The two
implementations are
[`drivers/mock_yam.py`](../backend/src/ctrl_pi/drivers/mock_yam.py) and
[`drivers/real_yam.py`](../backend/src/ctrl_pi/drivers/real_yam.py).
Single-process command arbitration is in
[`rig.py`](../backend/src/ctrl_pi/rig.py).

## Interface contract

```python
class YAMDriver(ABC):
    def startup(self) -> None: ...
    def shutdown(self) -> None: ...
    def diagnostic(self) -> YAMDriverDiagnostic: ...
    def list_arms(self) -> list[ArmTelemetry]: ...
    def get_arm(self, arm_id: str) -> ArmTelemetry: ...
    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry: ...
    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry: ...
```

| Method | Required behavior |
| --- | --- |
| `startup()` | Open resources and start sampling. A failure leaves the driver safely unavailable; it does not select a different implementation. |
| `shutdown()` | Reject new commands, stop sampling, put the follower in the vendor safe mode, and close both devices. Repeated calls are safe. |
| `diagnostic()` | Return a sanitized `configured`, `connected`, `missing`, or `error` summary without device paths or vendor payloads. |
| `list_arms()` | Return a copied, point-in-time snapshot for every configured arm. An empty list is valid for another implementation. |
| `get_arm(arm_id)` | Return one copied snapshot or raise `ArmNotFoundError`. |
| `jog(arm_id, command)` | Apply one validated relative command atomically and return the resulting snapshot. Raise `JogLimitError` for an unsafe or unsupported target. |
| `apply_action(arm_id, action)` | Apply one absolute six-joint-plus-gripper target atomically and return the resulting snapshot. Raise `ActionLimitError` rather than silently clipping it. |

Known driver failures are translated to bounded, operator-facing messages.
Raw exceptions, serial paths, CAN frames, and device payloads must not be
returned to the browser.

## Supported V1 hardware

The real adapter is deliberately narrow. It integrates one standard YAM
follower on SocketCAN with one GELLO/Dynamixel YAM leader and the
`crank_4310` gripper, using these pinned plugins:

- `lerobot-robot-yam==0.1.1`;
- `lerobot-teleoperator-yam-gello==0.1.1`; and
- `yam-common==0.1.1`.

YAM Pro, YAM Ultra, other leader devices, multiple pairs, and the linear
gripper are not supported by this V1 adapter. In particular,
`YAM_GRIPPER_TYPE` must be `crank_4310`; any other value fails configuration
before a vendor object or device is opened. Configure the SocketCAN interface,
leader serial port, MuJoCo model, and existing leader calibration as described
in [Setup and configuration](setup.md).

Vendor imports and device construction are lazy. Merely importing ctrl-π, or
running the default preflight probe, does not construct a vendor object or
open a bus. During FastAPI lifespan startup the real driver connects both
devices with `calibrate=False`, takes a first sample, and only then reports
connected. A missing calibration is an error to correct outside the server;
ctrl-π never starts an interactive calibration routine.

Even when startup cannot connect, the backend remains available. `/api/arms`
continues to expose the stable `yam-leader` and `yam-follower` identities as
disconnected, Settings shows the sanitized diagnostic, and every motion call
fails closed. Correct the host/configuration issue and restart the process.

## IDs, mapping, and units

Roles are exactly `leader` and `follower`. Teleoperation reads
`yam-leader` and writes the distinct `yam-follower`; only the follower accepts
manual or inference commands.

The pinned plugins and ctrl-π use different names and a different wrist order,
so the adapter maps by name in both directions:

| Plugin field/order | ctrl-π joint |
| --- | --- |
| `shoulder_pan` | `shoulder_yaw` |
| `shoulder_lift` | `shoulder_pitch` |
| `elbow_flex` | `elbow_pitch` |
| `wrist_flex` | `wrist_pitch` |
| `wrist_roll` | `wrist_roll` |
| `wrist_yaw` | `wrist_yaw` |

Follower positions and command targets are radians. The GELLO leader reports
each rotary joint on `[-100, 100]`; the adapter maps that value linearly into
the corresponding pinned follower range. Its gripper `[0, 100]` becomes
ctrl-π `[0, 1]` after inversion at the vendor boundary: ctrl-π preserves
`0=closed` and `1=open`, while the pinned crank plugin uses `0=open` and
`1=closed`. This conversion, the GELLO gripper endpoint orientation, and the
`wrist_flex`/`wrist_roll` reordering must be checked on the physical pair
before enabling teleop.

## Cached telemetry and unavailable fields

The real adapter samples the devices on a background loop with a 50 Hz target
and serves copied snapshots from memory. API and WebSocket reads do not start
device discovery or issue a new motion command. A sampling or vendor-worker
failure stops command acceptance, marks both cached arms disconnected, and
enters the failure shutdown path instead of presenting the last sample as
healthy.

Real fields have these sources:

| Public telemetry | Real-driver source |
| --- | --- |
| Follower joint position, velocity, and effort | Pinned follower observation. The physical test must confirm effort units and signs. |
| Leader joint position | GELLO normalized action mapped to the pinned follower ranges. |
| Leader joint velocity | Difference between consecutive mapped samples. |
| Leader joint effort | Unavailable (`null`); the GELLO plugin exposes no effort measurement. |
| End-effector pose | Forward kinematics from the configured MuJoCo XML's `joint1`–`joint6` and `grasp_site`; it is model-derived, not a measured Cartesian sensor. |
| Gripper position and velocity | Normalized plugin position and follower `joint_vel[6]`; leader velocity is derived between samples. |
| Gripper closed state | Derived from normalized position (`<= 0.15`), not an independent contact switch. |
| Loop frequency, cycle time, jitter, drops | The ctrl-π sampling loop, not an internal motor-control-loop diagnostic. |
| CAN/serial state | `active` only while sampling succeeds; otherwise `disconnected`. It is not a kernel CAN error-counter query. |

The pinned interfaces do not expose leader effort, do not consistently expose
motor temperature, do not expose gripper force, and do not expose CAN
transmit/receive error counters. Those fields are nullable: the UI renders
`null` as an em dash. Leader effort and temperatures are unavailable; follower
temperatures are present only when the plugin supplies `temp_mos`. Real
gripper force and both CAN counters remain unavailable. `MockYAMDriver`
continues to provide numeric values for all of them, as does the real follower
for effort, so mock behavior and available diagnostics remain intact.

## Commands and failure safety

`JogCommand` is a single non-zero relative change. Schema validation rejects
unknown axes, non-finite values, and deltas over these API bounds:

| Kind | Axes | Maximum absolute delta |
| --- | --- | --- |
| `joint` | the six named joints | `0.25` radians |
| `cartesian` translation | `x`, `y`, `z` | `0.05` metres |
| `cartesian` rotation | `roll`, `pitch`, `yaw` | `0.15` radians |
| `gripper` | `position` | `0.10` normalized units |

The real adapter supports follower joint and gripper jogs. Cartesian jogs are
explicitly rejected because the pinned plugin accepts joint-space commands.
It enforces the pinned per-joint hard ranges, and an `ArmAction` additionally
limits each change from the latest follower sample to `0.35` radians per joint
and `0.20` gripper units. The entire command is rejected before the write if
any value is invalid.

The first accepted motion command switches the follower into the plugin's
position-control mode before writing the seven-element target in plugin order.
A write, telemetry, or vendor-worker failure latches the driver unavailable,
rejects subsequent commands, stops sampling, calls the plugin's
`zero_torque_mode()`, closes the follower, and disconnects the leader. Recovery
requires a backend restart; there is no automatic reconnect during V1.

The plugin safe-mode call is not a claim that the arm is electrically
torque-free: its gravity-compensation worker or controller state may still be
active until shutdown completes. Software bounds and shutdown calls do not
replace firmware current/torque limits, collision constraints, physical
guards, or an independently accessible emergency stop.

## Threading and rig ownership

One application-scoped, thread-safe driver is shared by HTTP handlers,
`/ws/arms`, teleop, recording, inference, and shutdown. Its public reads return
immutable copies, and writes are serialized. FastAPI starts it off the event
loop before the recording and inference managers; shutdown stops inference and
recording writes before shutting down the driver.

`RigLease` prevents two ctrl-π modes from commanding the same process-local
rig. Acquisition is non-blocking: a conflict fails instead of waiting or
queueing. Manual jog holds a lease for one call, teleoperation holds it for
the session, and inference holds it through its robot loop. Passive inference
recording reuses the inference owner and never commands the arm. Run exactly
one Uvicorn worker; this lease is not CAN arbitration or a distributed lock.

## Probe commands

With hardware mode and all `YAM_*` settings loaded, this performs configuration
preflight only:

```bash
make yam-probe
```

It constructs the fail-closed adapter and checks, in order:

- required values and the supported `crank_4310` configuration;
- installed distribution metadata for all three exact `0.1.1` package pins;
- that the configured MuJoCo XML and leader calibration JSON are readable
  regular files;
- that the leader serial device exists and is readable and writable; and
- that the configured network-interface name resolves on the host.

It then prints a small JSON report and shuts down without importing vendor
modules, constructing their devices, or opening the leader/follower. These
read-only host checks do **not** parse the XML/calibration contents, validate
referenced model assets, prove the named interface is SocketCAN at 1 Mbit/s,
or prove that either physical device responds.

Only after securing the arm and positioning an operator at the emergency stop,
opt into one connection and sample:

```bash
PYTHONPATH=backend/src .venv/bin/python -m ctrl_pi.yam_probe --connect
```

The connected probe does not jog or call `apply_action`, but it is not
electrically read-only. The leader plugin configures its serial bus and the
follower starts its gravity-compensation/control worker. The probe prints only
safe identity/connection/bus state and always attempts the driver's bounded
shutdown path. A vendor I/O thread that does not return still requires the
operator to make the hardware safe and terminate the process.

## Required Ubuntu/YAM validation

No physical YAM is attached to the cloud development environment. Fake-vendor
tests validate adapter logic but cannot establish physical safety or device
compatibility. Before describing the real integration as hardware-validated,
perform and record all of the following on the target Ubuntu/YAM box:

1. Confirm the exact standard YAM follower, GELLO leader, and `crank_4310`
   gripper; reject Pro, Ultra, linear-gripper, or unreviewed variants.
2. Configure the intended SocketCAN interface at 1 Mbit/s and verify CAN and
   stable leader serial-device permissions for the non-root process/container.
3. Verify the pinned packages, MuJoCo XML and referenced assets, `joint1`–
   `joint6`, `grasp_site`, and the existing GELLO calibration file.
4. With power/motion controls secured and an operator on the emergency stop,
   run preflight and then the connected probe. Confirm clean close and the
   expected vendor gravity-compensation/safe-mode behavior.
5. Compare each reported joint against independent physical motion. Verify
   zero offsets, directions, radians, the GELLO normalization endpoints, and
   especially the distinct wrist-flex/wrist-roll mapping. Confirm FK pose
   direction and scale against the physical tool point.
6. Check follower velocity/effort units and signs, optional temperatures,
   gripper position/velocity, and observed ctrl-π sampling frequency. Do not
   expect force or CAN error counters from this plugin version.
7. Align leader and follower safely, then issue individually bounded, low-speed
   joint and gripper jogs. Verify the position-control transition, hard limits,
   per-command limits, emergency stop, current/torque limits, and collision
   protections before teleoperation.
8. Exercise teleop start/stop and process restart, then unplug serial, unplug
   or bus-off CAN, stop the vendor worker, and induce a command failure. Each
   case must stop motion, disconnect telemetry, latch commands unavailable,
   close resources, and leave the shared `RigLease` free.
9. Repeat the connected probe and a bounded mock-policy-compatible flow inside
   the documented production container with the exact serial/CAN/model mounts.
   Only run a real policy after independently verifying its observation,
   action, and safety compatibility with this YAM pair.

Container setup is in [Docker deployment](docker-deployment.md). Recording
ownership is in [Recording and teleoperation](recording.md).
