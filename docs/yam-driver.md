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
    def setup_config(self) -> YAMSetupConfig | None: ...
    def discover_setup(self) -> YAMDiscoveryResult: ...
    def preflight_setup(self, config: YAMSetupConfig) -> YAMPreflightResult: ...
    def apply_setup(self, config: YAMSetupConfig) -> YAMDriverDiagnostic: ...
    def reset_setup(self) -> YAMDriverDiagnostic: ...
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
| `setup_config()` | Return the selected bounded, non-secret single-rig configuration, or `None`; no I/O. |
| `discover_setup()` | Enumerate bounded OS/mock candidates without importing vendor modules or opening devices. |
| `preflight_setup(config)` | Check a candidate without opening devices. Calibration readiness means a bounded, readable artifact with the pinned structural fields—not physical calibration proof. |
| `apply_setup(config)` | Select an already validated configuration in place while resources remain closed. |
| `reset_setup()` | Close resources and restore the driver's process bootstrap configuration. |
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
open a bus. The application-scoped `YAMSetupManager` applies the one saved
physical setup to the same driver instance before other arm consumers start.
If explicit automatic connection was enabled, it connects both devices with
`calibrate=False`, takes a first sample, and only then reports connected. A
missing calibration is an error to correct outside the server; ctrl-π never
starts an interactive calibration routine.
Use the safety-framed, external pinned-CLI
[leader calibration procedure](setup.md#creating-the-leader-calibration-artifact),
then return to onboarding and rerun read-only preflight. The procedure has not
been executed on physical YAM hardware in this development environment.

When no physical row exists, a complete `YAM_*` environment bootstrap selects
configuration but does not open the driver during lifespan startup. The
operator must preflight and save it, then use acknowledged Connect or opt in to
saved automatic connection. Environment configuration cannot bypass persisted
hardware-motion consent.

Even when startup cannot connect, the backend remains available. `/api/arms`
continues to expose the stable `yam-leader` and `yam-follower` identities as
disconnected, Settings shows the sanitized diagnostic, and every motion call
fails closed. For a saved auto-connect setup, a passive monitor repeats only
non-opening preflight while the diagnostic is `missing` and makes one
`RigLease`-protected connection attempt when prerequisites become ready. It
does not retry a latched runtime/vendor `error`; that requires a manual Connect
or process restart.

## Setup and restoration boundary

The `/api/yam/setup` service is the only browser-facing configuration path.
The frontend does not inspect USB, serial, SocketCAN, files, or vendor objects.
Discovery lists at most 32 real SocketCAN interfaces and 32 stable leader
devices under `/dev/serial/by-id` (plus the documented `/dev/yam-leader`
container path), with sanitized labels. Hardware preflight accepts only a
currently discovered CAN/serial pair and bounded absolute model/calibration
paths. Preflight requires Linux ARPHRD_CAN and a character serial device. It
schema-parses a bounded JSON object with exactly the seven pinned motors and
their exact integer fields, motor IDs, drive mode, and ordered ranges. Neither
discovery nor preflight opens a device.

One physical setup is persisted in PostgreSQL. Mock-mode setup uses the same
driver methods with deterministic candidates but never mutates or deletes that
physical row. Saving an update first passes preflight and rolls back both the
database and in-memory selection on failure. A changed setup disconnects the
old selection; changing only auto-restore on the identical connected setup
does not. Setup changes, reset, explicit Connect, and automatic restoration all
compete for the same `RigLease` as jog,
teleop, recording, and inference; a conflict fails immediately rather than
queueing.

Automatic connection is disabled on a new saved setup. Enabling it requires a
backend-enforced acknowledgment that a later boot or hot-plug connection can
engage the follower gravity-compensation/control worker. Explicit Connect has
its own required hardware-motion acknowledgment. These acknowledgments do not
claim the calibration, physical model, bus, limits, or emergency stop were
validated. Disabling automatic connection for the unchanged persisted config
is a no-I/O transaction that works while hardware is missing and does not
disconnect an already connected driver. See
[Setup and configuration](setup.md) for the operator flow.

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
the corresponding pinned follower range. The crank follower and the GELLO
plugin's intended follower-command space use `0=open` and `1=closed` (GELLO
`0` and `100`). ctrl-π preserves its public `0=closed`, `1=open` convention by
inverting both sources at the vendor boundary. This conversion, the GELLO
mounting/calibration endpoint orientation, and the `wrist_flex`/`wrist_roll`
reordering must be checked on the physical pair before enabling teleop.

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
| Gripper position and velocity | Public `0=closed`, `1=open`: follower position is `1 - vendor_position`, follower velocity is `-joint_vel[6]`, and GELLO position is `1 - raw/100`; leader velocity is derived between samples. |
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
is intentionally deliberate: automatic restoration does not retry a latched
runtime/vendor error, and the operator must use manual Connect after review or
restart the backend.

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
- that the configured MuJoCo XML is readable and the bounded leader calibration
  JSON matches the exact seven pinned motors and field/ID/drive-mode/range
  structure;
- that the leader serial device is a readable/writable character device; and
- that the configured network interface is Linux ARPHRD_CAN.

It then prints a small JSON report and shuts down without importing vendor
modules, constructing their devices, or opening the leader/follower. These
read-only host checks do **not** parse the MuJoCo XML or referenced model
assets, validate the calibration values against the physical leader, prove the
named SocketCAN interface runs at 1 Mbit/s, or prove that either physical
device responds.

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
   run preflight and then the connected probe. Verify whether bounded shutdown
   completes and observe the vendor gravity-compensation/safe-mode behavior;
   treat incomplete cleanup as a blocker and make the hardware safe manually.
5. Compare each reported joint against independent physical motion. Verify
   zero offsets, directions, radians, the GELLO normalization endpoints, and
   especially the distinct wrist-flex/wrist-roll mapping. Confirm FK pose
   direction and scale against the physical tool point.
6. Check follower velocity/effort units and signs, optional temperatures, and
   observed ctrl-π sampling frequency. Verify that crank vendor `0=open` and
   `1=closed` becomes public `1=open` and `0=closed`, that follower gripper
   velocity changes sign with the inversion, and that the physically mounted
   GELLO/calibration endpoints follow the same public direction. Do not expect
   force or CAN error counters from this plugin version.
7. Align leader and follower safely, then issue individually bounded, low-speed
   joint and gripper jogs. Verify the position-control transition, hard limits,
   per-command limits, emergency stop, current/torque limits, and collision
   protections before teleoperation.
8. Exercise teleop start/stop and process restart, then unplug serial, unplug
   or bus-off CAN, stop the vendor worker, and induce a command failure. Each
   case must stop motion, disconnect telemetry, latch commands unavailable,
   close resources, and leave the shared `RigLease` free.
9. With the rig independently safe, exercise Settings discovery and preflight
   while observing the devices: confirm neither operation configures a bus,
   starts a controller, nor causes motion. Save the selected setup, restart
   ctrl-π, and confirm the same non-secret configuration survives while
   automatic connection remains off by default.
10. Explicitly opt into automatic connection and confirm exactly one connection
    occurs when the saved devices are ready. Then boot once with the rig
    unplugged, hot-plug it, and confirm exactly one connection occurs. Induce an
    active/vendor failure and confirm it latches without background retries or
    repeated controller engagement.
11. While the saved rig is unplugged, revoke automatic connection without
    forgetting the setup; restart and hot-plug to confirm no connection occurs.
    Boot once in mock mode and then return to hardware mode, confirming the
    physical PostgreSQL row was neither overwritten nor deleted.
12. Repeat the connected probe and a bounded mock-policy-compatible flow inside
   the documented production container with the exact serial/CAN/model mounts.
   Restart the container after a serial-device remap and verify the stable
   mounted path is rediscovered before reconnecting. Only run a real policy
   after independently verifying its observation, action, and safety
   compatibility with this YAM pair.

The ROS-58 lifecycle checks in items 9–12 have not been performed on a physical
Ubuntu/YAM box in this development environment; automated mock and fake-vendor
coverage is not a substitute for recording those observations on the target.

Container setup is in [Docker deployment](docker-deployment.md). Recording
ownership is in [Recording and teleoperation](recording.md).
