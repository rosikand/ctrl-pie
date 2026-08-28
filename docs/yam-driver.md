# YAM driver interface

`YAMDriver` is ctrl-π's synchronous hardware boundary. HTTP handlers,
teleoperation, recording, and inference depend on this interface; CAN/USB
protocol details do not cross it. The current application always injects the
thread-safe, process-local `MockYAMDriver`. Selecting and validating a real
hardware implementation is a Milestone 14 integration point, not a hidden
effect of setting `CTRL_PI_MOCK_MODE=false`.

The authoritative types are in
[`drivers/yam.py`](../backend/src/ctrl_pi/drivers/yam.py), the reference mock
is in [`drivers/mock_yam.py`](../backend/src/ctrl_pi/drivers/mock_yam.py), and
single-process command arbitration is in
[`rig.py`](../backend/src/ctrl_pi/rig.py).

## Interface contract

```python
class YAMDriver(ABC):
    def list_arms(self) -> list[ArmTelemetry]: ...
    def get_arm(self, arm_id: str) -> ArmTelemetry: ...
    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry: ...
    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry: ...
```

All four methods are synchronous and return a point-in-time snapshot. They
must not return a reference to mutable internal state.

| Method | Required behavior |
| --- | --- |
| `list_arms()` | Return a snapshot for every configured arm. An empty list is valid. |
| `get_arm(arm_id)` | Return one snapshot or raise `ArmNotFoundError`. |
| `jog(arm_id, command)` | Apply one validated relative manual command atomically and return the resulting snapshot. Raise `JogLimitError` if the target is unsafe. |
| `apply_action(arm_id, action)` | Apply one absolute six-joint-plus-gripper target atomically and return the resulting snapshot. Raise `ActionLimitError` rather than clipping an unsafe target silently. |

Unexpected hardware and transport exceptions must not contain credentials or
device payloads. The API and control loops convert known driver exceptions to
sanitized errors, but the driver remains the final hardware safety boundary.

## IDs, roles, and telemetry

An arm's `id` is a stable hardware-facing identity such as `yam-follower`; it
is not the PostgreSQL robot UUID. Roles are exactly `leader` and `follower`.
Teleoperation reads a leader and writes a distinct follower. Inference writes
one connected follower only.

Every `ArmTelemetry` snapshot contains:

- an aware UTC `timestamp`, connection state, driver name, ID, display name,
  and role;
- exactly the six joint names `shoulder_yaw`, `shoulder_pitch`,
  `elbow_pitch`, `wrist_roll`, `wrist_pitch`, and `wrist_yaw`, with position
  in radians, velocity in radians per second, effort in newton-metres, and
  temperature in Celsius;
- end-effector `x`, `y`, and `z` in metres and roll, pitch, and yaw in radians;
- gripper position normalized to `[0, 1]`, plus velocity, non-negative force
  in newtons, and closed state;
- CAN interface, positive bitrate, non-negative transmit/receive error counts,
  and state `active`, `warning`, `bus_off`, or `disconnected`; and
- target/observed control frequency, cycle time, jitter, and dropped cycles.

A real driver must report loss of communication as disconnected or a CAN
fault, not reuse an old snapshot as healthy. Numeric telemetry and command
targets must be finite.

## Commands and safety bounds

`JogCommand` is a single non-zero relative change. Schema validation rejects
unknown axes, non-finite values, and deltas over these API bounds:

| Kind | Axes | Maximum absolute delta |
| --- | --- | --- |
| `joint` | the six named joints | `0.25` radians |
| `cartesian` translation | `x`, `y`, `z` | `0.05` metres |
| `cartesian` rotation | `roll`, `pitch`, `yaw` | `0.15` radians |
| `gripper` | `position` | `0.10` normalized units |

`ArmAction` is an absolute target with exactly the six joint keys and a
gripper position in `[0, 1]`. The robot inference loop adds its own default
per-step bounds of `0.35` radians per joint and `0.20` gripper units, rejects
stale or malformed chunks, and applies actions at a target of 50 Hz. Those
software checks do not replace firmware limits, torque/current limits,
workspace constraints, an emergency stop, or collision avoidance. A real
driver must independently validate hardware-specific ranges and reject the
whole command before writing if any target is unsafe.

The mock provides one leader and one follower, uses mock CAN interfaces at
1 Mbit/s, reports a nominal 200 Hz device loop, constrains joint positions to
`[-pi, pi]`, and constrains its synthetic pose and gripper. Those values are
test behavior, not a specification for real YAM hardware.

## Threading and latency

One driver instance is shared by several execution contexts:

- synchronous HTTP handlers may run concurrently in Starlette worker threads;
- the arm WebSocket reads telemetry every 50 ms;
- teleoperation and recording read and write at their configured frame rate;
- the inference loop reads observations and applies actions at a target of
  50 Hz; and
- shutdown may race with an outstanding command.

Consequently, every method must be thread-safe and bounded in time.
`list_arms()` and `get_arm()` are called directly from asynchronous loops, so
they must return from an in-memory snapshot and must never perform blocking
CAN/USB discovery, reconnects, sleeps, or unbounded retries. `jog()` and
`apply_action()` must serialize writes and provide a bounded
acknowledgement/failure.

The intended real-driver shape is one dedicated hardware I/O thread (or
process) that owns the bus. It refreshes immutable snapshots and consumes a
bounded command handoff. Public methods use short critical sections to copy a
snapshot or submit one command; they do not hold a lock while doing blocking
device I/O. Shutdown rejects new commands, wakes the I/O owner, joins it with a
deadline, marks arms disconnected, and leaves no writer thread behind.

## RigLease arbitration

`RigLease` prevents two ctrl-π control modes from commanding the rig in the
same backend process. Acquisition is non-blocking: a conflict fails
immediately instead of waiting or queueing. A lease token contains the owner
literal (`manual`, `teleop`, or `inference`), a 1–200 character owner ID, and a
random nonce. Only that exact token can release the lease.

| Producer | Lease lifetime | Driver writes |
| --- | --- | --- |
| Manual jog | One `jog()` call | One selected arm |
| Teleoperation | From successful session start through stop, failure, or shutdown | Leader snapshots to the selected follower |
| Inference | Before runtime identity checks through loop/transport shutdown | Selected connected follower |
| Passive inference recording | No separate lease; verifies the active inference owner | None; observes already-applied actions |

Every command-producing service must receive the same application-scoped
`RigLease` instance. Do not construct a private lease inside a driver or a new
service. The lease is deliberately process-local: run one Uvicorn worker and
do not horizontally scale the hardware owner. It is not a CAN arbitration
mechanism, inter-process lock, distributed lock, or substitute for driver and
firmware safety checks.

## Real-driver integration checklist

Before replacing the mock, verify all of the following on the target Ubuntu
host:

1. IDs and roles remain stable across reconnects, and device discovery cannot
   swap leader/follower identities.
2. Telemetry units and all six joint mappings match the typed contract.
3. Reads are snapshot-only and fast under concurrent HTTP, WebSocket,
   teleoperation, and inference access.
4. Writes are serialized, atomic at the interface boundary, bounded by a
   timeout, and fail closed on disconnect, CAN `bus_off`, invalid targets, or
   partial acknowledgement.
5. Hardware range/rate/torque limits and an independent emergency-stop path
   remain effective even if application validation fails.
6. Start/stop/reconnect tests leave zero command writers and a free shared
   `RigLease`.

Container USB and SocketCAN visibility is documented in
[Docker deployment](docker-deployment.md). Recording ownership and recovery
are documented in [Recording and teleoperation](recording.md).
