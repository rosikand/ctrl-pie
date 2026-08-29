---
title: "YAM driver interface"
description: "Understand the multi-arm cell boundary, supervised i2rt workers, routing, and failure safety."
icon: "cpu"
---

`YAMDriver` is ctrl-π's synchronous, YAM-specific boundary around arm
hardware. HTTP handlers, WebSockets, recording, teleop, and inference use
stable logical arm IDs. They do not import i2rt, inspect sysfs, open CAN, or
hold vendor objects.

`MockYAMDriver` is the deterministic default. Hardware mode uses the
cell-aware driver for V1.2 configurations and retains `RealYAMDriver` as the
legacy serial-GELLO adapter. Hardware failures never fall back to mocks.

## Interface contract

The V1.1 read/action methods remain, with additive cell lifecycle methods:

```python
class YAMDriver(ABC):
    def startup(self) -> None: ...
    def shutdown(self) -> None: ...
    def diagnostic(self) -> YAMDriverDiagnostic: ...
    def setup_config(self) -> YAMConfiguration | None: ...
    def discover_setup(self) -> YAMDiscoveryResult: ...
    def preflight_setup(self, config: YAMConfiguration) -> YAMPreflightResult: ...
    def apply_setup(self, config: YAMConfiguration) -> YAMDriverDiagnostic: ...
    def connect_arms(self, arm_ids: list[str] | None = None) -> None: ...
    def disconnect_arms(self, arm_ids: list[str] | None = None) -> None: ...
    def check_teaching_handle_range(
        self, arm_id: str, duration_seconds: float = 10.0
    ) -> YAMHandleRangeResult: ...
    def list_arms(self) -> list[ArmTelemetry]: ...
    def get_arm(self, arm_id: str) -> ArmTelemetry: ...
    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry: ...
    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry: ...
    def prepare_teleop_action(self, leader_id: str, follower_id: str) -> ArmAction: ...
    def safe_idle(self, arm_ids: list[str] | None = None) -> None: ...
```

All reads return copied cached telemetry. Commands are serialized and accept
followers only. Known failures become bounded operator diagnostics; raw CAN
frames, vendor exceptions, local paths, and motor commands are not public.

## Stable identity and discovery

Configured SocketCAN arms store adapter serial, not `canN`. Bounded sysfs
discovery maps serials to current interfaces and rejects:

- missing or duplicate configured identities;
- one identity resolving to more than one interface;
- two configured arms resolving to one interface;
- unsupported role/transport/end-effector combinations; and
- invalid pair/group/side or duplicate pair-port topology.

Discovery and preflight never create a worker or vendor object. They do not
open hardware, ping/enable motors, calibrate, or mutate links. The active
teaching-handle range test is deliberately separate.

## Supervised direct-i2rt adapter

The all-CAN path uses direct Python i2rt calls behind one child process per
connected arm. It does not supervise Lux shell scripts and does not bind the
reference pair RPC ports. Pair ports remain validated routing metadata, so
pair identities cannot silently collide if an adapter requires them later.

Before importing i2rt, every worker proves:

1. the configured checkout root is absolute and its filesystem reports the
   kernel `ST_RDONLY` mount flag;
2. expected commit is a full exact Git object;
3. the checkout is at that commit and has no modified, staged, or untracked
   source; and
4. the loaded `i2rt` modules resolve below that verified root.

Only `ST_RDONLY` is accepted as positive read-only proof. Host `chmod` modes,
ownership, and an effective-permission check such as `os.access` are not mount
immutability and cannot satisfy this gate. Docker's `/opt/i2rt:ro` bind mount
provides the required kernel-visible flag inside the container.

The all-CAN Docker image additionally records
`CTRL_PI_I2RT_DEPENDENCY_COMMIT`, the exact local checkout used to resolve and
bake dependencies. Preflight requires the image marker, cell commit, and
mounted checkout to agree. There is no public-upstream/latest fallback.

The child owns its CAN object and all nested i2rt threads. Parent/child IPC is
bounded and typed:

- exact logical/physical/configuration ready handshake;
- heartbeats and fresh cached telemetry;
- observed named loop-rate/liveness diagnostics;
- latest-wins, timestamped seven-position action mailbox;
- explicit command-enable/revoke, safe-idle, and shutdown acknowledgements.

No raw CAN/motor action is exposed through REST, SDK, or a network pair
server. ctrl-π applies declared frame mapping and soft limits at the driver
boundary before a follower command.

### Command cadence

The supervised command path is a freshness contract, not a loop-frequency
health threshold. While teleop or inference is actively writing, the parent
must refresh the latest target at least every 125 ms. If it does not, the
parent safe-idles the arm before the child's fault watchdog. Independently,
the child refuses an action older than 250 ms. A queued or previously accepted
target never authorizes motion after either boundary expires.

Policy execution may resume only through a fresh, identity-checked command
boundary. Teleop may resume only through the explicit synchronization boundary
with current leader state; stale tracking is not replayed. One-shot jog remains
disabled for this all-CAN adapter. These 125/250 ms safety deadlines must not be
confused with, or derived from, the field-observed control-loop rates.

## Worker lifecycle and faults

Connect re-resolves the stable identity immediately before starting a worker
and acquires one-writer bus ownership. A multi-arm request starts followers
before leaders and rolls back only workers created by that failed request.

Runtime faults never auto-respawn. A crash, stale heartbeat/telemetry, stopped
inner loop, wrong identity, invalid protocol message, or action failure
revokes writes and latches the affected arm/pair unavailable.

Disconnect/shutdown proceeds in this order:

1. reject/revoke new commands;
2. request cooperative safe-idle and device close;
3. wait a bounded interval;
4. terminate, then kill, a non-cooperative child; and
5. reap it.

If safe device shutdown or process death cannot be proven, the bus remains
blocked/error. This is intentionally fail-closed. An operator must make the
rig safe and investigate; restarting is not proof that motor torque cleared.

## Telemetry

Each snapshot exposes stable identity metadata and current runtime state:

- role, pair/group/side, transport, end effector;
- current resolved CAN interface as nullable telemetry;
- connected/control state plus energized/holding flags;
- six joint values and normalized gripper/handle state;
- CAN/link and warnings, including `NO SASH GUARD`; and
- observed control-loop source/frequency/cycle/jitter/drop diagnostics.

The earlier physical i2rt/Lux session observed follower/gravity-compensation
loops around 263–275 Hz and a leader around 419–426 Hz at
`bilateral_kp=0.0`. ctrl-π displays rates for inspection but does not encode
those measurements as universal pass/fail thresholds.

## Frame and gripper normalization

Teleop uses a declared leader/follower pair. The follower target is:

```text
mapped_joint = sign * leader_joint + offset
bounded_joint = clamp(mapped_joint, optional follower soft limits)
```

Exactly six arm joints are mapped/clamped. The CAN teaching-handle trigger is
normalized to follower gripper command as `1 - trigger`; public ctrl-π
gripper semantics remain `0=closed`, `1=open`. Recorded actions contain the
post-map/post-limit command actually routed to the follower.

An absent frame-map path is valid identity mapping. An absent soft-limit path
is not a default limit and produces `NO SASH GUARD`.

## Pair-safe teleop

Teleop start validates the exact declared pair, requires both arms connected,
and acquires leases for those two arm IDs. It reads telemetry only. Sync starts
disabled and there is no follower write—not even one initial pose copy.

A separate acknowledged sync action interpolates over approximately three
seconds toward the latest mapped/clamped leader state, then enables live
latest-state tracking. Disable cancels any correction and stops writes while
observation continues. Stop safe-idles before releasing the pair lease.

Two disjoint pairs can operate independently. Cross-pair leader/follower
selection is rejected before leasing or writing. True bimanual recording and
multi-follower policy action shapes remain deferred.

## All-CAN inference and jog compatibility

Inference selects one logical follower ID, never a runtime bus, and holds a
resource-scoped lease for that follower. Two workflows may use disjoint
followers, while a second owner of the same follower fails immediately rather
than queueing.

The supervised all-CAN adapter explicitly rejects one-shot `jog()` in V1.2.
Its i2rt command path is exercised only by declared-pair synchronization and
follower inference after their stronger lifecycle gates. The public jog method
remains for deterministic mock mode and the retained legacy adapter, preserving
V1.1 compatibility without presenting jog as validated all-CAN behavior.

Policy actions are absolute six-joint-plus-normalized-gripper targets. The
driver rejects leader targets, non-finite/wrong-sized actions, stale sessions,
and unavailable/holding-state mismatches before vendor mutation. Final
follower soft limits remain the last software guard before the worker action.

## Teaching-handle check

The range test is an explicitly acknowledged active CAN diagnostic separate
from passive preflight and motor connection. It reads trigger/button encoder
state for a bounded 1–15 second window and classifies reachability/range.
Connectivity and health are distinct.

It never exposes encoder zeroing. If a handle is stuck, stop and use the
reviewed external i2rt encoder-manager maintenance workflow with separate
operator authority.

## Rig ownership

`RigLease` owns sets of logical arm resources. Teleop owns its declared
leader/follower pair; inference owns one follower. Mock/legacy jog owns one
follower. Disjoint sets can coexist and overlapping sets conflict. A legacy
wildcard lease remains for older single-rig workflows.

This protects one ctrl-π process. The per-worker bus lock and stable identity
checks protect local CAN ownership. Neither is a distributed lock; run exactly
one Uvicorn worker and do not run a second external controller on the same bus.

## Physical validation boundary

Fake-worker tests can prove protocol validation, rollback, routing, bounded
teardown, and zero-write teleop start. They cannot establish physical safety.
Complete the exact [H0–H7 field acceptance](/yam-field-test), including
least-privileged container access, directions/offsets/units, calibration,
loop behavior, first slow sync, disconnect/limp behavior, E-stop, bus faults,
recording semantics, and small controlled inference before a hardware claim.

## Legacy V1.1 compatibility contract

V1.2 retains `RealYAMDriver` for one serial GELLO leader and one SocketCAN
`crank_4310` follower. This adapter remains thread-safe and process-local,
keeps its legacy 50 Hz ctrl-π sampling cache, and preserves `jog()` plus
`apply_action()` behavior. Those facts do not apply one-shot jog support to
the supervised all-CAN adapter.

The retained non-opening `make yam-probe` and active
`python -m ctrl_pi.yam_probe --connect` (`yam_probe --connect`) commands apply
only to that legacy topology. Create its leader artifact through the reviewed
[external calibration boundary](setup.md#creating-the-leader-calibration-artifact),
not through all-CAN cell onboarding.

Legacy regression acceptance still verifies all of these behaviors:

1. Settings discovery and preflight remain non-opening.
2. automatic connection remains off by default.
3. With explicit consent, boot once with the rig
    unplugged, hot-plug it, and observe one bounded restoration attempt
    without background retries.
4. An operator can revoke automatic connection without
    forgetting the setup.
5. Switching through mock mode proves the physical PostgreSQL row was neither overwritten nor deleted.
6. A serial-device remap fails closed rather than selecting another leader.

These legacy checks have not been performed on a physical
Ubuntu/YAM box in this development environment. They remain compatibility
requirements alongside, not substitutes for, the V1.2 all-CAN H0–H7 gates.
