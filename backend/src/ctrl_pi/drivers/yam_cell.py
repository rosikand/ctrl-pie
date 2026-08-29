"""Composite, fail-closed hardware driver for a configurable YAM cell.

The parent process never imports or constructs i2rt robot objects.  Passive
setup work is limited to bounded sysfs/config inspection and verification of
an operator-provided, exact, read-only checkout.  Each explicitly connected
SocketCAN arm is owned by one supervised child worker.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Protocol

from ctrl_pi.drivers.yam import (
    JOINT_NAMES,
    ActionLimitError,
    ArmAction,
    ArmNotFoundError,
    ArmTelemetry,
    CANTelemetry,
    ControlLoopTelemetry,
    EndEffectorPose,
    GripperTelemetry,
    JogCommand,
    JogLimitError,
    JointTelemetry,
    LegacyYAMSetupConfig,
    TeachingHandleTelemetry,
    YAMArmPreflight,
    YAMArmResolution,
    YAMCellArmConfig,
    YAMCellConfig,
    YAMDiscoveryDevice,
    YAMDiscoveryResult,
    YAMDriver,
    YAMDriverDiagnostic,
    YAMDriverUnavailableError,
    YAMHandleRangeResult,
    YAMPreflightResult,
)
from ctrl_pi.drivers.yam_cell_discovery import (
    ConfiguredSocketCANArm,
    ResolvedSocketCANArm,
    SocketCANAdapter,
    SocketCANDiscoveryIssue,
    SocketCANInventory,
    discover_socketcan_adapters,
    resolve_configured_socketcan_arms,
)
from ctrl_pi.drivers.yam_cell_safety import (
    NO_SASH_GUARD,
    YAMFollowerSafety,
    YAMSafetyConfigError,
    load_yam_follower_safety,
)
from ctrl_pi.drivers.yam_handle_diagnostic import (
    HandleDiagnosticConfig,
    HandleDiagnosticSample,
    HandleRangeReader,
    SupervisedHandleRangeReader,
    YAMHandleDiagnosticError,
    YAMHandleDiagnosticTeardownUncertainError,
)
from ctrl_pi.drivers.yam_i2rt_worker import (
    I2RTArmWorker,
    I2RTArmWorkerConfig,
    I2RTCheckoutIdentity,
    I2RTWorkerError,
    I2RTWorkerShutdownResult,
    I2RTWorkerSnapshot,
    WorkerTelemetry,
    verify_i2rt_checkout,
)


_LOCK_ROOT = Path("/tmp/ctrl-pi-yam-bus-locks")
_POSE_WARNING = "End-effector pose is unavailable from the i2rt arm worker."
_FAULT_DETAIL = (
    "A YAM arm worker fault is latched; explicitly disconnect and review the arm."
)
_PAIR_FAULT_WARNING = "Pair motion is latched off after an arm worker fault."
_HANDLE_DIAGNOSTIC_FAULT_DETAIL = (
    "A teaching-handle diagnostic child could not be reaped; its CAN bus remains "
    "owned and blocked until ctrl-pi restarts."
)
_RETRYABLE_PREREQUISITE_ISSUES = frozenset(
    {"configured_identity_missing", "runtime_link_down", "runtime_link_unknown"}
)
_DEPENDENCY_MARKER_WARNING = (
    "The optional ctrl-pi i2rt dependency commit marker is absent; the mounted "
    "checkout identity was still verified directly."
)


class ArmWorker(Protocol):
    config: I2RTArmWorkerConfig

    def start(self) -> None: ...

    def snapshot(self) -> I2RTWorkerSnapshot: ...

    def telemetry(self) -> WorkerTelemetry: ...

    def enable_commands(self) -> None: ...

    def send_action(self, positions: Sequence[float]) -> int: ...

    def safe_idle(self) -> None: ...

    def shutdown(self) -> I2RTWorkerShutdownResult: ...


class BusLease(Protocol):
    def release(self) -> None: ...


class BusLockManager(Protocol):
    def acquire(self, *, stable_identity: str, runtime_interface: str) -> BusLease: ...


class _FileBusLease:
    def __init__(self, handles: list[int]) -> None:
        self._handles = handles
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for handle in reversed(self._handles):
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                os.close(handle)


class FileBusLockManager:
    """Cooperative process lock for both durable identity and runtime bus.

    This prevents two ctrl-pi processes using the same configured arm or
    current interface.  It deliberately does not claim to detect unrelated
    vendor processes that ignore this advisory lock.
    """

    def __init__(self, root: Path = _LOCK_ROOT) -> None:
        self._root = root

    def acquire(self, *, stable_identity: str, runtime_interface: str) -> BusLease:
        try:
            self._root.mkdir(mode=0o700, parents=False, exist_ok=True)
            if self._root.is_symlink() or not self._root.is_dir():
                raise OSError
        except OSError as error:
            raise YAMDriverUnavailableError(
                "The local YAM bus ownership lock directory is unavailable."
            ) from error

        keys = sorted((f"identity:{stable_identity}", f"bus:{runtime_interface}"))
        handles: list[int] = []
        try:
            for key in keys:
                name = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".lock"
                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                handle = os.open(self._root / name, flags, 0o600)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(handle)
                    raise
                handles.append(handle)
        except (OSError, BlockingIOError) as error:
            _FileBusLease(handles).release()
            raise YAMDriverUnavailableError(
                "The selected SocketCAN adapter already has a ctrl-pi owner."
            ) from error
        return _FileBusLease(handles)


@dataclass(slots=True)
class _ArmRuntime:
    arm: YAMCellArmConfig
    resolution: ResolvedSocketCANArm
    worker: ArmWorker
    bus_lease: BusLease
    last_telemetry: WorkerTelemetry | None = None
    blocked: bool = False
    command_generation: int = 0
    idle_timer: threading.Timer | None = None


@dataclass(frozen=True, slots=True)
class _CellInspection:
    result: YAMPreflightResult
    inventory: SocketCANInventory
    resolutions: dict[str, ResolvedSocketCANArm]
    safety: dict[str, YAMFollowerSafety]


InventoryProvider = Callable[[], SocketCANInventory]
CheckoutVerifier = Callable[[str, str], I2RTCheckoutIdentity]
WorkerFactory = Callable[[I2RTArmWorkerConfig], ArmWorker]


def _default_checkout_verifier(root: str, commit: str) -> I2RTCheckoutIdentity:
    return verify_i2rt_checkout(root, commit, require_read_only=True)


class YAMCellDriver(YAMDriver):
    """General cell parent plus the retained V1.1 legacy-pair delegate."""

    def __init__(
        self,
        legacy_driver: YAMDriver,
        *,
        inventory_provider: InventoryProvider = discover_socketcan_adapters,
        checkout_verifier: CheckoutVerifier = _default_checkout_verifier,
        worker_factory: WorkerFactory = I2RTArmWorker,
        bus_locks: BusLockManager | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        dependency_commit_marker: str | None = None,
        handle_range_reader: HandleRangeReader | None = None,
    ) -> None:
        self._legacy = legacy_driver
        self._inventory_provider = inventory_provider
        self._checkout_verifier = checkout_verifier
        self._worker_factory = worker_factory
        self._bus_locks = bus_locks or FileBusLockManager()
        self._monotonic = monotonic
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._dependency_commit_marker = (
            os.environ.get("CTRL_PI_I2RT_DEPENDENCY_COMMIT")
            if dependency_commit_marker is None
            else dependency_commit_marker
        )
        self._handle_range_reader = handle_range_reader or SupervisedHandleRangeReader()
        self._lock = threading.RLock()
        self._cell: YAMCellConfig | None = None
        self._mode = (
            "legacy"
            if isinstance(legacy_driver.setup_config(), LegacyYAMSetupConfig)
            else "empty"
        )
        self._runtimes: dict[str, _ArmRuntime] = {}
        self._faults: dict[str, str] = {}
        self._pair_faults: set[str] = set()
        self._safety: dict[str, YAMFollowerSafety] = {}
        self._resolutions: dict[str, ResolvedSocketCANArm] = {}
        self._handle_results: dict[str, HandleDiagnosticSample] = {}
        self._handle_bus_leases: dict[str, BusLease] = {}
        self._handle_teardown_uncertain: set[str] = set()
        self._passive_diagnostic: YAMDriverDiagnostic | None = None
        self._diagnostic = (
            legacy_driver.diagnostic()
            if self._mode == "legacy"
            else YAMDriverDiagnostic(
                status="missing", detail="No physical YAM cell is configured."
            )
        )

    def setup_config(self) -> LegacyYAMSetupConfig | YAMCellConfig | None:
        with self._lock:
            if self._mode == "cell":
                return self._cell
            if self._mode == "legacy":
                return self._legacy.setup_config()
            return None

    def discover_setup(self) -> YAMDiscoveryResult:
        with self._lock:
            legacy_discovery = (
                self._legacy.discover_setup() if self._mode == "legacy" else None
            )
            config = self._cell
        inventory = self._safe_inventory()
        configured = [] if config is None else self._socket_requests(config)
        resolution = resolve_configured_socketcan_arms(
            configured, inventory, require_link_up=True
        )
        by_id = resolution.by_logical_id
        identity_counts = Counter(
            adapter.stable_identity
            for adapter in inventory.adapters
            if adapter.stable_identity is not None
        )
        devices = [
            YAMDiscoveryDevice(
                transport_kind="socketcan",
                stable_identity=adapter.stable_identity or f"unidentified:{adapter.interface}",
                product=adapter.product,
                runtime_interface=adapter.interface,
                link_state=self._link_state(adapter),
                duplicate_identity=(
                    adapter.stable_identity is not None
                    and identity_counts[adapter.stable_identity] > 1
                ),
            )
            for adapter in inventory.adapters
        ]
        arm_resolutions: list[YAMArmResolution] = []
        if config is not None:
            for arm in config.arms:
                resolved = by_id.get(arm.logical_id)
                if arm.transport_kind == "serial":
                    arm_resolutions.append(
                        YAMArmResolution(
                            arm_id=arm.logical_id,
                            transport_kind="serial",
                            stable_identity=arm.stable_identity,
                            runtime_interface=arm.stable_identity,
                            resolved=False,
                            detail=(
                                "Serial GELLO arms are available through the retained "
                                "legacy_pair setup adapter."
                            ),
                        )
                    )
                    continue
                arm_issues = self._issues_for_arm(
                    arm.logical_id, arm.stable_identity, resolution.issues
                )
                conflict = any(
                    issue.code
                    in {
                        "duplicate_discovered_identity",
                        "configured_identity_ambiguous",
                        "runtime_interface_collision",
                    }
                    for issue in arm_issues
                )
                arm_resolutions.append(
                    YAMArmResolution(
                        arm_id=arm.logical_id,
                        transport_kind="socketcan",
                        stable_identity=arm.stable_identity,
                        runtime_interface=(
                            None if resolved is None else resolved.runtime_interface
                        ),
                        resolved=resolved is not None,
                        conflict=conflict,
                        detail=(
                            "Stable adapter identity resolved passively."
                            if resolved is not None and not arm_issues
                            else "Stable adapter identity is not ready for connection."
                        ),
                    )
                )
        return YAMDiscoveryResult(
            mode="hardware",
            devices=devices,
            resolutions=arm_resolutions,
            can_interfaces=(
                [] if legacy_discovery is None else legacy_discovery.can_interfaces
            ),
            leader_ports=(
                [] if legacy_discovery is None else legacy_discovery.leader_ports
            ),
            suggested_config=(
                config
                if legacy_discovery is None
                else legacy_discovery.suggested_config
            ),
            detail=(
                "SocketCAN adapters were inventoried passively; no device was opened."
                + (
                    " Retained legacy setup candidates remain available during migration."
                    if legacy_discovery is not None
                    else ""
                )
                if not inventory.errors
                else "Passive SocketCAN inventory contains blocking identity errors."
            ),
        )

    def preflight_setup(
        self, config: LegacyYAMSetupConfig | YAMCellConfig
    ) -> YAMPreflightResult:
        if isinstance(config, LegacyYAMSetupConfig):
            return self._legacy.preflight_setup(config)
        return self._inspect_cell(config).result

    def apply_setup(
        self, config: LegacyYAMSetupConfig | YAMCellConfig
    ) -> YAMDriverDiagnostic:
        with self._lock:
            if self._mode == "cell" and self._runtimes:
                raise YAMDriverUnavailableError(
                    "Disconnect all current YAM cell arms before changing its setup."
                )
            if self._handle_bus_leases:
                raise YAMDriverUnavailableError(
                    "A teaching-handle diagnostic still owns a YAM bus; restart "
                    "ctrl-pi before changing the setup."
                )
            if isinstance(config, LegacyYAMSetupConfig):
                self._require_cell_stopped_locked()
                self._cell = None
                self._safety = {}
                self._resolutions = {}
                self._handle_results = {}
                self._faults = {}
                self._pair_faults = set()
                self._passive_diagnostic = None
                self._mode = "legacy"
                self._diagnostic = self._legacy.apply_setup(config)
                return self._diagnostic

            if self._mode == "cell" and self._cell == config:
                return self.diagnostic()
            self._require_cell_stopped_locked()
            self._legacy.shutdown()
            inspection = self._inspect_cell(config)
            self._cell = config
            self._mode = "cell"
            self._safety = inspection.safety
            self._resolutions = inspection.resolutions
            self._handle_results = {}
            self._faults = {}
            self._pair_faults = set()
            self._diagnostic = inspection.result.diagnostic
            self._passive_diagnostic = inspection.result.diagnostic
            return self._diagnostic

    def reset_setup(self) -> YAMDriverDiagnostic:
        with self._lock:
            self._require_cell_stopped_locked()
            self._cell = None
            self._safety = {}
            self._resolutions = {}
            self._handle_results = {}
            self._faults = {}
            self._pair_faults = set()
            self._passive_diagnostic = None
            legacy = self._legacy.reset_setup()
            self._mode = (
                "legacy"
                if isinstance(self._legacy.setup_config(), LegacyYAMSetupConfig)
                else "empty"
            )
            self._diagnostic = (
                legacy
                if self._mode == "legacy"
                else YAMDriverDiagnostic(
                    status="missing", detail="No physical YAM cell is configured."
                )
            )
            return self._diagnostic

    def startup(self) -> None:
        with self._lock:
            mode = self._mode
        if mode == "legacy":
            self._legacy.startup()
        elif mode == "cell":
            self.connect_arms()

    def shutdown(self) -> None:
        with self._lock:
            mode = self._mode
        if mode == "legacy":
            self._legacy.shutdown()
        elif mode == "cell":
            self.disconnect_arms()

    def connect_arms(self, arm_ids: list[str] | None = None) -> list[ArmTelemetry]:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.connect_arms(arm_ids)
            config = self._require_cell_locked()
            selected = self._select_arms_locked(config, arm_ids)
            if selected & self._handle_bus_leases.keys():
                raise YAMDriverUnavailableError(
                    "A selected teaching-handle diagnostic still owns its CAN bus."
                )
            if any(arm_id in self._faults for arm_id in selected):
                raise YAMDriverUnavailableError(
                    "A selected arm has a latched worker fault; disconnect it first."
                )
            ordered = sorted(
                (self._arm_config_locked(arm_id) for arm_id in selected),
                key=lambda arm: (0 if arm.role == "follower" else 1, arm.logical_id),
            )
            started: list[str] = []
            try:
                for arm in ordered:
                    if arm.logical_id in self._runtimes:
                        runtime = self._runtimes[arm.logical_id]
                        if runtime.blocked:
                            raise YAMDriverUnavailableError(
                                "A selected YAM bus remains blocked after uncertain teardown."
                            )
                        self._refresh_runtime_locked(runtime)
                        continue
                    self._connect_one_locked(config, arm)
                    started.append(arm.logical_id)
            except BaseException as error:
                rollback_uncertain: list[str] = []
                for arm_id in reversed(started):
                    try:
                        self._disconnect_one_locked(arm_id)
                    except YAMDriverUnavailableError:
                        rollback_uncertain.append(arm_id)
                if rollback_uncertain:
                    raise YAMDriverUnavailableError(
                        "YAM cell connection failed and rollback is uncertain for: "
                        + ", ".join(sorted(rollback_uncertain))
                    ) from error
                if isinstance(error, (ArmNotFoundError, YAMDriverUnavailableError)):
                    raise
                raise YAMDriverUnavailableError(
                    "A selected YAM arm worker could not be started safely."
                ) from error
            self._update_diagnostic_locked()
            return self.list_arms()

    def disconnect_arms(self, arm_ids: list[str] | None = None) -> list[ArmTelemetry]:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.disconnect_arms(arm_ids)
            config = self._require_cell_locked()
            selected = self._select_arms_locked(config, arm_ids)
            uncertain: list[str] = []
            for arm_id in sorted(selected):
                if arm_id not in self._runtimes:
                    if arm_id in self._handle_teardown_uncertain:
                        uncertain.append(arm_id)
                    else:
                        self._faults.pop(arm_id, None)
                    continue
                try:
                    self._disconnect_one_locked(arm_id)
                except YAMDriverUnavailableError:
                    uncertain.append(arm_id)
            self._rebuild_pair_faults_locked()
            self._update_diagnostic_locked()
            if uncertain:
                raise YAMDriverUnavailableError(
                    "YAM teardown is uncertain; these buses remain blocked: "
                    + ", ".join(uncertain)
                )
            return self.list_arms()

    def diagnostic(self) -> YAMDriverDiagnostic:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.diagnostic()
            self._update_diagnostic_locked()
            return self._diagnostic.model_copy(deep=True)

    def list_arms(self) -> list[ArmTelemetry]:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.list_arms()
            if self._mode != "cell" or self._cell is None:
                return []
            return [self._arm_snapshot_locked(arm) for arm in self._cell.arms]

    def get_arm(self, arm_id: str) -> ArmTelemetry:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.get_arm(arm_id)
            arm = self._arm_config_locked(arm_id)
            return self._arm_snapshot_locked(arm)

    def validate_teleop_pair(self, leader_id: str, follower_id: str) -> None:
        with self._lock:
            if self._mode == "legacy":
                self._legacy.validate_teleop_pair(leader_id, follower_id)
                return
            leader = self._arm_config_locked(leader_id)
            follower = self._arm_config_locked(follower_id)
            if leader.role != "leader" or follower.role != "follower":
                raise ValueError("teleoperation requires one leader and one follower")
            if (
                leader.pair_id is None
                or follower.pair_id is None
                or leader.pair_id != follower.pair_id
            ):
                raise ValueError("leader and follower must be the exact declared pair")
            if leader.pair_id in self._pair_faults:
                raise YAMDriverUnavailableError(
                    "The selected teleoperation pair has a latched arm fault."
                )

    def prepare_teleop_action(
        self, leader_id: str, follower_id: str, action: ArmAction
    ) -> ArmAction:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.prepare_teleop_action(
                    leader_id, follower_id, action
                )
            self.validate_teleop_pair(leader_id, follower_id)
            follower = self._arm_config_locked(follower_id)
            safety = self._safety.get(follower.logical_id)
            if safety is None:
                raise YAMDriverUnavailableError(
                    "Follower safety configuration is unavailable."
                )
            source = [
                action.joint_positions_radians[name] for name in JOINT_NAMES
            ] + [action.gripper_position]
            mapped = safety.apply(source)
            return ArmAction(
                timestamp=action.timestamp,
                joint_positions_radians=dict(zip(JOINT_NAMES, mapped[:6], strict=True)),
                gripper_position=mapped[6],
            )

    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.apply_action(arm_id, action)
            arm = self._arm_config_locked(arm_id)
            if arm.role != "follower":
                raise ActionLimitError("Position actions are follower-only.")
            runtime = self._commandable_runtime_locked(arm)
            safety = self._safety.get(arm_id)
            if safety is None:
                raise YAMDriverUnavailableError(
                    "Follower safety configuration is unavailable."
                )
            # Inference and jog targets are already in follower coordinates;
            # only final follower soft limits apply here.  Pair mapping occurs
            # exactly once in prepare_teleop_action.
            source = [
                action.joint_positions_radians[name] for name in JOINT_NAMES
            ] + [action.gripper_position]
            limited = safety.soft_limits.clamp(source)
            try:
                snapshot = runtime.worker.snapshot()
                if not snapshot.accepting_commands:
                    runtime.worker.enable_commands()
                runtime.worker.send_action(limited)
                self._arm_command_idle_timer_locked(runtime)
                self._refresh_runtime_locked(runtime)
            except (I2RTWorkerError, OSError, RuntimeError) as error:
                self._latch_fault_locked(arm_id)
                raise YAMDriverUnavailableError(_FAULT_DETAIL) from error
            return self._arm_snapshot_locked(arm)

    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.jog(arm_id, command)
            arm = self._arm_config_locked(arm_id)
            if arm.role != "follower":
                raise JogLimitError("Manual commands are follower-only.")
            del command
            raise JogLimitError(
                "One-shot manual jog is disabled for i2rt cell arms because position "
                "control requires a continuous fresh command stream."
            )

    def safe_idle(self, arm_id: str) -> ArmTelemetry:
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.safe_idle(arm_id)
            arm = self._arm_config_locked(arm_id)
            runtime = self._runtimes.get(arm_id)
            if arm_id in self._handle_bus_leases:
                raise YAMDriverUnavailableError(
                    "A previous teaching-handle diagnostic still owns this CAN bus."
                )
            if runtime is None:
                return self._arm_snapshot_locked(arm)
            try:
                self._cancel_command_idle_timer_locked(runtime)
                runtime.worker.safe_idle()
            except (I2RTWorkerError, OSError, RuntimeError) as error:
                self._latch_fault_locked(arm_id)
                raise YAMDriverUnavailableError(
                    "The YAM arm could not confirm safe idle."
                ) from error
            return self._arm_snapshot_locked(arm)

    def latch_fault(self, arm_ids: list[str], detail: str) -> None:
        """Revoke selected writers and latch their exact pair routes.

        The caller's detail is intentionally not surfaced: it may contain a
        provider/vendor payload.  A confirmed cooperative safe-idle leaves the
        worker live but uncommandable for explicit disconnect; an unconfirmed
        idle is immediately taken through bounded worker shutdown and its bus
        stays blocked if teardown is uncertain.
        """

        del detail
        with self._lock:
            if self._mode == "legacy":
                legacy_latch = getattr(self._legacy, "latch_fault", None)
                if callable(legacy_latch):
                    legacy_latch(arm_ids, _FAULT_DETAIL)
                else:
                    self._legacy.shutdown()
                return
            config = self._require_cell_locked()
            selected = self._select_arms_locked(config, arm_ids)
            failed: list[str] = []
            for arm_id in sorted(selected):
                self._latch_fault_locked(arm_id, attempt_pair_idle=False)
                runtime = self._runtimes.get(arm_id)
                if runtime is None:
                    continue
                try:
                    runtime.worker.safe_idle()
                except BaseException:
                    result = runtime.worker.shutdown()
                    if result.clean:
                        runtime.bus_lease.release()
                        self._runtimes.pop(arm_id, None)
                    else:
                        runtime.blocked = True
                        failed.append(arm_id)
            for arm_id in sorted(selected):
                arm = self._arm_config_locked(arm_id)
                if arm.pair_id is not None:
                    self._idle_pair_counterparts_locked(arm.pair_id, exclude=arm_id)
            self._update_diagnostic_locked()
            if failed:
                raise YAMDriverUnavailableError(
                    "Fault shutdown is uncertain; these buses remain blocked: "
                    + ", ".join(failed)
                )

    def check_handle_range(
        self, arm_id: str, *, duration_seconds: float = 10.0
    ) -> YAMHandleRangeResult:
        if not math.isfinite(duration_seconds) or not 0.1 <= duration_seconds <= 30.0:
            raise ValueError("duration_seconds must be between 0.1 and 30")
        with self._lock:
            if self._mode == "legacy":
                return self._legacy.check_handle_range(
                    arm_id, duration_seconds=duration_seconds
                )
            arm = self._arm_config_locked(arm_id)
            if arm.role != "leader" or arm.end_effector_kind != "yam_teaching_handle":
                raise YAMDriverUnavailableError(
                    "Handle range checks require a CAN teaching-handle leader."
                )
            runtime = self._runtimes.get(arm_id)
            if runtime is None:
                inventory = self._safe_inventory()
                resolution = resolve_configured_socketcan_arms(
                    [ConfiguredSocketCANArm(arm.logical_id, arm.stable_identity)],
                    inventory,
                    require_link_up=True,
                )
                current = resolution.by_logical_id.get(arm.logical_id)
                errors = [
                    issue for issue in resolution.issues if issue.severity == "error"
                ]
                if current is None or current.link_up is not True or errors:
                    raise YAMDriverUnavailableError(
                        "The teaching-handle adapter is not uniquely link-UP."
                    )
                for existing in self._runtimes.values():
                    if existing.resolution.runtime_interface == current.runtime_interface:
                        raise YAMDriverUnavailableError(
                            "The teaching-handle CAN bus already has a live arm owner."
                        )
                lease = self._bus_locks.acquire(
                    stable_identity=arm.stable_identity,
                    runtime_interface=current.runtime_interface,
                )
                self._handle_bus_leases[arm_id] = lease
            else:
                raise YAMDriverUnavailableError(
                    "Disconnect the teaching-handle arm before its isolated range check."
                )

        retain_lease = False
        try:
            sample = self._handle_range_reader.sample(
                HandleDiagnosticConfig(
                    logical_id=arm.logical_id,
                    stable_identity=arm.stable_identity,
                    runtime_interface=current.runtime_interface,
                    duration_seconds=duration_seconds,
                )
            )
        except YAMHandleDiagnosticTeardownUncertainError as error:
            retain_lease = True
            with self._lock:
                self._handle_teardown_uncertain.add(arm_id)
                self._faults[arm_id] = _HANDLE_DIAGNOSTIC_FAULT_DETAIL
                self._mark_pair_fault_locked(arm)
                self._update_diagnostic_locked()
            raise YAMDriverUnavailableError(
                "The isolated teaching-handle diagnostic teardown is uncertain; "
                "its CAN bus remains blocked until ctrl-pi restarts."
            ) from error
        except (YAMHandleDiagnosticError, OSError, RuntimeError) as error:
            raise YAMDriverUnavailableError(
                "The isolated teaching-handle range check failed safely."
            ) from error
        finally:
            if not retain_lease:
                with self._lock:
                    owned_lease = self._handle_bus_leases.pop(arm_id, None)
                if owned_lease is not None:
                    owned_lease.release()
        with self._lock:
            self._handle_results[arm_id] = sample
        if not sample.reachable:
            return YAMHandleRangeResult(
                arm_id=arm_id,
                reachable=False,
                healthy=False,
                detail="Teaching-handle encoder telemetry was not reachable.",
            )
        minimum = sample.observed_minimum
        maximum = sample.observed_maximum
        healthy = bool(
            minimum is not None
            and maximum is not None
            and minimum <= 0.15
            and maximum >= 0.85
        )
        return YAMHandleRangeResult(
            arm_id=arm_id,
            reachable=True,
            observed_minimum=minimum,
            observed_maximum=maximum,
            healthy=healthy,
            detail=(
                "Teaching-handle trigger range changed across the active check."
                if healthy
                else "Teaching-handle trigger appears stuck; use the documented CLI maintenance procedure."
            ),
        )

    def _inspect_cell(self, config: YAMCellConfig) -> _CellInspection:
        inventory = self._safe_inventory()
        resolution = resolve_configured_socketcan_arms(
            self._socket_requests(config), inventory, require_link_up=True
        )
        resolved = resolution.by_logical_id
        safety: dict[str, YAMFollowerSafety] = {}
        safety_errors: dict[str, str] = {}
        warnings: list[str] = []

        checkout_ready = False
        try:
            identity = self._checkout_verifier(config.i2rt_root, config.i2rt_commit)
            checkout_ready = (
                identity.commit == config.i2rt_commit and identity.read_only
            )
        except BaseException:
            checkout_ready = False
        marker = self._dependency_commit_marker
        marker_matches = marker is None or marker == config.i2rt_commit
        checkout_ready = checkout_ready and marker_matches
        if marker is None:
            warnings.append(_DEPENDENCY_MARKER_WARNING)

        for arm in config.arms:
            if arm.role != "follower":
                continue
            try:
                loaded = load_yam_follower_safety(
                    frame_map_path=arm.frame_map_path,
                    soft_limits_path=arm.soft_limits_path,
                    expected_stable_identity=arm.stable_identity,
                )
                safety[arm.logical_id] = loaded
                warnings.extend(loaded.warnings)
            except YAMSafetyConfigError:
                safety_errors[arm.logical_id] = (
                    "Follower frame-map or soft-limit configuration is invalid."
                )

        arms: list[YAMArmPreflight] = []
        transiently_missing: set[str] = set()
        for arm in config.arms:
            current = resolved.get(arm.logical_id)
            issue_errors = [
                issue
                for issue in self._issues_for_arm(
                    arm.logical_id, arm.stable_identity, resolution.issues
                )
                if issue.severity == "error"
            ]
            retryable_prerequisite = bool(issue_errors) and all(
                issue.code in _RETRYABLE_PREREQUISITE_ISSUES
                for issue in issue_errors
            )
            if retryable_prerequisite:
                transiently_missing.add(arm.logical_id)
            unsupported_serial = arm.transport_kind == "serial"
            ready = bool(
                not unsupported_serial
                and checkout_ready
                and current is not None
                and current.link_up is True
                and not issue_errors
                and arm.logical_id not in safety_errors
            )
            arm_warnings: list[str] = []
            loaded = safety.get(arm.logical_id)
            if loaded is not None:
                arm_warnings.extend(loaded.warnings)
            if unsupported_serial:
                detail = "Serial GELLO topology must use the legacy_pair adapter."
            elif marker is not None and not marker_matches:
                detail = "The ctrl-pi i2rt dependency marker does not match this cell."
            elif not checkout_ready:
                detail = "The exact read-only i2rt checkout could not be verified."
            elif arm.logical_id in safety_errors:
                detail = safety_errors[arm.logical_id]
            elif retryable_prerequisite:
                detail = (
                    "Stable SocketCAN identity is absent or its link is not yet UP."
                )
            elif current is None or issue_errors:
                detail = "Stable SocketCAN identity is missing, ambiguous, or not link-UP."
            else:
                detail = "Arm identity and passive prerequisites are ready."
            if arm.role == "follower" and loaded is not None:
                frame_status = "active" if loaded.frame_map.active else "identity"
                limits_status = "active" if loaded.soft_limits.active else "missing"
            elif arm.role == "follower":
                frame_status = "error"
                limits_status = "error"
            else:
                frame_status = "not_applicable"
                limits_status = "not_applicable"
            arms.append(
                YAMArmPreflight(
                    arm_id=arm.logical_id,
                    ready=ready,
                    runtime_interface=(
                        arm.stable_identity
                        if unsupported_serial
                        else (None if current is None else current.runtime_interface)
                    ),
                    link_state=(
                        "not_applicable"
                        if unsupported_serial
                        else self._resolved_link_state(current)
                    ),
                    frame_map_status=frame_status,
                    soft_limits_status=limits_status,
                    handle_status=(
                        "not_checked" if arm.role == "leader" else "not_applicable"
                    ),
                    warnings=arm_warnings,
                    diagnostic=YAMDriverDiagnostic(
                        status=(
                            "configured"
                            if ready
                            else (
                                "missing"
                                if retryable_prerequisite
                                else "error"
                            )
                        ),
                        detail=detail,
                    ),
                )
            )

        empty_draft = not arms
        ready = checkout_ready and not empty_draft and all(arm.ready for arm in arms)
        if ready:
            diagnostic = YAMDriverDiagnostic(
                status="configured",
                detail=(
                    "The exact read-only i2rt checkout, stable identities, links, and "
                    "safety artifacts passed passive preflight."
                ),
            )
        elif marker is not None and not marker_matches:
            diagnostic = YAMDriverDiagnostic(
                status="error",
                detail=(
                    "The ctrl-pi i2rt dependency commit marker does not match the "
                    "configured operator checkout."
                ),
            )
        elif not checkout_ready:
            diagnostic = YAMDriverDiagnostic(
                status="error",
                detail="The exact read-only operator i2rt checkout could not be verified.",
            )
        elif empty_draft:
            diagnostic = YAMDriverDiagnostic(
                status="configured",
                detail=(
                    "The exact read-only i2rt checkout is verified; add at least one "
                    "arm before connecting this topology draft."
                ),
            )
        elif transiently_missing and all(
            arm.ready or arm.arm_id in transiently_missing for arm in arms
        ):
            diagnostic = YAMDriverDiagnostic(
                status="missing",
                detail=(
                    "One or more configured SocketCAN adapters are absent or not link-UP; "
                    "passive discovery will recheck them."
                ),
            )
        else:
            diagnostic = YAMDriverDiagnostic(
                status="error",
                detail="One or more configured YAM cell arms failed passive preflight.",
            )
        return _CellInspection(
            result=YAMPreflightResult(
                ready=ready,
                calibration_ready=ready,
                diagnostic=diagnostic,
                i2rt_ready=checkout_ready,
                arms=arms,
                warnings=list(dict.fromkeys(warnings)),
            ),
            inventory=inventory,
            resolutions=resolved,
            safety=safety,
        )

    def _connect_one_locked(
        self, config: YAMCellConfig, arm: YAMCellArmConfig
    ) -> None:
        if arm.transport_kind != "socketcan":
            raise YAMDriverUnavailableError(
                "Serial GELLO arms must use the retained legacy_pair adapter."
            )
        inventory = self._safe_inventory()
        resolution = resolve_configured_socketcan_arms(
            self._socket_requests(config), inventory, require_link_up=True
        )
        current = resolution.by_logical_id.get(arm.logical_id)
        arm_errors = [
            issue
            for issue in self._issues_for_arm(
                arm.logical_id, arm.stable_identity, resolution.issues
            )
            if issue.severity == "error"
        ]
        if current is None or current.link_up is not True or arm_errors:
            raise YAMDriverUnavailableError(
                "The selected stable adapter identity is no longer uniquely link-UP."
            )
        for existing in self._runtimes.values():
            latest = resolution.by_logical_id.get(existing.arm.logical_id)
            if (
                latest is None
                or latest.runtime_interface != existing.resolution.runtime_interface
            ):
                self._latch_fault_locked(existing.arm.logical_id)
                raise YAMDriverUnavailableError(
                    "A connected arm's stable adapter mapping changed; motion is latched off."
                )
            if existing.resolution.runtime_interface == current.runtime_interface:
                raise YAMDriverUnavailableError(
                    "The selected runtime CAN interface already has an arm owner."
                )

        lease = self._bus_locks.acquire(
            stable_identity=arm.stable_identity,
            runtime_interface=current.runtime_interface,
        )
        try:
            worker_config = I2RTArmWorkerConfig(
                logical_id=arm.logical_id,
                role=arm.role,
                stable_identity=arm.stable_identity,
                runtime_interface=current.runtime_interface,
                end_effector_kind=arm.end_effector_kind,  # type: ignore[arg-type]
                checkout_root=config.i2rt_root,
                expected_commit=config.i2rt_commit,
                require_read_only_checkout=True,
            )
            worker = self._worker_factory(worker_config)
        except BaseException as error:
            lease.release()
            raise YAMDriverUnavailableError(
                "The selected i2rt arm worker could not be constructed safely."
            ) from error
        runtime = _ArmRuntime(
            arm=arm,
            resolution=current,
            worker=worker,
            bus_lease=lease,
        )
        try:
            worker.start()
            runtime.last_telemetry = worker.telemetry()
        except BaseException as error:
            try:
                shutdown = worker.shutdown()
            except BaseException:
                shutdown = None
            if shutdown is not None and shutdown.clean:
                lease.release()
            else:
                runtime.blocked = True
                self._runtimes[arm.logical_id] = runtime
                self._faults[arm.logical_id] = _FAULT_DETAIL
                self._mark_pair_fault_locked(arm)
            raise YAMDriverUnavailableError(
                "The selected i2rt arm worker failed during verified startup."
            ) from error
        self._runtimes[arm.logical_id] = runtime
        self._resolutions[arm.logical_id] = current

    def _disconnect_one_locked(self, arm_id: str) -> None:
        runtime = self._runtimes.get(arm_id)
        if runtime is None:
            self._faults.pop(arm_id, None)
            return
        self._cancel_command_idle_timer_locked(runtime)
        try:
            result = runtime.worker.shutdown()
        except BaseException as error:
            runtime.blocked = True
            self._faults[arm_id] = _FAULT_DETAIL
            self._mark_pair_fault_locked(runtime.arm)
            raise YAMDriverUnavailableError(
                "The i2rt arm worker shutdown could not be confirmed."
            ) from error
        if not result.clean:
            runtime.blocked = True
            self._faults[arm_id] = _FAULT_DETAIL
            self._mark_pair_fault_locked(runtime.arm)
            raise YAMDriverUnavailableError(result.detail)
        runtime.bus_lease.release()
        self._runtimes.pop(arm_id, None)
        self._faults.pop(arm_id, None)

    def _commandable_runtime_locked(self, arm: YAMCellArmConfig) -> _ArmRuntime:
        if arm.logical_id in self._faults or (
            arm.pair_id is not None and arm.pair_id in self._pair_faults
        ):
            raise YAMDriverUnavailableError(_FAULT_DETAIL)
        runtime = self._runtimes.get(arm.logical_id)
        if runtime is None or runtime.blocked:
            raise YAMDriverUnavailableError("The selected YAM follower is disconnected.")
        self._refresh_runtime_locked(runtime)
        if arm.logical_id in self._faults:
            raise YAMDriverUnavailableError(_FAULT_DETAIL)
        return runtime

    def _refresh_runtime_locked(self, runtime: _ArmRuntime) -> None:
        if runtime.blocked:
            return
        try:
            snapshot = runtime.worker.snapshot()
            if (
                snapshot.phase != "ready"
                or snapshot.fault is not None
                or snapshot.shutdown_uncertain
            ):
                raise RuntimeError
            runtime.last_telemetry = runtime.worker.telemetry()
        except (I2RTWorkerError, OSError, RuntimeError):
            self._latch_fault_locked(runtime.arm.logical_id)

    def _latch_fault_locked(
        self, arm_id: str, *, attempt_pair_idle: bool = True
    ) -> None:
        if arm_id in self._faults:
            return
        arm = self._arm_config_locked(arm_id)
        self._faults[arm_id] = _FAULT_DETAIL
        self._mark_pair_fault_locked(arm)
        runtime = self._runtimes.get(arm_id)
        if runtime is not None:
            try:
                self._cancel_command_idle_timer_locked(runtime)
                runtime.worker.safe_idle()
            except BaseException:
                pass
        if attempt_pair_idle and arm.pair_id is not None:
            self._idle_pair_counterparts_locked(arm.pair_id, exclude=arm_id)
        self._diagnostic = YAMDriverDiagnostic(status="error", detail=_FAULT_DETAIL)

    def _idle_pair_counterparts_locked(self, pair_id: str, *, exclude: str) -> None:
        if self._cell is None:
            return
        for counterpart in self._cell.arms:
            if counterpart.logical_id == exclude or counterpart.pair_id != pair_id:
                continue
            runtime = self._runtimes.get(counterpart.logical_id)
            if runtime is None:
                continue
            try:
                self._cancel_command_idle_timer_locked(runtime)
                runtime.worker.safe_idle()
            except BaseException:
                self._faults[counterpart.logical_id] = _FAULT_DETAIL

    def _mark_pair_fault_locked(self, arm: YAMCellArmConfig) -> None:
        if arm.pair_id is not None:
            self._pair_faults.add(arm.pair_id)

    def _rebuild_pair_faults_locked(self) -> None:
        self._pair_faults = set()
        if self._cell is None:
            return
        for arm in self._cell.arms:
            if arm.logical_id in self._faults and arm.pair_id is not None:
                self._pair_faults.add(arm.pair_id)

    def _arm_snapshot_locked(self, arm: YAMCellArmConfig) -> ArmTelemetry:
        runtime = self._runtimes.get(arm.logical_id)
        if runtime is not None:
            self._refresh_runtime_locked(runtime)
        telemetry = None if runtime is None else runtime.last_telemetry
        faulted = arm.logical_id in self._faults
        pair_faulted = arm.pair_id is not None and arm.pair_id in self._pair_faults
        connected = bool(runtime is not None and not runtime.blocked and not faulted)
        worker_snapshot: I2RTWorkerSnapshot | None = None
        if runtime is not None:
            try:
                worker_snapshot = runtime.worker.snapshot()
            except BaseException:
                faulted = True
                connected = False

        safety = self._safety.get(arm.logical_id)
        warnings = list(safety.warnings if safety is not None else ())
        if connected:
            warnings.append(_POSE_WARNING)
        if pair_faulted:
            warnings.append(_PAIR_FAULT_WARNING)
        if arm.logical_id in self._handle_teardown_uncertain:
            warnings.insert(0, _HANDLE_DIAGNOSTIC_FAULT_DETAIL)
        if runtime is not None and runtime.blocked:
            warnings.append("Bus ownership is blocked after uncertain worker teardown.")
        resolution = (
            runtime.resolution if runtime is not None else self._resolutions.get(arm.logical_id)
        )
        positions = (
            telemetry.joint_positions if telemetry is not None else (0.0,) * 6
        )
        velocities = (
            telemetry.joint_velocities if telemetry is not None else (0.0,) * 6
        )
        efforts = (
            telemetry.joint_efforts if telemetry is not None else (None,) * 6
        )
        gripper_position = 0.0 if telemetry is None else telemetry.gripper_position
        gripper_velocity = 0.0 if telemetry is None else telemetry.gripper_velocity
        frequency = 0.0 if telemetry is None else telemetry.control_loop_frequency_hz
        accepting = bool(worker_snapshot and worker_snapshot.accepting_commands)
        if faulted or (runtime is not None and runtime.blocked):
            control_state = "error"
        elif not connected:
            control_state = "disconnected"
        elif arm.role == "follower" and accepting:
            control_state = "position_control"
        else:
            control_state = "gravity_comp"
        handle = None
        if arm.end_effector_kind == "yam_teaching_handle":
            range_result = self._handle_results.get(arm.logical_id)
            range_healthy = bool(
                range_result is not None
                and range_result.reachable
                and range_result.observed_minimum is not None
                and range_result.observed_maximum is not None
                and range_result.observed_minimum <= 0.15
                and range_result.observed_maximum >= 0.85
            )
            handle = TeachingHandleTelemetry(
                reachable=bool(
                    connected
                    and telemetry is not None
                    and telemetry.handle_trigger_position is not None
                ),
                trigger_position=(
                    None if telemetry is None else telemetry.handle_trigger_position
                ),
                buttons=([] if telemetry is None else list(telemetry.handle_buttons)),
                range_status=(
                    "not_tested"
                    if range_result is None
                    else ("healthy" if range_healthy else "unhealthy")
                ),
                observed_minimum=(
                    None if range_result is None else range_result.observed_minimum
                ),
                observed_maximum=(
                    None if range_result is None else range_result.observed_maximum
                ),
                calibration_warning=(
                    None
                    if range_result is None or range_healthy
                    else "Handle range is unhealthy; use the documented CLI maintenance procedure."
                ),
            )
        can_state = (
            "warning"
            if faulted or (runtime is not None and runtime.blocked)
            else ("active" if connected else "disconnected")
        )
        return ArmTelemetry(
            id=arm.logical_id,
            name=arm.name,
            role=arm.role,
            pair_id=arm.pair_id,
            group_id=arm.group_id,
            side=arm.side,
            transport_kind=arm.transport_kind,
            stable_identity=arm.stable_identity,
            end_effector_kind=arm.end_effector_kind,
            driver="i2rt-worker",
            connected=connected,
            control_state=control_state,
            energized=bool(runtime is not None),
            holding=bool(runtime is not None and arm.role == "follower"),
            timestamp=self._wall_clock(),
            joints=[
                JointTelemetry(
                    name=name,
                    position_radians=float(positions[index]),
                    velocity_radians_per_second=float(velocities[index]),
                    effort_newton_meters=(
                        None if efforts[index] is None else float(efforts[index])
                    ),
                    temperature_celsius=None,
                )
                for index, name in enumerate(JOINT_NAMES)
            ],
            pose=EndEffectorPose(
                x_m=0.0,
                y_m=0.0,
                z_m=0.0,
                roll_radians=0.0,
                pitch_radians=0.0,
                yaw_radians=0.0,
            ),
            gripper=GripperTelemetry(
                position=float(gripper_position),
                velocity=float(gripper_velocity),
                force_newtons=(
                    None
                    if telemetry is None or telemetry.gripper_effort is None
                    else max(0.0, float(telemetry.gripper_effort))
                ),
                is_closed=gripper_position <= 0.15,
            ),
            can=(
                None
                if arm.transport_kind != "socketcan" or resolution is None
                else CANTelemetry(
                    interface=resolution.runtime_interface,
                    state=can_state,
                    bitrate=None,
                    tx_error_count=None,
                    rx_error_count=None,
                )
            ),
            control_loop=ControlLoopTelemetry(
                target_frequency_hz=max(1.0, frequency),
                frequency_hz=frequency,
                cycle_time_ms=(0.0 if frequency <= 0.0 else 1_000.0 / frequency),
                jitter_ms=0.0,
                dropped_cycles=(
                    0 if worker_snapshot is None else worker_snapshot.dropped_commands
                ),
                source="i2rt-motor-chain",
            ),
            handle=handle,
            frame_map_active=bool(safety and safety.frame_map.active),
            soft_limits_active=bool(safety and safety.soft_limits.active),
            warnings=list(dict.fromkeys(warnings)),
        )

    def _update_diagnostic_locked(self) -> None:
        if self._mode != "cell" or self._cell is None:
            return
        if self._handle_teardown_uncertain:
            self._diagnostic = YAMDriverDiagnostic(
                status="error", detail=_HANDLE_DIAGNOSTIC_FAULT_DETAIL
            )
        elif self._faults or any(runtime.blocked for runtime in self._runtimes.values()):
            self._diagnostic = YAMDriverDiagnostic(status="error", detail=_FAULT_DETAIL)
        elif self._runtimes and len(self._runtimes) == len(self._cell.arms):
            self._diagnostic = YAMDriverDiagnostic(
                status="connected", detail="All configured YAM cell arm workers are live."
            )
        else:
            if (
                not self._runtimes
                and self._passive_diagnostic is not None
                and self._passive_diagnostic.status in {"missing", "error"}
            ):
                self._diagnostic = self._passive_diagnostic
            else:
                self._diagnostic = YAMDriverDiagnostic(
                    status="configured",
                    detail=(
                        "The YAM cell is configured; explicit per-arm connection controls "
                        "which workers are live."
                    ),
                )

    def _arm_command_idle_timer_locked(self, runtime: _ArmRuntime) -> None:
        """Safe-idle a stream before the worker's 250 ms freshness watchdog.

        Teleop and inference must refresh actions faster than
        ``command_max_age_seconds``.  If their queue underflows, this parent
        timer explicitly revokes position writes and asks for gravity-comp idle
        before the child would latch a watchdog fault.
        """

        self._cancel_command_idle_timer_locked(runtime)
        runtime.command_generation += 1
        generation = runtime.command_generation
        delay = runtime.worker.config.command_max_age_seconds * 0.5

        def idle_if_current() -> None:
            with self._lock:
                current = self._runtimes.get(runtime.arm.logical_id)
                if (
                    current is not runtime
                    or current.command_generation != generation
                    or current.blocked
                ):
                    return
                current.idle_timer = None
                try:
                    current.worker.safe_idle()
                except BaseException:
                    self._latch_fault_locked(current.arm.logical_id)

        timer = threading.Timer(delay, idle_if_current)
        timer.daemon = True
        runtime.idle_timer = timer
        timer.start()

    @staticmethod
    def _cancel_command_idle_timer_locked(runtime: _ArmRuntime) -> None:
        runtime.command_generation += 1
        timer = runtime.idle_timer
        runtime.idle_timer = None
        if timer is not None:
            timer.cancel()

    def _require_cell_stopped_locked(self) -> None:
        if self._handle_bus_leases:
            raise YAMDriverUnavailableError(
                "Cannot change YAM setup while a teaching-handle diagnostic owns a bus."
            )
        if not self._runtimes:
            return
        uncertain: list[str] = []
        for arm_id in sorted(tuple(self._runtimes)):
            try:
                self._disconnect_one_locked(arm_id)
            except YAMDriverUnavailableError:
                uncertain.append(arm_id)
        if uncertain:
            raise YAMDriverUnavailableError(
                "Cannot change YAM setup while teardown is uncertain for: "
                + ", ".join(uncertain)
            )

    def _require_cell_locked(self) -> YAMCellConfig:
        if self._mode != "cell" or self._cell is None:
            raise YAMDriverUnavailableError("No physical YAM cell is configured.")
        return self._cell

    def _arm_config_locked(self, arm_id: str) -> YAMCellArmConfig:
        config = self._require_cell_locked()
        for arm in config.arms:
            if arm.logical_id == arm_id:
                return arm
        raise ArmNotFoundError(arm_id)

    def _select_arms_locked(
        self, config: YAMCellConfig, arm_ids: list[str] | None
    ) -> set[str]:
        configured = {arm.logical_id for arm in config.arms}
        if arm_ids is None:
            selected = configured
        else:
            selected = set(arm_ids)
        if not selected:
            raise YAMDriverUnavailableError(
                "Add and select at least one configured YAM cell arm before connecting."
            )
        unknown = selected - configured
        if unknown:
            raise ArmNotFoundError(sorted(unknown)[0])
        return selected

    @staticmethod
    def _socket_requests(config: YAMCellConfig) -> list[ConfiguredSocketCANArm]:
        return [
            ConfiguredSocketCANArm(arm.logical_id, arm.stable_identity)
            for arm in config.arms
            if arm.transport_kind == "socketcan"
        ]

    def _safe_inventory(self) -> SocketCANInventory:
        try:
            return self._inventory_provider()
        except BaseException as error:
            raise YAMDriverUnavailableError(
                "Passive SocketCAN inventory could not be read safely."
            ) from error

    @staticmethod
    def _issues_for_arm(
        logical_id: str,
        stable_identity: str,
        issues: Sequence[SocketCANDiscoveryIssue],
    ) -> list[SocketCANDiscoveryIssue]:
        return [
            issue
            for issue in issues
            if getattr(issue, "logical_id", None) in {None, logical_id}
            and getattr(issue, "stable_identity", None) in {None, stable_identity}
        ]

    @staticmethod
    def _link_state(adapter: SocketCANAdapter) -> str:
        if adapter.link_up is True:
            return "up"
        if adapter.link_up is False:
            return "down"
        return "unknown"

    @staticmethod
    def _resolved_link_state(arm: ResolvedSocketCANArm | None) -> str:
        if arm is None or arm.link_up is None:
            return "unknown"
        return "up" if arm.link_up else "down"
