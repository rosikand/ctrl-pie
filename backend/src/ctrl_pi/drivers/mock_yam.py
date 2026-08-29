from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ctrl_pi.drivers.yam import (
    JOINT_NAMES,
    ActionLimitError,
    ArmAction,
    ArmNotFoundError,
    ArmRole,
    ArmTelemetry,
    CANTelemetry,
    ControlLoopTelemetry,
    EndEffectorPose,
    GripperTelemetry,
    JogCommand,
    JogLimitError,
    JointTelemetry,
    TeachingHandleTelemetry,
    YAMArmPreflight,
    YAMArmResolution,
    YAMCellArmConfig,
    YAMCellConfig,
    YAMDiscoveryCandidate,
    YAMDiscoveryDevice,
    YAMDiscoveryResult,
    YAMDriver,
    YAMDriverDiagnostic,
    YAMHandleRangeResult,
    YAMPreflightResult,
)

JOINT_LIMIT_RADIANS = (-math.pi, math.pi)
POSE_LIMITS = {
    "x": (0.10, 0.70),
    "y": (-0.50, 0.50),
    "z": (0.02, 0.70),
    "roll": (-math.pi, math.pi),
    "pitch": (-math.pi, math.pi),
    "yaw": (-math.pi, math.pi),
}

MOCK_I2RT_COMMIT = "0" * 40
MOCK_CELL_CONFIG = YAMCellConfig(
    name="Mock two-pair YAM cell",
    i2rt_root="/opt/ctrl-pi/mock-i2rt",
    i2rt_commit=MOCK_I2RT_COMMIT,
    pair_ports={"right": 21_333, "left": 21_334},
    arms=[
        YAMCellArmConfig(
            logical_id="yam-leader",
            name="Right YAM Leader",
            role="leader",
            pair_id="right",
            group_id="bimanual",
            side="right",
            transport_kind="socketcan",
            stable_identity="mock-adapter-right-leader",
            end_effector_kind="yam_teaching_handle",
        ),
        YAMCellArmConfig(
            logical_id="yam-follower",
            name="Right YAM Follower",
            role="follower",
            pair_id="right",
            group_id="bimanual",
            side="right",
            transport_kind="socketcan",
            stable_identity="mock-adapter-right-follower",
            end_effector_kind="linear_4310",
        ),
        YAMCellArmConfig(
            logical_id="yam-leader-left",
            name="Left YAM Leader",
            role="leader",
            pair_id="left",
            group_id="bimanual",
            side="left",
            transport_kind="socketcan",
            stable_identity="mock-adapter-left-leader",
            end_effector_kind="yam_teaching_handle",
        ),
        YAMCellArmConfig(
            logical_id="yam-follower-left",
            name="Left YAM Follower",
            role="follower",
            pair_id="left",
            group_id="bimanual",
            side="left",
            transport_kind="socketcan",
            stable_identity="mock-adapter-left-follower",
            end_effector_kind="linear_4310",
        ),
    ],
)
# Kept as a source-compatible import name for V1.1 tests and SDK examples.
MOCK_SETUP_CONFIG = MOCK_CELL_CONFIG

DEFAULT_RUNTIME_INTERFACES = {
    "mock-adapter-right-leader": "can0",
    "mock-adapter-right-follower": "can2",
    "mock-adapter-left-leader": "can3",
    "mock-adapter-left-follower": "can1",
}


@dataclass
class _MockArmState:
    id: str
    name: str
    role: ArmRole
    pair_id: str | None
    group_id: str | None
    side: str | None
    stable_identity: str
    end_effector_kind: str
    can_interface: str | None
    joints: dict[str, float]
    pose: dict[str, float]
    gripper_position: float
    connected: bool = True
    control_state: str = "gravity_comp"
    holding: bool = False
    energized: bool = field(init=False)
    joint_velocities: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0)
    )
    gripper_velocity: float = 0.0
    last_command_at: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    sample_count: int = 0
    joint_origins: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.joint_origins = self.joints.copy()
        self.energized = self.connected


class MockYAMDriver(YAMDriver):
    """Thread-safe, process-local YAM simulator used by every V1 mock flow."""

    def __init__(
        self,
        *,
        runtime_interfaces: dict[str, str] | None = None,
        handle_ranges: dict[str, tuple[float, float] | None] | None = None,
        initially_connected: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._setup_config = MOCK_CELL_CONFIG
        self._runtime_interfaces = dict(
            DEFAULT_RUNTIME_INTERFACES
            if runtime_interfaces is None
            else runtime_interfaces
        )
        self._handle_ranges = {
            "yam-leader": (0.04, 0.96),
            "yam-leader-left": (0.03, 0.97),
            **(handle_ranges or {}),
        }
        self.action_counts: dict[str, int] = {}
        self._arms: dict[str, _MockArmState] = {
            "yam-leader": _MockArmState(
                id="yam-leader",
                name="Right YAM Leader",
                role="leader",
                pair_id="right",
                group_id="bimanual",
                side="right",
                stable_identity="mock-adapter-right-leader",
                end_effector_kind="yam_teaching_handle",
                can_interface=self._runtime_interfaces.get("mock-adapter-right-leader"),
                joints=dict(zip(JOINT_NAMES, (0.0, -0.35, 0.70, 0.0, 0.15, 0.0))),
                pose={"x": 0.38, "y": -0.16, "z": 0.31, "roll": 0.0, "pitch": 0.1, "yaw": 0.0},
                gripper_position=0.62,
                connected=initially_connected,
            ),
            "yam-follower": _MockArmState(
                id="yam-follower",
                name="Right YAM Follower",
                role="follower",
                pair_id="right",
                group_id="bimanual",
                side="right",
                stable_identity="mock-adapter-right-follower",
                end_effector_kind="linear_4310",
                can_interface=self._runtime_interfaces.get("mock-adapter-right-follower"),
                joints=dict(zip(JOINT_NAMES, (0.05, -0.30, 0.65, 0.0, -0.10, 0.0))),
                pose={"x": 0.39, "y": 0.16, "z": 0.30, "roll": 0.0, "pitch": -0.1, "yaw": 0.0},
                gripper_position=0.45,
                connected=initially_connected,
                holding=initially_connected,
            ),
            "yam-leader-left": _MockArmState(
                id="yam-leader-left",
                name="Left YAM Leader",
                role="leader",
                pair_id="left",
                group_id="bimanual",
                side="left",
                stable_identity="mock-adapter-left-leader",
                end_effector_kind="yam_teaching_handle",
                can_interface=self._runtime_interfaces.get("mock-adapter-left-leader"),
                joints=dict(zip(JOINT_NAMES, (-0.02, -0.33, 0.68, 0.01, 0.12, -0.02))),
                pose={"x": 0.38, "y": 0.18, "z": 0.31, "roll": 0.0, "pitch": 0.1, "yaw": 0.0},
                gripper_position=0.58,
                connected=initially_connected,
            ),
            "yam-follower-left": _MockArmState(
                id="yam-follower-left",
                name="Left YAM Follower",
                role="follower",
                pair_id="left",
                group_id="bimanual",
                side="left",
                stable_identity="mock-adapter-left-follower",
                end_effector_kind="linear_4310",
                can_interface=self._runtime_interfaces.get("mock-adapter-left-follower"),
                joints=dict(zip(JOINT_NAMES, (0.01, -0.31, 0.66, -0.01, -0.08, 0.01))),
                pose={"x": 0.39, "y": -0.18, "z": 0.30, "roll": 0.0, "pitch": -0.1, "yaw": 0.0},
                gripper_position=0.48,
                connected=initially_connected,
                holding=initially_connected,
            ),
        }
        self.action_counts = dict.fromkeys(self._arms, 0)

    def setup_config(self) -> YAMCellConfig:
        with self._lock:
            return self._setup_config.model_copy(deep=True)

    def discover_setup(self) -> YAMDiscoveryResult:
        interface_counts: dict[str, int] = {}
        for interface in self._runtime_interfaces.values():
            interface_counts[interface] = interface_counts.get(interface, 0) + 1
        devices = [
            YAMDiscoveryDevice(
                transport_kind="socketcan",
                stable_identity=identity,
                product="ctrl-pi deterministic mock adapter",
                runtime_interface=interface,
                link_state="up",
            )
            for identity, interface in self._runtime_interfaces.items()
        ]
        resolutions = [
            YAMArmResolution(
                arm_id=arm.logical_id,
                transport_kind=arm.transport_kind,
                stable_identity=arm.stable_identity,
                runtime_interface=self._runtime_interfaces.get(arm.stable_identity),
                resolved=arm.stable_identity in self._runtime_interfaces
                and interface_counts.get(
                    self._runtime_interfaces.get(arm.stable_identity, ""), 0
                )
                == 1,
                conflict=interface_counts.get(
                    self._runtime_interfaces.get(arm.stable_identity, ""), 0
                )
                > 1,
                detail=(
                    "Mock adapter resolved."
                    if arm.stable_identity in self._runtime_interfaces
                    and interface_counts.get(
                        self._runtime_interfaces.get(arm.stable_identity, ""), 0
                    )
                    == 1
                    else "Mock adapter is missing or conflicts with another arm."
                ),
            )
            for arm in self._setup_config.arms
        ]
        return YAMDiscoveryResult(
            mode="mock",
            devices=devices,
            resolutions=resolutions,
            can_interfaces=[
                YAMDiscoveryCandidate(
                    id=device.runtime_interface or device.stable_identity,
                    label=f"Mock CAN adapter {device.stable_identity}",
                )
                for device in devices
            ],
            leader_ports=[],
            suggested_config=self.setup_config(),
            detail="Four deterministic mock CAN adapters were inspected without hardware access.",
        )

    def preflight_setup(self, config: YAMCellConfig) -> YAMPreflightResult:
        if config != MOCK_CELL_CONFIG:
            return YAMPreflightResult(
                ready=False,
                calibration_ready=False,
                diagnostic=YAMDriverDiagnostic(
                    status="error",
                    detail="Mock mode accepts only its deterministic built-in YAM setup.",
                ),
            )
        discovery = self.discover_setup()
        resolution_by_id = {item.arm_id: item for item in discovery.resolutions}
        arm_results = []
        warnings: list[str] = []
        for arm in config.arms:
            resolution = resolution_by_id[arm.logical_id]
            arm_warnings = []
            soft_limits_status = "not_applicable"
            frame_map_status = "not_applicable"
            if arm.role == "follower":
                frame_map_status = "identity"
                soft_limits_status = "missing"
                arm_warnings.append("NO SASH GUARD")
                warnings.append(f"{arm.logical_id}: NO SASH GUARD")
            arm_results.append(
                YAMArmPreflight(
                    arm_id=arm.logical_id,
                    ready=resolution.resolved and not resolution.conflict,
                    runtime_interface=resolution.runtime_interface,
                    link_state=("up" if resolution.resolved else "unknown"),
                    frame_map_status=frame_map_status,
                    soft_limits_status=soft_limits_status,
                    handle_status=(
                        "not_checked" if arm.role == "leader" else "not_applicable"
                    ),
                    warnings=arm_warnings,
                    diagnostic=YAMDriverDiagnostic(
                        status=("configured" if resolution.resolved else "missing"),
                        detail=resolution.detail,
                    ),
                )
            )
        ready = all(arm.ready for arm in arm_results)
        return YAMPreflightResult(
            ready=ready,
            calibration_ready=ready,
            diagnostic=YAMDriverDiagnostic(
                status="configured" if ready else "missing",
                detail=(
                    "Mock YAM cell passed passive preflight."
                    if ready
                    else "One or more mock CAN adapter assignments are unresolved."
                ),
            ),
            i2rt_ready=True,
            arms=arm_results,
            warnings=warnings,
        )

    def apply_setup(self, config: YAMCellConfig) -> YAMDriverDiagnostic:
        result = self.preflight_setup(config)
        if not result.ready:
            return result.diagnostic
        with self._lock:
            self._setup_config = config.model_copy(deep=True)
        return result.diagnostic

    def reset_setup(self) -> YAMDriverDiagnostic:
        with self._lock:
            self._setup_config = MOCK_CELL_CONFIG
        return self.preflight_setup(MOCK_CELL_CONFIG).diagnostic

    def diagnostic(self) -> YAMDriverDiagnostic:
        with self._lock:
            faulted = [state.id for state in self._arms.values() if state.control_state == "error"]
            connected = sum(state.connected for state in self._arms.values())
            total = len(self._arms)
        if faulted:
            return YAMDriverDiagnostic(
                status="error",
                detail="Mock YAM command path is fault-latched for: "
                + ", ".join(sorted(faulted))
                + ".",
            )
        if connected == total:
            return YAMDriverDiagnostic(
                status="connected", detail=f"All {total} mock YAM arms are connected."
            )
        return YAMDriverDiagnostic(
            status="configured",
            detail=f"Mock YAM cell configured; {connected} of {total} arms connected.",
        )

    def startup(self) -> None:
        self.connect_arms()

    def shutdown(self) -> None:
        self.disconnect_arms()

    def connect_arms(self, arm_ids: list[str] | None = None) -> list[ArmTelemetry]:
        with self._lock:
            selected = set(self._arms) if arm_ids is None else set(arm_ids)
            unknown = selected - set(self._arms)
            if unknown:
                raise ArmNotFoundError(sorted(unknown)[0])
            for arm_id in selected:
                state = self._arms[arm_id]
                if state.control_state == "error":
                    raise RuntimeError(
                        f"mock arm {arm_id} is fault-latched; disconnect it before reconnecting"
                    )
                if state.can_interface is None:
                    raise RuntimeError(f"mock adapter for {arm_id} is missing")
                state.connected = True
                state.energized = True
                state.control_state = "gravity_comp"
                state.holding = state.role == "follower"
            return [self._snapshot(self._arms[arm_id]) for arm_id in sorted(selected)]

    def disconnect_arms(self, arm_ids: list[str] | None = None) -> list[ArmTelemetry]:
        with self._lock:
            selected = set(self._arms) if arm_ids is None else set(arm_ids)
            unknown = selected - set(self._arms)
            if unknown:
                raise ArmNotFoundError(sorted(unknown)[0])
            for arm_id in selected:
                state = self._arms[arm_id]
                state.connected = False
                state.energized = False
                state.control_state = "disconnected"
                state.holding = False
            return [self._snapshot(self._arms[arm_id]) for arm_id in sorted(selected)]

    def check_handle_range(
        self, arm_id: str, *, duration_seconds: float = 10.0
    ) -> YAMHandleRangeResult:
        del duration_seconds
        with self._lock:
            state = self._find(arm_id)
            if state.role != "leader" or state.end_effector_kind != "yam_teaching_handle":
                raise ValueError("handle range checks require a teaching-handle leader")
            observed = self._handle_ranges.get(arm_id)
            if observed is None:
                return YAMHandleRangeResult(
                    arm_id=arm_id,
                    reachable=False,
                    healthy=False,
                    detail="Mock teaching handle did not return a fresh sample.",
                )
            minimum, maximum = observed
            healthy = minimum <= 0.15 and maximum >= 0.85
            return YAMHandleRangeResult(
                arm_id=arm_id,
                reachable=True,
                observed_minimum=minimum,
                observed_maximum=maximum,
                healthy=healthy,
                detail=(
                    "Teaching-handle release/squeeze range is healthy."
                    if healthy
                    else "Teaching-handle range is incomplete; use the documented CLI maintenance procedure."
                ),
            )

    def safe_idle(self, arm_id: str) -> ArmTelemetry:
        with self._lock:
            state = self._find(arm_id)
            if state.connected:
                state.control_state = "gravity_comp"
                # Gravity-comp idle revokes position writes; it is not a
                # torque-off/limp state. Keep follower holding semantics in
                # parity with a live supervised i2rt worker until disconnect.
                state.holding = state.role == "follower"
            return self._snapshot(state)

    def latch_fault(self, arm_ids: list[str], detail: str) -> None:
        del detail
        with self._lock:
            selected = set(arm_ids)
            unknown = selected - set(self._arms)
            if unknown:
                raise ArmNotFoundError(sorted(unknown)[0])
            for arm_id in selected:
                state = self._arms[arm_id]
                state.connected = False
                state.control_state = "error"
                # A faulted runtime may still own its bus and remain live.
                # Only explicit disconnect proves mock de-energization too.
                state.holding = state.energized and state.role == "follower"

    def list_arms(self) -> list[ArmTelemetry]:
        with self._lock:
            return [self._snapshot(state) for state in self._arms.values()]

    def get_arm(self, arm_id: str) -> ArmTelemetry:
        with self._lock:
            return self._snapshot(self._find(arm_id))

    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry:
        with self._lock:
            state = self._find(arm_id)
            if state.role != "follower":
                raise JogLimitError(
                    "Leader arms are observation-only and cannot be jogged."
                )
            if not state.connected:
                raise RuntimeError(f"arm {arm_id} is disconnected")
            now = time.monotonic()
            elapsed = max(now - state.last_command_at, 0.05) if state.last_command_at else 0.10

            if command.kind == "joint":
                current = state.joints[command.axis]
                target = current + command.delta
                self._require_within(command.axis, target, *JOINT_LIMIT_RADIANS)
                state.joints[command.axis] = target
                state.joint_velocities[command.axis] = command.delta / elapsed
            elif command.kind == "cartesian":
                current = self._derived_pose(state)[command.axis]
                target = current + command.delta
                self._require_within(command.axis, target, *POSE_LIMITS[command.axis])
                state.pose[command.axis] += command.delta
            else:
                target = state.gripper_position + command.delta
                self._require_within("gripper position", target, 0.0, 1.0)
                state.gripper_position = target
                state.gripper_velocity = command.delta / elapsed

            state.last_command_at = now
            return self._snapshot(state)

    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry:
        with self._lock:
            state = self._find(arm_id)
            if state.role != "follower":
                raise ActionLimitError(
                    "Leader arms are observation-only and cannot accept actions."
                )
            if not state.connected:
                raise RuntimeError(f"arm {arm_id} is disconnected")
            for name, target in action.joint_positions_radians.items():
                if not JOINT_LIMIT_RADIANS[0] <= target <= JOINT_LIMIT_RADIANS[1]:
                    raise ActionLimitError(f"action target for {name} is outside its safe range")

            now = time.monotonic()
            elapsed = max(now - state.last_command_at, 0.005) if state.last_command_at else 0.05
            state.joint_velocities = {
                name: (action.joint_positions_radians[name] - state.joints[name]) / elapsed
                for name in JOINT_NAMES
            }
            state.gripper_velocity = (
                action.gripper_position - state.gripper_position
            ) / elapsed
            state.joints.update(action.joint_positions_radians)
            state.gripper_position = action.gripper_position
            state.last_command_at = now
            state.control_state = "position_control"
            state.holding = state.role == "follower"
            self.action_counts[arm_id] += 1
            return self._snapshot(state)

    def _find(self, arm_id: str) -> _MockArmState:
        try:
            return self._arms[arm_id]
        except KeyError as error:
            raise ArmNotFoundError(arm_id) from error

    @staticmethod
    def _require_within(axis: str, value: float, lower: float, upper: float) -> None:
        if not lower <= value <= upper:
            raise JogLimitError(
                f"jog would move {axis} outside its safe range [{lower:.2f}, {upper:.2f}]"
            )

    def _snapshot(self, state: _MockArmState) -> ArmTelemetry:
        now = time.monotonic()
        state.sample_count += 1
        command_age = now - state.last_command_at if state.last_command_at else math.inf
        if command_age > 0.20:
            state.joint_velocities = dict.fromkeys(JOINT_NAMES, 0.0)
            state.gripper_velocity = 0.0

        phase = now - state.started_at
        nominal_frequency = 420.0 if state.role == "leader" else 270.0
        frequency = (
            nominal_frequency - 0.5 + math.sin(phase * 2.0) * 0.4
            if state.connected
            else 0.0
        )
        cycle_time = 1000.0 / frequency if frequency else 0.0
        joints = [
            JointTelemetry(
                name=name,
                position_radians=position,
                velocity_radians_per_second=state.joint_velocities[name],
                effort_newton_meters=0.3 + index * 0.08 + abs(math.sin(phase + index)) * 0.1,
                temperature_celsius=32.0 + index * 0.7 + abs(math.sin(phase / 5.0)),
            )
            for index, (name, position) in enumerate(state.joints.items())
        ]
        pose = self._derived_pose(state)
        return ArmTelemetry(
            id=state.id,
            name=state.name,
            role=state.role,
            pair_id=state.pair_id,
            group_id=state.group_id,
            side=state.side,
            transport_kind="socketcan",
            stable_identity=state.stable_identity,
            end_effector_kind=state.end_effector_kind,
            driver="mock",
            connected=state.connected,
            control_state=state.control_state,
            energized=state.energized,
            holding=state.holding,
            timestamp=datetime.now(UTC),
            joints=joints,
            pose=EndEffectorPose(
                x_m=pose["x"],
                y_m=pose["y"],
                z_m=pose["z"],
                roll_radians=pose["roll"],
                pitch_radians=pose["pitch"],
                yaw_radians=pose["yaw"],
            ),
            gripper=GripperTelemetry(
                position=state.gripper_position,
                velocity=state.gripper_velocity,
                force_newtons=max(0.0, (1.0 - state.gripper_position) * 18.0),
                is_closed=state.gripper_position <= 0.15,
            ),
            can=CANTelemetry(
                interface=state.can_interface or "unresolved",
                state="active" if state.connected else "disconnected",
                bitrate=1_000_000 if state.connected else None,
                tx_error_count=0,
                rx_error_count=0,
            ),
            control_loop=ControlLoopTelemetry(
                target_frequency_hz=nominal_frequency,
                frequency_hz=frequency,
                cycle_time_ms=cycle_time,
                jitter_ms=(
                    abs(cycle_time - (1000.0 / nominal_frequency))
                    if state.connected
                    else 0.0
                ),
                dropped_cycles=0,
                source="mock i2rt CAN worker",
            ),
            handle=(
                TeachingHandleTelemetry(
                    reachable=state.connected,
                    trigger_position=1.0 - state.gripper_position,
                    buttons=[False, False],
                    range_status="not_tested",
                    calibration_warning=(
                        None
                        if self._handle_ranges.get(state.id) is not None
                        else "Teaching handle has not returned a fresh sample."
                    ),
                )
                if state.role == "leader"
                else None
            ),
            frame_map_active=False,
            soft_limits_active=False,
            warnings=(["NO SASH GUARD"] if state.role == "follower" else []),
        )

    @staticmethod
    def _derived_pose(state: _MockArmState) -> dict[str, float]:
        """Small deterministic kinematic approximation for useful mock feedback."""

        joint_delta = {
            name: state.joints[name] - state.joint_origins[name] for name in JOINT_NAMES
        }
        return {
            "x": state.pose["x"]
            - 0.04 * joint_delta["shoulder_pitch"]
            - 0.025 * joint_delta["elbow_pitch"],
            "y": state.pose["y"] + 0.04 * joint_delta["shoulder_yaw"],
            "z": state.pose["z"]
            + 0.04 * joint_delta["shoulder_pitch"]
            + 0.03 * joint_delta["elbow_pitch"],
            "roll": state.pose["roll"] + joint_delta["wrist_roll"],
            "pitch": state.pose["pitch"] + joint_delta["wrist_pitch"],
            "yaw": state.pose["yaw"]
            + joint_delta["shoulder_yaw"]
            + joint_delta["wrist_yaw"],
        }
