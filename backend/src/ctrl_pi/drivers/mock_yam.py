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
    YAMDriver,
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


@dataclass
class _MockArmState:
    id: str
    name: str
    role: ArmRole
    can_interface: str
    joints: dict[str, float]
    pose: dict[str, float]
    gripper_position: float
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


class MockYAMDriver(YAMDriver):
    """Thread-safe, process-local YAM simulator used by every V1 mock flow."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._arms: dict[str, _MockArmState] = {
            "yam-leader": _MockArmState(
                id="yam-leader",
                name="YAM Leader",
                role="leader",
                can_interface="mock-can0",
                joints=dict(zip(JOINT_NAMES, (0.0, -0.35, 0.70, 0.0, 0.15, 0.0))),
                pose={"x": 0.38, "y": -0.16, "z": 0.31, "roll": 0.0, "pitch": 0.1, "yaw": 0.0},
                gripper_position=0.62,
            ),
            "yam-follower": _MockArmState(
                id="yam-follower",
                name="YAM Follower",
                role="follower",
                can_interface="mock-can1",
                joints=dict(zip(JOINT_NAMES, (0.05, -0.30, 0.65, 0.0, -0.10, 0.0))),
                pose={"x": 0.39, "y": 0.16, "z": 0.30, "roll": 0.0, "pitch": -0.1, "yaw": 0.0},
                gripper_position=0.45,
            ),
        }

    def list_arms(self) -> list[ArmTelemetry]:
        with self._lock:
            return [self._snapshot(state) for state in self._arms.values()]

    def get_arm(self, arm_id: str) -> ArmTelemetry:
        with self._lock:
            return self._snapshot(self._find(arm_id))

    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry:
        with self._lock:
            state = self._find(arm_id)
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
        frequency = 199.5 + math.sin(phase * 2.0) * 0.4
        cycle_time = 1000.0 / frequency
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
            driver="mock",
            connected=True,
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
                interface=state.can_interface,
                state="active",
                bitrate=1_000_000,
                tx_error_count=0,
                rx_error_count=0,
            ),
            control_loop=ControlLoopTelemetry(
                target_frequency_hz=200.0,
                frequency_hz=frequency,
                cycle_time_ms=cycle_time,
                jitter_ms=abs(cycle_time - 5.0),
                dropped_cycles=0,
            ),
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
