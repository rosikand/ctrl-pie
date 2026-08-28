from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArmRole = Literal["leader", "follower"]
JogKind = Literal["joint", "cartesian", "gripper"]

JOINT_NAMES = (
    "shoulder_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
CARTESIAN_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
GRIPPER_AXES = ("position",)


class JointTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    position_radians: float
    velocity_radians_per_second: float
    effort_newton_meters: float
    temperature_celsius: float


class EndEffectorPose(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_m: float
    y_m: float
    z_m: float
    roll_radians: float
    pitch_radians: float
    yaw_radians: float


class GripperTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    position: float = Field(ge=0.0, le=1.0)
    velocity: float
    force_newtons: float = Field(ge=0.0)
    is_closed: bool


class CANTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    interface: str
    state: Literal["active", "warning", "bus_off", "disconnected"]
    bitrate: int = Field(gt=0)
    tx_error_count: int = Field(ge=0)
    rx_error_count: int = Field(ge=0)


class ControlLoopTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_frequency_hz: float = Field(gt=0.0)
    frequency_hz: float = Field(ge=0.0)
    cycle_time_ms: float = Field(ge=0.0)
    jitter_ms: float = Field(ge=0.0)
    dropped_cycles: int = Field(ge=0)


class ArmTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    role: ArmRole
    driver: str
    connected: bool
    timestamp: datetime
    joints: list[JointTelemetry]
    pose: EndEffectorPose
    gripper: GripperTelemetry
    can: CANTelemetry
    control_loop: ControlLoopTelemetry


class ArmsResponse(BaseModel):
    arms: list[ArmTelemetry]


class TelemetryFrame(BaseModel):
    type: Literal["telemetry"] = "telemetry"
    timestamp: datetime
    arms: list[ArmTelemetry]


class ArmAction(BaseModel):
    """Absolute actuator targets shared by teleop and future policy runtimes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime
    joint_positions_radians: dict[str, float]
    gripper_position: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_joint_targets(self) -> ArmAction:
        if set(self.joint_positions_radians) != set(JOINT_NAMES):
            raise ValueError("joint targets must contain exactly the six YAM joint names")
        if not all(math.isfinite(value) for value in self.joint_positions_radians.values()):
            raise ValueError("joint targets must be finite")
        return self

    @classmethod
    def from_telemetry(cls, telemetry: ArmTelemetry) -> ArmAction:
        return cls(
            timestamp=telemetry.timestamp,
            joint_positions_radians={
                joint.name: joint.position_radians for joint in telemetry.joints
            },
            gripper_position=telemetry.gripper.position,
        )


class JogCommand(BaseModel):
    """One bounded relative jog; units depend on the selected kind."""

    model_config = ConfigDict(extra="forbid")

    kind: JogKind
    axis: str = Field(min_length=1, max_length=32)
    delta: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_axis_and_delta(self) -> JogCommand:
        allowed_axes: dict[JogKind, tuple[str, ...]] = {
            "joint": JOINT_NAMES,
            "cartesian": CARTESIAN_AXES,
            "gripper": GRIPPER_AXES,
        }
        if self.axis not in allowed_axes[self.kind]:
            allowed = ", ".join(allowed_axes[self.kind])
            raise ValueError(f"axis must be one of: {allowed}")

        if self.kind == "joint":
            maximum = 0.25
        elif self.kind == "cartesian" and self.axis in {"roll", "pitch", "yaw"}:
            maximum = 0.15
        elif self.kind == "cartesian":
            maximum = 0.05
        else:
            maximum = 0.10

        if self.delta == 0 or abs(self.delta) > maximum:
            raise ValueError(f"delta must be non-zero and no greater than {maximum}")
        return self


class ArmNotFoundError(LookupError):
    pass


class JogLimitError(ValueError):
    pass


class ActionLimitError(ValueError):
    pass


class YAMDriver(ABC):
    """Hardware boundary used by APIs, teleoperation, and inference loops."""

    @abstractmethod
    def list_arms(self) -> list[ArmTelemetry]:
        """Return a point-in-time snapshot for all configured arms."""

    @abstractmethod
    def get_arm(self, arm_id: str) -> ArmTelemetry:
        """Return a point-in-time snapshot for one arm."""

    @abstractmethod
    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry:
        """Apply a validated, relative manual command and return new state."""

    @abstractmethod
    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry:
        """Apply absolute actuator targets and return the resulting state."""
