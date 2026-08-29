from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    position_radians: float = Field(allow_inf_nan=False)
    velocity_radians_per_second: float = Field(allow_inf_nan=False)
    effort_newton_meters: float | None = Field(default=None, allow_inf_nan=False)
    temperature_celsius: float | None = Field(default=None, allow_inf_nan=False)


class EndEffectorPose(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(allow_inf_nan=False)
    roll_radians: float = Field(allow_inf_nan=False)
    pitch_radians: float = Field(allow_inf_nan=False)
    yaw_radians: float = Field(allow_inf_nan=False)


class GripperTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    position: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    velocity: float = Field(allow_inf_nan=False)
    force_newtons: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    is_closed: bool


class CANTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    interface: str
    state: Literal["active", "warning", "bus_off", "disconnected"]
    bitrate: int = Field(gt=0)
    tx_error_count: int | None = Field(default=None, ge=0)
    rx_error_count: int | None = Field(default=None, ge=0)


class ControlLoopTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_frequency_hz: float = Field(gt=0.0, allow_inf_nan=False)
    frequency_hz: float = Field(ge=0.0, allow_inf_nan=False)
    cycle_time_ms: float = Field(ge=0.0, allow_inf_nan=False)
    jitter_ms: float = Field(ge=0.0, allow_inf_nan=False)
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


class YAMDriverUnavailableError(RuntimeError):
    """A sanitized, operator-actionable hardware availability failure."""


class YAMDriverDiagnostic(BaseModel):
    """Public driver health without device paths or vendor exception payloads."""

    model_config = ConfigDict(frozen=True)

    status: Literal["connected", "configured", "missing", "error"]
    detail: str = Field(min_length=1, max_length=240)


_CAN_INTERFACE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,14})")
_CALIBRATION_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,118}[A-Za-z0-9])?")


class YAMSetupConfig(BaseModel):
    """Bounded, non-secret configuration for one YAM leader/follower rig."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    can_interface: str = Field(min_length=1, max_length=15)
    leader_port: str = Field(min_length=1, max_length=200)
    mujoco_xml_path: str = Field(min_length=1, max_length=1_024)
    leader_calibration_id: str = Field(min_length=1, max_length=64)
    leader_calibration_dir: str = Field(min_length=1, max_length=1_024)

    @field_validator("can_interface")
    @classmethod
    def validate_can_interface(cls, value: str) -> str:
        if _CAN_INTERFACE.fullmatch(value) is None or len(value.encode("utf-8")) > 15:
            raise ValueError("can_interface must be a Linux-safe interface name")
        return value

    @field_validator("leader_calibration_id")
    @classmethod
    def validate_calibration_id(cls, value: str) -> str:
        if _CALIBRATION_ID.fullmatch(value) is None:
            raise ValueError("leader_calibration_id contains unsupported characters")
        return value

    @field_validator("leader_port", "mujoco_xml_path", "leader_calibration_dir")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("hardware paths may not contain control characters")
        if len(value.encode("utf-8")) > 1_024:
            raise ValueError("hardware path is too long")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("hardware paths must be absolute")
        if ".." in path.parts:
            raise ValueError("hardware paths may not traverse parent directories")
        return str(path)

    @field_validator("leader_port")
    @classmethod
    def validate_leader_device_path(cls, value: str) -> str:
        if Path(value).parts[:2] != ("/", "dev"):
            raise ValueError("leader_port must select a device below /dev")
        return value


class YAMDiscoveryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=1_024)
    label: str = Field(min_length=1, max_length=160)


class YAMDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["mock", "hardware"]
    can_interfaces: list[YAMDiscoveryCandidate] = Field(max_length=32)
    leader_ports: list[YAMDiscoveryCandidate] = Field(max_length=32)
    suggested_config: YAMSetupConfig | None
    detail: str = Field(min_length=1, max_length=240)


class YAMPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    calibration_ready: bool
    diagnostic: YAMDriverDiagnostic


class YAMDriver(ABC):
    """Hardware boundary used by APIs, teleoperation, and inference loops."""

    def startup(self) -> None:
        """Open driver resources.

        Mock and test implementations need no resources, so lifecycle hooks are
        deliberately safe no-ops by default.
        """

    def shutdown(self) -> None:
        """Stop command writers and close resources; repeated calls are safe."""

    def diagnostic(self) -> YAMDriverDiagnostic:
        """Return a sanitized readiness summary from in-memory state only."""

        arms = self.list_arms()
        if arms and all(arm.connected for arm in arms):
            return YAMDriverDiagnostic(
                status="connected",
                detail=f"{self.__class__.__name__} is ready.",
            )
        return YAMDriverDiagnostic(
            status="missing",
            detail=f"{self.__class__.__name__} is not connected.",
        )

    def setup_config(self) -> YAMSetupConfig | None:
        """Return the selected non-secret setup, when this driver supports it."""

        return None

    def discover_setup(self) -> YAMDiscoveryResult:
        """Enumerate safe setup candidates without opening hardware."""

        return YAMDiscoveryResult(
            mode="hardware",
            can_interfaces=[],
            leader_ports=[],
            suggested_config=self.setup_config(),
            detail="YAM setup discovery is unavailable for this driver.",
        )

    def preflight_setup(self, config: YAMSetupConfig) -> YAMPreflightResult:
        """Inspect a candidate configuration without opening hardware."""

        return YAMPreflightResult(
            ready=False,
            calibration_ready=False,
            diagnostic=YAMDriverDiagnostic(
                status="error",
                detail="YAM setup preflight is unavailable for this driver.",
            ),
        )

    def apply_setup(self, config: YAMSetupConfig) -> YAMDriverDiagnostic:
        """Select a validated setup while leaving device resources closed."""

        raise YAMDriverUnavailableError("YAM setup cannot be changed for this driver.")

    def reset_setup(self) -> YAMDriverDiagnostic:
        """Restore the driver's process configuration with resources closed."""

        self.shutdown()
        return self.diagnostic()

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
