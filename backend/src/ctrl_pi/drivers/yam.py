from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ArmRole = Literal["leader", "follower"]
TransportKind = Literal["socketcan", "serial"]
EndEffectorKind = Literal[
    "yam_teaching_handle", "linear_4310", "crank_4310", "gello", "none"
]
ArmControlState = Literal[
    "disconnected",
    "connecting",
    "gravity_comp",
    "position_control",
    "stopping",
    "error",
]
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
    bitrate: int | None = Field(default=None, gt=0)
    tx_error_count: int | None = Field(default=None, ge=0)
    rx_error_count: int | None = Field(default=None, ge=0)


class ControlLoopTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_frequency_hz: float = Field(gt=0.0, allow_inf_nan=False)
    frequency_hz: float = Field(ge=0.0, allow_inf_nan=False)
    cycle_time_ms: float = Field(ge=0.0, allow_inf_nan=False)
    jitter_ms: float = Field(ge=0.0, allow_inf_nan=False)
    dropped_cycles: int = Field(ge=0)
    source: str = Field(default="driver", min_length=1, max_length=64)


class TeachingHandleTelemetry(BaseModel):
    """Live handle input kept distinct from CAN adapter connectivity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reachable: bool
    trigger_position: float | None = Field(default=None, ge=0.0, le=1.0)
    buttons: list[bool] = Field(default_factory=list, max_length=8)
    range_status: Literal["not_tested", "healthy", "unhealthy"] = "not_tested"
    observed_minimum: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_maximum: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_warning: str | None = Field(default=None, max_length=240)


class ArmTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    role: ArmRole
    pair_id: str | None = None
    group_id: str | None = None
    side: str | None = None
    transport_kind: TransportKind | Literal["mock"] = "mock"
    stable_identity: str | None = None
    end_effector_kind: EndEffectorKind = "none"
    driver: str
    connected: bool
    control_state: ArmControlState = "disconnected"
    energized: bool = False
    holding: bool = False
    timestamp: datetime
    joints: list[JointTelemetry]
    pose: EndEffectorPose
    gripper: GripperTelemetry
    can: CANTelemetry | None = None
    control_loop: ControlLoopTelemetry
    handle: TeachingHandleTelemetry | None = None
    frame_map_active: bool = False
    soft_limits_active: bool = False
    warnings: list[str] = Field(default_factory=list, max_length=16)


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
_LOGICAL_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,118}[A-Za-z0-9])?")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class LegacyYAMSetupConfig(BaseModel):
    """Bounded, non-secret configuration for one YAM leader/follower rig."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    # Excluded from legacy serialization so existing V1.1 payloads remain
    # byte-for-byte compatible; the ordinary union still accepts tagged input.
    kind: Literal["legacy_pair"] = Field(default="legacy_pair", exclude=True)
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


# Public source compatibility for V1.1 callers. New code should use
# YAMCellConfig; the setup API accepts both shapes.
YAMSetupConfig = LegacyYAMSetupConfig


class YAMCellArmConfig(BaseModel):
    """One durable arm assignment in a YAM-specific cell."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    logical_id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    role: ArmRole
    pair_id: str | None = Field(default=None, min_length=1, max_length=120)
    group_id: str | None = Field(default=None, min_length=1, max_length=120)
    side: str | None = Field(default=None, min_length=1, max_length=64)
    transport_kind: TransportKind
    stable_identity: str = Field(min_length=1, max_length=256)
    end_effector_kind: EndEffectorKind
    frame_map_path: str | None = Field(default=None, max_length=1_024)
    soft_limits_path: str | None = Field(default=None, max_length=1_024)
    mujoco_xml_path: str | None = Field(default=None, max_length=1_024)
    calibration_id: str | None = Field(default=None, max_length=64)
    calibration_dir: str | None = Field(default=None, max_length=1_024)

    @field_validator("logical_id", "pair_id", "group_id")
    @classmethod
    def validate_logical_id(cls, value: str | None) -> str | None:
        if value is not None and _LOGICAL_ID.fullmatch(value) is None:
            raise ValueError("logical identifiers contain unsupported characters")
        return value

    @field_validator("stable_identity")
    @classmethod
    def validate_stable_identity(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("stable identity may not contain control characters")
        if value != value.strip():
            raise ValueError("stable identity may not contain surrounding whitespace")
        if len(value.encode("utf-8")) > 256:
            raise ValueError("stable identity is too long")
        if value.startswith("can") and value[3:].isdigit():
            raise ValueError("persist the USB-CAN adapter serial, not an ephemeral canN")
        return value

    @field_validator(
        "frame_map_path", "soft_limits_path", "mujoco_xml_path", "calibration_dir"
    )
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("hardware paths may not contain control characters")
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("hardware paths must be absolute and may not traverse parents")
        return str(path)

    @field_validator("calibration_id")
    @classmethod
    def validate_optional_calibration_id(cls, value: str | None) -> str | None:
        if value is not None and _CALIBRATION_ID.fullmatch(value) is None:
            raise ValueError("calibration_id contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_supported_shape(self) -> YAMCellArmConfig:
        if self.transport_kind == "socketcan":
            if self.stable_identity.startswith("/dev/"):
                raise ValueError(
                    "SocketCAN stable identity must be the USB adapter serial"
                )
            if self.stable_identity.startswith("can") and self.stable_identity[3:].isdigit():
                raise ValueError(
                    "persist the USB-CAN adapter serial, not an ephemeral canN"
                )
            if self.role == "leader" and self.end_effector_kind != "yam_teaching_handle":
                raise ValueError("SocketCAN leaders require yam_teaching_handle")
            if self.role == "follower" and self.end_effector_kind not in {
                "linear_4310",
                "crank_4310",
            }:
                raise ValueError("SocketCAN followers require a supported actuated gripper")
        elif self.role != "leader" or self.end_effector_kind != "gello":
            raise ValueError("serial transport is supported only for GELLO leaders")
        if self.transport_kind == "serial" and not self.stable_identity.startswith(
            "/dev/serial/by-id/"
        ):
            raise ValueError("serial stable identity must be a /dev/serial/by-id path")
        if self.role == "leader" and (
            self.frame_map_path is not None or self.soft_limits_path is not None
        ):
            raise ValueError("frame maps and soft limits belong to follower arms")
        if self.calibration_id is None and self.calibration_dir is not None:
            raise ValueError("calibration_dir requires calibration_id")
        if self.calibration_id is not None and self.calibration_dir is None:
            raise ValueError("calibration_id requires calibration_dir")
        return self


class YAMCellConfig(BaseModel):
    """Persisted topology for the one primary V1.2 YAM cell."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    kind: Literal["cell"] = "cell"
    name: str = Field(min_length=1, max_length=120)
    i2rt_root: str = Field(min_length=1, max_length=1_024)
    i2rt_commit: str = Field(min_length=40, max_length=40)
    arms: list[YAMCellArmConfig] = Field(default_factory=list, max_length=16)
    pair_ports: dict[str, int] = Field(default_factory=dict, max_length=16)

    @field_validator("i2rt_root")
    @classmethod
    def validate_i2rt_root(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("i2rt_root must be an absolute bounded path")
        return str(path)

    @field_validator("i2rt_commit")
    @classmethod
    def validate_i2rt_commit(cls, value: str) -> str:
        normalized = value.lower()
        if _GIT_COMMIT.fullmatch(normalized) is None:
            raise ValueError("i2rt_commit must be a full lowercase Git commit")
        return normalized

    @model_validator(mode="after")
    def validate_topology(self) -> YAMCellConfig:
        ids = [arm.logical_id for arm in self.arms]
        if len(ids) != len(set(ids)):
            raise ValueError("arm logical IDs must be unique")
        names = [arm.name.casefold() for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("arm names must be unique")
        identities = [
            (arm.transport_kind, arm.stable_identity.casefold()) for arm in self.arms
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("durable physical identities must be unique")

        pairs: dict[str, list[YAMCellArmConfig]] = {}
        for arm in self.arms:
            if arm.pair_id is not None:
                pairs.setdefault(arm.pair_id, []).append(arm)
        for pair_id, pair_arms in pairs.items():
            roles = [arm.role for arm in pair_arms]
            if sorted(roles) != ["follower", "leader"]:
                raise ValueError(
                    f"pair {pair_id!r} must contain exactly one leader and one follower"
                )
            sides = {arm.side for arm in pair_arms if arm.side is not None}
            if len(sides) > 1:
                raise ValueError(f"pair {pair_id!r} has conflicting side metadata")
            groups = {arm.group_id for arm in pair_arms if arm.group_id is not None}
            if len(groups) > 1:
                raise ValueError(f"pair {pair_id!r} has conflicting group metadata")

        if any(_LOGICAL_ID.fullmatch(pair_id) is None for pair_id in self.pair_ports):
            raise ValueError("pair port keys must be valid pair IDs")
        unknown_ports = set(self.pair_ports) - set(pairs)
        if unknown_ports:
            raise ValueError("pair ports may reference only configured pairs")
        ports = list(self.pair_ports.values())
        if any(isinstance(port, bool) or not 1_024 <= port <= 65_535 for port in ports):
            raise ValueError("pair ports must be integers from 1024 through 65535")
        if len(ports) != len(set(ports)):
            raise ValueError("pair ports must be unique")
        return self


YAMConfiguration = LegacyYAMSetupConfig | YAMCellConfig


class YAMDiscoveryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=1_024)
    label: str = Field(min_length=1, max_length=160)


class YAMDiscoveryDevice(BaseModel):
    """One OS-visible transport endpoint; role is never inferred here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport_kind: TransportKind
    stable_identity: str = Field(min_length=1, max_length=256)
    product: str | None = Field(default=None, max_length=160)
    runtime_interface: str | None = Field(default=None, max_length=200)
    link_state: Literal["up", "down", "unknown", "not_applicable"] = "unknown"
    duplicate_identity: bool = False


class YAMArmResolution(BaseModel):
    """Passive resolution of one configured identity to current runtime state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1, max_length=120)
    transport_kind: TransportKind
    stable_identity: str = Field(min_length=1, max_length=256)
    runtime_interface: str | None = Field(default=None, max_length=200)
    resolved: bool
    conflict: bool = False
    detail: str = Field(min_length=1, max_length=240)


class YAMDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["mock", "hardware"]
    devices: list[YAMDiscoveryDevice] = Field(default_factory=list, max_length=32)
    resolutions: list[YAMArmResolution] = Field(default_factory=list, max_length=16)
    # Deprecated V1.1 candidate lists remain during the serial-GELLO migration.
    can_interfaces: list[YAMDiscoveryCandidate] = Field(default_factory=list, max_length=32)
    leader_ports: list[YAMDiscoveryCandidate] = Field(default_factory=list, max_length=32)
    suggested_config: YAMConfiguration | None
    detail: str = Field(min_length=1, max_length=240)


class YAMArmPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1, max_length=120)
    ready: bool
    runtime_interface: str | None = Field(default=None, max_length=200)
    link_state: Literal["up", "down", "unknown", "not_applicable"] = "unknown"
    frame_map_status: Literal["identity", "active", "error", "not_applicable"] = (
        "not_applicable"
    )
    soft_limits_status: Literal["active", "missing", "error", "not_applicable"] = (
        "not_applicable"
    )
    handle_status: Literal["not_checked", "healthy", "unhealthy", "not_applicable"] = (
        "not_checked"
    )
    warnings: list[str] = Field(default_factory=list, max_length=16)
    diagnostic: YAMDriverDiagnostic


class YAMPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    calibration_ready: bool
    diagnostic: YAMDriverDiagnostic
    i2rt_ready: bool | None = None
    arms: list[YAMArmPreflight] = Field(default_factory=list, max_length=16)
    warnings: list[str] = Field(default_factory=list, max_length=32)


class YAMHandleRangeResult(BaseModel):
    """Result of a separately acknowledged, active CAN input range test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1, max_length=120)
    reachable: bool
    observed_minimum: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_maximum: float | None = Field(default=None, ge=0.0, le=1.0)
    healthy: bool
    detail: str = Field(min_length=1, max_length=240)


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

    def setup_config(self) -> LegacyYAMSetupConfig | YAMCellConfig | None:
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

    def preflight_setup(
        self, config: LegacyYAMSetupConfig | YAMCellConfig
    ) -> YAMPreflightResult:
        """Inspect a candidate configuration without opening hardware."""

        return YAMPreflightResult(
            ready=False,
            calibration_ready=False,
            diagnostic=YAMDriverDiagnostic(
                status="error",
                detail="YAM setup preflight is unavailable for this driver.",
            ),
        )

    def apply_setup(
        self, config: LegacyYAMSetupConfig | YAMCellConfig
    ) -> YAMDriverDiagnostic:
        """Select a validated setup while leaving device resources closed."""

        raise YAMDriverUnavailableError("YAM setup cannot be changed for this driver.")

    def reset_setup(self) -> YAMDriverDiagnostic:
        """Restore the driver's process configuration with resources closed."""

        self.shutdown()
        return self.diagnostic()

    def connect_arms(self, arm_ids: list[str] | None = None) -> list[ArmTelemetry]:
        """Connect selected logical arms after consent is enforced above the driver.

        Legacy drivers support only an all-arms startup. Cell drivers override
        this method to preserve partial-connect and per-arm fault isolation.
        """

        configured = self.list_arms()
        configured_ids = {arm.id for arm in configured}
        selected = configured_ids if arm_ids is None else set(arm_ids)
        if selected != configured_ids:
            raise YAMDriverUnavailableError(
                "This legacy YAM adapter can connect only its complete pair."
            )
        self.startup()
        return self.list_arms()

    def disconnect_arms(self, arm_ids: list[str] | None = None) -> list[ArmTelemetry]:
        """Disconnect selected logical arms; repeated calls must be safe."""

        configured = self.list_arms()
        configured_ids = {arm.id for arm in configured}
        selected = configured_ids if arm_ids is None else set(arm_ids)
        if selected != configured_ids:
            raise YAMDriverUnavailableError(
                "This legacy YAM adapter can disconnect only its complete pair."
            )
        self.shutdown()
        return self.list_arms()

    def check_handle_range(
        self, arm_id: str, *, duration_seconds: float = 10.0
    ) -> YAMHandleRangeResult:
        """Run the explicit active handle-input diagnostic, never discovery."""

        raise YAMDriverUnavailableError(
            "Teaching-handle range checking is unavailable for this driver."
        )

    def validate_teleop_pair(self, leader_id: str, follower_id: str) -> None:
        """Reject cross-pair routes before a recording writer can start."""

        leader = self.get_arm(leader_id)
        follower = self.get_arm(follower_id)
        if leader.role != "leader" or follower.role != "follower":
            raise ValueError("teleoperation requires one leader and one follower")
        if leader.pair_id is not None or follower.pair_id is not None:
            if leader.pair_id is None or leader.pair_id != follower.pair_id:
                raise ValueError("leader and follower must belong to the same pair")
            return
        arms = self.list_arms()
        if len(arms) != 2:
            raise ValueError("unpaired arms cannot be routed in a multi-arm cell")

    def prepare_teleop_action(
        self, leader_id: str, follower_id: str, action: ArmAction
    ) -> ArmAction:
        """Map and clamp a pair action at the driver safety boundary."""

        self.validate_teleop_pair(leader_id, follower_id)
        return action

    def safe_idle(self, arm_id: str) -> ArmTelemetry:
        """Stop position writes while retaining an explicitly visible live state."""

        return self.get_arm(arm_id)

    def latch_fault(self, arm_ids: list[str], detail: str) -> None:
        """Revoke future commands after a control transition becomes uncertain.

        Cell drivers override this to isolate the exact arm/pair.  The legacy
        fallback closes the complete adapter rather than allowing a new owner
        to acquire a command path whose safe-idle state was not confirmed.
        ``detail`` must already be sanitized for operator display.
        """

        del detail
        try:
            self.disconnect_arms(arm_ids)
        except Exception:
            self.shutdown()
        selected = set(arm_ids)
        if any(arm.id in selected and arm.connected for arm in self.list_arms()):
            raise YAMDriverUnavailableError(
                "The YAM command path could not be fault-latched safely."
            )

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
