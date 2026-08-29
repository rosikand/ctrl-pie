from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
ConsoleLogSource = Literal["stdout", "stderr", "system"]
Runtime = Literal["stub", "lerobot", "openpi"]
ComputeSize = Literal["CPU", "Modal: A10G", "Modal: A100", "Modal: H100"]
ManagedTrainingComputeSize = Literal[
    "Modal: A10G",
    "Modal: A100",
    "Modal: 2xA100",
    "Modal: 4xA100",
    "Modal: 8xA100",
    "Modal: H100",
    "Modal: 2xH100",
    "Modal: 4xH100",
    "Modal: 8xH100",
]
ManagedTrainingJobStatus = Literal[
    "created",
    "launching",
    "running",
    "finalizing",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
]
ManagedTrainingOutcome = Literal["pending", "succeeded", "failed", "cancelled"]
ManagedTrainingProviderState = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "stopping",
    "stopped",
    "unknown",
]
ImmutableRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
RecordingStatus = Literal[
    "draft", "teleop", "recording", "ready", "uploading", "uploaded", "failed"
]
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


class SDKModel(BaseModel):
    """Strict, frozen response value returned by the ctrl-pi SDK."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class HealthStatus(SDKModel):
    status: Literal["ok"]
    mode: Literal["mock", "hardware"]


class ServiceStatus(SDKModel):
    id: Literal["postgres", "huggingface", "modal", "arms"]
    label: str
    status: Literal["connected", "configured", "missing", "error"]
    detail: str
    required: bool


class InferenceReadiness(SDKModel):
    mock_mode: bool
    hf_configured: bool
    modal_configured: bool
    modal_proxy_configured: bool


class SystemStatus(SDKModel):
    mode: Literal["mock", "hardware"]
    setup_complete: bool
    services: list[ServiceStatus]
    inference: InferenceReadiness


class PublicSettings(SDKModel):
    hf_namespace: str | None
    recording_fps: int = Field(ge=1, le=60)
    default_runtime: Literal["lerobot", "openpi"]
    default_compute: Literal["Modal: A10G", "Modal: A100", "Modal: H100"]
    modal_timeout_minutes: int = Field(ge=1, le=30)


_CAN_INTERFACE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,14})")
_CALIBRATION_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,118}[A-Za-z0-9])?")
_LOGICAL_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,118}[A-Za-z0-9])?")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class YAMSetupConfig(SDKModel):
    """Deprecated V1.1 serial-GELLO plus SocketCAN-follower topology."""

    can_interface: str = Field(min_length=1, max_length=15)
    leader_port: str = Field(min_length=1, max_length=200)
    mujoco_xml_path: str = Field(min_length=1, max_length=1_024)
    leader_calibration_id: str = Field(min_length=1, max_length=64)
    leader_calibration_dir: str = Field(min_length=1, max_length=1_024)

    @field_validator("can_interface")
    @classmethod
    def validate_can_interface(cls, value: str) -> str:
        if _CAN_INTERFACE.fullmatch(value) is None or len(value.encode()) > 15:
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
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("hardware paths must be absolute and bounded")
        return str(path)

    @field_validator("leader_port")
    @classmethod
    def validate_leader_port(cls, value: str) -> str:
        if Path(value).parts[:2] != ("/", "dev"):
            raise ValueError("leader_port must select a device below /dev")
        return value


class YAMCellArmConfig(SDKModel):
    """One durable physical-arm assignment in a YAM cell."""

    model_config = ConfigDict(str_strip_whitespace=True)

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
            if self.role == "leader" and self.end_effector_kind != "yam_teaching_handle":
                raise ValueError("SocketCAN leaders require yam_teaching_handle")
            if self.role == "follower" and self.end_effector_kind not in {
                "linear_4310",
                "crank_4310",
            }:
                raise ValueError(
                    "SocketCAN followers require a supported actuated gripper"
                )
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


class YAMCellConfig(SDKModel):
    """Stable topology for the one primary configurable YAM cell."""

    model_config = ConfigDict(str_strip_whitespace=True)

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
        if set(self.pair_ports) - set(pairs):
            raise ValueError("pair ports may reference only configured pairs")
        ports = list(self.pair_ports.values())
        if any(isinstance(port, bool) or not 1_024 <= port <= 65_535 for port in ports):
            raise ValueError("pair ports must be integers from 1024 through 65535")
        if len(ports) != len(set(ports)):
            raise ValueError("pair ports must be unique")
        return self


YAMConfiguration = YAMSetupConfig | YAMCellConfig


class YAMDiagnostic(SDKModel):
    status: Literal["connected", "configured", "missing", "error"]
    detail: str = Field(min_length=1, max_length=240)


class YAMSetupCandidate(SDKModel):
    id: str = Field(min_length=1, max_length=1_024)
    label: str = Field(min_length=1, max_length=160)


class YAMDiscoveryDevice(SDKModel):
    transport_kind: TransportKind
    stable_identity: str = Field(min_length=1, max_length=256)
    product: str | None = Field(default=None, max_length=160)
    runtime_interface: str | None = Field(default=None, max_length=200)
    link_state: Literal["up", "down", "unknown", "not_applicable"] = "unknown"
    duplicate_identity: bool = False


class YAMArmResolution(SDKModel):
    arm_id: str = Field(min_length=1, max_length=120)
    transport_kind: TransportKind
    stable_identity: str = Field(min_length=1, max_length=256)
    runtime_interface: str | None = Field(default=None, max_length=200)
    resolved: bool
    conflict: bool = False
    detail: str = Field(min_length=1, max_length=240)


class YAMDiscovery(SDKModel):
    mode: Literal["mock", "hardware"]
    devices: list[YAMDiscoveryDevice] = Field(default_factory=list, max_length=32)
    resolutions: list[YAMArmResolution] = Field(default_factory=list, max_length=16)
    can_interfaces: list[YAMSetupCandidate] = Field(default_factory=list, max_length=32)
    leader_ports: list[YAMSetupCandidate] = Field(default_factory=list, max_length=32)
    suggested_config: YAMConfiguration | None
    detail: str = Field(min_length=1, max_length=240)


class YAMArmPreflight(SDKModel):
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
    handle_status: Literal[
        "not_checked", "healthy", "unhealthy", "not_applicable"
    ] = "not_checked"
    warnings: list[str] = Field(default_factory=list, max_length=16)
    diagnostic: YAMDiagnostic


class YAMPreflight(SDKModel):
    ready: bool
    calibration_ready: bool
    diagnostic: YAMDiagnostic
    i2rt_ready: bool | None = None
    arms: list[YAMArmPreflight] = Field(default_factory=list, max_length=16)
    warnings: list[str] = Field(default_factory=list, max_length=32)


class YAMHandleRangeResult(SDKModel):
    arm_id: str = Field(min_length=1, max_length=120)
    reachable: bool
    observed_minimum: float | None = Field(default=None, ge=0, le=1)
    observed_maximum: float | None = Field(default=None, ge=0, le=1)
    healthy: bool
    detail: str = Field(min_length=1, max_length=240)


class YAMSetupArmStatus(SDKModel):
    arm_id: str
    role: ArmRole
    pair_id: str | None
    group_id: str | None
    side: str | None
    connected: bool
    control_state: ArmControlState
    energized: bool
    holding: bool
    runtime_interface: str | None
    error: str | None = None


class YAMSetupStatus(SDKModel):
    mode: Literal["mock", "hardware"]
    state: Literal[
        "needs_setup",
        "awaiting_hardware",
        "ready_to_connect",
        "partially_connected",
        "ready",
        "error",
    ]
    configured: bool
    saved: bool
    connected: bool
    any_connected: bool = False
    all_connected: bool = False
    configured_arm_count: int = Field(default=0, ge=0, le=16)
    connected_arm_count: int = Field(default=0, ge=0, le=16)
    arms: list[YAMSetupArmStatus] = Field(default_factory=list, max_length=16)
    calibration_ready: bool
    auto_restore: bool
    restored_on_boot: bool
    config: YAMConfiguration | None
    diagnostic: YAMDiagnostic
    last_attempt_at: datetime | None
    last_connected_at: datetime | None
    requires_physical_validation: bool


# Canonical V1.2 names. The legacy names remain aliases to the same strict
# wire models because both route families deliberately share one response
# contract during migration.
YAMCellDiscovery = YAMDiscovery
YAMCellPreflight = YAMPreflight
YAMCellStatus = YAMSetupStatus


class JointTelemetry(SDKModel):
    name: str
    position_radians: float
    velocity_radians_per_second: float
    effort_newton_meters: float | None
    temperature_celsius: float | None


class EndEffectorPose(SDKModel):
    x_m: float
    y_m: float
    z_m: float
    roll_radians: float
    pitch_radians: float
    yaw_radians: float


class GripperTelemetry(SDKModel):
    position: float = Field(ge=0, le=1)
    velocity: float
    force_newtons: float | None
    is_closed: bool


class CANTelemetry(SDKModel):
    interface: str
    state: Literal["active", "warning", "bus_off", "disconnected"]
    bitrate: int | None = Field(default=None, gt=0)
    tx_error_count: int | None
    rx_error_count: int | None


class ControlLoopTelemetry(SDKModel):
    target_frequency_hz: float = Field(gt=0)
    frequency_hz: float = Field(ge=0)
    cycle_time_ms: float = Field(ge=0)
    jitter_ms: float = Field(ge=0)
    dropped_cycles: int = Field(ge=0)
    source: str = Field(default="driver", min_length=1, max_length=64)


class TeachingHandleTelemetry(SDKModel):
    reachable: bool
    trigger_position: float | None = Field(default=None, ge=0, le=1)
    buttons: list[bool] = Field(default_factory=list, max_length=8)
    range_status: Literal["not_tested", "healthy", "unhealthy"] = "not_tested"
    observed_minimum: float | None = Field(default=None, ge=0, le=1)
    observed_maximum: float | None = Field(default=None, ge=0, le=1)
    calibration_warning: str | None = Field(default=None, max_length=240)


class ArmTelemetry(SDKModel):
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


class Recording(SDKModel):
    id: UUID
    name: str
    task: str
    status: RecordingStatus
    leader_robot_id: str
    follower_robot_id: str
    episode_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    hf_repo_id: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class RecordingState(SDKModel):
    recording_id: UUID
    teleop_active: bool
    sync_enabled: bool
    sync_in_progress: bool
    joint_deltas_radians: dict[str, float]
    episode_active: bool
    current_episode_index: int | None
    episode_duration_seconds: float = Field(ge=0)
    episode_count: int = Field(ge=0)
    status: RecordingStatus


class RecordingUploadResult(SDKModel):
    recording: Recording
    repo_id: str
    repo_url: str
    revision: str | None


class DatasetCard(SDKModel):
    title: str | None
    description: str | None
    license: str | None
    task_categories: list[str]


class LeRobotDatasetMetadata(SDKModel):
    codebase_version: str | None
    robot_type: str | None
    fps: float | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    features: list[str]


class Dataset(SDKModel):
    repo_id: str
    name: str
    revision: str | None
    hub_url: str
    private: bool
    gated: bool
    created_at: datetime | None
    last_modified: datetime | None
    tags: list[str]
    card: DatasetCard | None
    lerobot: LeRobotDatasetMetadata | None


class DatasetPage(SDKModel):
    namespace: str
    datasets: list[Dataset]
    total: int = Field(ge=0)
    next_cursor: str | None
    fetched_at: datetime


class EpisodeSummary(SDKModel):
    episode_index: int = Field(ge=0)
    tasks: list[str]
    frame_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    dataset_from_index: int = Field(ge=0)
    dataset_to_index: int = Field(ge=0)
    video_from_timestamp: float | None
    video_to_timestamp: float | None


class DatasetEpisodes(SDKModel):
    repo_id: str
    revision: str
    fps: float = Field(gt=0)
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    total_episodes: int = Field(ge=0)
    episodes: list[EpisodeSummary]


class TimelineFrame(SDKModel):
    timestamp: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    state: list[float]
    action: list[float]


class DatasetEpisode(SDKModel):
    repo_id: str
    revision: str
    fps: float = Field(gt=0)
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    episode: EpisodeSummary
    frames: list[TimelineFrame]
    sampled_frame_count: int = Field(ge=0)
    frames_truncated: bool
    video_url: str | None


class MetricPoint(SDKModel):
    step: int = Field(ge=0)
    value: float


class Checkpoint(SDKModel):
    repo_id: str
    revision: str
    step: int = Field(ge=0)


class ConsoleLog(SDKModel):
    sequence: int = Field(ge=1)
    source: ConsoleLogSource
    line: str
    step: int | None
    timestamp: datetime


class ConsoleLogPage(SDKModel):
    logs: list[ConsoleLog]
    oldest_sequence: int | None
    latest_sequence: int | None
    next_sequence: int = Field(ge=0)
    truncated: bool
    has_more: bool


class ManagedTrainingJobSummary(SDKModel):
    id: UUID
    status: ManagedTrainingJobStatus
    outcome: ManagedTrainingOutcome
    target_kind: Literal["stub", "modal"]
    compute_size: ManagedTrainingComputeSize
    deadline_at: datetime
    provider_state: ManagedTrainingProviderState
    teardown_verified: bool
    output_model_repo: str = Field(min_length=3, max_length=255)
    output_marker_revision: ImmutableRevision | None
    output_revision: ImmutableRevision | None
    last_error: str | None = Field(max_length=240)
    event_gap: bool


class TrainingRun(SDKModel):
    id: UUID
    name: str
    status: RunStatus
    current_step: int = Field(ge=0)
    dataset_repo: str | None
    base_model: str | None
    runtime: str | None
    framework: str | None
    output_model_repo: str | None
    checkpoint_revision: str | None
    config: dict[str, Any]
    metrics: dict[str, list[MetricPoint]]
    checkpoints: list[Checkpoint]
    managed_job: ManagedTrainingJobSummary | None = None
    created_at: datetime
    updated_at: datetime


class ManagedTrainingJob(SDKModel):
    id: UUID
    training_run_id: UUID
    idempotency_key: UUID
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ManagedTrainingJobStatus
    outcome: ManagedTrainingOutcome
    target_kind: Literal["stub", "modal"]
    provider_state: ManagedTrainingProviderState
    compute_size: ManagedTrainingComputeSize
    runtime: Literal["lerobot"]
    dataset_repo: str = Field(min_length=3, max_length=255)
    requested_dataset_revision: str | None = Field(max_length=128)
    dataset_revision: ImmutableRevision | None
    base_model: str = Field(min_length=3, max_length=255)
    requested_base_model_revision: str | None = Field(max_length=128)
    base_model_revision: ImmutableRevision | None
    output_model_repo: str = Field(min_length=3, max_length=255)
    output_private: bool
    output_marker_revision: ImmutableRevision | None
    output_revision: ImmutableRevision | None
    max_steps: int = Field(ge=1, le=2_147_483_647)
    batch_size: int = Field(ge=1, le=4_096)
    log_every: int = Field(ge=1)
    save_every: int = Field(ge=1)
    seed: int = Field(ge=0, le=2_147_483_647)
    num_workers: int = Field(ge=0, le=64)
    timeout_seconds: int = Field(ge=60, le=86_400)
    deadline_at: datetime
    provider_app_id: str | None = Field(min_length=1, max_length=255)
    provider_function_call_id: str | None = Field(min_length=1, max_length=255)
    last_event_sequence: int = Field(ge=0, le=9_007_199_254_740_991)
    event_gap: bool
    launch_attempted_at: datetime | None
    provider_launch_started_at: datetime | None
    started_at: datetime | None
    execution_finished_at: datetime | None
    cancel_requested_at: datetime | None
    teardown_verified: bool
    teardown_verified_at: datetime | None
    last_error: str | None = Field(max_length=240)
    created_at: datetime
    updated_at: datetime


class ManagedTrainingJobPage(SDKModel):
    jobs: list[ManagedTrainingJob]
    next_cursor: str | None


class ManagedTrainingMetrics(SDKModel):
    job_id: UUID
    training_run_id: UUID
    current_step: int = Field(ge=0, le=2_147_483_647)
    metrics: dict[str, list[MetricPoint]]


class ManagedTrainingCheckpoints(SDKModel):
    job_id: UUID
    training_run_id: UUID
    checkpoints: list[Checkpoint]


class ModelCard(SDKModel):
    description: str | None
    base_model: list[str]
    datasets: list[str]


class Model(SDKModel):
    repo_id: str
    name: str
    revision: str | None
    hub_url: str
    private: bool
    gated: bool
    last_modified: datetime | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    card: ModelCard | None
    checkpoints: list[str]


class ModelPage(SDKModel):
    namespace: str
    models: list[Model]
    total: int = Field(ge=0)
    fetched_at: datetime


class Deployment(SDKModel):
    id: UUID
    endpoint_id: UUID
    name: str
    target_kind: Literal["stub", "modal"]
    status: Literal[
        "created", "deploying", "running", "stopping", "stopped", "failed"
    ]
    model_repo: str
    checkpoint_revision: str | None
    runtime: Runtime
    compute_size: ComputeSize
    timeout_seconds: int = Field(ge=1, le=1800)
    endpoint_url: str | None
    provider_app_id: str | None
    arm_id: str | None
    record_session: bool
    recording_id: UUID | None
    started_at: datetime | None
    stopped_at: datetime | None
    created_at: datetime
    updated_at: datetime


class InferenceRecording(SDKModel):
    enabled: bool
    status: Literal[
        "disabled", "starting", "recording", "finalizing", "ready", "failed"
    ]
    recording_id: UUID | None
    episode_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    hf_repo_id: str | None


class InferenceState(Deployment):
    session_status: Literal[
        "idle", "starting", "running", "stopping", "stopped", "failed"
    ]
    endpoint_healthy: bool
    teardown_verified: bool
    steps_executed: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    dropped_chunks: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
    last_latency_ms: float | None = Field(default=None, ge=0)
    average_latency_ms: float | None = Field(default=None, ge=0)
    frequency_hz: float = Field(ge=0)
    last_error: str | None
    session_started_at: datetime | None
    session_stopped_at: datetime | None
    recording: InferenceRecording
