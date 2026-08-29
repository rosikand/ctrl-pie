from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class YAMSetupConfig(SDKModel):
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


class YAMDiagnostic(SDKModel):
    status: Literal["connected", "configured", "missing", "error"]
    detail: str = Field(min_length=1, max_length=240)


class YAMSetupCandidate(SDKModel):
    id: str = Field(min_length=1, max_length=1_024)
    label: str = Field(min_length=1, max_length=160)


class YAMDiscovery(SDKModel):
    mode: Literal["mock", "hardware"]
    can_interfaces: list[YAMSetupCandidate] = Field(max_length=32)
    leader_ports: list[YAMSetupCandidate] = Field(max_length=32)
    suggested_config: YAMSetupConfig | None
    detail: str = Field(min_length=1, max_length=240)


class YAMPreflight(SDKModel):
    ready: bool
    calibration_ready: bool
    diagnostic: YAMDiagnostic


class YAMSetupStatus(SDKModel):
    mode: Literal["mock", "hardware"]
    state: Literal[
        "needs_setup", "awaiting_hardware", "ready_to_connect", "ready", "error"
    ]
    configured: bool
    saved: bool
    connected: bool
    calibration_ready: bool
    auto_restore: bool
    restored_on_boot: bool
    config: YAMSetupConfig | None
    diagnostic: YAMDiagnostic
    last_attempt_at: datetime | None
    last_connected_at: datetime | None
    requires_physical_validation: bool


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
    bitrate: int = Field(gt=0)
    tx_error_count: int | None
    rx_error_count: int | None


class ControlLoopTelemetry(SDKModel):
    target_frequency_hz: float = Field(gt=0)
    frequency_hz: float = Field(ge=0)
    cycle_time_ms: float = Field(ge=0)
    jitter_ms: float = Field(ge=0)
    dropped_cycles: int = Field(ge=0)


class ArmTelemetry(SDKModel):
    id: str
    name: str
    role: Literal["leader", "follower"]
    driver: str
    connected: bool
    timestamp: datetime
    joints: list[JointTelemetry]
    pose: EndEffectorPose
    gripper: GripperTelemetry
    can: CANTelemetry
    control_loop: ControlLoopTelemetry


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
