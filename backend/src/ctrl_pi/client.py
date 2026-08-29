from __future__ import annotations

import math
import re
from types import TracebackType
from typing import Any, Literal, TypeVar
from urllib.parse import quote
from uuid import UUID

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ctrl_pi._http import SafeHttpClient
from ctrl_pi.sdk_models import (
    ArmTelemetry,
    Checkpoint,
    ComputeSize,
    ConsoleLog,
    ConsoleLogPage,
    ConsoleLogSource,
    DatasetEpisode,
    DatasetEpisodes,
    DatasetPage,
    Deployment,
    HealthStatus,
    InferenceState,
    ModelPage,
    PublicSettings,
    Recording,
    RecordingState,
    RecordingUploadResult,
    RunStatus,
    Runtime,
    SDKModel,
    SystemStatus,
    TrainingRun,
    YAMDiscovery,
    YAMPreflight,
    YAMSetupConfig,
    YAMSetupStatus,
)


class CtrlPiError(RuntimeError):
    """Sanitized SDK configuration, request, API, or transport failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ArmsResponse(SDKModel):
    arms: list[ArmTelemetry]


class _RecordingsResponse(SDKModel):
    recordings: list[Recording]


class _RunsResponse(SDKModel):
    runs: list[TrainingRun]


class _DeploymentsResponse(SDKModel):
    deployments: list[Deployment]


class _SettingsUpdate(_InputModel):
    recording_fps: int | None = Field(default=None, ge=1, le=60)
    default_runtime: Literal["lerobot", "openpi"] | None = None
    default_compute: Literal["Modal: A10G", "Modal: A100", "Modal: H100"] | None = None
    modal_timeout_minutes: int | None = Field(default=None, ge=1, le=30)

    @model_validator(mode="after")
    def reject_nulls(self) -> _SettingsUpdate:
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("settings updates cannot be null")
        return self


class _RecordingCreate(_InputModel):
    name: str = Field(min_length=1, max_length=160)
    task: str = Field(min_length=1, max_length=2_000)
    leader_robot_id: str = Field(min_length=1, max_length=120)
    follower_robot_id: str = Field(min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "task", "leader_robot_id", "follower_robot_id")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("metadata")
    @classmethod
    def reject_reserved_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if {"episodes", "upload"}.intersection(value):
            raise ValueError("recording metadata contains a reserved key")
        return value


class _EpisodeMetadata(_InputModel):
    operator: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=2_000)

    @field_validator("operator", "notes")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if "\x00" in value:
            raise ValueError("metadata contains an invalid control character")
        return value or None


class _RunCreate(_InputModel):
    name: str = Field(min_length=1, max_length=160)
    status: RunStatus = "created"
    current_step: int = Field(default=0, ge=0, le=2_147_483_647)
    dataset_repo: str | None = Field(default=None, max_length=255)
    base_model: str | None = Field(default=None, max_length=255)
    runtime: str | None = Field(default=None, max_length=64)
    framework: str | None = Field(default=None, max_length=64)
    output_model_repo: str | None = Field(default=None, max_length=255)
    checkpoint_revision: str | None = Field(default=None, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)


class _MetricLog(_InputModel):
    step: int = Field(ge=0, le=2_147_483_647)
    metrics: dict[str, float]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if not 1 <= len(value) <= 64:
            raise ValueError("metrics must contain 1-64 values")
        for name, number in value.items():
            if not re.fullmatch(r"[A-Za-z0-9_.\-/]{1,64}", name):
                raise ValueError("metric name is invalid")
            if not math.isfinite(number):
                raise ValueError("metric value must be finite")
        return value


class _DeploymentCreate(_InputModel):
    name: str = Field(min_length=1, max_length=160)
    model_repo: str = Field(min_length=3, max_length=255)
    checkpoint_revision: str | None = Field(default=None, max_length=128)
    runtime: Runtime = "stub"
    compute_size: ComputeSize = "CPU"

    @model_validator(mode="after")
    def runtime_matches_compute(self) -> _DeploymentCreate:
        if (self.runtime == "stub") != (self.compute_size == "CPU"):
            raise ValueError("runtime and compute size are incompatible")
        return self


class _InferenceStart(_InputModel):
    arm_id: str = Field(min_length=1, max_length=120)
    task: str = Field(min_length=1, max_length=512)
    record_session: bool = False
    recording_name: str | None = Field(default=None, max_length=160)
    recording_metadata: _EpisodeMetadata | None = None

    @model_validator(mode="after")
    def recording_fields_match(self) -> _InferenceStart:
        if not self.record_session and (
            self.recording_name is not None or self.recording_metadata is not None
        ):
            raise ValueError("recording fields require recording")
        return self


_ModelT = TypeVar("_ModelT", bound=SDKModel)


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _UnsetType()
_REPO_NAME = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?")
_ARM_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,119})")
_HUB_SHA = re.compile(r"[0-9a-f]{40}")


class CtrlPiClient:
    """Typed synchronous client for the public ctrl-pi REST workflows."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = SafeHttpClient(
            base_url,
            timeout=timeout,
            error_type=CtrlPiError,
            _transport=_transport,
        )

    @property
    def is_closed(self) -> bool:
        return self._http.is_closed

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> CtrlPiClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # System and settings
    def health(self) -> HealthStatus:
        return self._get(HealthStatus, "/api/health")

    def get_system_status(self) -> SystemStatus:
        return self._get(SystemStatus, "/api/settings/status")

    def get_settings(self) -> PublicSettings:
        return self._get(PublicSettings, "/api/settings")

    def update_settings(
        self,
        *,
        recording_fps: int | _UnsetType = _UNSET,
        default_runtime: Literal["lerobot", "openpi"] | _UnsetType = _UNSET,
        default_compute: Literal[
            "Modal: A10G", "Modal: A100", "Modal: H100"
        ]
        | _UnsetType = _UNSET,
        modal_timeout_minutes: int | _UnsetType = _UNSET,
    ) -> PublicSettings:
        values = {
            "recording_fps": recording_fps,
            "default_runtime": default_runtime,
            "default_compute": default_compute,
            "modal_timeout_minutes": modal_timeout_minutes,
        }
        payload = {key: value for key, value in values.items() if value is not _UNSET}
        if not payload:
            raise CtrlPiError("At least one settings field is required.") from None
        return self._send_model(
            PublicSettings, "PATCH", "/api/settings", _SettingsUpdate, payload
        )

    # YAM setup and arms
    def get_yam_setup(self) -> YAMSetupStatus:
        return self._get(YAMSetupStatus, "/api/yam/setup")

    def discover_yam(self) -> YAMDiscovery:
        return self._post(YAMDiscovery, "/api/yam/setup/discover")

    def preflight_yam(self, config: YAMSetupConfig) -> YAMPreflight:
        config = self._yam_config(config)
        return self._post(
            YAMPreflight,
            "/api/yam/setup/preflight",
            json={"config": config.model_dump(mode="json")},
        )

    def save_yam_setup(
        self,
        config: YAMSetupConfig,
        *,
        auto_restore: bool,
        acknowledge_automatic_motion_risk: bool,
    ) -> YAMSetupStatus:
        config = self._yam_config(config)
        self._require_bool(auto_restore, "Auto-restore flag")
        self._require_bool(
            acknowledge_automatic_motion_risk, "Automatic-motion acknowledgement"
        )
        return self._post(
            YAMSetupStatus,
            "/api/yam/setup",
            method="PUT",
            json={
                "config": config.model_dump(mode="json"),
                "auto_restore": auto_restore,
                "acknowledge_automatic_motion_risk": acknowledge_automatic_motion_risk,
            },
        )

    def connect_yam(
        self, *, acknowledge_hardware_motion_risk: bool
    ) -> YAMSetupStatus:
        self._require_bool(
            acknowledge_hardware_motion_risk, "Hardware-motion acknowledgement"
        )
        return self._post(
            YAMSetupStatus,
            "/api/yam/setup/connect",
            json={
                "acknowledge_hardware_motion_risk": acknowledge_hardware_motion_risk
            },
        )

    def reset_yam_setup(self) -> YAMSetupStatus:
        return self._post(YAMSetupStatus, "/api/yam/setup", method="DELETE")

    def list_arms(self) -> list[ArmTelemetry]:
        return self._get(_ArmsResponse, "/api/arms").arms

    def get_arm(self, arm_id: str) -> ArmTelemetry:
        return self._get(ArmTelemetry, f"/api/arms/{self._component(arm_id, 'Arm ID')}")

    def jog_arm(
        self,
        arm_id: str,
        *,
        kind: Literal["joint", "cartesian", "gripper"],
        axis: str,
        delta: float,
    ) -> ArmTelemetry:
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
        ):
            raise CtrlPiError("Jog delta is invalid.") from None
        return self._post(
            ArmTelemetry,
            f"/api/arms/{self._component(arm_id, 'Arm ID')}/jog",
            json={"kind": kind, "axis": axis, "delta": delta},
        )

    # Recording and datasets
    def list_recordings(self) -> list[Recording]:
        return self._get(_RecordingsResponse, "/api/recordings").recordings

    def get_recording(self, recording_id: UUID | str) -> Recording:
        return self._get(Recording, f"/api/recordings/{self._uuid(recording_id, 'Recording')}")

    def create_recording(
        self,
        name: str,
        *,
        task: str,
        leader_robot_id: str,
        follower_robot_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Recording:
        return self._send_model(
            Recording,
            "POST",
            "/api/recordings",
            _RecordingCreate,
            {
                "name": name,
                "task": task,
                "leader_robot_id": leader_robot_id,
                "follower_robot_id": follower_robot_id,
                "metadata": {} if metadata is None else metadata,
            },
        )

    def get_recording_state(self, recording_id: UUID | str) -> RecordingState:
        return self._get(
            RecordingState,
            f"/api/recordings/{self._uuid(recording_id, 'Recording')}/state",
        )

    def start_teleop(self, recording_id: UUID | str) -> RecordingState:
        return self._post(
            RecordingState,
            f"/api/recordings/{self._uuid(recording_id, 'Recording')}/teleop/start",
        )

    def stop_teleop(self, recording_id: UUID | str) -> RecordingState:
        return self._post(
            RecordingState,
            f"/api/recordings/{self._uuid(recording_id, 'Recording')}/teleop/stop",
        )

    def start_episode(
        self,
        recording_id: UUID | str,
        *,
        operator: str | None = None,
        notes: str | None = None,
    ) -> RecordingState:
        metadata = self._input(
            _EpisodeMetadata, {"operator": operator, "notes": notes}, "episode"
        )
        return self._post(
            RecordingState,
            f"/api/recordings/{self._uuid(recording_id, 'Recording')}/episodes/start",
            json={"metadata": metadata.model_dump(mode="json", exclude_none=True)},
        )

    def stop_episode(
        self,
        recording_id: UUID | str,
        *,
        success: bool = True,
        notes: str | None = None,
    ) -> RecordingState:
        self._require_bool(success, "Episode success flag")
        return self._post(
            RecordingState,
            f"/api/recordings/{self._uuid(recording_id, 'Recording')}/episodes/stop",
            json={"success": success, "notes": notes},
        )

    def upload_recording(
        self,
        recording_id: UUID | str,
        *,
        repo_name: str,
        private: bool,
    ) -> RecordingUploadResult:
        self._require_bool(private, "Repository visibility flag")
        return self._post(
            RecordingUploadResult,
            f"/api/recordings/{self._uuid(recording_id, 'Recording')}/upload",
            json={"repo_name": self._repo_name(repo_name), "private": private},
        )

    def list_datasets(
        self,
        *,
        limit: int = 24,
        cursor: str | None = None,
        refresh: bool = False,
    ) -> DatasetPage:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise CtrlPiError("Dataset page limit is invalid.") from None
        params: dict[str, Any] = {"limit": limit, "refresh": refresh}
        if cursor is not None:
            params["cursor"] = self._bounded_query(cursor, "Dataset cursor", 1_024)
        return self._get(DatasetPage, "/api/datasets", params=params)

    def list_dataset_episodes(self, repo_name: str) -> DatasetEpisodes:
        return self._get(
            DatasetEpisodes, f"/api/datasets/{self._repo_name(repo_name)}/episodes"
        )

    def get_dataset_episode(
        self, repo_name: str, episode_index: int, *, revision: str
    ) -> DatasetEpisode:
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise CtrlPiError("Dataset episode index is invalid.") from None
        return self._get(
            DatasetEpisode,
            f"/api/datasets/{self._repo_name(repo_name)}/episodes/{episode_index}",
            params={"revision": self._hub_sha(revision)},
        )

    # Models and externally reported training
    def list_models(self, *, refresh: bool = False) -> ModelPage:
        return self._get(ModelPage, "/api/models", params={"refresh": refresh})

    def create_run(
        self,
        name: str,
        *,
        status: RunStatus = "created",
        current_step: int = 0,
        dataset_repo: str | None = None,
        base_model: str | None = None,
        runtime: str | None = None,
        framework: str | None = None,
        output_model_repo: str | None = None,
        checkpoint_revision: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingRun:
        return self._send_model(
            TrainingRun,
            "POST",
            "/api/trainer/runs",
            _RunCreate,
            {
                "name": name,
                "status": status,
                "current_step": current_step,
                "dataset_repo": dataset_repo,
                "base_model": base_model,
                "runtime": runtime,
                "framework": framework,
                "output_model_repo": output_model_repo,
                "checkpoint_revision": checkpoint_revision,
                "config": {} if config is None else config,
            },
        )

    def list_runs(self, *, status: RunStatus | None = None) -> list[TrainingRun]:
        params = None if status is None else {"status": status}
        return self._get(_RunsResponse, "/api/trainer/runs", params=params).runs

    def get_run(self, run_id: UUID | str) -> TrainingRun:
        return self._get(TrainingRun, f"/api/trainer/runs/{self._uuid(run_id, 'Training run')}")

    def update_run(
        self,
        run_id: UUID | str,
        *,
        status: RunStatus | _UnsetType = _UNSET,
        current_step: int | _UnsetType = _UNSET,
        dataset_repo: str | None | _UnsetType = _UNSET,
        base_model: str | None | _UnsetType = _UNSET,
        runtime: str | None | _UnsetType = _UNSET,
        framework: str | None | _UnsetType = _UNSET,
        output_model_repo: str | None | _UnsetType = _UNSET,
        checkpoint_revision: str | None | _UnsetType = _UNSET,
        config: dict[str, Any] | _UnsetType = _UNSET,
    ) -> TrainingRun:
        values = {
            "status": status,
            "current_step": current_step,
            "dataset_repo": dataset_repo,
            "base_model": base_model,
            "runtime": runtime,
            "framework": framework,
            "output_model_repo": output_model_repo,
            "checkpoint_revision": checkpoint_revision,
            "config": config,
        }
        payload = {key: value for key, value in values.items() if value is not _UNSET}
        if not payload:
            raise CtrlPiError("At least one training run field is required.") from None
        return self._patch(
            TrainingRun,
            f"/api/trainer/runs/{self._uuid(run_id, 'Training run')}",
            json=payload,
        )

    def log_metrics(
        self, run_id: UUID | str, *, step: int, metrics: dict[str, float]
    ) -> TrainingRun:
        payload = self._input(_MetricLog, {"step": step, "metrics": metrics}, "metrics")
        return self._post(
            TrainingRun,
            f"/api/trainer/runs/{self._uuid(run_id, 'Training run')}/metrics",
            json=payload.model_dump(mode="json"),
        )

    def list_console_logs(
        self,
        run_id: UUID | str,
        *,
        after_sequence: int | None = None,
        limit: int = 200,
    ) -> ConsoleLogPage:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise CtrlPiError("Console log page limit is invalid.") from None
        params: dict[str, Any] = {"limit": limit}
        if after_sequence is not None:
            if (
                isinstance(after_sequence, bool)
                or not isinstance(after_sequence, int)
                or after_sequence < 0
            ):
                raise CtrlPiError("Console sequence is invalid.") from None
            params["after_sequence"] = after_sequence
        return self._get(
            ConsoleLogPage,
            f"/api/trainer/runs/{self._uuid(run_id, 'Training run')}/logs",
            params=params,
        )

    def log_console(
        self,
        run_id: UUID | str,
        *,
        line: str,
        source: ConsoleLogSource = "stdout",
        step: int | None = None,
    ) -> ConsoleLog:
        return self._post(
            ConsoleLog,
            f"/api/trainer/runs/{self._uuid(run_id, 'Training run')}/logs",
            json={"source": source, "line": line, "step": step},
        )

    def register_checkpoint(
        self,
        run_id: UUID | str,
        *,
        repo_id: str,
        revision: str,
        step: int,
    ) -> TrainingRun:
        return self._post(
            TrainingRun,
            f"/api/trainer/runs/{self._uuid(run_id, 'Training run')}/checkpoints",
            json={"repo_id": repo_id, "revision": revision, "step": step},
        )

    # Inference
    def deploy(
        self,
        name: str,
        *,
        model_repo: str,
        checkpoint_revision: str | None = None,
        runtime: Runtime = "stub",
        compute_size: ComputeSize = "CPU",
    ) -> Deployment:
        return self._send_model(
            Deployment,
            "POST",
            "/api/inference/deployments",
            _DeploymentCreate,
            {
                "name": name,
                "model_repo": model_repo,
                "checkpoint_revision": checkpoint_revision,
                "runtime": runtime,
                "compute_size": compute_size,
            },
        )

    def list_deployments(self) -> list[Deployment]:
        return self._get(
            _DeploymentsResponse, "/api/inference/deployments"
        ).deployments

    def get_deployment(self, deployment_id: UUID | str) -> Deployment:
        return self._get(
            Deployment,
            f"/api/inference/deployments/{self._uuid(deployment_id, 'Deployment')}",
        )

    def get_inference_state(self, deployment_id: UUID | str) -> InferenceState:
        return self._get(
            InferenceState,
            f"/api/inference/deployments/{self._uuid(deployment_id, 'Deployment')}/state",
        )

    def start_inference(
        self,
        deployment_id: UUID | str,
        *,
        arm_id: str,
        task: str,
        record_session: bool = False,
        recording_name: str | None = None,
        recording_operator: str | None = None,
        recording_notes: str | None = None,
    ) -> InferenceState:
        self._require_bool(record_session, "Inference recording flag")
        metadata = None
        if recording_operator is not None or recording_notes is not None:
            metadata = {
                "operator": recording_operator,
                "notes": recording_notes,
            }
        payload = self._input(
            _InferenceStart,
            {
                "arm_id": arm_id,
                "task": task,
                "record_session": record_session,
                "recording_name": recording_name,
                "recording_metadata": metadata,
            },
            "inference start",
        )
        return self._post(
            InferenceState,
            f"/api/inference/deployments/{self._uuid(deployment_id, 'Deployment')}/start",
            json=payload.model_dump(mode="json", exclude_none=True),
        )

    def stop_inference(
        self,
        deployment_id: UUID | str,
        *,
        recording_success: bool = True,
        recording_notes: str | None = None,
    ) -> InferenceState:
        self._require_bool(recording_success, "Inference recording success flag")
        return self._post(
            InferenceState,
            f"/api/inference/deployments/{self._uuid(deployment_id, 'Deployment')}/stop",
            json={
                "recording_success": recording_success,
                "recording_notes": recording_notes,
            },
        )

    def _get(
        self,
        model: type[_ModelT],
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> _ModelT:
        return self._request(model, "GET", path, params=params)

    def _post(
        self,
        model: type[_ModelT],
        path: str,
        *,
        method: str = "POST",
        json: Any = _UNSET,
    ) -> _ModelT:
        return self._request(model, method, path, json=json)

    def _patch(self, model: type[_ModelT], path: str, *, json: Any) -> _ModelT:
        return self._request(model, "PATCH", path, json=json)

    def _request(
        self,
        model: type[_ModelT],
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = _UNSET,
    ) -> _ModelT:
        body = self._http.request(
            method,
            path,
            params=params,
            json=None if json is _UNSET else json,
            json_supplied=json is not _UNSET,
        )
        parsed: _ModelT | None = None
        try:
            parsed = model.model_validate_json(body)
        except (ValidationError, ValueError):
            pass
        if parsed is None:
            raise CtrlPiError("ctrl-pi returned an invalid response.") from None
        return parsed

    def _send_model(
        self,
        response_model: type[_ModelT],
        method: str,
        path: str,
        input_model: type[_InputModel],
        values: dict[str, Any],
    ) -> _ModelT:
        payload = self._input(input_model, values, "SDK")
        body: dict[str, Any] | None = None
        try:
            body = payload.model_dump(mode="json", exclude_unset=True)
        except Exception:
            pass
        if body is None:
            raise CtrlPiError("The SDK request is invalid.") from None
        return self._request(
            response_model,
            method,
            path,
            json=body,
        )

    @staticmethod
    def _input(
        model: type[_InputModel], values: dict[str, Any], label: str
    ) -> _InputModel:
        parsed: _InputModel | None = None
        try:
            parsed = model.model_validate(values)
        except (ValidationError, ValueError):
            pass
        if parsed is None:
            raise CtrlPiError(f"The {label} request is invalid.") from None
        return parsed

    @staticmethod
    def _uuid(value: UUID | str, label: str) -> str:
        parsed: UUID | None = None
        try:
            parsed = value if isinstance(value, UUID) else UUID(value)
        except (TypeError, ValueError, AttributeError):
            pass
        if parsed is None:
            raise CtrlPiError(f"{label} ID is invalid.") from None
        return str(parsed)

    @staticmethod
    def _component(value: str, label: str) -> str:
        if (
            not isinstance(value, str)
            or _ARM_ID.fullmatch(value) is None
        ):
            raise CtrlPiError(f"{label} is invalid.") from None
        return quote(value, safe="")

    @staticmethod
    def _repo_name(value: str) -> str:
        if (
            not isinstance(value, str)
            or _REPO_NAME.fullmatch(value) is None
            or ".." in value
            or "--" in value
        ):
            raise CtrlPiError("Dataset repository name is invalid.") from None
        return quote(value, safe="")

    @staticmethod
    def _bounded_query(value: str, label: str, maximum: int) -> str:
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= maximum
            or "\x00" in value
        ):
            raise CtrlPiError(f"{label} is invalid.") from None
        return value

    @staticmethod
    def _hub_sha(value: str) -> str:
        if not isinstance(value, str) or _HUB_SHA.fullmatch(value) is None:
            raise CtrlPiError("Dataset revision is invalid.") from None
        return value

    @staticmethod
    def _yam_config(value: YAMSetupConfig) -> YAMSetupConfig:
        if not isinstance(value, YAMSetupConfig):
            raise CtrlPiError("YAM setup config is invalid.") from None
        return value

    @staticmethod
    def _require_bool(value: bool, label: str) -> None:
        if not isinstance(value, bool):
            raise CtrlPiError(f"{label} is invalid.") from None


__all__ = ["CtrlPiClient", "CtrlPiError"]
