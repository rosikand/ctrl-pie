from __future__ import annotations

import asyncio
import re
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import get_db
from ctrl_pi.deployments import (
    DeploymentConfigurationError,
    DeploymentConflictError,
    DeploymentNotFoundError,
    DeploymentProviderError,
    DeploymentRecord,
    DeploymentService,
    DeploymentStorageError,
)
from ctrl_pi.inference_sessions import (
    InferenceSessionConfigurationError,
    InferenceSessionConflictError,
    InferenceSessionManager,
    InferenceSessionRuntimeError,
    InferenceSessionSnapshot,
    InferenceSessionStorageError,
    InferenceSessionUnavailableError,
    InferenceStartOptions,
    InferenceStopOptions,
)
from ctrl_pi.models import AppSetting

router = APIRouter(prefix="/api/inference/deployments", tags=["inference"])

_REVISION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?")


class DeploymentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    model_repo: str = Field(min_length=3, max_length=255)
    checkpoint_revision: str | None = Field(default=None, max_length=128)
    runtime: Literal["stub", "lerobot", "openpi"] = "stub"
    compute_size: Literal[
        "CPU", "Modal: A10G", "Modal: A100", "Modal: H100"
    ] = "CPU"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("model_repo")
    @classmethod
    def validate_model_repo(cls, value: str) -> str:
        value = value.strip()
        if value.count("/") != 1:
            raise ValueError("model_repo must contain one namespace")
        try:
            from huggingface_hub.utils import HFValidationError, validate_repo_id

            validate_repo_id(value)
        except HFValidationError as error:
            raise ValueError("model_repo is not a valid Hugging Face repo ID") from error
        return value

    @field_validator("checkpoint_revision")
    @classmethod
    def validate_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if (
            not value
            or not _REVISION.fullmatch(value)
            or ".." in value
            or "//" in value
            or "--" in value
        ):
            raise ValueError("checkpoint_revision is not a safe Hub revision")
        return value

    @model_validator(mode="after")
    def validate_runtime_compute_pair(self) -> DeploymentCreate:
        if self.runtime == "stub" and self.compute_size != "CPU":
            raise ValueError("the stub runtime requires CPU compute")
        if self.runtime != "stub" and self.compute_size == "CPU":
            raise ValueError("the inference runtime requires a Modal GPU")
        return self


class DeploymentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    endpoint_id: uuid.UUID
    name: str
    target_kind: Literal["stub", "modal"]
    status: Literal[
        "created", "deploying", "running", "stopping", "stopped", "failed"
    ]
    model_repo: str
    checkpoint_revision: str | None
    runtime: Literal["stub", "lerobot", "openpi"]
    compute_size: Literal[
        "CPU", "Modal: A10G", "Modal: A100", "Modal: H100"
    ]
    timeout_seconds: int = Field(ge=1, le=1800)
    endpoint_url: str | None
    provider_app_id: str | None
    arm_id: str | None
    record_session: bool
    recording_id: uuid.UUID | None
    started_at: datetime | None
    stopped_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DeploymentsRead(BaseModel):
    deployments: list[DeploymentRead]


class InferenceRecordingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    enabled: bool
    status: Literal[
        "disabled", "starting", "recording", "finalizing", "ready", "failed"
    ]
    recording_id: uuid.UUID | None
    episode_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    hf_repo_id: str | None


class InferenceStateRead(DeploymentRead):
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
    recording: InferenceRecordingRead


class InferenceStartMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("operator", "notes")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if "\x00" in value:
            raise ValueError("metadata contains an invalid control character")
        return value or None


class InferenceStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_id: str = Field(min_length=1, max_length=120)
    task: str = Field(min_length=1, max_length=512)
    record_session: bool = False
    recording_name: str | None = Field(default=None, max_length=160)
    recording_metadata: InferenceStartMetadata | None = None

    @field_validator("arm_id", "task", "recording_name")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("value must not be blank or contain NUL")
        return value

    @model_validator(mode="after")
    def recording_fields_require_recording(self) -> InferenceStart:
        if not self.record_session and (
            self.recording_name is not None or self.recording_metadata is not None
        ):
            raise ValueError("recording fields require record_session=true")
        return self


class InferenceStop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_success: bool = True
    recording_notes: str | None = Field(default=None, max_length=2000)

    @field_validator("recording_notes")
    @classmethod
    def strip_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if "\x00" in value:
            raise ValueError("recording notes contain an invalid control character")
        return value or None


class InferenceStreamFrame(BaseModel):
    type: Literal["inference_state"] = "inference_state"
    timestamp: datetime
    state: InferenceStateRead


def get_deployment_service(request: Request) -> DeploymentService:
    return request.app.state.deployment_service


def get_inference_session_manager(request: Request) -> InferenceSessionManager:
    return request.app.state.inference_session_manager


def _timeout_seconds(db: Session) -> int:
    try:
        setting = db.get(AppSetting, "modal_timeout_minutes")
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL could not read deployment settings.",
        ) from None
    value = 30 if setting is None else setting.value
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 30:
        value = 30
    return value * 60


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, DeploymentNotFoundError):
        return HTTPException(status_code=404, detail="Deployment was not found.")
    if isinstance(error, DeploymentConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, InferenceSessionConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, InferenceSessionConfigurationError):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, InferenceSessionStorageError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, InferenceSessionUnavailableError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, InferenceSessionRuntimeError):
        return HTTPException(status_code=502, detail=str(error))
    if isinstance(error, (DeploymentConfigurationError, DeploymentStorageError)):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(
        status_code=502,
        detail="The compute target operation failed safely.",
    )


def _state_read(snapshot: InferenceSessionSnapshot) -> InferenceStateRead:
    deployment = DeploymentRead.model_validate(snapshot.deployment).model_dump()
    return InferenceStateRead(
        **deployment,
        session_status=snapshot.session_status,
        endpoint_healthy=snapshot.endpoint_healthy,
        teardown_verified=snapshot.teardown_verified,
        steps_executed=snapshot.steps_executed,
        requests_completed=snapshot.requests_completed,
        dropped_chunks=snapshot.dropped_chunks,
        queue_depth=snapshot.queue_depth,
        last_latency_ms=snapshot.last_latency_ms,
        average_latency_ms=snapshot.average_latency_ms,
        frequency_hz=snapshot.frequency_hz,
        last_error=snapshot.last_error,
        session_started_at=snapshot.session_started_at,
        session_stopped_at=snapshot.session_stopped_at,
        recording=InferenceRecordingRead.model_validate(snapshot.recording),
    )


@router.post("", response_model=DeploymentRead, status_code=201)
async def deploy(
    payload: DeploymentCreate,
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[DeploymentService, Depends(get_deployment_service)],
    config: Annotated[AppConfig, Depends(get_config)],
) -> DeploymentRecord:
    if payload.runtime == "openpi" and not config.mock_mode:
        raise HTTPException(
            status_code=422,
            detail="The real OpenPI runtime is not available in V1.",
        )
    try:
        return await service.deploy(
            db,
            name=payload.name,
            model_repo=payload.model_repo,
            checkpoint_revision=payload.checkpoint_revision,
            runtime=payload.runtime,
            compute_size=payload.compute_size,
            timeout_seconds=_timeout_seconds(db),
        )
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
    ) as error:
        raise _http_error(error) from None


@router.get("", response_model=DeploymentsRead)
def list_deployments(
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[DeploymentService, Depends(get_deployment_service)],
) -> DeploymentsRead:
    try:
        return DeploymentsRead(
            deployments=[
                DeploymentRead.model_validate(record)
                for record in service.list(db)
            ]
        )
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
    ) as error:
        raise _http_error(error) from None


@router.get("/{deployment_id}", response_model=DeploymentRead)
def detail(
    deployment_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[DeploymentService, Depends(get_deployment_service)],
) -> DeploymentRecord:
    try:
        return service.get(db, deployment_id)
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
    ) as error:
        raise _http_error(error) from None


@router.post("/{deployment_id}/start", response_model=InferenceStateRead)
async def start_inference(
    deployment_id: uuid.UUID,
    payload: InferenceStart,
    db: Annotated[Session, Depends(get_db)],
    manager: Annotated[
        InferenceSessionManager, Depends(get_inference_session_manager)
    ],
) -> InferenceStateRead:
    try:
        setting = db.get(AppSetting, "recording_fps")
        fps = None if setting is None else setting.value
        if fps is not None and (
            isinstance(fps, bool)
            or not isinstance(fps, int)
            or not 1 <= fps <= 60
        ):
            raise HTTPException(
                status_code=503,
                detail="The recording FPS setting is invalid.",
            )
        options = InferenceStartOptions(
            arm_id=payload.arm_id,
            task=payload.task,
            record_session=payload.record_session,
            recording_name=payload.recording_name,
            recording_metadata=(
                None
                if payload.recording_metadata is None
                else payload.recording_metadata.model_dump(exclude_none=True)
            ),
        )
        return _state_read(
            await manager.start(
                db,
                deployment_id,
                options,
                recording_fps=fps,
            )
        )
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
        InferenceSessionConflictError,
        InferenceSessionConfigurationError,
        InferenceSessionRuntimeError,
        InferenceSessionStorageError,
        InferenceSessionUnavailableError,
    ) as error:
        raise _http_error(error) from None
    except (SQLAlchemyError, TypeError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL could not read inference settings.",
        ) from None


@router.get("/{deployment_id}/state", response_model=InferenceStateRead)
async def inference_state(
    deployment_id: uuid.UUID,
    manager: Annotated[
        InferenceSessionManager, Depends(get_inference_session_manager)
    ],
) -> InferenceStateRead:
    try:
        return _state_read(await manager.read(deployment_id))
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
        InferenceSessionConflictError,
        InferenceSessionConfigurationError,
        InferenceSessionRuntimeError,
        InferenceSessionStorageError,
        InferenceSessionUnavailableError,
    ) as error:
        raise _http_error(error) from None


@router.post("/{deployment_id}/stop", response_model=InferenceStateRead)
async def stop(
    deployment_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    manager: Annotated[
        InferenceSessionManager, Depends(get_inference_session_manager)
    ],
    payload: InferenceStop | None = None,
) -> InferenceStateRead:
    try:
        payload = payload or InferenceStop()
        return _state_read(
            await manager.stop(
                db,
                deployment_id,
                InferenceStopOptions(
                    recording_success=payload.recording_success,
                    recording_notes=payload.recording_notes,
                ),
            )
        )
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
        InferenceSessionConflictError,
        InferenceSessionConfigurationError,
        InferenceSessionRuntimeError,
        InferenceSessionStorageError,
        InferenceSessionUnavailableError,
    ) as error:
        raise _http_error(error) from None


@router.websocket("/{deployment_id}/stream")
async def stream_inference_state(
    websocket: WebSocket,
    deployment_id: uuid.UUID,
) -> None:
    await websocket.accept()
    manager: InferenceSessionManager = websocket.app.state.inference_session_manager
    try:
        while True:
            try:
                snapshot = await manager.read(deployment_id)
            except DeploymentNotFoundError:
                await websocket.send_json(
                    {"type": "error", "detail": "Deployment was not found."}
                )
                await websocket.close(code=1008)
                return
            except Exception:
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": "Inference state is temporarily unavailable.",
                    }
                )
                await websocket.close(code=1011)
                return
            frame = InferenceStreamFrame(
                timestamp=datetime.now(UTC),
                state=_state_read(snapshot),
            )
            await websocket.send_json(frame.model_dump(mode="json"))
            if snapshot.session_status in {"stopped", "failed"}:
                await websocket.close(code=1000)
                return
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
