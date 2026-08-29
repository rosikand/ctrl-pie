from __future__ import annotations

import copy
import math
import re
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from ctrl_pi.api.trainer import (
    CheckpointRead,
    ConsoleLogsRead,
    MetricPointRead,
    list_console_logs,
)
from ctrl_pi.db import get_db
from ctrl_pi.managed_training import (
    ManagedTrainingConflictError,
    ManagedTrainingLaunch,
    ManagedTrainingManager,
    ManagedTrainingNotFoundError,
    ManagedTrainingRecord,
    ManagedTrainingStorageError,
    parse_managed_training_cursor,
)
from ctrl_pi.models import TrainingRun
from ctrl_pi.training_compute import ManagedTrainingComputeSize

router = APIRouter(prefix="/api/trainer/jobs", tags=["managed-training"])

_REVISION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?")
_MAX_STEP = 2_147_483_647
_IMMUTABLE_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class ManagedTrainingLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    idempotency_key: uuid.UUID
    name: str = Field(min_length=1, max_length=160)
    dataset_repo: str = Field(min_length=3, max_length=255)
    dataset_revision: str | None = Field(default=None, min_length=1, max_length=128)
    base_model: str = Field(min_length=3, max_length=255)
    base_model_revision: str | None = Field(default=None, min_length=1, max_length=128)
    output_model_repo: str = Field(min_length=3, max_length=255)
    output_private: bool = Field(default=True, strict=True)
    acknowledge_public_model_risk: bool = Field(default=False, strict=True)
    acknowledge_compute_cost: bool = Field(default=False, strict=True)
    runtime: Literal["lerobot"] = "lerobot"
    compute_size: ManagedTrainingComputeSize
    max_steps: int = Field(ge=1, le=_MAX_STEP, strict=True)
    batch_size: int = Field(default=8, ge=1, le=4096, strict=True)
    log_every: int = Field(default=10, ge=1, le=_MAX_STEP, strict=True)
    save_every: int = Field(default=1000, ge=1, le=_MAX_STEP, strict=True)
    seed: int = Field(default=42, ge=0, le=_MAX_STEP, strict=True)
    num_workers: int = Field(default=4, ge=0, le=64, strict=True)
    timeout_minutes: int = Field(default=60, ge=1, le=1440, strict=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("dataset_repo", "base_model", "output_model_repo")
    @classmethod
    def validate_repo(cls, value: str) -> str:
        if value.strip() != value or value.count("/") != 1:
            raise ValueError("repository ID must contain exactly one namespace")
        try:
            from huggingface_hub.utils import HFValidationError, validate_repo_id

            validate_repo_id(value)
        except HFValidationError:
            raise ValueError("repository ID is invalid for Hugging Face") from None
        return value

    @field_validator("dataset_revision", "base_model_revision")
    @classmethod
    def validate_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            value.strip() != value
            or _REVISION.fullmatch(value) is None
            or ".." in value
            or "//" in value
            or "--" in value
        ):
            raise ValueError("revision is invalid for Hugging Face")
        return value

    @model_validator(mode="after")
    def validate_safety_and_storage_bounds(self) -> ManagedTrainingLaunchRequest:
        if not self.acknowledge_compute_cost:
            raise ValueError("acknowledge_compute_cost must be true")
        if not self.output_private and not self.acknowledge_public_model_risk:
            raise ValueError(
                "acknowledge_public_model_risk must be true for a public output"
            )
        if self.log_every > self.max_steps:
            raise ValueError("log_every must not exceed max_steps")
        if math.ceil(self.max_steps / self.log_every) * 64 > 10_000:
            raise ValueError("log interval would exceed managed metric storage")
        if self.save_every > self.max_steps:
            raise ValueError("save_every must not exceed max_steps")
        if math.ceil(self.max_steps / self.save_every) > 512:
            raise ValueError("save interval would exceed managed checkpoint storage")
        return self

    def launch(self) -> ManagedTrainingLaunch:
        return ManagedTrainingLaunch(**self.model_dump())


class ManagedTrainingJobRead(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str
    training_run_id: str
    idempotency_key: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal[
        "created",
        "launching",
        "running",
        "finalizing",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
    ]
    outcome: Literal["pending", "succeeded", "failed", "cancelled"]
    target_kind: Literal["stub", "modal"]
    provider_state: Literal[
        "pending",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "stopping",
        "stopped",
        "unknown",
    ]
    compute_size: ManagedTrainingComputeSize
    runtime: Literal["lerobot"]
    dataset_repo: str = Field(min_length=3, max_length=255)
    requested_dataset_revision: str | None = Field(max_length=128)
    dataset_revision: _IMMUTABLE_SHA | None
    base_model: str = Field(min_length=3, max_length=255)
    requested_base_model_revision: str | None = Field(max_length=128)
    base_model_revision: _IMMUTABLE_SHA | None
    output_model_repo: str = Field(min_length=3, max_length=255)
    output_private: bool
    output_marker_revision: _IMMUTABLE_SHA | None
    output_revision: _IMMUTABLE_SHA | None
    max_steps: int
    batch_size: int
    log_every: int
    save_every: int
    seed: int
    num_workers: int
    timeout_seconds: int
    deadline_at: datetime
    provider_app_id: str | None = Field(min_length=1, max_length=255)
    provider_function_call_id: str | None = Field(
        min_length=1, max_length=255
    )
    last_event_sequence: int
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


class ManagedTrainingJobsRead(BaseModel):
    jobs: list[ManagedTrainingJobRead]
    next_cursor: str | None


class ManagedTrainingMetricsRead(BaseModel):
    job_id: str
    training_run_id: str
    current_step: int
    metrics: dict[str, list[MetricPointRead]]


class ManagedTrainingCheckpointsRead(BaseModel):
    job_id: str
    training_run_id: str
    checkpoints: list[CheckpointRead]


def _manager(request: Request) -> ManagedTrainingManager:
    manager = getattr(request.app.state, "managed_training_manager", None)
    if not isinstance(manager, ManagedTrainingManager):
        raise HTTPException(
            status_code=503, detail="Managed training is not configured."
        )
    return manager


def _read(record: ManagedTrainingRecord) -> ManagedTrainingJobRead:
    return ManagedTrainingJobRead(
        **{
            **record.__dict__,
            "id": str(record.id),
            "training_run_id": str(record.training_run_id),
            "idempotency_key": str(record.idempotency_key),
            "teardown_verified": record.teardown_verified,
        }
    )


def _record_or_http(
    operation: Any,
) -> Any:
    try:
        return operation()
    except ManagedTrainingNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ManagedTrainingConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except ManagedTrainingStorageError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None


@router.post("", response_model=ManagedTrainingJobRead, status_code=202)
async def launch_managed_training(
    payload: ManagedTrainingLaunchRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> ManagedTrainingJobRead:
    manager = _manager(request)
    try:
        record = await manager.create(db, payload.launch())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except ManagedTrainingConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except ManagedTrainingStorageError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    response.headers["Cache-Control"] = "private, no-store"
    return _read(record)


@router.get("", response_model=ManagedTrainingJobsRead)
def list_managed_training_jobs(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    db: Session = Depends(get_db),
) -> ManagedTrainingJobsRead:
    before_at = None
    before_id = None
    if cursor is not None:
        try:
            before_at, before_id = parse_managed_training_cursor(cursor)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
    page = _manager(request).list(
        db,
        limit=limit,
        before_created_at=before_at,
        before_id=before_id,
    )
    response.headers["Cache-Control"] = "private, no-store"
    return ManagedTrainingJobsRead(
        jobs=[_read(record) for record in page.jobs],
        next_cursor=page.next_cursor,
    )


@router.get("/{job_id}", response_model=ManagedTrainingJobRead)
def get_managed_training_job(
    job_id: uuid.UUID,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> ManagedTrainingJobRead:
    record = _record_or_http(lambda: _manager(request).get(db, job_id))
    response.headers["Cache-Control"] = "private, no-store"
    return _read(record)


@router.post("/{job_id}/cancel", response_model=ManagedTrainingJobRead, status_code=202)
async def cancel_managed_training_job(
    job_id: uuid.UUID,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> ManagedTrainingJobRead:
    manager = _manager(request)
    try:
        record = await manager.cancel(db, job_id)
    except ManagedTrainingNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ManagedTrainingStorageError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    response.headers["Cache-Control"] = "private, no-store"
    return _read(record)


@router.get("/{job_id}/logs", response_model=ConsoleLogsRead)
def list_managed_training_logs(
    job_id: uuid.UUID,
    request: Request,
    response: Response,
    after_sequence: Annotated[int | None, Query(ge=0, le=9_007_199_254_740_991)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 200,
    db: Session = Depends(get_db),
) -> ConsoleLogsRead:
    record = _record_or_http(lambda: _manager(request).get(db, job_id))
    return list_console_logs(
        record.training_run_id,
        response,
        after_sequence,
        limit,
        db,
    )


@router.get("/{job_id}/metrics", response_model=ManagedTrainingMetricsRead)
def get_managed_training_metrics(
    job_id: uuid.UUID,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> ManagedTrainingMetricsRead:
    record = _record_or_http(lambda: _manager(request).get(db, job_id))
    run = db.get(TrainingRun, record.training_run_id)
    if run is None:
        raise HTTPException(status_code=503, detail="Managed training run is unavailable.")
    response.headers["Cache-Control"] = "private, no-store"
    return ManagedTrainingMetricsRead(
        job_id=str(record.id),
        training_run_id=str(record.training_run_id),
        current_step=run.current_step,
        metrics=copy.deepcopy(run.metrics or {}),
    )


@router.get("/{job_id}/checkpoints", response_model=ManagedTrainingCheckpointsRead)
def list_managed_training_checkpoints(
    job_id: uuid.UUID,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> ManagedTrainingCheckpointsRead:
    record = _record_or_http(lambda: _manager(request).get(db, job_id))
    run = db.get(TrainingRun, record.training_run_id)
    if run is None:
        raise HTTPException(status_code=503, detail="Managed training run is unavailable.")
    try:
        checkpoints = [CheckpointRead.model_validate(item) for item in run.checkpoints or []]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=503, detail="Stored training checkpoints are invalid."
        ) from None
    response.headers["Cache-Control"] = "private, no-store"
    return ManagedTrainingCheckpointsRead(
        job_id=str(record.id),
        training_run_id=str(record.training_run_id),
        checkpoints=checkpoints,
    )


__all__ = [
    "ManagedTrainingCheckpointsRead",
    "ManagedTrainingJobRead",
    "ManagedTrainingJobsRead",
    "ManagedTrainingLaunchRequest",
    "ManagedTrainingMetricsRead",
    "router",
]
