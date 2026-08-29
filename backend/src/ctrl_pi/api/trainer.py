from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.db import get_db
from ctrl_pi.models import ManagedTrainingJob, TrainingRun
from ctrl_pi.training_compute import ManagedTrainingComputeSize
from ctrl_pi.training_store import (
    ManagedTrainingRunMutationError,
    TrainingStoreCorruptError,
    TrainingStoreError,
    append_checkpoint,
    append_console_log,
    append_metrics,
    assert_external_run_mutable,
)

router = APIRouter(prefix="/api/trainer", tags=["trainer"])

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
ConsoleLogSource = Literal["stdout", "stderr", "system"]
_METRIC_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.\-/]{0,63})")
_REVISION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?")
_CONSOLE_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})"
    r"\s*[:=](?=\s*\S+)"
)
_SECRET_TOKEN = re.compile(
    r"(?i)(?:\bhf_[a-z0-9]{8,}\b|\b(?:ak|as|wk|ws)-[a-z0-9_-]{8,}\b|"
    r"://[^/\s:@]+:[^/\s@]+@|\bbearer\s+\S+)"
)
_CONSOLE_SENSITIVE_KEY_PARTS = {
    "authorization",
    "cookie",
    "credential",
    "dsn",
    "passwd",
    "password",
    "secret",
}
_CONSOLE_SENSITIVE_KEY_SUFFIXES = {
    "accesstoken",
    "apikey",
    "databaseurl",
    "dburl",
    "hftoken",
    "modaltoken",
    "privatekey",
}
_SENSITIVE_PARTS = {
    "auth",
    "authorization",
    "bearer",
    "cookie",
    "key",
    "oauth",
    "passwd",
    "session",
}
_SENSITIVE_KEY_FRAGMENTS = {
    "accesskey",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "cookie",
    "connectionstring",
    "credential",
    "databaseurl",
    "dburl",
    "dsn",
    "password",
    "passwd",
    "privatekey",
    "secret",
    "session",
}
_SAFE_TOKEN_CONFIG_KEYS = {
    "inputtokens",
    "maxnewtokens",
    "maxtokens",
    "mintokens",
    "numtokens",
    "numtraintokens",
    "outputtokens",
    "tokendropout",
    "tokenlength",
    "tokensperbatch",
    "totaltokens",
}
_SAFE_TOKENIZER_KEYS = {
    "tokenizer",
    "tokenizerclassname",
    "tokenizerconfig",
    "tokenizermaxlength",
    "tokenizername",
    "tokenizerpadding",
    "tokenizerpath",
    "tokenizertruncation",
    "tokenizertype",
    "tokenizerusefast",
    "tokenizervocabfile",
    "tokenizervocabsize",
}
_SAFE_KEY_CONFIG_KEYS = {
    "actionkey",
    "imagekey",
    "observationkey",
    "statekey",
    "videokey",
}
_MAX_STEP = 2_147_483_647
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_METRIC_NAMES = 128
_MAX_METRIC_POINTS = 10_000
_MAX_CHECKPOINTS = 512
_MAX_CONSOLE_LOGS = 1_000
_MAX_CONSOLE_LOG_BYTES = 512 * 1024
_MAX_CONSOLE_LINE_BYTES = 4 * 1024
_MAX_CONSOLE_SEQUENCE = 9_007_199_254_740_991
_IMMUTABLE_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def _contains_secret_assignment(value: str) -> bool:
    for match in _CONSOLE_ASSIGNMENT.finditer(value):
        key = match.group("key")
        separated_key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        separated_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated_key)
        parts = {
            part
            for part in re.split(r"[^a-z0-9]+", separated_key.casefold())
            if part
        }
        normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
        if parts.intersection(_CONSOLE_SENSITIVE_KEY_PARTS) or any(
            normalized.endswith(suffix)
            for suffix in _CONSOLE_SENSITIVE_KEY_SUFFIXES
        ):
            return True
    return False


class MetricPointRead(BaseModel):
    step: int
    value: float


class CheckpointRead(BaseModel):
    repo_id: str
    revision: str
    step: int


class ConsoleLogRead(BaseModel):
    sequence: int = Field(ge=1, le=_MAX_CONSOLE_SEQUENCE)
    source: ConsoleLogSource
    line: str = Field(min_length=1, max_length=_MAX_CONSOLE_LINE_BYTES)
    step: int | None = Field(default=None, ge=0, le=_MAX_STEP, strict=True)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("console timestamp must include a timezone")
        return value.astimezone(UTC)


class ManagedTrainingJobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str
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
    compute_size: ManagedTrainingComputeSize
    deadline_at: datetime
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
    teardown_verified: bool
    output_model_repo: str
    output_marker_revision: _IMMUTABLE_SHA | None
    output_revision: _IMMUTABLE_SHA | None
    last_error: str | None = Field(max_length=240)
    event_gap: bool


class TrainingRunRead(BaseModel):
    id: str
    name: str
    status: RunStatus
    current_step: int
    dataset_repo: str | None
    base_model: str | None
    runtime: str | None
    framework: str | None
    output_model_repo: str | None
    checkpoint_revision: str | None
    config: dict[str, Any]
    metrics: dict[str, list[MetricPointRead]]
    checkpoints: list[CheckpointRead]
    managed_job: ManagedTrainingJobSummary | None = None
    created_at: datetime
    updated_at: datetime


class TrainingRunsRead(BaseModel):
    runs: list[TrainingRunRead]


class ConsoleLogsRead(BaseModel):
    logs: list[ConsoleLogRead]
    oldest_sequence: int | None
    latest_sequence: int | None
    next_sequence: int
    truncated: bool
    has_more: bool


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    status: RunStatus = "created"
    current_step: int = Field(default=0, ge=0, le=_MAX_STEP)
    dataset_repo: str | None = Field(default=None, max_length=255)
    base_model: str | None = Field(default=None, max_length=255)
    runtime: str | None = Field(default=None, max_length=64)
    framework: str | None = Field(default=None, max_length=64)
    output_model_repo: str | None = Field(default=None, max_length=255)
    checkpoint_revision: str | None = Field(default=None, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("runtime", "framework")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("dataset_repo", "base_model", "output_model_repo")
    @classmethod
    def validate_repository(cls, value: str | None) -> str | None:
        return _repo_id(value)

    @field_validator("checkpoint_revision")
    @classmethod
    def validate_checkpoint_revision(cls, value: str | None) -> str | None:
        return _revision(value)

    @field_validator("config")
    @classmethod
    def validate_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _safe_config(value)


class RunUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus | None = None
    current_step: int | None = Field(default=None, ge=0, le=_MAX_STEP)
    dataset_repo: str | None = Field(default=None, max_length=255)
    base_model: str | None = Field(default=None, max_length=255)
    runtime: str | None = Field(default=None, max_length=64)
    framework: str | None = Field(default=None, max_length=64)
    output_model_repo: str | None = Field(default=None, max_length=255)
    checkpoint_revision: str | None = Field(default=None, max_length=128)
    config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def reject_null_required_updates(self) -> RunUpdate:
        for field in ("status", "current_step", "config"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self

    @field_validator("runtime", "framework")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("dataset_repo", "base_model", "output_model_repo")
    @classmethod
    def validate_repository(cls, value: str | None) -> str | None:
        return _repo_id(value)

    @field_validator("checkpoint_revision")
    @classmethod
    def validate_checkpoint_revision(cls, value: str | None) -> str | None:
        return _revision(value)

    @field_validator("config")
    @classmethod
    def validate_config(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return None if value is None else _safe_config(value)


class MetricLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0, le=_MAX_STEP)
    metrics: dict[str, float]

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict) or not 1 <= len(value) <= 64:
            raise ValueError("metrics must contain between 1 and 64 scalar values")
        result: dict[str, float] = {}
        for name, raw in value.items():
            if not isinstance(name, str) or not _METRIC_NAME.fullmatch(name):
                raise ValueError("metric names must be 1-64 safe characters")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("metric values must be numeric scalars")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError("metric values must be finite")
            result[name] = number
        return result


class CheckpointCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(min_length=3, max_length=255)
    revision: str = Field(min_length=1, max_length=128)
    step: int = Field(ge=0, le=_MAX_STEP)

    @field_validator("repo_id")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        result = _repo_id(value)
        if result is None:
            raise ValueError("repo_id is required")
        return result

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        result = _revision(value)
        if result is None:
            raise ValueError("revision is required")
        return result


class ConsoleLogCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: ConsoleLogSource = "stdout"
    line: str = Field(min_length=1, max_length=_MAX_CONSOLE_LINE_BYTES, strict=True)
    step: int | None = Field(default=None, ge=0, le=_MAX_STEP, strict=True)

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("console line must not be blank")
        if len(value.encode("utf-8")) > _MAX_CONSOLE_LINE_BYTES:
            raise ValueError("console line must be at most 4 KiB")
        if any(
            character != "\t" and unicodedata.category(character).startswith("C")
            for character in value
        ):
            raise ValueError("console line must contain printable single-line text")
        if _contains_secret_assignment(value) or _SECRET_TOKEN.search(value):
            raise ValueError("console line must not contain secret-like values")
        return value

def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _repo_id(value: str | None) -> str | None:
    value = _optional_text(value)
    if value is None:
        return None
    if value.count("/") != 1:
        raise ValueError("repository IDs must include exactly one namespace")
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        validate_repo_id(value)
    except HFValidationError as error:
        raise ValueError("repository ID is not valid for Hugging Face") from error
    return value


def _revision(value: str | None) -> str | None:
    value = _optional_text(value)
    if value is None:
        return None
    if (
        not _REVISION.fullmatch(value)
        or ".." in value
        or "//" in value
        or "--" in value
    ):
        raise ValueError("revision is not a valid Hugging Face revision")
    return value


def _safe_config(value: dict[str, Any]) -> dict[str, Any]:
    def validate(item: Any, *, depth: int) -> None:
        if depth > 10:
            raise ValueError("config nesting is too deep")
        if item is None or isinstance(item, (bool, int, str)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("config numbers must be finite")
            return
        if isinstance(item, list):
            if len(item) > 1000:
                raise ValueError("config lists are too large")
            for child in item:
                validate(child, depth=depth + 1)
            return
        if isinstance(item, dict):
            if len(item) > 256:
                raise ValueError("config objects are too large")
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 128:
                    raise ValueError("config keys must be non-empty strings")
                separated_key = re.sub(
                    r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key
                )
                separated_key = re.sub(
                    r"([a-z0-9])([A-Z])", r"\1_\2", separated_key
                )
                parts = {
                    part
                    for part in re.split(
                        r"[^a-z0-9]+", separated_key.casefold()
                    )
                    if part
                }
                normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
                safe_token_key = (
                    normalized in _SAFE_TOKENIZER_KEYS
                    or normalized in _SAFE_TOKEN_CONFIG_KEYS
                )
                if (
                    (
                        parts.intersection(_SENSITIVE_PARTS)
                        and normalized not in _SAFE_KEY_CONFIG_KEYS
                    )
                    or any(
                        fragment in normalized
                        for fragment in _SENSITIVE_KEY_FRAGMENTS
                    )
                    or ("token" in normalized and not safe_token_key)
                ):
                    raise ValueError("config must not contain secret-like keys")
                validate(child, depth=depth + 1)
            return
        raise ValueError("config values must be JSON-compatible")

    validate(value, depth=0)
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("config values must be JSON-compatible") from error
    if len(serialized) > _MAX_CONFIG_BYTES:
        raise ValueError("config must be at most 64 KiB")
    return copy.deepcopy(value)


def _append_console_log(
    stored_logs: list[dict[str, Any]], payload: ConsoleLogCreate
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logs = copy.deepcopy(stored_logs)
    last_sequence = 0
    for entry in logs:
        sequence = entry.get("sequence") if isinstance(entry, dict) else None
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= last_sequence
            or sequence > _MAX_CONSOLE_SEQUENCE
        ):
            raise HTTPException(
                status_code=503,
                detail="Stored training console metadata is invalid.",
            )
        last_sequence = sequence
    if last_sequence >= _MAX_CONSOLE_SEQUENCE:
        raise HTTPException(
            status_code=409,
            detail="Training run console sequence limit was reached.",
        )
    entry = {
        "sequence": last_sequence + 1,
        "source": payload.source,
        "line": payload.line,
        "step": payload.step,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    logs.append(entry)
    while len(logs) > _MAX_CONSOLE_LOGS:
        logs.pop(0)
    try:
        while (
            len(
                json.dumps(
                    logs,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > _MAX_CONSOLE_LOG_BYTES
        ):
            logs.pop(0)
    except (AttributeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="Stored training console metadata is invalid.",
        ) from error
    return logs, entry


def _run_or_404(db: Session, run_id: uuid.UUID, *, lock: bool) -> TrainingRun:
    statement = select(TrainingRun).where(TrainingRun.id == run_id)
    if lock:
        statement = statement.with_for_update()
    run = db.scalar(statement)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found.")
    return run


def _managed_summary(job: ManagedTrainingJob | None) -> ManagedTrainingJobSummary | None:
    if job is None:
        return None
    deadline = job.deadline_at
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return ManagedTrainingJobSummary(
        id=str(job.id),
        status=job.status,
        outcome=job.outcome,
        target_kind=job.target_kind,
        compute_size=job.compute_size,
        deadline_at=deadline,
        provider_state=job.provider_state,
        teardown_verified=job.teardown_verified_at is not None,
        output_model_repo=job.output_model_repo,
        output_marker_revision=job.output_marker_revision,
        output_revision=job.output_revision,
        last_error=job.last_error,
        event_gap=job.event_gap,
    )


def _run_read(
    run: TrainingRun, managed_job: ManagedTrainingJob | None = None
) -> TrainingRunRead:
    return TrainingRunRead(
        id=str(run.id),
        name=run.name,
        status=run.status,
        current_step=run.current_step,
        dataset_repo=run.dataset_repo,
        base_model=run.base_model,
        runtime=run.runtime,
        framework=run.framework,
        output_model_repo=run.output_model_repo,
        checkpoint_revision=run.checkpoint_revision,
        config=copy.deepcopy(run.config or {}),
        metrics=copy.deepcopy(run.metrics or {}),
        checkpoints=copy.deepcopy(run.checkpoints or []),
        managed_job=_managed_summary(managed_job),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _commit(db: Session) -> None:
    try:
        db.commit()
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL could not persist the training run.",
        ) from error


def _assert_external_mutation(db: Session, run_id: uuid.UUID) -> None:
    try:
        assert_external_run_mutable(db, run_id)
    except ManagedTrainingRunMutationError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None


def _store_or_http(operation: Any) -> Any:
    try:
        return operation()
    except TrainingStoreCorruptError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except TrainingStoreError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None


@router.post("/runs", response_model=TrainingRunRead, status_code=201)
def create_run(payload: RunCreate, db: Session = Depends(get_db)) -> TrainingRunRead:
    run = TrainingRun(
        name=payload.name,
        status=payload.status,
        current_step=payload.current_step,
        dataset_repo=payload.dataset_repo,
        base_model=payload.base_model,
        runtime=payload.runtime,
        framework=payload.framework,
        output_model_repo=payload.output_model_repo,
        checkpoint_revision=payload.checkpoint_revision,
        config=payload.config,
        metrics={},
        checkpoints=[],
        console_logs=[],
    )
    db.add(run)
    _commit(db)
    db.refresh(run)
    return _run_read(run, None)


@router.get("/runs", response_model=TrainingRunsRead)
def list_runs(
    status: Annotated[RunStatus | None, Query()] = None,
    db: Session = Depends(get_db),
) -> TrainingRunsRead:
    statement = select(TrainingRun)
    if status is not None:
        statement = statement.where(TrainingRun.status == status)
    statement = statement.order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc())
    runs = db.scalars(statement).all()
    run_ids = [run.id for run in runs]
    managed = {
        job.training_run_id: job
        for job in (
            db.scalars(
                select(ManagedTrainingJob).where(
                    ManagedTrainingJob.training_run_id.in_(run_ids)
                )
            ).all()
            if run_ids
            else []
        )
    }
    return TrainingRunsRead(
        runs=[_run_read(run, managed.get(run.id)) for run in runs]
    )


@router.get("/runs/{run_id}", response_model=TrainingRunRead)
def get_run(
    run_id: uuid.UUID,
    response: Response,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    response.headers["Cache-Control"] = "private, no-store"
    run = _run_or_404(db, run_id, lock=False)
    job = db.scalar(
        select(ManagedTrainingJob).where(
            ManagedTrainingJob.training_run_id == run.id
        )
    )
    return _run_read(run, job)


@router.patch("/runs/{run_id}", response_model=TrainingRunRead)
def update_run(
    run_id: uuid.UUID,
    payload: RunUpdate,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    if not payload.model_fields_set:
        raise HTTPException(status_code=422, detail="At least one update field is required.")
    run = _run_or_404(db, run_id, lock=True)
    _assert_external_mutation(db, run_id)
    if (
        "current_step" in payload.model_fields_set
        and payload.current_step is not None
        and payload.current_step < run.current_step
    ):
        raise HTTPException(status_code=409, detail="current_step cannot regress.")

    if "status" in payload.model_fields_set:
        run.status = payload.status
    if "current_step" in payload.model_fields_set:
        run.current_step = payload.current_step
    for field in (
        "dataset_repo",
        "base_model",
        "runtime",
        "framework",
        "output_model_repo",
        "checkpoint_revision",
        "config",
    ):
        if field in payload.model_fields_set:
            setattr(run, field, copy.deepcopy(getattr(payload, field)))
    _commit(db)
    db.refresh(run)
    return _run_read(run, None)


@router.post("/runs/{run_id}/metrics", response_model=TrainingRunRead)
def log_metrics(
    run_id: uuid.UUID,
    payload: MetricLog,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    run = _run_or_404(db, run_id, lock=True)
    _assert_external_mutation(db, run_id)
    _store_or_http(
        lambda: append_metrics(run, step=payload.step, metrics=payload.metrics)
    )
    _commit(db)
    db.refresh(run)
    return _run_read(run, None)


@router.post("/runs/{run_id}/checkpoints", response_model=TrainingRunRead)
def register_checkpoint(
    run_id: uuid.UUID,
    payload: CheckpointCreate,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    run = _run_or_404(db, run_id, lock=True)
    _assert_external_mutation(db, run_id)
    _store_or_http(
        lambda: append_checkpoint(
            run,
            repo_id=payload.repo_id,
            revision=payload.revision,
            step=payload.step,
        )
    )
    _commit(db)
    db.refresh(run)
    return _run_read(run, None)


@router.get("/runs/{run_id}/logs", response_model=ConsoleLogsRead)
def list_console_logs(
    run_id: uuid.UUID,
    response: Response,
    after_sequence: Annotated[
        int | None, Query(ge=0, le=_MAX_CONSOLE_SEQUENCE)
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 200,
    db: Session = Depends(get_db),
) -> ConsoleLogsRead:
    run = _run_or_404(db, run_id, lock=False)
    stored_logs = copy.deepcopy(run.console_logs or [])
    try:
        logs = [ConsoleLogRead.model_validate(item) for item in stored_logs]
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="Stored training console metadata is invalid.",
        ) from error
    sequences = [item.sequence for item in logs]
    if sequences != sorted(set(sequences)):
        raise HTTPException(
            status_code=503,
            detail="Stored training console metadata is invalid.",
        )
    oldest_sequence = sequences[0] if sequences else None
    latest_sequence = sequences[-1] if sequences else None
    if after_sequence is None:
        selected = logs[-limit:]
        truncated = len(selected) < len(logs) or (
            oldest_sequence is not None and oldest_sequence > 1
        )
    else:
        selected = [item for item in logs if item.sequence > after_sequence][:limit]
        truncated = (
            oldest_sequence is not None
            and after_sequence < oldest_sequence - 1
        )
    next_sequence = (
        selected[-1].sequence
        if selected
        else latest_sequence
        if latest_sequence is not None
        else after_sequence or 0
    )
    has_more = latest_sequence is not None and latest_sequence > next_sequence
    response.headers["Cache-Control"] = "private, no-store"
    return ConsoleLogsRead(
        logs=selected,
        oldest_sequence=oldest_sequence,
        latest_sequence=latest_sequence,
        next_sequence=next_sequence,
        truncated=truncated,
        has_more=has_more,
    )


@router.post(
    "/runs/{run_id}/logs",
    response_model=ConsoleLogRead,
    status_code=201,
)
def log_console(
    run_id: uuid.UUID,
    payload: ConsoleLogCreate,
    db: Session = Depends(get_db),
) -> ConsoleLogRead:
    run = _run_or_404(db, run_id, lock=True)
    _assert_external_mutation(db, run_id)
    entry = _store_or_http(
        lambda: append_console_log(
            run,
            source=payload.source,
            line=payload.line,
            step=payload.step,
            max_logs=_MAX_CONSOLE_LOGS,
            max_bytes=_MAX_CONSOLE_LOG_BYTES,
        )
    )
    if payload.step is not None:
        run.current_step = max(run.current_step, payload.step)
    _commit(db)
    return ConsoleLogRead.model_validate(entry)
