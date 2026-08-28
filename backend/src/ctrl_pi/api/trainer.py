from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import get_db
from ctrl_pi.hf_models import (
    HFModelBrowser,
    ModelAuthenticationError,
    ModelHubError,
)
from ctrl_pi.models import TrainingRun

router = APIRouter(prefix="/api/trainer", tags=["trainer"])

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
_METRIC_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.\-/]{0,63})")
_REVISION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?")
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


class MetricPointRead(BaseModel):
    step: int
    value: float


class CheckpointRead(BaseModel):
    repo_id: str
    revision: str
    step: int


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
    created_at: datetime
    updated_at: datetime


class TrainingRunsRead(BaseModel):
    runs: list[TrainingRunRead]


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


class ModelCardRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    description: str | None
    base_model: list[str]
    datasets: list[str]


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    card: ModelCardRead | None
    checkpoints: list[str]


class ModelsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    namespace: str
    models: list[ModelRead]
    total: int
    fetched_at: datetime


def get_model_browser(request: Request) -> HFModelBrowser:
    return request.app.state.hf_model_browser


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


def _run_or_404(db: Session, run_id: uuid.UUID, *, lock: bool) -> TrainingRun:
    statement = select(TrainingRun).where(TrainingRun.id == run_id)
    if lock:
        statement = statement.with_for_update()
    run = db.scalar(statement)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found.")
    return run


def _run_read(run: TrainingRun) -> TrainingRunRead:
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
    )
    db.add(run)
    _commit(db)
    db.refresh(run)
    return _run_read(run)


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
    return TrainingRunsRead(runs=[_run_read(run) for run in runs])


@router.get("/runs/{run_id}", response_model=TrainingRunRead)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> TrainingRunRead:
    return _run_read(_run_or_404(db, run_id, lock=False))


@router.patch("/runs/{run_id}", response_model=TrainingRunRead)
def update_run(
    run_id: uuid.UUID,
    payload: RunUpdate,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    if not payload.model_fields_set:
        raise HTTPException(status_code=422, detail="At least one update field is required.")
    run = _run_or_404(db, run_id, lock=True)
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
    return _run_read(run)


@router.post("/runs/{run_id}/metrics", response_model=TrainingRunRead)
def log_metrics(
    run_id: uuid.UUID,
    payload: MetricLog,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    run = _run_or_404(db, run_id, lock=True)
    metrics = copy.deepcopy(run.metrics or {})
    for name, value in payload.metrics.items():
        existing = metrics.get(name, [])
        series = [point for point in existing if point.get("step") != payload.step]
        series.append({"step": payload.step, "value": value})
        series.sort(key=lambda point: point["step"])
        metrics[name] = series
    if len(metrics) > _MAX_METRIC_NAMES or sum(map(len, metrics.values())) > _MAX_METRIC_POINTS:
        raise HTTPException(
            status_code=409,
            detail="Training run metric storage limit was reached.",
        )
    run.metrics = metrics
    run.current_step = max(run.current_step, payload.step)
    _commit(db)
    db.refresh(run)
    return _run_read(run)


@router.post("/runs/{run_id}/checkpoints", response_model=TrainingRunRead)
def register_checkpoint(
    run_id: uuid.UUID,
    payload: CheckpointCreate,
    db: Session = Depends(get_db),
) -> TrainingRunRead:
    run = _run_or_404(db, run_id, lock=True)
    checkpoints = copy.deepcopy(run.checkpoints or [])
    checkpoint = payload.model_dump()
    if checkpoint not in checkpoints:
        if len(checkpoints) >= _MAX_CHECKPOINTS:
            raise HTTPException(
                status_code=409,
                detail="Training run checkpoint storage limit was reached.",
            )
        checkpoints.append(checkpoint)
        checkpoints.sort(
            key=lambda item: (item["step"], item["repo_id"], item["revision"])
        )
    run.checkpoints = checkpoints
    run.output_model_repo = payload.repo_id
    run.checkpoint_revision = payload.revision
    run.current_step = max(run.current_step, payload.step)
    _commit(db)
    db.refresh(run)
    return _run_read(run)


@router.get("/models", response_model=ModelsRead)
async def list_models(
    response: Response,
    refresh: bool = False,
    config: AppConfig = Depends(get_config),
    browser: HFModelBrowser = Depends(get_model_browser),
) -> ModelsRead:
    namespace = (config.hf_namespace or "").strip()
    if not namespace:
        raise HTTPException(
            status_code=503,
            detail="Set HF_NAMESPACE before browsing models.",
        )
    if config.hf_token is None or not config.hf_token.get_secret_value().strip():
        raise HTTPException(
            status_code=503,
            detail="Set HF_TOKEN before browsing models.",
        )
    token = config.hf_token.get_secret_value().strip()
    try:
        page = await asyncio.to_thread(
            browser.list_namespace,
            namespace=namespace,
            token=token,
            refresh=refresh,
        )
    except ModelAuthenticationError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ModelHubError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    response.headers["Cache-Control"] = (
        "private, no-store" if refresh else "private, max-age=30"
    )
    return ModelsRead.model_validate(page)
