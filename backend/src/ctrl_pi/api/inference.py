from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

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
from ctrl_pi.models import AppSetting

router = APIRouter(prefix="/api/inference/deployments", tags=["inference"])

_REVISION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?")


class DeploymentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    model_repo: str = Field(min_length=3, max_length=255)
    checkpoint_revision: str | None = Field(default=None, max_length=128)
    runtime: Literal["stub"] = "stub"
    compute_size: Literal["CPU"] = "CPU"

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
    runtime: str
    compute_size: str
    endpoint_url: str | None
    provider_app_id: str | None
    started_at: datetime | None
    stopped_at: datetime | None
    created_at: datetime
    updated_at: datetime


def get_deployment_service(request: Request) -> DeploymentService:
    return request.app.state.deployment_service


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
    if isinstance(error, (DeploymentConfigurationError, DeploymentStorageError)):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(
        status_code=502,
        detail="The compute target operation failed safely.",
    )


@router.post("", response_model=DeploymentRead, status_code=201)
async def deploy(
    payload: DeploymentCreate,
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[DeploymentService, Depends(get_deployment_service)],
) -> DeploymentRecord:
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


@router.post("/{deployment_id}/stop", response_model=DeploymentRead)
async def stop(
    deployment_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[DeploymentService, Depends(get_deployment_service)],
) -> DeploymentRecord:
    try:
        return await service.stop(db, deployment_id)
    except (
        DeploymentNotFoundError,
        DeploymentConflictError,
        DeploymentConfigurationError,
        DeploymentProviderError,
        DeploymentStorageError,
    ) as error:
        raise _http_error(error) from None
