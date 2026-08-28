from __future__ import annotations

from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import engine_for_url, get_db
from ctrl_pi.models import AppSetting

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ServiceStatus(BaseModel):
    id: Literal["postgres", "huggingface", "modal", "arms"]
    label: str
    status: Literal["connected", "configured", "missing", "error"]
    detail: str
    required: bool = True


class InferenceReadiness(BaseModel):
    mock_mode: bool
    hf_configured: bool
    modal_configured: bool
    modal_proxy_configured: bool


class SettingsStatus(BaseModel):
    mode: Literal["mock", "hardware"]
    setup_complete: bool
    services: list[ServiceStatus]
    inference: InferenceReadiness


class PublicSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_namespace: str | None = None
    recording_fps: int = Field(default=20, ge=1, le=60)
    default_runtime: Literal["lerobot", "openpi"] = "lerobot"
    default_compute: Literal["Modal: A10G", "Modal: A100", "Modal: H100"] = (
        "Modal: A10G"
    )
    modal_timeout_minutes: int = Field(default=30, ge=1, le=30)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_fps: int | None = Field(default=None, ge=1, le=60)
    default_runtime: Literal["lerobot", "openpi"] | None = None
    default_compute: Literal["Modal: A10G", "Modal: A100", "Modal: H100"] | None = None
    modal_timeout_minutes: int | None = Field(default=None, ge=1, le=30)


DEFAULTS = PublicSettings().model_dump(exclude={"hf_namespace"})
EDITABLE_KEYS = frozenset(DEFAULTS)


def postgres_status(config: AppConfig) -> ServiceStatus:
    value = config.database_url
    if value is None or not value.get_secret_value().strip():
        return ServiceStatus(
            id="postgres",
            label="PostgreSQL",
            status="missing",
            detail="Set DATABASE_URL in .env.",
        )
    try:
        engine = engine_for_url(value.get_secret_value())
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return ServiceStatus(
            id="postgres",
            label="PostgreSQL",
            status="connected",
            detail="Database connection is healthy.",
        )
    except SQLAlchemyError:
        return ServiceStatus(
            id="postgres",
            label="PostgreSQL",
            status="error",
            detail="Database connection failed. Check DATABASE_URL.",
        )


def huggingface_status(config: AppConfig) -> ServiceStatus:
    namespace = (config.hf_namespace or "").strip()
    if config.hf_token is None or not config.hf_token.get_secret_value().strip():
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="missing",
            detail="Set HF_TOKEN in .env.",
        )
    if not namespace:
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="missing",
            detail="Set HF_NAMESPACE in .env; ctrl-π never guesses it.",
        )
    try:
        response = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {config.hf_token.get_secret_value()}"},
            timeout=5.0,
        )
        response.raise_for_status()
    except (httpx.HTTPError, ValueError):
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="error",
            detail=f"Could not validate the token for namespace {namespace}.",
        )
    return ServiceStatus(
        id="huggingface",
        label="Hugging Face",
        status="connected",
        detail=f"Authenticated; artifacts use {namespace}.",
    )


def modal_status(config: AppConfig) -> ServiceStatus:
    token_id = config.modal_token_id
    token_secret = config.modal_token_secret
    if token_id and token_secret:
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="configured",
            detail="Credentials found; live validation is available with deployment support.",
        )
    if (Path.home() / ".modal.toml").exists():
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="configured",
            detail="Using credentials from ~/.modal.toml.",
        )
    return ServiceStatus(
        id="modal",
        label="Modal",
        status="missing",
        detail="Configure Modal credentials using its normal local mechanism.",
    )


def get_connection_status(config: AppConfig | None = None) -> SettingsStatus:
    config = config or get_config()
    services = [
        postgres_status(config),
        huggingface_status(config),
        modal_status(config),
        ServiceStatus(
            id="arms",
            label="YAM arms",
            status="connected" if config.mock_mode else "configured",
            detail=(
                "MockYAMDriver is ready."
                if config.mock_mode
                else "Hardware mode selected; live driver validation is pending."
            ),
        ),
    ]
    ready = {"connected", "configured"}
    hf_configured = bool(
        (config.hf_namespace or "").strip()
        and config.hf_token is not None
        and config.hf_token.get_secret_value().strip()
    )
    modal_configured = bool(
        (
            config.modal_token_id is not None
            and config.modal_token_id.get_secret_value().strip()
            and config.modal_token_secret is not None
            and config.modal_token_secret.get_secret_value().strip()
        )
        or (Path.home() / ".modal.toml").exists()
    )
    proxy_id = (
        ""
        if config.modal_proxy_token_id is None
        else config.modal_proxy_token_id.get_secret_value().strip()
    )
    proxy_secret = (
        ""
        if config.modal_proxy_token_secret is None
        else config.modal_proxy_token_secret.get_secret_value().strip()
    )
    return SettingsStatus(
        mode="mock" if config.mock_mode else "hardware",
        setup_complete=all(item.status in ready for item in services if item.required),
        services=services,
        inference=InferenceReadiness(
            mock_mode=config.mock_mode,
            hf_configured=hf_configured,
            modal_configured=modal_configured,
            modal_proxy_configured=(
                proxy_id.startswith("wk-")
                and len(proxy_id) > 3
                and proxy_secret.startswith("ws-")
                and len(proxy_secret) > 3
            ),
        ),
    )


def read_public_settings(db: Session, config: AppConfig) -> PublicSettings:
    stored = {
        item.key: item.value
        for item in db.scalars(select(AppSetting).where(AppSetting.key.in_(EDITABLE_KEYS)))
    }
    return PublicSettings(
        hf_namespace=config.hf_namespace,
        **{**DEFAULTS, **stored},
    )


@router.get("/status", response_model=SettingsStatus)
def settings_status() -> SettingsStatus:
    return get_connection_status()


@router.get("", response_model=PublicSettings)
def get_public_settings(
    db: Session = Depends(get_db), config: AppConfig = Depends(get_config)
) -> PublicSettings:
    return read_public_settings(db, config)


@router.patch("", response_model=PublicSettings)
def update_public_settings(
    payload: SettingsUpdate,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
) -> PublicSettings:
    for key, value in payload.model_dump(exclude_none=True).items():
        setting = db.get(AppSetting, key)
        if setting is None:
            db.add(AppSetting(key=key, value=value))
        else:
            setting.value = value
    db.commit()
    return read_public_settings(db, config)
