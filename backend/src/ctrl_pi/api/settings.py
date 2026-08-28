from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import re
import tomllib
from typing import Literal

from fastapi import APIRouter, Depends, Request
import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import engine_for_url, get_db
from ctrl_pi.drivers.yam import YAMDriver
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

_HF_NAMESPACE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?")
_MODAL_TOKEN_ID = re.compile(r"ak-[A-Za-z0-9_-]{1,508}")
_MODAL_TOKEN_SECRET = re.compile(r"as-[A-Za-z0-9_-]{1,508}")
_MODAL_PROXY_TOKEN_ID = re.compile(r"wk-[A-Za-z0-9_-]{1,253}")
_MODAL_PROXY_TOKEN_SECRET = re.compile(r"ws-[A-Za-z0-9_-]{1,253}")
_MAX_MODAL_PROFILE_BYTES = 64 * 1024
_MAX_MODAL_CONFIG_PATH_CHARS = 4_096
_MAX_MODAL_PROFILE_NAME_CHARS = 128
_MAX_HF_IDENTITY_BYTES = 64 * 1024


def _secret_value(value: object | None) -> str:
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    if not callable(getter):
        return ""
    raw = getter()
    return raw.strip() if isinstance(raw, str) else ""


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
    except (SQLAlchemyError, ValueError):
        return ServiceStatus(
            id="postgres",
            label="PostgreSQL",
            status="error",
            detail="Database connection failed. Check DATABASE_URL.",
        )


def huggingface_status(
    config: AppConfig,
    *,
    required: bool = True,
    http_get: Callable[..., object] | None = None,
) -> ServiceStatus:
    namespace = (config.hf_namespace or "").strip()
    token = _secret_value(config.hf_token)
    if not token:
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="missing",
            detail="Set HF_TOKEN in .env.",
            required=required,
        )
    if not namespace:
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="missing",
            detail="Set HF_NAMESPACE in .env; ctrl-π never guesses it.",
            required=required,
        )
    if (
        _HF_NAMESPACE.fullmatch(namespace) is None
        or ".." in namespace
        or "--" in namespace
    ):
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="error",
            detail="HF_NAMESPACE is invalid.",
            required=required,
        )
    try:
        response = (http_get or httpx.get)(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
            follow_redirects=False,
            trust_env=False,
        )
        response.raise_for_status()  # type: ignore[attr-defined]
        content = response.content  # type: ignore[attr-defined]
        if not isinstance(content, bytes) or len(content) > _MAX_HF_IDENTITY_BYTES:
            raise ValueError
        identity = response.json()  # type: ignore[attr-defined]
    except Exception:
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="error",
            detail="HF_TOKEN could not be validated.",
            required=required,
        )
    if not isinstance(identity, Mapping):
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="error",
            detail="Hugging Face returned an invalid identity response.",
            required=required,
        )
    available: set[str] = set()
    user_name = identity.get("name")
    if isinstance(user_name, str) and _HF_NAMESPACE.fullmatch(user_name.strip()):
        available.add(user_name.strip().casefold())
    organizations = identity.get("orgs", [])
    if not isinstance(organizations, (list, tuple)):
        organizations = []
    for organization in organizations:
        if not isinstance(organization, Mapping):
            continue
        name = organization.get("name")
        if isinstance(name, str) and _HF_NAMESPACE.fullmatch(name.strip()):
            available.add(name.strip().casefold())
    if not available:
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="error",
            detail="Hugging Face returned an invalid identity response.",
            required=required,
        )
    if namespace.casefold() not in available:
        return ServiceStatus(
            id="huggingface",
            label="Hugging Face",
            status="error",
            detail="HF_TOKEN does not belong to the configured HF_NAMESPACE.",
            required=required,
        )
    return ServiceStatus(
        id="huggingface",
        label="Hugging Face",
        status="connected",
        detail="HF_TOKEN is authenticated for the configured namespace.",
        required=required,
    )


def _modal_api_status(
    config: AppConfig,
    *,
    config_path: Path | None = None,
    profile: str | None = None,
) -> ServiceStatus:
    token_id = _secret_value(config.modal_token_id)
    token_secret = _secret_value(config.modal_token_secret)
    env_declared = (
        config.modal_token_id is not None or config.modal_token_secret is not None
    )
    if env_declared:
        if not token_id or not token_secret:
            return ServiceStatus(
                id="modal",
                label="Modal",
                status="error",
                detail="Set Modal API credentials as one complete non-blank pair.",
            )
        if (
            _MODAL_TOKEN_ID.fullmatch(token_id) is None
            or _MODAL_TOKEN_SECRET.fullmatch(token_secret) is None
        ):
            return ServiceStatus(
                id="modal",
                label="Modal",
                status="error",
                detail="Modal API credentials have an invalid format.",
            )
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="configured",
            detail="Modal API credentials are configured from the environment.",
        )

    try:
        raw_path = (
            str(config_path)
            if config_path is not None
            else os.environ.get("MODAL_CONFIG_PATH", "")
        )
        if raw_path and (
            len(raw_path) > _MAX_MODAL_CONFIG_PATH_CHARS
            or any(ord(character) < 32 for character in raw_path)
        ):
            raise ValueError
        raw_path = raw_path.strip()
        selected_path = (
            Path(raw_path).expanduser()
            if raw_path
            else Path.home() / ".modal.toml"
        )
        if not selected_path.exists():
            return ServiceStatus(
                id="modal",
                label="Modal",
                status="missing",
                detail="Configure a complete Modal API credential pair.",
            )
        if not selected_path.is_file():
            raise ValueError
        size = selected_path.stat().st_size
        if size < 0 or size > _MAX_MODAL_PROFILE_BYTES:
            raise ValueError
        with selected_path.open("rb") as handle:
            profile_bytes = handle.read(_MAX_MODAL_PROFILE_BYTES + 1)
        if len(profile_bytes) > _MAX_MODAL_PROFILE_BYTES:
            raise ValueError
        raw_profiles = tomllib.loads(profile_bytes.decode("utf-8"))
    except (OSError, RuntimeError, UnicodeError, ValueError, tomllib.TOMLDecodeError):
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The Modal profile file is unavailable or invalid.",
        )
    if not raw_profiles or not all(
        isinstance(name, str)
        and 0 < len(name) <= _MAX_MODAL_PROFILE_NAME_CHARS
        and not any(ord(character) < 32 for character in name)
        and isinstance(values, Mapping)
        for name, values in raw_profiles.items()
    ):
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The Modal profile file has no valid profiles.",
        )
    active_profiles = [
        name
        for name, values in raw_profiles.items()
        if values.get("active") is True
    ]
    raw_requested_profile = (
        profile if profile is not None else os.environ.get("MODAL_PROFILE", "")
    )
    if not isinstance(raw_requested_profile, str):
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The selected Modal profile is invalid.",
        )
    requested_profile = raw_requested_profile.strip()
    if (
        len(requested_profile) > _MAX_MODAL_PROFILE_NAME_CHARS
        or any(ord(character) < 32 for character in requested_profile)
    ):
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The selected Modal profile is invalid.",
        )
    if len(active_profiles) > 1:
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="Select exactly one Modal profile.",
        )
    if requested_profile:
        selected_profile = requested_profile
    elif len(active_profiles) == 1:
        selected_profile = active_profiles[0]
    elif len(raw_profiles) > 1:
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="Select exactly one Modal profile.",
        )
    else:
        selected_profile = "default"
    values = raw_profiles.get(selected_profile)
    if not isinstance(values, Mapping):
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The selected Modal profile is unavailable.",
        )
    profile_id = values.get("token_id")
    profile_secret = values.get("token_secret")
    profile_id = profile_id.strip() if isinstance(profile_id, str) else ""
    profile_secret = (
        profile_secret.strip() if isinstance(profile_secret, str) else ""
    )
    if not profile_id or not profile_secret:
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The selected Modal profile has an incomplete credential pair.",
        )
    if (
        _MODAL_TOKEN_ID.fullmatch(profile_id) is None
        or _MODAL_TOKEN_SECRET.fullmatch(profile_secret) is None
    ):
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="The selected Modal profile credentials have an invalid format.",
        )
    return ServiceStatus(
        id="modal",
        label="Modal",
        status="configured",
        detail="Modal API credentials are configured from the selected profile.",
    )


def _modal_proxy_status(
    config: AppConfig,
) -> tuple[bool, Literal["missing", "error"] | None]:
    token_id = _secret_value(config.modal_proxy_token_id)
    token_secret = _secret_value(config.modal_proxy_token_secret)
    declared = (
        config.modal_proxy_token_id is not None
        or config.modal_proxy_token_secret is not None
    )
    if not declared:
        return False, "missing"
    if not token_id or not token_secret:
        return False, "error"
    if (
        _MODAL_PROXY_TOKEN_ID.fullmatch(token_id) is None
        or _MODAL_PROXY_TOKEN_SECRET.fullmatch(token_secret) is None
    ):
        return False, "error"
    return True, None


def modal_status(
    config: AppConfig,
    *,
    required: bool = True,
    config_path: Path | None = None,
    profile: str | None = None,
) -> tuple[ServiceStatus, bool, bool]:
    api_status = _modal_api_status(
        config,
        config_path=config_path,
        profile=profile,
    )
    proxy_ready, proxy_problem = _modal_proxy_status(config)
    api_ready = api_status.status == "configured"
    if not api_ready:
        return api_status.model_copy(update={"required": required}), False, proxy_ready
    if proxy_ready:
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="configured",
            detail="Modal API and endpoint proxy credential pairs are configured.",
            required=required,
        ), True, True
    if proxy_problem == "error":
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="error",
            detail="Modal endpoint proxy credentials are incomplete or invalid.",
            required=required,
        ), True, False
    if required:
        return ServiceStatus(
            id="modal",
            label="Modal",
            status="missing",
            detail="Set the complete Modal endpoint proxy credential pair.",
            required=True,
        ), True, False
    return ServiceStatus(
        id="modal",
        label="Modal",
        status="configured",
        detail="Modal API credentials are ready; endpoint proxy tokens are optional in mock mode.",
        required=False,
    ), True, False


def get_connection_status(
    config: AppConfig | None = None,
    driver: YAMDriver | None = None,
    *,
    hf_http_get: Callable[..., object] | None = None,
    modal_config_path: Path | None = None,
    modal_profile: str | None = None,
) -> SettingsStatus:
    config = config or get_config()
    if driver is None:
        arms_status = ServiceStatus(
            id="arms",
            label="YAM arms",
            status="connected" if config.mock_mode else "missing",
            detail=(
                "MockYAMDriver is ready."
                if config.mock_mode
                else "YAM hardware driver is unavailable."
            ),
        )
    else:
        try:
            diagnostic = driver.diagnostic()
            arms = driver.list_arms()
            all_connected = bool(arms) and all(arm.connected for arm in arms)
            if diagnostic.status == "connected":
                status = "connected" if all_connected else "error"
            elif diagnostic.status == "configured":
                status = "missing"
            else:
                status = diagnostic.status
            arms_status = ServiceStatus(
                id="arms",
                label="YAM arms",
                status=status,
                detail=diagnostic.detail,
            )
        except Exception:
            arms_status = ServiceStatus(
                id="arms",
                label="YAM arms",
                status="error",
                detail="YAM driver status is unavailable.",
            )
    external_required = not config.mock_mode
    hf_status = huggingface_status(
        config,
        required=external_required,
        http_get=hf_http_get,
    )
    modal_service, modal_configured, proxy_configured = modal_status(
        config,
        required=external_required,
        config_path=modal_config_path,
        profile=modal_profile,
    )
    services = [postgres_status(config), hf_status, modal_service, arms_status]
    ready = {"connected", "configured"}
    hf_configured = hf_status.status == "connected"
    return SettingsStatus(
        mode="mock" if config.mock_mode else "hardware",
        setup_complete=all(item.status in ready for item in services if item.required),
        services=services,
        inference=InferenceReadiness(
            mock_mode=config.mock_mode,
            hf_configured=hf_configured,
            modal_configured=modal_configured,
            modal_proxy_configured=proxy_configured,
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
def settings_status(
    request: Request,
    config: AppConfig = Depends(get_config),
) -> SettingsStatus:
    return get_connection_status(config, request.app.state.yam_driver)


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
