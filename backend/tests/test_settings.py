from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ctrl_pi.api.settings import (
    DEFAULTS,
    SettingsUpdate,
    get_connection_status,
    read_public_settings,
    update_public_settings,
)
from ctrl_pi.config import AppConfig
from ctrl_pi.db import Base


def test_missing_configuration_is_reported_without_exposing_secrets() -> None:
    config = AppConfig(
        _env_file=None,
        database_url=None,
        hf_token=None,
        hf_namespace=None,
        modal_token_id=None,
        modal_token_secret=None,
    )

    status = get_connection_status(config)
    payload = status.model_dump()

    assert status.mode == "mock"
    assert status.setup_complete is False
    assert {service.status for service in status.services} >= {"missing", "connected"}
    assert set(payload) == {"mode", "setup_complete", "services"}
    assert all(set(service) == {"id", "label", "status", "detail", "required"} for service in payload["services"])


def test_environment_secrets_are_redacted() -> None:
    config = AppConfig(
        _env_file=None,
        database_url="postgresql://user:secret@db/ctrl_pi",
        hf_token="hf_private",
        modal_token_id="id_private",
        modal_token_secret="secret_private",
    )

    representation = repr(config)

    assert "hf_private" not in representation
    assert "secret_private" not in representation
    assert "**********" in representation


def test_non_secret_settings_round_trip() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    config = AppConfig(_env_file=None, hf_namespace="configured-namespace")

    with Session(engine) as session:
        initial = read_public_settings(session, config)
        assert initial.model_dump(exclude={"hf_namespace"}) == DEFAULTS

        updated = update_public_settings(
            SettingsUpdate(
                recording_fps=30,
                default_runtime="openpi",
                default_compute="Modal: A100",
                modal_timeout_minutes=10,
            ),
            session,
            config,
        )

        assert updated.hf_namespace == "configured-namespace"
        assert updated.recording_fps == 30
        assert updated.default_runtime == "openpi"
        assert updated.default_compute == "Modal: A100"
        assert updated.modal_timeout_minutes == 10
