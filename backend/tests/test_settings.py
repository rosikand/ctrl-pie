from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import ctrl_pi.api.settings as settings_module
from ctrl_pi.api.settings import (
    DEFAULTS,
    ServiceStatus,
    SettingsUpdate,
    get_connection_status,
    huggingface_status,
    modal_status,
    read_public_settings,
    update_public_settings,
)
from ctrl_pi.config import AppConfig
from ctrl_pi.db import Base
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.real_yam import RealYAMDriver


class FakeHfResponse:
    def __init__(self, identity: Any) -> None:
        self.identity = identity
        self.content = (
            json.dumps(identity).encode("utf-8")
            if not isinstance(identity, Exception)
            else b"{}"
        )

    def raise_for_status(self) -> None:
        if isinstance(self.identity, Exception):
            raise self.identity

    def json(self) -> Any:
        return self.identity


def test_missing_configuration_is_reported_without_exposing_secrets(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        _env_file=None,
        database_url=None,
        hf_token=None,
        hf_namespace=None,
        modal_token_id=None,
        modal_token_secret=None,
        modal_proxy_token_id=None,
        modal_proxy_token_secret=None,
    )

    status = get_connection_status(
        config,
        modal_config_path=tmp_path / "missing-modal.toml",
    )
    payload = status.model_dump()

    assert status.mode == "mock"
    assert status.setup_complete is False
    assert {service.status for service in status.services} >= {"missing", "connected"}
    assert set(payload) == {"mode", "setup_complete", "services", "inference"}
    assert payload["inference"] == {
        "mock_mode": True,
        "hf_configured": False,
        "modal_configured": False,
        "modal_proxy_configured": False,
    }
    required = {service.id: service.required for service in status.services}
    assert required == {
        "postgres": True,
        "huggingface": False,
        "modal": False,
        "arms": True,
    }
    assert all(set(service) == {"id", "label", "status", "detail", "required"} for service in payload["services"])


def test_mock_setup_requires_only_database_and_arms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        settings_module,
        "postgres_status",
        lambda _config: ServiceStatus(
            id="postgres",
            label="PostgreSQL",
            status="connected",
            detail="Database connection is healthy.",
        ),
    )
    status = get_connection_status(
        AppConfig(
            _env_file=None,
            database_url=None,
            hf_token=None,
            hf_namespace=None,
            modal_token_id=None,
            modal_token_secret=None,
            modal_proxy_token_id=None,
            modal_proxy_token_secret=None,
        ),
        modal_config_path=tmp_path / "missing-modal.toml",
    )

    assert status.setup_complete is True
    services = {service.id: service for service in status.services}
    assert services["postgres"].required is True
    assert services["arms"].required is True
    assert services["huggingface"].required is False
    assert services["modal"].required is False
    assert services["huggingface"].status == "missing"
    assert services["modal"].status == "missing"


def test_hugging_face_status_validates_exact_user_or_org_with_explicit_token() -> None:
    secret = "hf_explicit_settings_secret"
    calls: list[dict[str, Any]] = []

    def http_get(url: str, **kwargs: Any) -> FakeHfResponse:
        calls.append({"url": url, **kwargs})
        return FakeHfResponse(
            {"name": "operator", "orgs": [{"name": "Acme-Robotics"}]},
        )

    connected = huggingface_status(
        AppConfig(
            _env_file=None,
            hf_token=f"  {secret}  ",
            hf_namespace="  acme-robotics  ",
        ),
        http_get=http_get,
    )
    mismatched = huggingface_status(
        AppConfig(
            _env_file=None,
            hf_token=secret,
            hf_namespace="another-org",
        ),
        http_get=http_get,
    )

    assert connected.status == "connected"
    assert mismatched.status == "error"
    assert calls[0] == {
        "url": "https://huggingface.co/api/whoami-v2",
        "headers": {"Authorization": f"Bearer {secret}"},
        "timeout": 5.0,
        "follow_redirects": False,
        "trust_env": False,
    }
    assert secret not in connected.model_dump_json()
    assert secret not in mismatched.model_dump_json()


def test_hugging_face_status_rejects_blank_and_sanitizes_provider_failure() -> None:
    calls = 0

    def unused_get(url: str, **kwargs: Any) -> FakeHfResponse:
        nonlocal calls
        calls += 1
        return FakeHfResponse({})

    blank = huggingface_status(
        AppConfig(_env_file=None, hf_token="   ", hf_namespace="acme"),
        http_get=unused_get,
    )
    secret = "hf_settings_failure_secret"
    failed = huggingface_status(
        AppConfig(_env_file=None, hf_token=secret, hf_namespace="acme"),
        http_get=lambda *_args, **_kwargs: FakeHfResponse(
            RuntimeError(f"provider leaked {secret}")
        ),
    )

    assert blank.status == "missing"
    assert calls == 0
    assert failed.status == "error"
    assert secret not in failed.model_dump_json()


@pytest.mark.parametrize("namespace", ["acme..robotics", "acme--robotics"])
def test_hugging_face_status_rejects_ambiguous_namespace_forms(
    namespace: str,
) -> None:
    calls = 0

    def unused_get(_url: str, **_kwargs: Any) -> FakeHfResponse:
        nonlocal calls
        calls += 1
        return FakeHfResponse({"name": namespace, "orgs": []})

    status = huggingface_status(
        AppConfig(_env_file=None, hf_token="hf-token", hf_namespace=namespace),
        http_get=unused_get,
    )

    assert status.status == "error"
    assert calls == 0


def test_modal_environment_and_profile_pairs_are_validated_without_secrets(
    tmp_path: Path,
) -> None:
    secret = "as-modal_settings_secret"
    partial, api_ready, proxy_ready = modal_status(
        AppConfig(
            _env_file=None,
            modal_token_id="ak-only",
            modal_token_secret="   ",
        ),
        config_path=tmp_path / "unused.toml",
    )
    assert partial.status == "error"
    assert api_ready is proxy_ready is False

    profile_path = tmp_path / ".modal.toml"
    profile_path.write_text(
        "[robot]\nactive = true\ntoken_id = 'ak-profile'\n"
        f"token_secret = '{secret}'\n",
        encoding="utf-8",
    )
    configured, api_ready, proxy_ready = modal_status(
        AppConfig(
            _env_file=None,
            modal_token_id=None,
            modal_token_secret=None,
            modal_proxy_token_id=None,
            modal_proxy_token_secret=None,
        ),
        required=False,
        config_path=profile_path,
    )

    assert configured.status == "configured"
    assert configured.required is False
    assert api_ready is True
    assert proxy_ready is False
    assert secret not in configured.model_dump_json()

    profile_path.write_text(
        "[robot]\nactive = true\ntoken_id = 'ak-profile'\ntoken_secret = '   '\n",
        encoding="utf-8",
    )
    invalid, api_ready, _ = modal_status(
        AppConfig(
            _env_file=None,
            modal_token_id=None,
            modal_token_secret=None,
            modal_proxy_token_id=None,
            modal_proxy_token_secret=None,
        ),
        config_path=profile_path,
    )
    assert invalid.status == "error"
    assert api_ready is False

    invalid_proxy, api_ready, proxy_ready = modal_status(
        AppConfig(
            _env_file=None,
            modal_token_id="ak-valid",
            modal_token_secret="as-valid",
            modal_proxy_token_id="wk-only",
            modal_proxy_token_secret="   ",
        ),
        required=False,
    )
    assert invalid_proxy.status == "error"
    assert invalid_proxy.required is False
    assert api_ready is True
    assert proxy_ready is False


@pytest.mark.parametrize(
    "hostile_path",
    [
        "~ctrl_pi_user_that_does_not_exist_6f53/.modal.toml",
        "x" * 4_097,
        "unsafe\npath",
    ],
)
def test_modal_config_path_failures_are_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    hostile_path: str,
) -> None:
    monkeypatch.setenv("MODAL_CONFIG_PATH", hostile_path)
    config = AppConfig(
        _env_file=None,
        modal_token_id=None,
        modal_token_secret=None,
        modal_proxy_token_id=None,
        modal_proxy_token_secret=None,
    )

    status, api_ready, proxy_ready = modal_status(config)
    payload = status.model_dump_json()

    assert status.status == "error"
    assert api_ready is proxy_ready is False
    assert hostile_path not in payload
    assert "ctrl_pi_user_that_does_not_exist" not in payload


def test_modal_config_path_rejects_an_embedded_null_without_os_access() -> None:
    hostile_path = "unsafe\0path"
    status, api_ready, proxy_ready = modal_status(
        AppConfig(
            _env_file=None,
            modal_token_id=None,
            modal_token_secret=None,
            modal_proxy_token_id=None,
            modal_proxy_token_secret=None,
        ),
        config_path=Path(hostile_path),
    )

    assert status.status == "error"
    assert api_ready is proxy_ready is False
    assert hostile_path not in status.model_dump_json()


def test_modal_profile_selection_matches_pinned_sdk_and_sanitizes_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "modal.toml"
    path.write_text(
        "[robot]\ntoken_id = 'ak-profile'\ntoken_secret = 'as-profile'\n",
        encoding="utf-8",
    )
    config = AppConfig(
        _env_file=None,
        modal_token_id=None,
        modal_token_secret=None,
        modal_proxy_token_id=None,
        modal_proxy_token_secret=None,
    )

    implicit, implicit_ready, _ = modal_status(
        config,
        required=False,
        config_path=path,
    )
    explicit, explicit_ready, _ = modal_status(
        config,
        required=False,
        config_path=path,
        profile="robot",
    )
    hostile_name = "profile\nsecret-profile-name"
    monkeypatch.setenv("MODAL_PROFILE", hostile_name)
    hostile, hostile_ready, _ = modal_status(
        config,
        required=False,
        config_path=path,
    )

    assert implicit.status == "error"
    assert implicit_ready is False
    assert explicit.status == "configured"
    assert explicit_ready is True
    assert hostile.status == "error"
    assert hostile_ready is False
    assert hostile_name not in hostile.model_dump_json()


def test_hardware_setup_requires_verified_hf_modal_proxy_database_and_arms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        settings_module,
        "postgres_status",
        lambda _config: ServiceStatus(
            id="postgres",
            label="PostgreSQL",
            status="connected",
            detail="Database connection is healthy.",
        ),
    )
    secret_values = (
        "hf_hardware_settings_secret",
        "ak-hardware",
        "as-hardware",
        "wk-hardware",
        "ws-hardware",
    )
    config = AppConfig(
        _env_file=None,
        ctrl_pi_mock_mode=False,
        hf_token=secret_values[0],
        hf_namespace="acme",
        modal_token_id=secret_values[1],
        modal_token_secret=secret_values[2],
        modal_proxy_token_id=secret_values[3],
        modal_proxy_token_secret=secret_values[4],
    )
    def http_get(_url: str, **_kwargs: Any) -> FakeHfResponse:
        return FakeHfResponse(
            {"name": "operator", "orgs": [{"name": "acme"}]}
        )
    ready = get_connection_status(
        config,
        MockYAMDriver(),
        hf_http_get=http_get,
        modal_config_path=tmp_path / "unused.toml",
    )
    incomplete_config = config.model_copy(
        update={"modal_proxy_token_secret": None}
    )
    incomplete = get_connection_status(
        incomplete_config,
        MockYAMDriver(),
        hf_http_get=http_get,
        modal_config_path=tmp_path / "unused.toml",
    )

    assert ready.setup_complete is True
    assert all(service.required for service in ready.services)
    assert ready.inference.model_dump() == {
        "mock_mode": False,
        "hf_configured": True,
        "modal_configured": True,
        "modal_proxy_configured": True,
    }
    assert incomplete.setup_complete is False
    assert incomplete.inference.modal_proxy_configured is False
    assert all(
        isinstance(value, bool)
        for value in incomplete.inference.model_dump().values()
    )
    payload = ready.model_dump_json()
    assert all(secret not in payload for secret in secret_values)


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


def test_hardware_arm_status_uses_sanitized_driver_diagnostic() -> None:
    config = AppConfig(
        _env_file=None,
        ctrl_pi_mock_mode=False,
        yam_leader_port="/dev/serial/by-id/operator-selected",
    )
    driver = RealYAMDriver.from_app_config(config)

    status = get_connection_status(config, driver)
    arms = next(service for service in status.services if service.id == "arms")

    assert status.mode == "hardware"
    assert arms.status == "missing"
    assert arms.detail == driver.diagnostic().detail
    assert "/dev/serial" not in arms.detail
    assert all(not arm.connected for arm in driver.list_arms())


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
