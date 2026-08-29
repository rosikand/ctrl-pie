from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_models import HFModelBrowser
from ctrl_pi.main import create_app

SHA_A = "a" * 40
SHA_B = "b" * 40


class FakeHubError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status_code)


class FakeModelHub:
    def __init__(self, items: list[Any], readmes: dict[tuple[str, str], Any]) -> None:
        self.items = items
        self.readmes = readmes
        self.identity: Any = {"name": "test-user", "orgs": [{"name": "acme"}]}
        self.whoami_error: Exception | None = None
        self.list_error: Exception | None = None
        self.info_errors: dict[str, Exception] = {}
        self.whoami_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.info_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []

    def whoami(self, **kwargs):
        self.whoami_calls.append(kwargs)
        if self.whoami_error is not None:
            raise self.whoami_error
        return self.identity

    def list_models(self, **kwargs):
        self.list_calls.append(kwargs)

        def results():
            if self.list_error is not None:
                raise self.list_error
            yield from self.items

        return results()

    def model_info(self, **kwargs):
        self.info_calls.append(kwargs)
        repo_id = kwargs["repo_id"]
        if repo_id in self.info_errors:
            raise self.info_errors[repo_id]
        sha = kwargs.get("revision") or (
            SHA_A if repo_id.endswith("/alpha") else SHA_B
        )
        return SimpleNamespace(
            id=repo_id,
            sha=sha,
            siblings=[SimpleNamespace(rfilename="checkpoints/100/model.safetensors")],
        )

    def hf_hub_download(self, **kwargs):
        self.download_calls.append(kwargs)
        value = self.readmes.get(
            (kwargs["repo_id"], kwargs["revision"]), FileNotFoundError("README")
        )
        if isinstance(value, Exception):
            raise value
        return value


def _model(
    repo_id: str,
    *,
    modified: datetime,
    private: bool | None = False,
    siblings: list[str] | None = None,
) -> SimpleNamespace:
    payload = {
        "id": repo_id,
        "sha": None,
        "gated": False,
        "last_modified": modified,
        "pipeline_tag": "robotics",
        "library_name": "lerobot",
        "tags": ["LeRobot", "robotics"],
        "siblings": [
            SimpleNamespace(rfilename=path)
            for path in (siblings or ["config.json", "checkpoint-50/model.safetensors"])
        ],
    }
    if private is not None:
        payload["private"] = private
    return SimpleNamespace(**payload)


@pytest.fixture
def model_app(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(
        """---
base_model: lerobot/act-base
datasets:
  - acme/yam-data
---
# ACT policy

Fine-tuned for block pickup on YAM.
"""
    )
    timestamp = datetime(2026, 8, 28, 10, 0, tzinfo=UTC)
    items = [
        _model("acme/zeta", modified=timestamp),
        _model("acme/alpha", modified=timestamp, private=True),
        _model("acme/broken", modified=datetime(2026, 1, 1, tzinfo=UTC)),
        _model("acme-extra/leak", modified=timestamp),
    ]
    api = FakeModelHub(items, {("acme/alpha", SHA_A): readme})
    api.info_errors["acme/broken"] = RuntimeError("per-item failure")
    tokens: list[str] = []
    browser = HFModelBrowser(
        hub_api_factory=lambda token: (tokens.append(token), api)[1]
    )
    app = create_app(hf_model_browser=browser)
    config = {
        "value": AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace="acme",
            hf_token="hf_model_secret",
        )
    }
    app.dependency_overrides[get_config] = lambda: config["value"]
    return app, api, tokens, config


def test_model_discovery_resolves_sha_pins_card_and_degrades_one_item(
    model_app,
) -> None:
    app, api, tokens, _ = model_app
    with TestClient(app) as client:
        response = client.get("/api/models")
        cached = client.get("/api/models")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, max-age=30"
    payload = response.json()
    assert payload["namespace"] == "acme"
    assert payload["total"] == 3
    assert [model["repo_id"] for model in payload["models"]] == [
        "acme/alpha",
        "acme/zeta",
        "acme/broken",
    ]
    alpha = payload["models"][0]
    assert alpha == {
        "repo_id": "acme/alpha",
        "name": "alpha",
        "revision": SHA_A,
        "hub_url": "https://huggingface.co/acme/alpha",
        "private": True,
        "gated": False,
        "last_modified": "2026-08-28T10:00:00Z",
        "pipeline_tag": "robotics",
        "library_name": "lerobot",
        "tags": ["LeRobot", "robotics"],
        "card": {
            "description": "Fine-tuned for block pickup on YAM.",
            "base_model": ["lerobot/act-base"],
            "datasets": ["acme/yam-data"],
        },
        "checkpoints": [
            "checkpoints/100/model.safetensors",
        ],
    }
    broken = payload["models"][2]
    assert broken["revision"] is None
    assert broken["card"] is None
    assert broken["checkpoints"] == []
    assert cached.json() == payload
    assert tokens and set(tokens) == {"hf_model_secret"}
    assert len(api.whoami_calls) == len(api.list_calls) == 1
    assert all(call["token"] == "hf_model_secret" for call in api.whoami_calls)
    assert all(call["token"] == "hf_model_secret" for call in api.list_calls)
    assert all(call["token"] == "hf_model_secret" for call in api.info_calls)
    assert all(call["token"] == "hf_model_secret" for call in api.download_calls)
    assert all(call["revision"] in {SHA_A, SHA_B} for call in api.download_calls)


def test_model_refresh_bypasses_caches_and_unknown_visibility_is_private(
    model_app,
) -> None:
    app, api, _, _ = model_app
    delattr(api.items[1], "private")
    with TestClient(app) as client:
        first = client.get("/api/models")
        refreshed = client.get("/api/models?refresh=true")

    assert first.status_code == refreshed.status_code == 200
    assert first.json()["models"][0]["private"] is True
    assert refreshed.headers["cache-control"] == "private, no-store"
    assert len(api.list_calls) == len(api.whoami_calls) == 2
    assert len(api.info_calls) == 6


def test_model_configuration_auth_namespace_and_hub_errors_are_safe(
    model_app,
) -> None:
    app, api, _, config = model_app
    secret = "hf_model_secret"
    with TestClient(app) as client:
        config["value"] = AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace=None,
            hf_token=None,
        )
        missing = client.get("/api/models")

        config["value"] = AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace="acme",
            hf_token=secret,
        )
        api.whoami_error = FakeHubError(f"bad {secret}", 401)
        unauthorized = client.get("/api/models?refresh=true")

        api.whoami_error = None
        api.identity = {"name": "different", "orgs": []}
        wrong_namespace = client.get("/api/models?refresh=true")

        api.identity = {"name": "test-user", "orgs": [{"name": "acme"}]}
        api.list_error = RuntimeError(f"network {secret}")
        hub_error = client.get("/api/models?refresh=true")

    assert missing.status_code == 503
    assert unauthorized.status_code == wrong_namespace.status_code == 403
    assert hub_error.status_code == 502
    assert hub_error.json()["detail"] == "Hugging Face model discovery failed."
    assert secret not in hub_error.text


def test_legacy_trainer_models_route_reuses_the_first_class_model_service(
    model_app,
) -> None:
    app, api, _, _ = model_app
    with TestClient(app) as client:
        canonical = client.get("/api/models")
        legacy = client.get("/api/trainer/models")

    assert canonical.status_code == legacy.status_code == 200
    assert canonical.json() == legacy.json()
    assert len(api.list_calls) == 1


def test_checkpoint_path_filter_rejects_unsafe_and_irrelevant_paths() -> None:
    siblings = [
        SimpleNamespace(rfilename="../checkpoint-1/model.safetensors"),
        SimpleNamespace(rfilename="config.json"),
        SimpleNamespace(rfilename="weights/model.safetensors"),
        SimpleNamespace(rfilename="checkpoints/20/model.safetensors"),
    ]

    result = HFModelBrowser._checkpoint_paths(siblings)

    assert result == [
        "checkpoints/20/model.safetensors",
        "weights/model.safetensors",
    ]


def test_listed_revision_is_pinned_during_model_info_resolution() -> None:
    item = _model(
        "acme/pinned",
        modified=datetime(2026, 8, 28, tzinfo=UTC),
    )
    item.sha = SHA_A
    api = FakeModelHub([item], {})
    browser = HFModelBrowser(hub_api_factory=lambda token: api)

    page = browser.list_namespace(
        namespace="acme",
        token="hf_token",
    )

    assert page.models[0].revision == SHA_A
    assert api.info_calls == [
        {
            "repo_id": "acme/pinned",
            "revision": SHA_A,
            "expand": ["sha", "siblings"],
            "token": "hf_token",
        }
    ]
