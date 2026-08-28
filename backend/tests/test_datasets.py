from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_datasets import DatasetCursorError, HFDatasetBrowser
from ctrl_pi.main import create_app


class FakeHubError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status_code)


class FakeHubApi:
    def __init__(self, items: list[Any], files: dict[tuple[str, str, str], Any]) -> None:
        self.items = items
        self.files = files
        self.identity: dict[str, Any] = {
            "name": "test-user",
            "orgs": [{"name": "acme"}],
        }
        self.whoami_error: Exception | None = None
        self.list_error: Exception | None = None
        self.whoami_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []

    def whoami(self, **kwargs):
        self.whoami_calls.append(kwargs)
        if self.whoami_error is not None:
            raise self.whoami_error
        return self.identity

    def list_datasets(self, **kwargs):
        self.list_calls.append(kwargs)

        def results():
            if self.list_error is not None:
                raise self.list_error
            yield from self.items

        return results()

    def hf_hub_download(self, **kwargs):
        self.download_calls.append(kwargs)
        key = (kwargs["repo_id"], kwargs["filename"], kwargs["revision"])
        value = self.files.get(key, FileNotFoundError(kwargs["filename"]))
        if isinstance(value, Exception):
            raise value
        return value


def _dataset(
    repo_id: str,
    *,
    revision: str | None,
    last_modified: datetime | None,
    tags: list[str] | None = None,
    private: bool = False,
    gated: bool | str = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=repo_id,
        sha=revision,
        private=private,
        gated=gated,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        last_modified=last_modified,
        tags=["LeRobot"] if tags is None else tags,
    )


@pytest.fixture
def dataset_app(tmp_path: Path):
    timestamp = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
    readme = tmp_path / "README-alpha.md"
    readme.write_text(
        """---
pretty_name: Alpha Dataset
license: apache-2.0
task_categories:
  - robotics
---
# Alpha fallback

Place the blue block into the tray.
"""
    )
    info = tmp_path / "info-alpha.json"
    info.write_text(
        """{
  "codebase_version": "v3.0",
  "robot_type": "yam",
  "fps": 20,
  "total_episodes": 3,
  "total_frames": 120,
  "total_tasks": 1,
  "features": {"observation.state": {}, "action": {}}
}
"""
    )
    malformed = tmp_path / "info-malformed.json"
    malformed.write_text("{not-json")

    items = [
        _dataset("acme/zeta", revision="sha-z", last_modified=timestamp, gated="manual"),
        _dataset("acme/alpha", revision="sha-a", last_modified=timestamp, private=True),
        _dataset(
            "acme/old",
            revision=None,
            last_modified=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        _dataset("acme-extra/leak", revision="sha-x", last_modified=timestamp),
        _dataset(
            "acme/not-lerobot",
            revision="sha-n",
            last_modified=timestamp,
            tags=["robotics"],
        ),
    ]
    files: dict[tuple[str, str, str], Any] = {
        ("acme/alpha", "README.md", "sha-a"): readme,
        ("acme/alpha", "meta/info.json", "sha-a"): info,
        ("acme/zeta", "README.md", "sha-z"): FileNotFoundError("README"),
        ("acme/zeta", "meta/info.json", "sha-z"): malformed,
    }
    api = FakeHubApi(items, files)
    factory_tokens: list[str] = []
    browser = HFDatasetBrowser(
        hub_api_factory=lambda token: (factory_tokens.append(token), api)[1],
    )
    app = create_app(hf_dataset_browser=browser)
    config = {
        "value": AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace="acme",
            hf_token="hf_dataset_secret",
        )
    }
    app.dependency_overrides[get_config] = lambda: config["value"]
    return app, api, factory_tokens, config, browser


def test_dataset_discovery_contract_keyset_pagination_and_pinned_metadata(
    dataset_app,
) -> None:
    app, api, factory_tokens, _, _ = dataset_app
    with TestClient(app) as client:
        first = client.get("/api/datasets?limit=1&namespace=ignored")
        cached = client.get("/api/datasets?limit=1")
        cursor = first.json()["next_cursor"]
        second = client.get(f"/api/datasets?limit=1&cursor={cursor}")

    assert first.status_code == 200, first.text
    assert first.headers["cache-control"] == "private, max-age=30"
    payload = first.json()
    assert payload["namespace"] == "acme"
    assert payload["total"] == 3
    assert payload["next_cursor"]
    assert payload["fetched_at"]
    assert [item["repo_id"] for item in payload["datasets"]] == ["acme/alpha"]
    alpha = payload["datasets"][0]
    assert alpha == {
        "repo_id": "acme/alpha",
        "name": "alpha",
        "revision": "sha-a",
        "hub_url": "https://huggingface.co/datasets/acme/alpha",
        "private": True,
        "gated": False,
        "created_at": "2026-01-01T00:00:00Z",
        "last_modified": "2026-08-20T12:00:00Z",
        "tags": ["LeRobot"],
        "card": {
            "title": "Alpha Dataset",
            "description": "Place the blue block into the tray.",
            "license": "apache-2.0",
            "task_categories": ["robotics"],
        },
        "lerobot": {
            "codebase_version": "v3.0",
            "robot_type": "yam",
            "fps": 20.0,
            "total_episodes": 3,
            "total_frames": 120,
            "total_tasks": 1,
            "features": ["action", "observation.state"],
        },
    }
    assert cached.json() == payload
    assert [item["repo_id"] for item in second.json()["datasets"]] == ["acme/zeta"]
    assert second.json()["datasets"][0]["gated"] is True
    assert second.json()["datasets"][0]["card"] is None
    assert second.json()["datasets"][0]["lerobot"] is None

    assert factory_tokens and set(factory_tokens) == {"hf_dataset_secret"}
    assert len(api.whoami_calls) == 1
    assert api.whoami_calls[0]["token"] == "hf_dataset_secret"
    assert len(api.list_calls) == 1
    assert api.list_calls[0] == {
        "author": "acme",
        "filter": "LeRobot",
        "sort": "last_modified",
        "direction": -1,
        "expand": HFDatasetBrowser._EXPAND,
        "token": "hf_dataset_secret",
    }
    assert all(call["revision"].startswith("sha-") for call in api.download_calls)
    assert all(call["token"] == "hf_dataset_secret" for call in api.download_calls)
    assert all(call["repo_type"] == "dataset" for call in api.download_calls)


def test_refresh_bypasses_enumeration_and_revision_caches(dataset_app) -> None:
    app, api, _, _, _ = dataset_app
    with TestClient(app) as client:
        first = client.get("/api/datasets?limit=1")
        refreshed = client.get("/api/datasets?limit=1&refresh=true")

    assert first.status_code == refreshed.status_code == 200
    assert first.headers["cache-control"] == "private, max-age=30"
    assert refreshed.headers["cache-control"] == "private, no-store"
    assert len(api.whoami_calls) == 2
    assert len(api.list_calls) == 2
    alpha_downloads = [
        call for call in api.download_calls if call["repo_id"] == "acme/alpha"
    ]
    assert len(alpha_downloads) == 4


def test_unknown_hub_visibility_is_conservatively_private() -> None:
    item = SimpleNamespace(
        id="acme/unknown-visibility",
        sha="sha",
        gated=False,
        created_at=None,
        last_modified=None,
        tags=["LeRobot"],
    )

    descriptor = HFDatasetBrowser._descriptor(item, "acme")

    assert descriptor is not None
    assert descriptor.private is True


def test_invalid_and_cross_namespace_cursors_are_rejected_without_hub_call(
    dataset_app,
) -> None:
    app, api, _, _, browser = dataset_app
    with TestClient(app) as client:
        invalid = client.get("/api/datasets?cursor=not-a-cursor")
        first = client.get("/api/datasets?limit=1")

    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "Dataset cursor is invalid or expired."
    assert len(api.list_calls) == 1
    with pytest.raises(DatasetCursorError):
        browser.list_namespace(
            namespace="other",
            token="hf_dataset_secret",
            limit=1,
            cursor=first.json()["next_cursor"],
            refresh=False,
        )
    assert len(api.list_calls) == 1


def test_dataset_discovery_configuration_and_safe_hub_errors(dataset_app) -> None:
    app, api, _, config, _ = dataset_app
    with TestClient(app) as client:
        config["value"] = AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace=None,
            hf_token=None,
        )
        missing = client.get("/api/datasets")

        config["value"] = AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace="acme",
            hf_token="hf_dataset_secret",
        )
        api.whoami_error = FakeHubError("bad hf_dataset_secret", 401)
        unauthenticated = client.get("/api/datasets?refresh=true")

        api.whoami_error = None
        api.identity = {"name": "someone-else", "orgs": []}
        wrong_namespace = client.get("/api/datasets?refresh=true")

        api.identity = {"name": "test-user", "orgs": [{"name": "acme"}]}
        api.list_error = RuntimeError("upstream leaked hf_dataset_secret")
        hub_failure = client.get("/api/datasets?refresh=true")

    assert missing.status_code == 503
    assert unauthenticated.status_code == 403
    assert wrong_namespace.status_code == 403
    assert hub_failure.status_code == 502
    assert hub_failure.json()["detail"] == "Hugging Face dataset discovery failed."
    assert "hf_dataset_secret" not in hub_failure.text


def test_independent_cache_expiry_for_enumeration_and_revision_metadata(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Cached\n\nCache test.")
    info = tmp_path / "info.json"
    info.write_text("{\"fps\": 20, \"features\": {}}")
    item = _dataset(
        "acme/cached",
        revision="cached-sha",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )
    api = FakeHubApi(
        [item],
        {
            ("acme/cached", "README.md", "cached-sha"): readme,
            ("acme/cached", "meta/info.json", "cached-sha"): info,
        },
    )
    clock = {"value": 0.0}
    browser = HFDatasetBrowser(
        hub_api_factory=lambda token: api,
        enumeration_ttl_seconds=10,
        revision_ttl_seconds=20,
        monotonic=lambda: clock["value"],
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    def browse() -> None:
        browser.list_namespace(
            namespace="acme",
            token="hf_token",
            limit=24,
            cursor=None,
            refresh=False,
        )

    browse()
    clock["value"] = 11
    browse()
    assert len(api.list_calls) == 2
    assert len(api.download_calls) == 2

    clock["value"] = 21
    browse()
    assert len(api.list_calls) == 3
    assert len(api.download_calls) == 4


def test_lerobot_card_description_uses_dataset_description_section(
    tmp_path: Path,
) -> None:
    from lerobot.datasets.lerobot_dataset import create_lerobot_dataset_card

    card = create_lerobot_dataset_card(
        tags=["ctrl-pi", "yam"],
        dataset_info={"fps": 20},
        license="apache-2.0",
        repo_id="acme/demo",
        dataset_description="Pick up the blue block.",
    )
    path = tmp_path / "README.md"
    card.save(path)

    parsed = HFDatasetBrowser._parse_card(path)

    assert parsed is not None
    assert parsed.description == "Pick up the blue block."
