from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_episodes import HFEpisodeBrowser
from ctrl_pi.main import create_app

REVISION_A = "a" * 40
REVISION_B = "b" * 40
REPO_ID = "acme/yam-demo"
VIDEO_PATH = "videos/observation.images.workspace/chunk-000/file-000.mp4"


class FakeHubError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status_code)


class FakeEpisodeHub:
    def __init__(self) -> None:
        self.head_revision = REVISION_A
        self.identity: Any = {
            "name": "test-user",
            "orgs": [{"name": "acme"}],
        }
        self.whoami_error: Exception | None = None
        self.dataset_error: Exception | None = None
        self.download_error: Exception | None = None
        self.artifacts: dict[tuple[str, str], Path] = {}
        self.repo_files: dict[str, list[str]] = {}
        self.whoami_calls: list[dict[str, Any]] = []
        self.info_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []

    def whoami(self, **kwargs):
        self.whoami_calls.append(kwargs)
        if self.whoami_error is not None:
            raise self.whoami_error
        return self.identity

    def dataset_info(self, **kwargs):
        self.info_calls.append(kwargs)
        if self.dataset_error is not None:
            raise self.dataset_error
        revision = kwargs.get("revision") or self.head_revision
        if revision not in self.repo_files:
            raise FakeHubError("missing revision", 404)
        return SimpleNamespace(id=kwargs["repo_id"], sha=revision, tags=["LeRobot"])

    def list_repo_files(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.repo_files[kwargs["revision"]]

    def hf_hub_download(self, **kwargs):
        self.download_calls.append(kwargs)
        if self.download_error is not None:
            raise self.download_error
        key = (kwargs["revision"], kwargs["filename"])
        if key not in self.artifacts:
            raise FakeHubError("artifact missing", 404)
        return self.artifacts[key]


def _write_revision(root: Path, revision: str, video: bytes) -> dict[str, Path]:
    revision_root = root / revision
    info_path = revision_root / "meta" / "info.json"
    episodes_path = revision_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_path = revision_root / "data" / "chunk-000" / "file-000.parquet"
    video_path = revision_root / VIDEO_PATH
    info_path.parent.mkdir(parents=True)
    episodes_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "yam",
                "total_episodes": 2,
                "total_frames": 5,
                "total_tasks": 2,
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [2],
                        "names": ["joint", "gripper"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [2],
                        "names": ["joint", "gripper"],
                    },
                    "observation.images.workspace": {
                        "dtype": "video",
                        "shape": [8, 8, 3],
                        "names": ["height", "width", "channels"],
                    },
                },
            }
        )
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "tasks": ["Pick"],
                    "length": 2,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 0,
                    "dataset_to_index": 2,
                    "videos/observation.images.workspace/chunk_index": 0,
                    "videos/observation.images.workspace/file_index": 0,
                    "videos/observation.images.workspace/from_timestamp": 0.0,
                    "videos/observation.images.workspace/to_timestamp": 0.2,
                },
                {
                    "episode_index": 1,
                    "tasks": ["Place"],
                    "length": 3,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 2,
                    "dataset_to_index": 5,
                    "videos/observation.images.workspace/chunk_index": 0,
                    "videos/observation.images.workspace/file_index": 0,
                    "videos/observation.images.workspace/from_timestamp": 0.2,
                    "videos/observation.images.workspace/to_timestamp": 0.5,
                },
            ]
        ),
        episodes_path,
    )
    rows = []
    global_index = 0
    for episode_index, frame_count in ((0, 2), (1, 3)):
        for frame_index in range(frame_count):
            rows.append(
                {
                    "observation.state": [
                        float(episode_index),
                        frame_index / 10,
                    ],
                    "action": [
                        float(episode_index + 1),
                        frame_index / 5,
                    ],
                    "timestamp": frame_index / 10,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": episode_index,
                }
            )
            global_index += 1
    pq.write_table(pa.Table.from_pylist(rows), data_path)
    video_path.write_bytes(video)
    return {
        "meta/info.json": info_path,
        "meta/episodes/chunk-000/file-000.parquet": episodes_path,
        "data/chunk-000/file-000.parquet": data_path,
        VIDEO_PATH: video_path,
    }


@pytest.fixture
def episode_app(tmp_path: Path):
    api = FakeEpisodeHub()
    revision_a = _write_revision(tmp_path, REVISION_A, bytes(range(100)))
    revision_b = _write_revision(tmp_path, REVISION_B, b"B" * 100)
    for revision, artifacts in ((REVISION_A, revision_a), (REVISION_B, revision_b)):
        api.repo_files[revision] = list(artifacts)
        for name, path in artifacts.items():
            api.artifacts[(revision, name)] = path
    tokens: list[str] = []
    browser = HFEpisodeBrowser(
        hub_api_factory=lambda token: (tokens.append(token), api)[1]
    )
    app = create_app(hf_episode_browser=browser)
    config = {
        "value": AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace="acme",
            hf_token="hf_episode_secret",
        )
    }
    app.dependency_overrides[get_config] = lambda: config["value"]
    return app, api, tokens, config


def test_episode_list_and_pinned_detail_timeline(episode_app) -> None:
    app, api, tokens, _ = episode_app
    with TestClient(app) as client:
        listing = client.get("/api/datasets/yam-demo/episodes")
        unpinned = client.get("/api/datasets/yam-demo/episodes/1")
        detail = client.get(
            f"/api/datasets/yam-demo/episodes/1?revision={REVISION_A}"
        )

    assert listing.status_code == 200, listing.text
    assert listing.json() == {
        "repo_id": REPO_ID,
        "revision": REVISION_A,
        "fps": 10.0,
        "state_names": ["joint", "gripper"],
        "action_names": ["joint", "gripper"],
        "video_key": "observation.images.workspace",
        "total_episodes": 2,
        "episodes": [
            {
                "episode_index": 0,
                "tasks": ["Pick"],
                "frame_count": 2,
                "duration_seconds": 0.2,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "video_from_timestamp": 0.0,
                "video_to_timestamp": 0.2,
            },
            {
                "episode_index": 1,
                "tasks": ["Place"],
                "frame_count": 3,
                "duration_seconds": 0.3,
                "dataset_from_index": 2,
                "dataset_to_index": 5,
                "video_from_timestamp": 0.2,
                "video_to_timestamp": 0.5,
            },
        ],
    }
    assert unpinned.status_code == 422
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload["revision"] == REVISION_A
    assert payload["episode"] == listing.json()["episodes"][1]
    assert payload["video_url"] == (
        f"/api/datasets/yam-demo/episodes/1/video?revision={REVISION_A}"
    )
    assert payload["frames"] == [
        {
            "timestamp": 0.0,
            "frame_index": 0,
            "state": [1.0, 0.0],
            "action": [2.0, 0.0],
        },
        {
            "timestamp": pytest.approx(0.1),
            "frame_index": 1,
            "state": [1.0, pytest.approx(0.1)],
            "action": [2.0, pytest.approx(0.2)],
        },
        {
            "timestamp": pytest.approx(0.2),
            "frame_index": 2,
            "state": [1.0, pytest.approx(0.2)],
            "action": [2.0, pytest.approx(0.4)],
        },
    ]
    assert tokens and set(tokens) == {"hf_episode_secret"}
    assert all(call["token"] == "hf_episode_secret" for call in api.whoami_calls)
    assert all(call["token"] == "hf_episode_secret" for call in api.info_calls)
    assert all(call["token"] == "hf_episode_secret" for call in api.list_calls)
    assert all(call["token"] == "hf_episode_secret" for call in api.download_calls)
    assert all(call["revision"] in {REVISION_A, REVISION_B} for call in api.download_calls)


def test_media_url_remains_on_detail_revision_when_head_advances(episode_app) -> None:
    app, api, _, _ = episode_app
    with TestClient(app) as client:
        listing = client.get("/api/datasets/yam-demo/episodes").json()
        api.head_revision = REVISION_B
        detail = client.get(
            f"/api/datasets/yam-demo/episodes/0?revision={listing['revision']}"
        )
        video = client.get(detail.json()["video_url"])

    assert detail.status_code == 200
    assert detail.json()["revision"] == REVISION_A
    assert video.status_code == 200
    assert video.content == bytes(range(100))
    assert api.info_calls[-1]["revision"] == REVISION_A
    assert api.download_calls[-1]["revision"] == REVISION_A


@pytest.mark.parametrize(
    ("range_header", "expected_start", "expected_end"),
    [
        ("bytes=10-19", 10, 19),
        ("bytes=90-", 90, 99),
        ("bytes=-7", 93, 99),
        ("bytes=95-999", 95, 99),
        ("bytes=-1000", 0, 99),
    ],
)
def test_video_single_range_forms(
    episode_app,
    range_header: str,
    expected_start: int,
    expected_end: int,
) -> None:
    app, _, _, _ = episode_app
    url = f"/api/datasets/yam-demo/episodes/0/video?revision={REVISION_A}"
    with TestClient(app) as client:
        response = client.get(url, headers={"Range": range_header})

    expected = bytes(range(100))[expected_start : expected_end + 1]
    assert response.status_code == 206
    assert response.content == expected
    assert response.headers["content-range"] == (
        f"bytes {expected_start}-{expected_end}/100"
    )
    assert response.headers["content-length"] == str(len(expected))
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["cache-control"] == (
        "private, max-age=31536000, immutable"
    )
    assert response.headers["etag"].startswith('"')


@pytest.mark.parametrize(
    "range_header",
    [
        "items=0-1",
        "bytes=",
        "bytes=1-2,4-5",
        "bytes=100-",
        "bytes=9-3",
        "bytes=-0",
        "bytes=" + "9" * 5000 + "-",
    ],
)
def test_invalid_or_unsatisfiable_video_ranges_return_416(
    episode_app, range_header: str
) -> None:
    app, _, _, _ = episode_app
    url = f"/api/datasets/yam-demo/episodes/0/video?revision={REVISION_A}"
    with TestClient(app) as client:
        response = client.get(url, headers={"Range": range_header})

    assert response.status_code == 416
    assert response.content == b""
    assert response.headers["content-range"] == "bytes */100"
    assert response.headers["content-length"] == "0"
    assert response.headers["accept-ranges"] == "bytes"


def test_video_get_and_head_have_parity_without_head_body(episode_app) -> None:
    app, _, _, _ = episode_app
    url = f"/api/datasets/yam-demo/episodes/0/video?revision={REVISION_A}"
    with TestClient(app) as client:
        get_response = client.get(url)
        head_response = client.head(url)
        range_head = client.head(url, headers={"Range": "bytes=4-8"})

    assert get_response.status_code == head_response.status_code == 200
    assert get_response.content == bytes(range(100))
    assert head_response.content == b""
    for header in (
        "content-length",
        "content-type",
        "accept-ranges",
        "etag",
        "cache-control",
    ):
        assert head_response.headers[header] == get_response.headers[header]
    assert range_head.status_code == 206
    assert range_head.content == b""
    assert range_head.headers["content-range"] == "bytes 4-8/100"
    assert range_head.headers["content-length"] == "5"


def test_episode_config_auth_hub_and_input_errors_are_safe(episode_app) -> None:
    app, api, _, config = episode_app
    secret = "hf_episode_secret"
    with TestClient(app) as client:
        bad_repo = client.get("/api/datasets/bad--repo/episodes")
        bad_revision = client.get(
            "/api/datasets/yam-demo/episodes/0?revision=main"
        )
        missing_episode = client.get(
            f"/api/datasets/yam-demo/episodes/99?revision={REVISION_A}"
        )

        config["value"] = AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace=None,
            hf_token=None,
        )
        missing_config = client.get("/api/datasets/yam-demo/episodes")

        config["value"] = AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace="acme",
            hf_token=secret,
        )
        api.whoami_error = FakeHubError(f"bad {secret}", 401)
        unauthenticated = client.get("/api/datasets/yam-demo/episodes")

        api.whoami_error = None
        api.identity = None
        malformed_identity = client.get("/api/datasets/yam-demo/episodes")

        api.identity = {"name": "test-user", "orgs": [{"name": "acme"}]}
        api.download_error = RuntimeError(f"network leaked {secret}")
        hub_failure = client.get("/api/datasets/yam-demo/episodes")

    assert bad_repo.status_code == 422
    assert bad_revision.status_code == 422
    assert missing_episode.status_code == 404
    assert missing_config.status_code == 503
    assert unauthenticated.status_code == 403
    assert malformed_identity.status_code == 502
    assert hub_failure.status_code == 502
    assert hub_failure.json()["detail"] == "Hugging Face dataset access failed."
    assert secret not in hub_failure.text


def test_unsafe_metadata_path_and_nonfinite_vector_are_rejected(
    episode_app,
) -> None:
    app, api, _, _ = episode_app
    info_path = api.artifacts[(REVISION_A, "meta/info.json")]
    original_info = info_path.read_text()
    unsafe = json.loads(original_info)
    unsafe["data_path"] = "../{file_index}.parquet"
    info_path.write_text(json.dumps(unsafe))
    with TestClient(app) as client:
        unsafe_response = client.get("/api/datasets/yam-demo/episodes")

    assert unsafe_response.status_code == 422

    info_path.write_text(original_info)
    data_path = api.artifacts[(REVISION_A, "data/chunk-000/file-000.parquet")]
    table = pq.read_table(data_path).to_pylist()
    table[0]["action"] = [float("nan"), 0.0]
    pq.write_table(pa.Table.from_pylist(table), data_path)
    with TestClient(app) as client:
        nonfinite = client.get(
            f"/api/datasets/yam-demo/episodes/0?revision={REVISION_A}"
        )

    assert nonfinite.status_code == 422
    assert "not finite" in nonfinite.json()["detail"]


def test_zero_episode_dataset_returns_empty_list(tmp_path: Path) -> None:
    api = FakeEpisodeHub()
    info_path = tmp_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 0,
                "total_frames": 0,
                "fps": 20,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {"shape": [1], "names": ["joint"]},
                    "action": {"shape": [1], "names": ["joint"]},
                },
            }
        )
    )
    api.repo_files[REVISION_A] = ["meta/info.json"]
    api.artifacts[(REVISION_A, "meta/info.json")] = info_path
    browser = HFEpisodeBrowser(hub_api_factory=lambda token: api)

    response = browser.list_episodes(
        namespace="acme",
        repo_name="yam-demo",
        token="hf_token",
    )

    assert response.total_episodes == 0
    assert response.episodes == []
