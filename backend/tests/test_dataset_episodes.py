from __future__ import annotations

import json
from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_episodes import (
    MAX_PARQUET_SCAN_ROWS,
    MAX_TIMELINE_SAMPLES,
    MAX_TIMELINE_SCALARS,
    EpisodeFormatError,
    HFEpisodeBrowser,
)
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


def _write_large_revision(
    root: Path,
    *,
    frame_count: int,
    state_size: int = 2,
    action_size: int = 2,
) -> tuple[FakeEpisodeHub, HFEpisodeBrowser, Path]:
    revision_root = root / f"large-{frame_count}"
    info_path = revision_root / "meta" / "info.json"
    episodes_path = (
        revision_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    data_path = revision_root / "data" / "chunk-000" / "file-000.parquet"
    info_path.parent.mkdir(parents=True)
    episodes_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 1,
                "total_frames": frame_count,
                "fps": 50,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [state_size],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [action_size],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "tasks": ["Long task"],
                    "length": frame_count,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 0,
                    "dataset_to_index": frame_count,
                }
            ]
        ),
        episodes_path,
    )
    rows = [
        {
            "observation.state": [float(index)] * state_size,
            "action": [float(index + 1)] * action_size,
            "timestamp": index / 50,
            "frame_index": index,
            "episode_index": 0,
            "index": index,
        }
        for index in range(frame_count)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows),
        data_path,
        row_group_size=113,
    )
    api = FakeEpisodeHub()
    artifacts = {
        "meta/info.json": info_path,
        "meta/episodes/chunk-000/file-000.parquet": episodes_path,
        "data/chunk-000/file-000.parquet": data_path,
    }
    api.repo_files[REVISION_A] = list(artifacts)
    for name, path in artifacts.items():
        api.artifacts[(REVISION_A, name)] = path
    return api, HFEpisodeBrowser(hub_api_factory=lambda _token: api), data_path


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
    assert payload["sampled_frame_count"] == 3
    assert payload["frames_truncated"] is False
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


@pytest.mark.parametrize(
    ("frame_count", "expected_truncated"),
    [
        (MAX_TIMELINE_SAMPLES, False),
        (MAX_TIMELINE_SAMPLES + 1, True),
        (MAX_TIMELINE_SAMPLES * 5 + 17, True),
    ],
)
def test_episode_detail_returns_a_strict_first_last_inclusive_sample(
    tmp_path: Path,
    frame_count: int,
    expected_truncated: bool,
) -> None:
    _api, browser, _data_path = _write_large_revision(
        tmp_path,
        frame_count=frame_count,
    )

    detail = browser.episode_detail(
        namespace="acme",
        repo_name="yam-demo",
        episode_index=0,
        revision=REVISION_A,
        token="hf_token",
    )

    expected_count = min(frame_count, MAX_TIMELINE_SAMPLES)
    frame_indices = [frame.frame_index for frame in detail.frames]
    assert detail.sampled_frame_count == expected_count
    assert detail.frames_truncated is expected_truncated
    assert len(detail.frames) == expected_count
    assert frame_indices[0] == 0
    assert frame_indices[-1] == frame_count - 1
    assert frame_indices == sorted(set(frame_indices))
    assert frame_indices == [
        sample_index * (frame_count - 1) // (expected_count - 1)
        for sample_index in range(expected_count)
    ]
    assert detail.frames[0].timestamp == 0
    assert detail.frames[-1].timestamp == pytest.approx((frame_count - 1) / 50)


def test_episode_detail_applies_scalar_budget_at_max_vector_dimensions(
    tmp_path: Path,
) -> None:
    vector_width = 2_048
    expected_count = MAX_TIMELINE_SCALARS // vector_width
    frame_count = expected_count + 1
    _api, browser, _data_path = _write_large_revision(
        tmp_path,
        frame_count=frame_count,
        state_size=1_024,
        action_size=1_024,
    )

    detail = browser.episode_detail(
        namespace="acme",
        repo_name="yam-demo",
        episode_index=0,
        revision=REVISION_A,
        token="hf_token",
    )

    indices = [frame.frame_index for frame in detail.frames]
    assert detail.sampled_frame_count == len(detail.frames) == expected_count
    assert detail.frames_truncated is True
    assert indices[0] == 0
    assert indices[-1] == frame_count - 1
    assert indices == sorted(set(indices))


def test_episode_detail_rejects_parquet_bounds_that_disagree_with_index(
    tmp_path: Path,
) -> None:
    frame_count = MAX_TIMELINE_SAMPLES + 1
    _api, browser, data_path = _write_large_revision(
        tmp_path,
        frame_count=frame_count,
    )
    table = pq.read_table(data_path).slice(0, frame_count - 1)
    pq.write_table(table, data_path, row_group_size=113)

    with pytest.raises(
        EpisodeFormatError,
        match="data parquet bounds do not match",
    ):
        browser.episode_detail(
            namespace="acme",
            repo_name="yam-demo",
            episode_index=0,
            revision=REVISION_A,
            token="hf_token",
        )


@pytest.mark.parametrize(
    ("consumed_rows", "consumed_row_groups", "metadata_rows"),
    [
        (0, 0, 100_000_000),
        (2, 2, 1),
    ],
)
def test_episode_index_rejects_metadata_bounds_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    consumed_rows: int,
    consumed_row_groups: int,
    metadata_rows: int,
) -> None:
    artifacts = _write_revision(tmp_path, REVISION_A, b"video")
    browser = HFEpisodeBrowser()
    layout = browser._parse_layout(artifacts["meta/info.json"])
    decode_attempted: list[bool] = []

    class OversizedMetadata:
        num_rows = metadata_rows
        num_row_groups = 1

        @staticmethod
        def row_group(_index: int) -> SimpleNamespace:
            raise AssertionError("row-group metadata must not be scanned past bounds")

    class OversizedParquetFile:
        metadata = OversizedMetadata()

        def __init__(self, _path: Path) -> None:
            pass

        def iter_batches(self, **_kwargs: Any):
            decode_attempted.append(True)
            return iter(())

    monkeypatch.setattr(pq, "ParquetFile", OversizedParquetFile)

    with pytest.raises(EpisodeFormatError, match="metadata bounds"):
        browser._read_episode_records(
            artifacts["meta/episodes/chunk-000/file-000.parquet"],
            layout,
            consumed_rows=consumed_rows,
            consumed_row_groups=consumed_row_groups,
        )

    assert decode_attempted == []


def test_episode_detail_sanitizes_parquet_reader_failures(tmp_path: Path) -> None:
    _api, browser, data_path = _write_large_revision(tmp_path, frame_count=3)
    data_path.write_bytes(b"not a parquet file")

    with pytest.raises(EpisodeFormatError) as caught:
        browser.episode_detail(
            namespace="acme",
            repo_name="yam-demo",
            episode_index=0,
            revision=REVISION_A,
            token="hf_token",
        )

    rendered = "".join(traceback.format_exception(caught.value))
    assert str(caught.value) == "The episode data parquet is malformed."
    assert str(data_path) not in rendered


def test_episode_detail_rejects_oversized_selected_row_group_before_scanning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, browser, data_path = _write_large_revision(tmp_path, frame_count=3)
    oversized_rows = max(100_000_000, MAX_PARQUET_SCAN_ROWS + 1)
    revision_root = data_path.parents[2]
    info_path = revision_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_frames"] = oversized_rows
    info_path.write_text(json.dumps(info), encoding="utf-8")
    episodes_path = (
        revision_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "tasks": ["Oversized row group"],
                    "length": oversized_rows,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 0,
                    "dataset_to_index": oversized_rows,
                }
            ]
        ),
        episodes_path,
    )
    dataset = browser._load_dataset(
        namespace="acme",
        repo_name="yam-demo",
        token="hf_token",
        revision=REVISION_A,
    )
    episode = browser._episode(dataset, 0)
    scan_attempted: list[bool] = []

    class OversizedMetadata:
        num_rows = oversized_rows
        num_row_groups = 1

        @staticmethod
        def row_group(index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(num_rows=oversized_rows)

    class OversizedParquetFile:
        metadata = OversizedMetadata()

        def __init__(self, _path: Path) -> None:
            pass

        def iter_batches(self, **_kwargs: Any):
            scan_attempted.append(True)
            return iter(())

    monkeypatch.setattr(pq, "ParquetFile", OversizedParquetFile)

    with pytest.raises(EpisodeFormatError, match="scan budget"):
        browser._load_frames(api, dataset, episode, "hf_token")

    assert scan_attempted == []


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
