from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ctrl_pi.camera import MockCamera
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import Base, get_db
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.hf import (
    ConvertedDataset,
    DatasetUploadResult,
    HFDatasetUploader,
    HubUploadError,
    RecordingUploadSource,
    UploadConflictError,
)
from ctrl_pi.main import create_app
from ctrl_pi.models import Recording
from ctrl_pi.recording import RecordingManager


class StubUploader:
    def __init__(self, failure: Exception | None = None) -> None:
        self.active: set[str] = set()
        self.failure = failure
        self.calls: list[dict[str, Any]] = []
        self.on_release: Callable[[str], None] | None = None

    def repo_id(self, namespace: str, repo_name: str) -> str:
        return f"{namespace}/{repo_name}"

    def reserve(self, recording_id: str, repo_id: str) -> None:
        self.active.add(recording_id)

    def release(self, recording_id: str, repo_id: str) -> None:
        if self.on_release is not None:
            self.on_release(recording_id)
        self.active.discard(recording_id)

    def is_active(self, recording_id: str) -> bool:
        return recording_id in self.active

    def upload(
        self,
        source: RecordingUploadSource,
        namespace: str,
        repo_name: str,
        private: bool,
        token: str,
    ) -> DatasetUploadResult:
        self.calls.append(
            {
                "source": source,
                "namespace": namespace,
                "repo_name": repo_name,
                "private": private,
                "token": token,
            }
        )
        if self.failure is not None:
            raise self.failure
        return DatasetUploadResult(
            repo_id=f"{namespace}/{repo_name}",
            repo_url=f"https://huggingface.co/datasets/{namespace}/{repo_name}",
            revision="abc123",
            total_frames=3,
            fps=10,
        )


@pytest.fixture
def upload_app(tmp_path: Path):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    driver = MockYAMDriver()
    camera = MockCamera(width=96, height=64)
    manager = RecordingManager(driver, camera, tmp_path / "recordings")
    uploader = StubUploader()
    app = create_app(driver, camera, manager, uploader)  # type: ignore[arg-type]

    def override_db() -> Iterator[Session]:
        with Session(engine, expire_on_commit=False) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_config] = lambda: AppConfig(
        _env_file=None,
        database_url=None,
        hf_namespace="test-user",
        hf_token="hf_super_secret",
        recording_staging_dir=tmp_path / "recordings",
    )
    return app, engine, uploader


def _ready_recording(client: TestClient, engine) -> dict[str, Any]:
    created = client.post(
        "/api/recordings",
        json={
            "name": "HF upload",
            "task": "Place the block",
            "leader_robot_id": "yam-leader",
            "follower_robot_id": "yam-follower",
        },
    )
    assert created.status_code == 201
    payload = created.json()
    with Session(engine) as db:
        recording = db.get(Recording, uuid.UUID(payload["id"]))
        assert recording is not None
        recording.status = "ready"
        recording.episode_count = 1
        recording.duration_seconds = 0.3
        recording.recording_metadata = {
            "episodes": [
                {
                    "index": 0,
                    "sample_count": 3,
                    "duration_seconds": 0.3,
                    "fps": 10,
                    "artifact_key": f"{payload['id']}/episode_000000",
                }
            ]
        }
        db.commit()
    return payload


def test_upload_api_defaults_private_and_persists_verified_result(upload_app) -> None:
    app, engine, uploader = upload_app
    statuses_at_release: list[str] = []

    def capture_status(recording_id: str) -> None:
        with Session(engine) as db:
            recording = db.get(Recording, uuid.UUID(recording_id))
            assert recording is not None
            statuses_at_release.append(recording.status)

    uploader.on_release = capture_status
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]
        response = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "yam-block-pick"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["repo_id"] == "test-user/yam-block-pick"
    assert payload["revision"] == "abc123"
    assert payload["recording"]["status"] == "uploaded"
    assert payload["recording"]["hf_repo_id"] == "test-user/yam-block-pick"
    assert uploader.calls[0]["private"] is True
    assert uploader.calls[0]["token"] == "hf_super_secret"
    assert statuses_at_release == ["uploaded"]

    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(recording_id))
        assert stored is not None
        assert stored.status == "uploaded"
        upload = stored.recording_metadata["upload"]
        assert upload["lerobot_version"] == "0.4.4"
        assert upload["owner_recording_id"] == recording_id
        assert upload["remote_repo_created"] is True
        assert "local_dataset_key" not in upload
        assert "hf_super_secret" not in json.dumps(stored.recording_metadata)


def test_create_recording_rejects_server_owned_upload_metadata(upload_app) -> None:
    app, _, _ = upload_app
    with TestClient(app) as client:
        response = client.post(
            "/api/recordings",
            json={
                "name": "Forged ownership",
                "task": "Do not overwrite",
                "leader_robot_id": "yam-leader",
                "follower_robot_id": "yam-follower",
                "metadata": {"upload": {"repo_id": "test-user/existing"}},
            },
        )

    assert response.status_code == 422


def test_upload_failure_is_sanitized_and_retry_target_is_locked(upload_app) -> None:
    app, engine, uploader = upload_app
    uploader.failure = RuntimeError("network rejected hf_super_secret")
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]
        failed = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "first-target"},
        )
        changed_target = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "different-target"},
        )

    assert failed.status_code == 502
    assert "hf_super_secret" not in failed.text
    assert changed_target.status_code == 409
    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(recording_id))
        assert stored is not None
        assert stored.status == "failed"
        assert "hf_super_secret" not in json.dumps(stored.recording_metadata)


def test_repo_collision_restores_session_without_claiming_target(upload_app) -> None:
    app, engine, uploader = upload_app
    uploader.failure = UploadConflictError("target already exists")
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]
        collision = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "occupied"},
        )
        uploader.failure = None
        alternate = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "available"},
        )

    assert collision.status_code == 409
    assert alternate.status_code == 200, alternate.text
    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(recording_id))
        assert stored is not None
        assert stored.hf_repo_id == "test-user/available"


def test_post_create_hub_failure_persists_retry_ownership(upload_app) -> None:
    app, engine, uploader = upload_app
    uploader.failure = HubUploadError(
        "settings failed", remote_repo_created=True
    )
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]
        failed = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "partial-target"},
        )

    assert failed.status_code == 502
    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(recording_id))
        assert stored is not None
        upload = stored.recording_metadata["upload"]
        assert upload["owner_recording_id"] == recording_id
        assert upload["remote_repo_created"] is True


def test_stale_upload_is_reconciled_and_live_upload_is_immutable(upload_app) -> None:
    app, engine, uploader = upload_app
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]
        with Session(engine) as db:
            stored = db.get(Recording, uuid.UUID(recording_id))
            assert stored is not None
            stored.status = "uploading"
            stored.recording_metadata = {
                **stored.recording_metadata,
                "upload": {"status": "uploading", "repo_id": "test-user/stale"},
            }
            db.commit()

        listed = client.get("/api/recordings")
        assert listed.json()["recordings"][0]["status"] == "failed"

        with Session(engine) as db:
            stored = db.get(Recording, uuid.UUID(recording_id))
            assert stored is not None
            stored.status = "uploading"
            db.commit()
        state = client.get(f"/api/recordings/{recording_id}/state")
        assert state.status_code == 200
        assert state.json()["status"] == "failed"

        with Session(engine) as db:
            stored = db.get(Recording, uuid.UUID(recording_id))
            assert stored is not None
            stored.status = "uploading"
            db.commit()
        uploader.active.add(recording_id)
        teleop = client.post(f"/api/recordings/{recording_id}/teleop/start")

    assert teleop.status_code == 409
    assert "immutable" in teleop.json()["detail"]


def test_failed_upload_target_blocks_new_episode_but_allows_retry(upload_app) -> None:
    app, engine, uploader = upload_app
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]
        with Session(engine) as db:
            stored = db.get(Recording, uuid.UUID(recording_id))
            assert stored is not None
            stored.status = "failed"
            stored.recording_metadata = {
                **stored.recording_metadata,
                "upload": {
                    "status": "failed",
                    "repo_id": "test-user/yam-data",
                },
            }
            db.commit()

        episode = client.post(
            f"/api/recordings/{recording_id}/episodes/start",
            json={"metadata": {}},
        )
        retry = client.post(
            f"/api/recordings/{recording_id}/upload",
            json={"repo_name": "yam-data"},
        )

    assert episode.status_code == 409
    assert "immutable" in episode.json()["detail"]
    assert retry.status_code == 200, retry.text


@pytest.mark.asyncio
async def test_cancelled_request_holds_reservation_until_terminal_commit(
    upload_app,
) -> None:
    app, engine, uploader = upload_app
    with TestClient(app) as client:
        recording_id = _ready_recording(client, engine)["id"]

    started = threading.Event()
    finish = threading.Event()
    original_upload = uploader.upload

    def blocking_upload(*args, **kwargs):
        started.set()
        if not finish.wait(timeout=3):
            raise RuntimeError("test upload timed out")
        return original_upload(*args, **kwargs)

    uploader.upload = blocking_upload  # type: ignore[method-assign]
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            request = asyncio.create_task(
                client.post(
                    f"/api/recordings/{recording_id}/upload",
                    json={"repo_name": "cancel-safe"},
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            request.cancel()
            await asyncio.sleep(0.05)
            assert recording_id in uploader.active
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await request
    finally:
        finish.set()

    assert recording_id not in uploader.active
    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(recording_id))
        assert stored is not None
        assert stored.status == "uploaded"


@pytest.mark.asyncio
async def test_real_lerobot_044_conversion_reopens_with_expected_counts(
    tmp_path: Path,
) -> None:
    recording_id = str(uuid.uuid4())
    recording_staging = tmp_path / "recordings"
    driver = MockYAMDriver()
    manager = RecordingManager(
        driver,
        MockCamera(width=96, height=64),
        recording_staging,
    )
    await manager.startup()
    await manager.start_teleop(
        recording_id,
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )
    await manager.start_episode(recording_id, fps=10, metadata={})
    await asyncio.sleep(0.13)
    _, episode = await manager.stop_episode(recording_id, success=True, notes=None)
    await manager.confirm_episode_persisted(recording_id, 1, "teleop")
    await manager.stop_teleop(recording_id)

    source = RecordingUploadSource(
        recording_id=recording_id,
        task="Place the block",
        episode_count=1,
        metadata={
            "episodes": [
                {
                    "index": episode.index,
                    "sample_count": episode.sample_count,
                    "duration_seconds": episode.duration_seconds,
                    "fps": episode.fps,
                    "artifact_key": episode.artifact_key,
                }
            ]
        },
    )
    uploader = HFDatasetUploader(
        recording_staging,
        dataset_staging_dir=tmp_path / "lerobot",
    )
    converted = await asyncio.to_thread(
        uploader.convert,
        source,
        "test-user/local-conversion",
        "local-conversion",
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    reopened = LeRobotDataset(repo_id="test-user/local-conversion", root=converted.root)
    assert reopened.num_episodes == 1
    assert reopened.num_frames == episode.sample_count
    assert set(reopened.features) >= {
        "observation.state",
        "action",
        "observation.images.workspace",
    }
    assert any((converted.root / "data").rglob("*.parquet"))
    assert any((converted.root / "videos").rglob("*.mp4"))
    await manager.shutdown()


def test_explicit_hub_upload_verifies_target_revision_and_required_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_staging = tmp_path / "lerobot"
    root = dataset_staging / "recording" / "converted"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"parquet")
    (root / "videos" / "chunk-000" / "video.mp4").write_bytes(b"video")
    converted = ConvertedDataset(
        dataset=SimpleNamespace(meta=SimpleNamespace(info={"fps": 10})),
        root=root,
        artifact_key="recording/local",
        fps=10,
        total_frames=3,
    )

    class FakeCard:
        def save(self, path: Path) -> None:
            path.write_text("# LeRobot")

    class FakeApi:
        mutations: list[tuple[str, Any]] = []

        def whoami(self):
            return {"name": "user", "orgs": [{"name": "test-org"}]}

        def repo_exists(self, **kwargs):
            return False

        def create_repo(self, **kwargs):
            self.mutations.append(("create", kwargs))
            return "https://huggingface.co/datasets/test-org/yam-data"

        def upload_file(self, **kwargs):
            assert kwargs["token"] == "hf_private"
            self.mutations.append(("marker", kwargs))
            return SimpleNamespace(oid="marker-sha")

        def update_repo_settings(self, **kwargs):
            self.mutations.append(("settings", kwargs))

        def upload_folder(self, **kwargs):
            assert (Path(kwargs["folder_path"]) / "README.md").is_file()
            self.mutations.append(("upload", kwargs))
            return SimpleNamespace(oid="commit-sha")

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="commit-sha")

        def list_repo_files(self, **kwargs):
            return [
                ".ctrl-pi.json",
                "README.md",
                "meta/info.json",
                "data/chunk-000/file-000.parquet",
                "videos/chunk-000/video.mp4",
            ]

        def hf_hub_download(self, **kwargs):
            assert kwargs["token"] == "hf_private"
            return root / ".ctrl-pi.json"

    tokens: list[str] = []
    api = FakeApi()
    uploader = HFDatasetUploader(
        tmp_path / "recordings",
        dataset_staging_dir=dataset_staging,
        hub_api_factory=lambda token: (tokens.append(token), api)[1],
        card_factory=lambda **kwargs: FakeCard(),
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted)
    source = RecordingUploadSource("recording", "Task", 1, {"episodes": [{}]})

    result = uploader.upload(source, "test-org", "yam-data", True, "hf_private")

    assert result.revision == "commit-sha"
    assert tokens == ["hf_private"]
    assert api.mutations[0][1]["private"] is True
    assert api.mutations[0][1]["exist_ok"] is False
    assert [item[0] for item in api.mutations] == [
        "create",
        "marker",
        "settings",
        "upload",
    ]
    assert api.mutations[2][1]["private"] is True
    assert not root.exists()


def test_target_reservation_serializes_recordings_by_repo(tmp_path: Path) -> None:
    uploader = HFDatasetUploader(tmp_path / "recordings")
    uploader.reserve("recording-one", "test-user/shared")
    with pytest.raises(UploadConflictError, match="repository"):
        uploader.reserve("recording-two", "test-user/shared")
    uploader.release("recording-one", "test-user/shared")


def test_post_create_409_preserves_remote_retry_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_staging = tmp_path / "lerobot"
    root = dataset_staging / "recording" / "converted"
    root.mkdir(parents=True)
    converted = ConvertedDataset(
        dataset=SimpleNamespace(meta=SimpleNamespace(info={})),
        root=root,
        artifact_key="recording/local",
        fps=10,
        total_frames=1,
    )

    class Http409(RuntimeError):
        response = SimpleNamespace(status_code=409)

    class FakeCard:
        def save(self, path: Path) -> None:
            path.write_text("# LeRobot")

    class SettingsConflictApi:
        def whoami(self):
            return {"name": "test-user", "orgs": []}

        def repo_exists(self, **kwargs):
            return False

        def create_repo(self, **kwargs):
            return "https://huggingface.co/datasets/test-user/yam-data"

        def upload_file(self, **kwargs):
            assert kwargs["token"] == "hf_private"
            return SimpleNamespace(oid="marker-sha")

        def hf_hub_download(self, **kwargs):
            return root / ".ctrl-pi.json"

        def update_repo_settings(self, **kwargs):
            raise Http409("settings conflict")

    uploader = HFDatasetUploader(
        tmp_path / "recordings",
        dataset_staging_dir=dataset_staging,
        hub_api_factory=lambda token: SettingsConflictApi(),
        card_factory=lambda **kwargs: FakeCard(),
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted)
    source = RecordingUploadSource("recording", "Task", 1, {"episodes": [{}]})

    with pytest.raises(HubUploadError) as raised:
        uploader.upload(source, "test-user", "yam-data", True, "hf_private")

    assert raised.value.remote_repo_created is True
    assert not root.exists()


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        (None, "nonempty.*no ctrl-pi ownership marker"),
        (
            {
                "schema": "ctrl-pi.recording-dataset",
                "version": 1,
                "recording_id": "another-recording",
            },
            "different recording",
        ),
    ],
)
def test_existing_retry_requires_matching_remote_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: dict[str, Any] | None,
    message: str,
) -> None:
    dataset_staging = tmp_path / "lerobot"
    root = dataset_staging / "recording" / "converted"
    root.mkdir(parents=True)
    remote_marker = tmp_path / "remote-marker.json"
    if marker is not None:
        remote_marker.write_text(json.dumps(marker))
    converted = ConvertedDataset(
        dataset=SimpleNamespace(meta=SimpleNamespace(info={})),
        root=root,
        artifact_key="recording/local",
        fps=10,
        total_frames=1,
    )

    class FakeCard:
        def save(self, path: Path) -> None:
            path.write_text("# LeRobot")

    class ExistingApi:
        mutated = False

        def whoami(self):
            return {"name": "test-user", "orgs": []}

        def repo_exists(self, **kwargs):
            return True

        def hf_hub_download(self, **kwargs):
            assert kwargs["token"] == "hf_private"
            if marker is None:
                raise FileNotFoundError("missing marker")
            return remote_marker

        def list_repo_files(self, **kwargs):
            return ["README.md"]

        def update_repo_settings(self, **kwargs):
            self.mutated = True

    api = ExistingApi()
    uploader = HFDatasetUploader(
        tmp_path / "recordings",
        dataset_staging_dir=dataset_staging,
        hub_api_factory=lambda token: api,
        card_factory=lambda **kwargs: FakeCard(),
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted)
    source = RecordingUploadSource(
        "recording",
        "Task",
        1,
        {
            "episodes": [{}],
            "upload": {
                "repo_id": "test-user/yam-data",
                "owner_recording_id": "recording",
                "remote_repo_created": True,
            },
        },
    )

    with pytest.raises(UploadConflictError, match=message):
        uploader.upload(source, "test-user", "yam-data", True, "hf_private")

    assert api.mutated is False


def test_existing_retry_accepts_matching_token_bound_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_staging = tmp_path / "lerobot"
    root = dataset_staging / "recording" / "converted"
    root.mkdir(parents=True)
    remote_marker = tmp_path / "remote-marker.json"
    remote_marker.write_text(
        json.dumps(
            {
                "schema": "ctrl-pi.recording-dataset",
                "version": 1,
                "recording_id": "recording",
            }
        )
    )
    converted = ConvertedDataset(
        dataset=SimpleNamespace(meta=SimpleNamespace(info={})),
        root=root,
        artifact_key="recording/local",
        fps=10,
        total_frames=1,
    )

    class FakeCard:
        def save(self, path: Path) -> None:
            path.write_text("# LeRobot")

    class ExistingApi:
        download_tokens: list[str] = []
        settings_updated = False

        def whoami(self):
            return {"name": "test-user", "orgs": []}

        def repo_exists(self, **kwargs):
            return True

        def hf_hub_download(self, **kwargs):
            self.download_tokens.append(kwargs["token"])
            return remote_marker

        def update_repo_settings(self, **kwargs):
            self.settings_updated = True

        def upload_folder(self, **kwargs):
            assert (Path(kwargs["folder_path"]) / ".ctrl-pi.json").is_file()
            return SimpleNamespace(oid="retry-sha")

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="retry-sha")

        def list_repo_files(self, **kwargs):
            return [
                ".ctrl-pi.json",
                "README.md",
                "meta/info.json",
                "data/chunk-000/file-000.parquet",
                "videos/chunk-000/video.mp4",
            ]

    api = ExistingApi()
    uploader = HFDatasetUploader(
        tmp_path / "recordings",
        dataset_staging_dir=dataset_staging,
        hub_api_factory=lambda token: api,
        card_factory=lambda **kwargs: FakeCard(),
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted)
    source = RecordingUploadSource(
        "recording",
        "Task",
        1,
        {
            "episodes": [{}],
            "upload": {
                "repo_id": "test-user/yam-data",
                "owner_recording_id": "recording",
                "remote_repo_created": True,
            },
        },
    )

    result = uploader.upload(
        source, "test-user", "yam-data", True, "hf_retry_token"
    )

    assert result.revision == "retry-sha"
    assert api.download_tokens == ["hf_retry_token", "hf_retry_token"]
    assert api.settings_updated is True


def test_retry_bootstraps_marker_only_for_empty_trusted_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_staging = tmp_path / "lerobot"
    root = dataset_staging / "recording" / "converted"
    root.mkdir(parents=True)
    remote_marker = tmp_path / "remote-marker.json"
    converted = ConvertedDataset(
        dataset=SimpleNamespace(meta=SimpleNamespace(info={})),
        root=root,
        artifact_key="recording/local",
        fps=10,
        total_frames=1,
    )

    class FakeCard:
        def save(self, path: Path) -> None:
            path.write_text("# LeRobot")

    class EmptyRepoApi:
        marker_tokens: list[str] = []
        bulk_uploaded = False

        def whoami(self):
            return {"name": "test-user", "orgs": []}

        def repo_exists(self, **kwargs):
            return True

        def hf_hub_download(self, **kwargs):
            if not remote_marker.exists():
                raise FileNotFoundError("marker not committed before crash")
            return remote_marker

        def list_repo_files(self, **kwargs):
            if self.bulk_uploaded:
                return [
                    ".ctrl-pi.json",
                    "README.md",
                    "meta/info.json",
                    "data/chunk-000/file-000.parquet",
                    "videos/chunk-000/video.mp4",
                ]
            return [".gitattributes"]

        def upload_file(self, **kwargs):
            self.marker_tokens.append(kwargs["token"])
            remote_marker.write_text(Path(kwargs["path_or_fileobj"]).read_text())
            return SimpleNamespace(oid="marker-sha")

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            self.bulk_uploaded = True
            return SimpleNamespace(oid="bootstrap-sha")

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="bootstrap-sha")

    api = EmptyRepoApi()
    uploader = HFDatasetUploader(
        tmp_path / "recordings",
        dataset_staging_dir=dataset_staging,
        hub_api_factory=lambda token: api,
        card_factory=lambda **kwargs: FakeCard(),
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted)
    source = RecordingUploadSource(
        "recording",
        "Task",
        1,
        {
            "episodes": [{}],
            "upload": {
                "repo_id": "test-user/yam-data",
                "owner_recording_id": "recording",
                "remote_repo_created": False,
            },
        },
    )

    result = uploader.upload(
        source, "test-user", "yam-data", True, "hf_bootstrap_token"
    )

    assert result.revision == "bootstrap-sha"
    assert api.marker_tokens == ["hf_bootstrap_token"]


def test_marker_commit_survives_bulk_failure_and_enables_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_staging = tmp_path / "lerobot"
    remote_marker = tmp_path / "remote-marker.json"
    conversion_count = 0

    def converted_dataset() -> ConvertedDataset:
        nonlocal conversion_count
        conversion_count += 1
        root = dataset_staging / "recording" / f"converted-{conversion_count}"
        root.mkdir(parents=True)
        return ConvertedDataset(
            dataset=SimpleNamespace(meta=SimpleNamespace(info={})),
            root=root,
            artifact_key=f"recording/local-{conversion_count}",
            fps=10,
            total_frames=1,
        )

    class FakeCard:
        def save(self, path: Path) -> None:
            path.write_text("# LeRobot")

    class RetryingApi:
        exists = False
        bulk_calls = 0
        marker_calls = 0

        def whoami(self):
            return {"name": "test-user", "orgs": []}

        def repo_exists(self, **kwargs):
            return self.exists

        def create_repo(self, **kwargs):
            self.exists = True
            return "https://huggingface.co/datasets/test-user/yam-data"

        def upload_file(self, **kwargs):
            self.marker_calls += 1
            remote_marker.write_text(Path(kwargs["path_or_fileobj"]).read_text())
            return SimpleNamespace(oid="marker-sha")

        def hf_hub_download(self, **kwargs):
            return remote_marker

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            self.bulk_calls += 1
            if self.bulk_calls == 1:
                raise RuntimeError("bulk transfer interrupted")
            return SimpleNamespace(oid="retry-sha")

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="retry-sha")

        def list_repo_files(self, **kwargs):
            return [
                ".ctrl-pi.json",
                "README.md",
                "meta/info.json",
                "data/chunk-000/file-000.parquet",
                "videos/chunk-000/video.mp4",
            ]

    api = RetryingApi()
    uploader = HFDatasetUploader(
        tmp_path / "recordings",
        dataset_staging_dir=dataset_staging,
        hub_api_factory=lambda token: api,
        card_factory=lambda **kwargs: FakeCard(),
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted_dataset())
    first_source = RecordingUploadSource(
        "recording", "Task", 1, {"episodes": [{}]}
    )

    with pytest.raises(HubUploadError) as first_error:
        uploader.upload(
            first_source, "test-user", "yam-data", True, "hf_retry_token"
        )
    assert first_error.value.remote_repo_created is True

    retry_source = RecordingUploadSource(
        "recording",
        "Task",
        1,
        {
            "episodes": [{}],
            "upload": {
                "repo_id": "test-user/yam-data",
                "owner_recording_id": "recording",
                "remote_repo_created": True,
            },
        },
    )
    result = uploader.upload(
        retry_source, "test-user", "yam-data", True, "hf_retry_token"
    )

    assert result.revision == "retry-sha"
    assert api.marker_calls == 1
    assert api.bulk_calls == 2


def test_existing_unrelated_hub_repo_is_rejected_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted = ConvertedDataset(
        dataset=SimpleNamespace(meta=SimpleNamespace(info={})),
        root=tmp_path,
        artifact_key="recording/local",
        fps=10,
        total_frames=1,
    )

    class ExistingRepoApi:
        mutated = False

        def whoami(self):
            return {"name": "test-user", "orgs": []}

        def repo_exists(self, **kwargs):
            return True

        def create_repo(self, **kwargs):
            self.mutated = True

    api = ExistingRepoApi()
    uploader = HFDatasetUploader(
        tmp_path,
        hub_api_factory=lambda token: api,
    )
    monkeypatch.setattr(uploader, "convert", lambda *args, **kwargs: converted)
    source = RecordingUploadSource("recording", "Task", 1, {"episodes": [{}]})

    with pytest.raises(UploadConflictError, match="already exists"):
        uploader.upload(source, "test-user", "existing", True, "hf_private")
    forged_retry = RecordingUploadSource(
        "recording",
        "Task",
        1,
        {"upload": {"repo_id": "test-user/existing"}, "episodes": [{}]},
    )
    with pytest.raises(UploadConflictError, match="already exists"):
        uploader.upload(forged_retry, "test-user", "existing", True, "hf_private")
    assert api.mutated is False
