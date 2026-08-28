from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ctrl_pi.api.camera import mjpeg_chunks
from ctrl_pi.api.recordings import (
    EpisodeStop,
    _reconcile_episode_manifests,
    reconcile_recordings_startup,
    stop_episode as stop_episode_api,
)
from ctrl_pi.camera import MockCamera
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import Base, get_db
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.main import create_app
from ctrl_pi.models import AppSetting, Recording, Robot
from ctrl_pi.recording import (
    EPISODE_MANIFEST_FILENAME,
    MAX_EPISODE_MANIFEST_BYTES,
    FFmpegVideoWriter,
    RecordingConflictError,
    RecordingManager,
    RecordingRuntimeError,
    episode_result_metadata,
    load_episode_manifest,
)


class _FailingActionDriver(MockYAMDriver):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.action_calls = 0

    def apply_action(self, arm_id, action):
        self.action_calls += 1
        if self.action_calls >= self.fail_on_call:
            raise RuntimeError("mock action failure")
        return super().apply_action(arm_id, action)


class _FakeVideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: int) -> None:
        del width, height, fps
        self.output_path = output_path
        output_path.write_bytes(b"")
        self.closed = False

    def write(self, frame) -> None:
        assert not self.closed and frame.rgb
        with self.output_path.open("ab") as output:
            output.write(b"frame")

    def close(self) -> None:
        self.closed = True
        if not self.output_path.read_bytes():
            raise RecordingRuntimeError("test video is empty")

    def terminate(self) -> None:
        self.closed = True


@pytest.fixture
def recording_app(tmp_path: Path):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    driver = MockYAMDriver()
    camera = MockCamera(width=160, height=120)
    manager = RecordingManager(driver, camera, tmp_path / "staging")
    app = create_app(driver, camera, manager)

    def override_db() -> Iterator[Session]:
        with Session(engine, expire_on_commit=False) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_config] = lambda: AppConfig(
        _env_file=None,
        recording_staging_dir=tmp_path / "staging",
        recording_fps=20,
    )
    return app, engine, manager, tmp_path / "staging"


def _create_recording(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/recordings",
        json={
            "name": "Pick red block",
            "task": "Move the red block into the tray",
            "leader_robot_id": "yam-leader",
            "follower_robot_id": "yam-follower",
            "metadata": {"operator": "Ada"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_lists_recording_and_maps_driver_ids_to_robot_foreign_keys(
    recording_app,
) -> None:
    app, engine, _, _ = recording_app
    with TestClient(app) as client:
        created = _create_recording(client)
        listed = client.get("/api/recordings")

    assert created["leader_robot_id"] == "yam-leader"
    assert created["follower_robot_id"] == "yam-follower"
    assert created["status"] == "draft"
    assert listed.status_code == 200
    assert listed.json()["recordings"][0]["id"] == created["id"]

    with Session(engine) as db:
        recording = db.scalar(select(Recording))
        assert recording is not None
        assert recording.leader_robot_id is not None
        assert recording.follower_robot_id is not None
        assert set(db.scalars(select(Robot.driver_id))) == {
            "yam-leader",
            "yam-follower",
        }


def test_recording_lifecycle_writes_atomic_mp4_and_synchronized_samples(
    recording_app,
) -> None:
    app, engine, manager, staging = recording_app
    with Session(engine) as db:
        db.add(AppSetting(key="recording_fps", value=10))
        db.commit()

    with TestClient(app) as client:
        recording = _create_recording(client)
        recording_id = recording["id"]

        assert client.post(f"/api/recordings/{recording_id}/episodes/start", json={}).status_code == 409
        started = client.post(f"/api/recordings/{recording_id}/teleop/start")
        assert started.status_code == 200
        assert started.json()["teleop_active"] is True
        blocked_jog = client.post(
            "/api/arms/yam-follower/jog",
            json={"kind": "joint", "axis": "shoulder_yaw", "delta": 0.1},
        )
        assert blocked_jog.status_code == 409
        assert "controlled by teleop" in blocked_jog.json()["detail"]

        episode = client.post(
            f"/api/recordings/{recording_id}/episodes/start",
            json={"metadata": {"operator": "Ada", "notes": "short test"}},
        )
        assert episode.status_code == 200, episode.text
        assert episode.json()["episode_active"] is True
        assert client.post(f"/api/recordings/{recording_id}/teleop/stop").status_code == 409

        time.sleep(0.24)
        stopped = client.post(
            f"/api/recordings/{recording_id}/episodes/stop",
            json={"success": True, "notes": "completed"},
        )
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["episode_count"] == 1
        assert stopped.json()["episode_active"] is False
        assert client.post(f"/api/recordings/{recording_id}/teleop/stop").status_code == 200
        released_jog = client.post(
            "/api/arms/yam-follower/jog",
            json={"kind": "joint", "axis": "shoulder_yaw", "delta": 0.1},
        )
        assert released_jog.status_code == 200

    episode_dir = staging / str(recording_id) / "episode_000000"
    video_path = episode_dir / "video.mp4"
    samples_path = episode_dir / "samples.jsonl"
    assert video_path.stat().st_size > 100
    assert not (episode_dir / "video.partial.mp4").exists()
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    assert len(decoded.stdout) == 160 * 120 * 3
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    assert 2 <= len(samples) <= 5
    assert all({"observation", "action", "camera_timestamp"} <= sample.keys() for sample in samples)
    assert [sample["frame_index"] for sample in samples] == list(range(len(samples)))
    assert [sample["timestamp_seconds"] for sample in samples] == sorted(
        sample["timestamp_seconds"] for sample in samples
    )
    for sample in samples:
        action = sample["action"]["joint_positions_radians"]
        observed = {
            joint["name"]: joint["position_radians"]
            for joint in sample["observation"]["joints"]
        }
        assert action == observed

    with Session(engine) as db:
        stored = db.scalar(select(Recording))
        assert stored is not None
        assert stored.episode_count == 1
        assert stored.duration_seconds == pytest.approx(len(samples) / 10)
        summary = stored.recording_metadata["episodes"][0]
        assert summary["sample_count"] == len(samples)
        assert summary["artifact_key"] == f"{recording_id}/episode_000000"
        assert not Path(summary["artifact_key"]).is_absolute()

    assert asyncio.run(manager.active_resource_counts()) == (0, 0)
    assert manager.rig_lease.current() is None


def test_app_shutdown_stops_active_teleop_and_ffmpeg(recording_app) -> None:
    app, _, manager, staging = recording_app
    with TestClient(app) as client:
        recording_id = _create_recording(client)["id"]
        assert client.post(f"/api/recordings/{recording_id}/teleop/start").status_code == 200
        assert client.post(f"/api/recordings/{recording_id}/episodes/start", json={}).status_code == 200
        time.sleep(0.05)

    assert asyncio.run(manager.active_resource_counts()) == (0, 0)
    assert manager.rig_lease.current() is None
    assert not (staging / str(recording_id) / "episode_000000" / "video.mp4").exists()


@pytest.mark.asyncio
async def test_teleop_releases_rig_when_the_first_driver_action_fails(
    tmp_path: Path,
) -> None:
    driver = _FailingActionDriver(fail_on_call=1)
    manager = RecordingManager(
        driver,
        MockCamera(width=96, height=64),
        tmp_path / "staging",
    )

    with pytest.raises(RuntimeError, match="mock action failure"):
        await manager.start_teleop(
            "session",
            "yam-leader",
            "yam-follower",
            episode_count=0,
            status="draft",
        )

    assert await manager.active_resource_counts() == (0, 0)
    assert manager.rig_lease.current() is None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_teleop_releases_rig_when_the_background_loop_fails(
    tmp_path: Path,
) -> None:
    driver = _FailingActionDriver(fail_on_call=2)
    manager = RecordingManager(
        driver,
        MockCamera(width=96, height=64),
        tmp_path / "staging",
        teleop_frequency_hz=100.0,
    )
    await manager.start_teleop(
        "session",
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )

    for _ in range(50):
        if manager.rig_lease.current() is None:
            break
        await asyncio.sleep(0.01)

    assert driver.action_calls == 2
    assert await manager.active_resource_counts() == (0, 0)
    assert manager.rig_lease.current() is None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_mock_camera_produces_timestamped_jpeg_mjpeg_frames() -> None:
    camera = MockCamera(width=96, height=64)
    first = camera.capture()
    second = camera.capture()

    assert first.timestamp <= second.timestamp
    assert first.rgb != second.rgb
    assert camera.jpeg(first).startswith(b"\xff\xd8")

    stream = mjpeg_chunks(camera, fps=60)
    chunk = await anext(stream)
    await stream.aclose()
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert b"\xff\xd8" in chunk


@pytest.mark.asyncio
async def test_episode_finalization_keeps_the_global_rig_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = MockYAMDriver()
    manager = RecordingManager(
        driver,
        MockCamera(width=96, height=64),
        tmp_path / "staging",
    )
    await manager.start_teleop(
        "session",
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )
    await manager.start_episode("session", fps=10, metadata={})

    entered = threading.Event()
    release = threading.Event()
    original_finalize = manager._finalize_episode

    def slow_finalize(episode, success, notes):
        entered.set()
        assert release.wait(timeout=3)
        return original_finalize(episode, success, notes)

    monkeypatch.setattr(manager, "_finalize_episode", slow_finalize)
    stop_task = asyncio.create_task(
        manager.stop_episode("session", success=True, notes=None)
    )
    assert await asyncio.to_thread(entered.wait, 3)

    polling_state = await manager.state(
        "session", "yam-leader", "yam-follower", 0, "recording"
    )
    assert polling_state.episode_active is True
    with pytest.raises(RecordingConflictError, match="still finalizing"):
        await manager.start_episode("session", fps=10, metadata={})

    release.set()
    stopped, _ = await stop_task
    assert stopped.episode_active is False
    await manager.confirm_episode_persisted("session", 1, "teleop")
    await manager.stop_teleop("session")
    await manager.shutdown()


@pytest.mark.asyncio
async def test_teleop_failure_during_finalization_is_not_restored_to_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _FailingActionDriver(fail_on_call=10_000)
    manager = RecordingManager(
        driver,
        MockCamera(width=96, height=64),
        tmp_path / "staging",
        teleop_frequency_hz=100.0,
    )
    await manager.start_teleop(
        "session",
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )
    await manager.start_episode("session", fps=10, metadata={})

    entered = threading.Event()
    release = threading.Event()
    original_finalize = manager._finalize_episode

    def slow_finalize(episode, success, notes):
        entered.set()
        assert release.wait(timeout=3)
        return original_finalize(episode, success, notes)

    monkeypatch.setattr(manager, "_finalize_episode", slow_finalize)
    stop_task = asyncio.create_task(
        manager.stop_episode("session", success=True, notes=None)
    )
    assert await asyncio.to_thread(entered.wait, 3)
    driver.fail_on_call = driver.action_calls + 1

    for _ in range(100):
        if manager.rig_lease.current() is None:
            break
        await asyncio.sleep(0.01)
    assert manager.rig_lease.current() is None

    release.set()
    stopped, _ = await stop_task
    assert stopped.status == "failed"
    assert stopped.teleop_active is False
    await manager.confirm_episode_persisted("session", 1, "failed")
    assert await manager.active_resource_counts() == (0, 0)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_persistence_confirmation_preserves_a_late_teleop_failure(
    tmp_path: Path,
) -> None:
    driver = _FailingActionDriver(fail_on_call=10_000)
    manager = RecordingManager(
        driver,
        MockCamera(width=96, height=64),
        tmp_path / "staging",
        teleop_frequency_hz=100.0,
    )
    await manager.start_teleop(
        "session",
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )
    await manager.start_episode("session", fps=10, metadata={})
    stopped, _ = await manager.stop_episode("session", success=True, notes=None)
    assert stopped.status == "teleop"

    driver.fail_on_call = driver.action_calls + 1
    for _ in range(100):
        if manager.rig_lease.current() is None:
            break
        await asyncio.sleep(0.01)
    assert manager.rig_lease.current() is None

    with pytest.raises(RecordingConflictError, match="still finalizing"):
        await manager.start_teleop(
            "session",
            "yam-leader",
            "yam-follower",
            episode_count=0,
            status="teleop",
        )
    confirmed = await manager.confirm_episode_persisted("session", 1, "teleop")
    assert confirmed.status == "failed"
    assert confirmed.teleop_active is False
    assert confirmed.episode_count == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_in_flight_episode_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RecordingManager(
        MockYAMDriver(),
        MockCamera(width=96, height=64),
        tmp_path / "staging",
        teleop_frequency_hz=100.0,
    )
    await manager.start_teleop(
        "session",
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )
    await manager.start_episode("session", fps=10, metadata={})

    entered = threading.Event()
    release = threading.Event()
    original_finalize = manager._finalize_episode

    def slow_finalize(episode, success, notes):
        entered.set()
        assert release.wait(timeout=3)
        return original_finalize(episode, success, notes)

    monkeypatch.setattr(manager, "_finalize_episode", slow_finalize)
    stop_task = asyncio.create_task(
        manager.stop_episode("session", success=True, notes=None)
    )
    assert await asyncio.to_thread(entered.wait, 3)

    shutdown_task = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0.05)
    assert shutdown_task.done() is False
    release.set()

    stopped, _ = await stop_task
    await shutdown_task
    assert stopped.status == "failed"
    assert await manager.active_resource_counts() == (0, 0)
    assert manager.rig_lease.current() is None
    episode_dir = tmp_path / "staging" / "session" / "episode_000000"
    assert (episode_dir / "video.mp4").is_file()
    assert (episode_dir / "samples.jsonl").is_file()
    assert not (episode_dir / "video.partial.mp4").exists()


def test_recording_create_validates_arm_roles_and_reserved_metadata(recording_app) -> None:
    app, _, _, _ = recording_app
    with TestClient(app) as client:
        reversed_pair = client.post(
            "/api/recordings",
            json={
                "name": "Bad pair",
                "task": "No-op",
                "leader_robot_id": "yam-follower",
                "follower_robot_id": "yam-leader",
            },
        )
        reserved_metadata = client.post(
            "/api/recordings",
            json={
                "name": "Bad metadata",
                "task": "No-op",
                "leader_robot_id": "yam-leader",
                "follower_robot_id": "yam-follower",
                "metadata": {"episodes": 1},
            },
        )

    assert reversed_pair.status_code == 422
    assert reserved_metadata.status_code == 422


def test_state_reconciles_stale_active_status_after_process_restart(recording_app) -> None:
    app, engine, _, _ = recording_app
    with TestClient(app) as client:
        recording_id = _create_recording(client)["id"]

    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(recording_id))
        assert stored is not None
        stored.status = "recording"
        db.commit()

    restarted_driver = MockYAMDriver()
    restarted_camera = MockCamera(width=160, height=120)
    restarted_manager = RecordingManager(
        restarted_driver, restarted_camera, recording_app[3]
    )
    restarted_app = create_app(
        restarted_driver, restarted_camera, restarted_manager
    )

    def override_db() -> Iterator[Session]:
        with Session(engine, expire_on_commit=False) as session:
            yield session

    restarted_app.dependency_overrides[get_db] = override_db
    with TestClient(restarted_app) as client:
        state = client.get(f"/api/recordings/{recording_id}/state")

    assert state.status_code == 200
    assert state.json()["episode_active"] is False
    assert state.json()["teleop_active"] is False
    assert state.json()["status"] == "draft"
    with Session(engine) as db:
        refreshed = db.get(Recording, uuid.UUID(recording_id))
        assert refreshed is not None
        assert refreshed.status == "draft"


def _commit_failing_database(engine, secret: str):
    def override_db() -> Iterator[Session]:
        with Session(engine, expire_on_commit=False) as session:
            def fail_commit() -> None:
                raise SQLAlchemyError(secret)

            session.commit = fail_commit  # type: ignore[method-assign]
            yield session

    return override_db


def test_teleop_commit_failure_releases_every_process_local_resource(
    recording_app,
) -> None:
    app, engine, manager, _ = recording_app
    secret = "postgresql://private-host/teleop"
    with TestClient(app) as client:
        recording_id = _create_recording(client)["id"]
        app.dependency_overrides[get_db] = _commit_failing_database(engine, secret)
        response = client.post(f"/api/recordings/{recording_id}/teleop/start")

    assert response.status_code == 503
    assert secret not in response.text
    assert asyncio.run(manager.active_resource_counts()) == (0, 0)
    assert manager.rig_lease.current() is None


def test_episode_start_commit_failure_stops_writer_teleop_and_cleans_partials(
    recording_app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, engine, manager, staging = recording_app
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    secret = "/private/database/episode-start"
    with TestClient(app) as client:
        recording_id = _create_recording(client)["id"]
        assert client.post(f"/api/recordings/{recording_id}/teleop/start").status_code == 200
        app.dependency_overrides[get_db] = _commit_failing_database(engine, secret)
        response = client.post(
            f"/api/recordings/{recording_id}/episodes/start", json={}
        )

    assert response.status_code == 503
    assert secret not in response.text
    assert asyncio.run(manager.active_resource_counts()) == (0, 0)
    assert manager.rig_lease.current() is None
    assert not (staging / str(recording_id) / "episode_000000").exists()


def test_finalized_manifest_recovers_failed_db_commit_exactly_once_after_restart(
    recording_app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, engine, manager, staging = recording_app
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    secret = "postgresql://private-host/finalize"
    with TestClient(app) as client:
        recording_id = _create_recording(client)["id"]
        assert client.post(f"/api/recordings/{recording_id}/teleop/start").status_code == 200
        assert client.post(
            f"/api/recordings/{recording_id}/episodes/start", json={}
        ).status_code == 200
        app.dependency_overrides[get_db] = _commit_failing_database(engine, secret)
        response = client.post(
            f"/api/recordings/{recording_id}/episodes/stop", json={}
        )

    assert response.status_code == 503
    assert secret not in response.text
    assert asyncio.run(manager.active_resource_counts()) == (0, 0)
    assert manager.rig_lease.current() is None
    episode_directory = staging / str(recording_id) / "episode_000000"
    manifest = episode_directory / EPISODE_MANIFEST_FILENAME
    assert manifest.is_file()
    assert 0 < manifest.stat().st_size <= MAX_EPISODE_MANIFEST_BYTES
    recovered = load_episode_manifest(
        episode_directory, expected_recording_id=str(recording_id)
    )
    assert recovered.index == 0 and recovered.sample_count >= 1

    restarted = RecordingManager(
        MockYAMDriver(), MockCamera(width=160, height=120), staging
    )
    def factory() -> Session:
        return Session(engine, expire_on_commit=False)

    asyncio.run(restarted.startup())
    asyncio.run(reconcile_recordings_startup(factory, restarted))
    asyncio.run(reconcile_recordings_startup(factory, restarted))

    with Session(engine) as db:
        stored = db.get(Recording, uuid.UUID(str(recording_id)))
        assert stored is not None
        assert stored.status == "ready"
        assert stored.episode_count == 1
        assert stored.duration_seconds == pytest.approx(recovered.duration_seconds)
        assert len(stored.recording_metadata["episodes"]) == 1
        assert stored.recording_metadata["episodes"][0]["artifact_key"] == (
            f"{recording_id}/episode_000000"
        )


@pytest.mark.asyncio
async def test_rename_failure_removes_mixed_final_and_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    manager = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), tmp_path / "staging"
    )
    await manager.startup()
    await manager.start_teleop(
        "session", "yam-leader", "yam-follower", 0, "draft"
    )
    await manager.start_episode("session", fps=10, metadata={})
    original_replace = Path.replace

    def fail_second_rename(path: Path, target: Path):
        if path.name == "samples.partial.jsonl":
            raise OSError("/private/staging/should-not-leak")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_rename)
    with pytest.raises(RecordingRuntimeError) as raised:
        await manager.stop_episode("session", success=True, notes=None)

    assert "/private/staging" not in str(raised.value)
    assert not (tmp_path / "staging" / "session" / "episode_000000").exists()
    await manager.compensate_persistence_failure("session")
    await manager.shutdown()


@pytest.mark.asyncio
async def test_restart_removes_partial_and_mixed_orphans_but_keeps_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    staging = tmp_path / "staging"
    manager = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), staging
    )
    await manager.startup()
    await manager.start_teleop(
        "valid", "yam-leader", "yam-follower", 0, "draft"
    )
    await manager.start_episode("valid", fps=10, metadata={})
    _, result = await manager.stop_episode("valid", success=True, notes=None)
    await manager.confirm_episode_persisted("valid", 1, "teleop")
    await manager.stop_teleop("valid")

    partial = staging / "partial" / "episode_000000"
    partial.mkdir(parents=True)
    (partial / "video.partial.mp4").write_bytes(b"partial")
    (partial / "samples.partial.jsonl").write_text("{}\n")
    mixed = staging / "mixed" / "episode_000000"
    mixed.mkdir(parents=True)
    (mixed / "video.mp4").write_bytes(b"final")
    (mixed / "samples.partial.jsonl").write_text("{}\n")

    restarted = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), staging
    )
    await restarted.startup()

    valid = staging / result.artifact_key
    assert (valid / EPISODE_MANIFEST_FILENAME).is_file()
    assert (valid / "video.mp4").is_file()
    assert (valid / "samples.jsonl").is_file()
    assert not partial.exists()
    assert not mixed.exists()


@pytest.mark.asyncio
async def test_episode_directory_fsyncs_cover_creation_and_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    staging = tmp_path / "staging"
    manager = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), staging
    )
    events: list[str] = []
    original_replace = Path.replace

    def relative(path: Path) -> str:
        return str(path.relative_to(staging))

    def tracked_directory_fsync(path: Path) -> None:
        events.append(f"fsync-dir:{relative(path)}")

    def tracked_file_fsync(path: Path) -> None:
        events.append(f"fsync-file:{relative(path)}")

    def tracked_replace(source: Path, target: Path) -> Path:
        events.append(f"replace:{relative(source)}->{relative(target)}")
        return original_replace(source, target)

    monkeypatch.setattr(manager, "_fsync_directory", tracked_directory_fsync)
    monkeypatch.setattr(manager, "_fsync_file", tracked_file_fsync)
    monkeypatch.setattr(Path, "replace", tracked_replace)

    await manager.start_teleop(
        "session", "yam-leader", "yam-follower", 0, "draft"
    )
    await manager.start_episode("session", fps=10, metadata={})
    assert events == ["fsync-dir:.", "fsync-dir:session"]

    events.clear()
    await manager.stop_episode("session", success=True, notes=None)
    episode = "session/episode_000000"
    assert events == [
        f"fsync-file:{episode}/video.partial.mp4",
        f"replace:{episode}/video.partial.mp4->{episode}/video.mp4",
        f"replace:{episode}/samples.partial.jsonl->{episode}/samples.jsonl",
        f"fsync-dir:{episode}",
        f"replace:{episode}/.episode.json.partial->{episode}/episode.json",
        f"fsync-dir:{episode}",
        "fsync-dir:session",
        "fsync-dir:.",
    ]

    await manager.confirm_episode_persisted("session", 1, "teleop")
    await manager.stop_teleop("session")
    await manager.shutdown()


@pytest.mark.asyncio
async def test_fresh_database_preserves_manifest_and_unrelated_shared_root_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    staging = tmp_path / "staging"
    manager = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), staging
    )
    await manager.startup()
    await manager.start_teleop(
        "preserved", "yam-leader", "yam-follower", 0, "draft"
    )
    await manager.start_episode("preserved", fps=10, metadata={})
    _, result = await manager.stop_episode("preserved", success=True, notes=None)
    await manager.confirm_episode_persisted("preserved", 1, "teleop")
    await manager.stop_teleop("preserved")
    await manager.shutdown()

    unrelated_file = staging / "shared-note.txt"
    unrelated_file.write_text("do not delete")
    coincidental_directory = staging / "shared" / "episode_000000"
    coincidental_directory.mkdir(parents=True)
    (coincidental_directory / "notes.txt").write_text("not ctrl-pi data")
    incomplete = staging / "incomplete" / "episode_000000"
    incomplete.mkdir(parents=True)
    (incomplete / "video.partial.mp4").write_bytes(b"partial")

    fresh_engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(fresh_engine)

    def fresh_factory() -> Session:
        return Session(fresh_engine, expire_on_commit=False)

    restarted = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), staging
    )
    await restarted.startup()
    await reconcile_recordings_startup(fresh_factory, restarted)

    finalized = staging / result.artifact_key
    assert (finalized / EPISODE_MANIFEST_FILENAME).is_file()
    assert (finalized / "video.mp4").is_file()
    assert (finalized / "samples.jsonl").is_file()
    assert unrelated_file.read_text() == "do not delete"
    assert (coincidental_directory / "notes.txt").read_text() == "not ctrl-pi data"
    assert not incomplete.exists()


def test_ffmpeg_failure_discards_stderr_and_exception_causes() -> None:
    secret = b"/private/staging/token=hf_secret"

    class FakeInput:
        def close(self) -> None:
            raise OSError(secret.decode())

    class FakeStderr:
        def read(self) -> bytes:
            return secret

        def close(self) -> None:
            return None

    class FakeProcess:
        stdin = FakeInput()
        stderr = FakeStderr()

        def wait(self, timeout: int) -> int:
            del timeout
            return 1

    writer = object.__new__(FFmpegVideoWriter)
    writer.output_path = Path("/private/staging/video.partial.mp4")
    writer.width = 1
    writer.height = 1
    writer._closed = False
    writer._process = FakeProcess()

    with pytest.raises(RecordingRuntimeError) as raised:
        writer.close()

    assert secret.decode() not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_cancelled_reconciliation_keeps_filesystem_ownership_until_worker_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    manager = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), tmp_path / "staging"
    )
    await manager.startup()
    await manager.start_teleop(
        "session", "yam-leader", "yam-follower", 0, "draft"
    )
    entered = threading.Event()
    release = threading.Event()
    original = manager._reconcile_recording_directory

    def blocked_reconcile(*args):
        entered.set()
        assert release.wait(timeout=3)
        return original(*args)

    monkeypatch.setattr(manager, "_reconcile_recording_directory", blocked_reconcile)
    reconciliation = asyncio.create_task(
        manager.reconcile_recording_artifacts("session", [])
    )
    assert await asyncio.to_thread(entered.wait, 1)
    reconciliation.cancel()
    await asyncio.sleep(0)

    assert reconciliation.done() is False
    with pytest.raises(RecordingConflictError, match="being reconciled"):
        await manager.start_episode("session", fps=10, metadata={})

    release.set()
    assert await reconciliation == []
    await manager.start_episode("session", fps=10, metadata={})
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_after_episode_db_commit_settles_runtime_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    driver = MockYAMDriver()
    manager = RecordingManager(
        driver, MockCamera(width=64, height=48), tmp_path / "staging"
    )
    with Session(engine, expire_on_commit=False) as db:
        leader = Robot(driver_id="yam-leader", name="YAM Leader", role="leader")
        follower = Robot(
            driver_id="yam-follower", name="YAM Follower", role="follower"
        )
        db.add_all((leader, follower))
        db.flush()
        recording = Recording(
            name="Cancellation safe",
            task="Finalize once",
            status="draft",
            leader_robot_id=leader.id,
            follower_robot_id=follower.id,
            episode_count=0,
            duration_seconds=0,
            recording_metadata={"episodes": []},
        )
        db.add(recording)
        db.commit()
        recording_id = recording.id

        await manager.startup()
        await manager.start_teleop(
            str(recording_id), "yam-leader", "yam-follower", 0, "draft"
        )
        await manager.start_episode(str(recording_id), fps=10, metadata={})
        recording.status = "recording"
        db.commit()

        entered = asyncio.Event()
        release = asyncio.Event()
        original_confirm = manager.confirm_episode_persisted

        async def blocked_confirm(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original_confirm(*args, **kwargs)

        monkeypatch.setattr(manager, "confirm_episode_persisted", blocked_confirm)
        request = asyncio.create_task(
            stop_episode_api(
                recording_id,
                EpisodeStop(),
                db=db,
                manager=manager,
            )
        )
        await entered.wait()
        request.cancel()
        await asyncio.sleep(0)
        assert request.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request

        db.expire_all()
        stored = db.get(Recording, recording_id)
        assert stored is not None and stored.episode_count == 1
        state = await manager.state(
            str(recording_id), "yam-leader", "yam-follower", 1, stored.status
        )
        assert state.episode_active is False
        assert state.episode_count == 1

    await manager.stop_teleop(str(recording_id))
    await manager.shutdown()


@pytest.mark.asyncio
async def test_noop_manifest_reconciliation_releases_persisted_finalizer_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    manager = RecordingManager(
        MockYAMDriver(), MockCamera(width=64, height=48), tmp_path / "staging"
    )
    recording_id = uuid.uuid4()
    await manager.startup()
    await manager.start_teleop(
        str(recording_id), "yam-leader", "yam-follower", 0, "draft"
    )
    await manager.start_episode(str(recording_id), fps=10, metadata={})
    _, result = await manager.stop_episode(
        str(recording_id), success=True, notes=None
    )

    with Session(engine, expire_on_commit=False) as db:
        recording = Recording(
            id=recording_id,
            name="Already persisted",
            task="Reconcile sentinel",
            status="teleop",
            episode_count=1,
            duration_seconds=result.duration_seconds,
            recording_metadata={"episodes": [episode_result_metadata(result)]},
        )
        db.add(recording)
        db.commit()

        assert await _reconcile_episode_manifests(db, recording, manager) is False

    state = await manager.state(
        str(recording_id), "yam-leader", "yam-follower", 1, "teleop"
    )
    assert state.episode_active is False
    assert state.episode_count == 1
    await manager.stop_teleop(str(recording_id))
    await manager.shutdown()
