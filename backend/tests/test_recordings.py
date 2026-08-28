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
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ctrl_pi.api.camera import mjpeg_chunks
from ctrl_pi.camera import MockCamera
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import Base, get_db
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.main import create_app
from ctrl_pi.models import AppSetting, Recording, Robot
from ctrl_pi.recording import RecordingConflictError, RecordingManager


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


def test_app_shutdown_stops_active_teleop_and_ffmpeg(recording_app) -> None:
    app, _, manager, staging = recording_app
    with TestClient(app) as client:
        recording_id = _create_recording(client)["id"]
        assert client.post(f"/api/recordings/{recording_id}/teleop/start").status_code == 200
        assert client.post(f"/api/recordings/{recording_id}/episodes/start", json={}).status_code == 200
        time.sleep(0.05)

    assert asyncio.run(manager.active_resource_counts()) == (0, 0)
    assert not (staging / str(recording_id) / "episode_000000" / "video.mp4").exists()


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
