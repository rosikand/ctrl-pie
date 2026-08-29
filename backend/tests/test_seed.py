from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.config import AppConfig
from ctrl_pi.db import Base
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.hf import DatasetUploadResult, UploadConflictError
from ctrl_pi.models import Recording, Robot, TrainingRun
from ctrl_pi.recording import EpisodeResult
from ctrl_pi.seed import (
    _SAMPLE_RECORDING_ID,
    _persist_seed_upload_success,
    _record_sample_episode,
    _upload_sample_recording,
    seed,
)


class _FakeVideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: int) -> None:
        del width, height, fps
        self.output_path = output_path
        self.output_path.write_bytes(b"")

    def write(self, frame) -> None:
        assert frame.rgb
        with self.output_path.open("ab") as stream:
            stream.write(b"frame")

    def close(self) -> None:
        if not self.output_path.read_bytes():
            raise RuntimeError("empty test video")

    def terminate(self) -> None:
        return None


def _engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_seed_is_idempotent_and_skips_external_upload_without_credentials(
    tmp_path: Path,
) -> None:
    engine = _engine()
    messages: list[str] = []
    config = AppConfig(
        _env_file=None,
        database_url=None,
        hf_namespace=None,
        hf_token=None,
        recording_staging_dir=tmp_path / "recordings",
    )

    first = seed(config=config, engine=engine, output=messages.append)
    second = seed(config=config, engine=engine, output=messages.append)

    assert first.database_seeded is second.database_seeded is True
    assert first.dataset_status == second.dataset_status == "skipped"
    assert any("HF_NAMESPACE and HF_TOKEN" in message for message in messages)
    assert messages.count("Seeded 4 mock robots and two example training runs.") == 2
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(Robot)) == 4
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 2
        assert db.scalar(select(func.count()).select_from(Recording)) == 0
        runs = db.scalars(select(TrainingRun).order_by(TrainingRun.name)).all()
        assert all(run.metrics for run in runs)
        assert {run.status for run in runs} == {"completed", "running"}
        robots = db.scalars(select(Robot).order_by(Robot.driver_id)).all()
        assert {
            robot.driver_id: (
                robot.role,
                robot.config["pair_id"],
                robot.config["group_id"],
                robot.config["side"],
                robot.config["end_effector_kind"],
            )
            for robot in robots
        } == {
            "yam-leader": (
                "leader",
                "right",
                "bimanual",
                "right",
                "yam_teaching_handle",
            ),
            "yam-follower": (
                "follower",
                "right",
                "bimanual",
                "right",
                "linear_4310",
            ),
            "yam-leader-left": (
                "leader",
                "left",
                "bimanual",
                "left",
                "yam_teaching_handle",
            ),
            "yam-follower-left": (
                "follower",
                "left",
                "bimanual",
                "left",
                "linear_4310",
            ),
        }
        assert all(robot.can_interface is None for robot in robots)
        stable_identities = {
            robot.config["stable_identity"] for robot in robots
        }
        assert len(stable_identities) == 4
        assert all(robot.config["transport_kind"] == "socketcan" for robot in robots)


@pytest.mark.asyncio
async def test_seed_sample_recording_crosses_explicit_sync_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    driver = MockYAMDriver()

    result = await _record_sample_episode(
        recording_id="seed-sync-test",
        driver=driver,
        staging_dir=tmp_path / "recordings",
        fps=10,
        duration_seconds=0.02,
    )

    assert result.sample_count >= 1
    assert driver.action_counts["yam-follower"] >= 5
    assert driver.action_counts["yam-follower-left"] == 0
    assert (tmp_path / "recordings" / result.artifact_key / "video.mp4").is_file()


class FakeUploader:
    def __init__(self, *, collision: bool = False, owned: bool = False) -> None:
        self.collision = collision
        self.owned = owned
        self.upload_sources = []
        self.reservations: list[tuple[str, str]] = []
        self.releases: list[tuple[str, str]] = []

    @staticmethod
    def repo_id(namespace: str, repo_name: str) -> str:
        return f"{namespace}/{repo_name}"

    def reserve(self, recording_id: str, repo_id: str) -> None:
        self.reservations.append((recording_id, repo_id))

    def release(self, recording_id: str, repo_id: str) -> None:
        self.releases.append((recording_id, repo_id))

    def repository_owned_by(
        self, *, repo_id: str, recording_id: str, token: str
    ) -> bool:
        assert repo_id == "acme/ctrl-pi-yam-sample"
        assert token == "hf_seed_secret"
        return self.owned

    def upload(self, source, namespace, repo_name, private, token):
        self.upload_sources.append(source)
        assert private is True
        assert token == "hf_seed_secret"
        if self.collision:
            self.collision = False
            raise UploadConflictError("unrelated")
        return DatasetUploadResult(
            repo_id=f"{namespace}/{repo_name}",
            repo_url=f"https://huggingface.co/datasets/{namespace}/{repo_name}",
            revision="a" * 40,
            total_frames=3,
            fps=10,
        )


async def _fake_episode(**kwargs) -> EpisodeResult:
    directory = (
        kwargs["staging_dir"]
        / kwargs["recording_id"]
        / "episode_000000"
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "video.mp4").write_bytes(b"mock-video")
    (directory / "samples.jsonl").write_text(
        json.dumps({"frame_index": 0}) + "\n",
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    return EpisodeResult(
        index=0,
        duration_seconds=0.3,
        sample_count=3,
        success=True,
        notes="seed",
        metadata={"operator": "seed"},
        artifact_key=f"{kwargs['recording_id']}/episode_000000",
        started_at=now,
        ended_at=now,
        fps=10,
    )


def _hf_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        _env_file=None,
        database_url=None,
        hf_namespace="acme",
        hf_token="hf_seed_secret",
        recording_staging_dir=tmp_path / "recordings",
        recording_fps=10,
    )


def test_seed_upload_is_idempotent_after_success(
    tmp_path: Path, monkeypatch
) -> None:
    engine = _engine()
    uploader = FakeUploader()
    monkeypatch.setattr("ctrl_pi.seed._record_sample_episode", _fake_episode)

    first = seed(
        config=_hf_config(tmp_path),
        engine=engine,
        uploader=uploader,
        output=lambda message: None,
    )
    second = seed(
        config=_hf_config(tmp_path),
        engine=engine,
        uploader=uploader,
        output=lambda message: None,
    )

    assert first.dataset_status == "uploaded"
    assert second.dataset_status == "already_uploaded"
    assert len(uploader.upload_sources) == 1
    assert uploader.reservations == uploader.releases
    with Session(engine) as db:
        recording = db.scalar(select(Recording))
        assert recording is not None
        assert recording.status == "uploaded"
        assert recording.episode_count == 1
        assert recording.hf_repo_id == "acme/ctrl-pi-yam-sample"


def test_seed_collision_never_claims_or_overwrites_unrelated_repo(
    tmp_path: Path, monkeypatch
) -> None:
    engine = _engine()
    uploader = FakeUploader(collision=True)
    monkeypatch.setattr("ctrl_pi.seed._record_sample_episode", _fake_episode)

    result = seed(
        config=_hf_config(tmp_path),
        engine=engine,
        uploader=uploader,
        output=lambda message: None,
    )

    assert result.dataset_status == "failed"
    assert len(uploader.upload_sources) == 1
    assert "upload" not in uploader.upload_sources[0].metadata
    assert uploader.reservations == uploader.releases
    with Session(engine) as db:
        recording = db.scalar(select(Recording))
        assert recording is not None
        assert recording.status == "ready"
        assert "upload" not in recording.recording_metadata


def test_seed_adopts_only_a_matching_remote_recording_marker(
    tmp_path: Path, monkeypatch
) -> None:
    engine = _engine()
    uploader = FakeUploader(collision=True, owned=True)
    monkeypatch.setattr("ctrl_pi.seed._record_sample_episode", _fake_episode)

    result = seed(
        config=_hf_config(tmp_path),
        engine=engine,
        uploader=uploader,
        output=lambda message: None,
    )

    assert result.dataset_status == "uploaded"
    assert len(uploader.upload_sources) == 2
    retry_upload = uploader.upload_sources[1].metadata["upload"]
    assert retry_upload["repo_id"] == "acme/ctrl-pi-yam-sample"
    assert retry_upload["owner_recording_id"] == str(_SAMPLE_RECORDING_ID)
    assert retry_upload["remote_repo_created"] is True


def test_seed_lost_attempt_is_not_reported_as_already_uploaded() -> None:
    engine = _engine()
    with Session(engine) as db:
        db.add(
            Recording(
                id=_SAMPLE_RECORDING_ID,
                name="seed",
                task="seed",
                status="failed",
                episode_count=1,
                duration_seconds=0.3,
                recording_metadata={},
            )
        )
        db.commit()
        outcome = _persist_seed_upload_success(
            db,
            result=DatasetUploadResult(
                repo_id="acme/ctrl-pi-yam-sample",
                repo_url="https://huggingface.co/datasets/acme/ctrl-pi-yam-sample",
                revision="a" * 40,
                total_frames=3,
                fps=10,
            ),
            started_at="2026-08-28T00:00:00+00:00",
        )

    assert outcome == "superseded"


def test_seed_preserves_unowned_existing_robot_configuration(tmp_path: Path) -> None:
    engine = _engine()
    robot_id = uuid.uuid4()
    with Session(engine) as db:
        db.add(
            Robot(
                id=robot_id,
                driver_id="yam-leader",
                name="My calibrated leader",
                role="follower",
                driver="custom-yam",
                can_interface="can9",
                enabled=False,
                config={"calibration": {"offset": 0.25}},
            )
        )
        db.commit()

    result = seed(
        config=AppConfig(
            _env_file=None,
            database_url=None,
            hf_namespace=None,
            hf_token=None,
            recording_staging_dir=tmp_path / "recordings",
        ),
        engine=engine,
        output=lambda message: None,
    )

    assert result.database_seeded is True
    with Session(engine) as db:
        robot = db.get(Robot, robot_id)
        assert robot is not None
        assert robot.name == "My calibrated leader"
        assert robot.role == "follower"
        assert robot.driver == "custom-yam"
        assert robot.can_interface == "can9"
        assert robot.enabled is False
        assert robot.config == {"calibration": {"offset": 0.25}}


def test_seed_releases_upload_reservation_when_claim_commit_fails(
    tmp_path: Path, monkeypatch
) -> None:
    engine = _engine()
    uploader = FakeUploader()
    with Session(engine) as db:
        db.add(
            Recording(
                id=_SAMPLE_RECORDING_ID,
                name="seed",
                task="seed",
                status="ready",
                episode_count=1,
                duration_seconds=0.3,
                recording_metadata={"episodes": [{"artifact_key": "unused"}]},
            )
        )
        db.commit()

    def fail_commit(self: Session) -> None:
        raise SQLAlchemyError("connection string must stay private")

    monkeypatch.setattr(Session, "commit", fail_commit)
    with pytest.raises(SQLAlchemyError):
        _upload_sample_recording(
            engine=engine,
            uploader=uploader,
            namespace="acme",
            repo_id="acme/ctrl-pi-yam-sample",
            token="hf_seed_secret",
            output=lambda message: None,
        )

    assert uploader.reservations == uploader.releases
