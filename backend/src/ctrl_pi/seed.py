from __future__ import annotations

import asyncio
import copy
import fcntl
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.camera import MockCamera
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import engine_for_url
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.hf import (
    DatasetConversionError,
    DatasetUploadResult,
    HFDatasetUploader,
    HubAuthenticationError,
    HubUploadError,
    RecordingUploadSource,
    UploadConflictError,
)
from ctrl_pi.models import Recording, Robot, TrainingRun
from ctrl_pi.recording import EpisodeResult, RecordingManager

_SAMPLE_RECORDING_ID = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://ctrl-pi.dev/seed/sample-recording/v1"
)
_RUN_IDS = (
    uuid.uuid5(uuid.NAMESPACE_URL, "https://ctrl-pi.dev/seed/act-baseline/v1"),
    uuid.uuid5(uuid.NAMESPACE_URL, "https://ctrl-pi.dev/seed/smolvla-run/v1"),
)
_SAMPLE_REPO_NAME = "ctrl-pi-yam-sample"


@dataclass(frozen=True)
class SeedResult:
    database_seeded: bool
    dataset_status: Literal["skipped", "uploaded", "already_uploaded", "failed"]


@contextmanager
def _exclusive_seed_lock(staging_dir: Path) -> Iterator[bool]:
    lock_directory = staging_dir.parent
    try:
        lock_directory.mkdir(parents=True, exist_ok=True)
        stream = (lock_directory / ".seed.lock").open("a+", encoding="utf-8")
    except OSError:
        yield False
        return
    acquired = False
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield acquired
    finally:
        if acquired:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def seed(
    *,
    config: AppConfig | None = None,
    engine: Engine | None = None,
    driver: MockYAMDriver | None = None,
    uploader: HFDatasetUploader | None = None,
    output: Callable[[str], None] = print,
    episode_seconds: float = 0.35,
) -> SeedResult:
    """Idempotently seed local mock control-plane state and an optional HF sample."""

    config = config or get_config()
    with _exclusive_seed_lock(config.recording_staging_dir) as acquired:
        if not acquired:
            output("Seed skipped: another `make seed` process is active.")
            return SeedResult(database_seeded=False, dataset_status="failed")
        try:
            return _seed_impl(
                config=config,
                engine=engine,
                driver=driver,
                uploader=uploader,
                output=output,
                episode_seconds=episode_seconds,
            )
        except SQLAlchemyError:
            output("Database seed failed without exposing connection details.")
            return SeedResult(database_seeded=False, dataset_status="failed")


def _seed_impl(
    *,
    config: AppConfig,
    engine: Engine | None,
    driver: MockYAMDriver | None,
    uploader: HFDatasetUploader | None,
    output: Callable[[str], None],
    episode_seconds: float,
) -> SeedResult:
    if engine is None and config.database_url is not None:
        database_url = config.database_url.get_secret_value().strip()
        if database_url:
            engine = engine_for_url(database_url)
    if engine is None:
        output("Database seed skipped: set DATABASE_URL and run `make seed` again.")
        return SeedResult(database_seeded=False, dataset_status="skipped")
    driver = driver or MockYAMDriver()
    with Session(engine) as db:
        robots = _seed_control_plane(db, driver, config.hf_namespace)
        db.commit()
        leader_id = robots["yam-leader"].id
        follower_id = robots["yam-follower"].id
    output(f"Seeded {len(robots)} mock robots and two example training runs.")

    namespace = (config.hf_namespace or "").strip()
    token = (
        ""
        if config.hf_token is None
        else config.hf_token.get_secret_value().strip()
    )
    if not namespace or not token:
        output("Sample dataset upload skipped: set HF_NAMESPACE and HF_TOKEN to enable it.")
        return SeedResult(database_seeded=True, dataset_status="skipped")

    uploader = uploader or HFDatasetUploader(config.recording_staging_dir)
    try:
        repo_id = uploader.repo_id(namespace, _SAMPLE_REPO_NAME)
    except ValueError:
        output("Sample dataset upload failed: HF_NAMESPACE is invalid.")
        return SeedResult(database_seeded=True, dataset_status="failed")

    with Session(engine) as db:
        recording = _sample_recording(
            db,
            leader_id=leader_id,
            follower_id=follower_id,
        )
        if recording.status == "uploaded" and recording.hf_repo_id == repo_id:
            output(f"Sample dataset already uploaded as {repo_id}.")
            return SeedResult(database_seeded=True, dataset_status="already_uploaded")
        if recording.status == "uploading":
            if _upload_is_recent(recording) and not _same_host_attempt(recording):
                output("Sample dataset upload is active on another ctrl-pi process.")
                return SeedResult(database_seeded=True, dataset_status="failed")
            recording.status = "failed"
        db.commit()
        needs_episode = not _has_finalized_episode(
            recording, config.recording_staging_dir
        )
        existing_target = _upload_target(recording)

    if needs_episode:
        if existing_target is not None:
            output("Sample dataset upload failed: its staged episode is unavailable.")
            return SeedResult(database_seeded=True, dataset_status="failed")
        try:
            episode = asyncio.run(
                _record_sample_episode(
                    recording_id=str(_SAMPLE_RECORDING_ID),
                    driver=driver,
                    staging_dir=config.recording_staging_dir,
                    fps=min(config.recording_fps, 10),
                    duration_seconds=episode_seconds,
                )
            )
        except Exception:
            output("Sample dataset recording failed; verify ffmpeg is installed.")
            return SeedResult(database_seeded=True, dataset_status="failed")
        with Session(engine) as db:
            recording = db.get(Recording, _SAMPLE_RECORDING_ID)
            if recording is None:
                output("Sample dataset recording state disappeared before persistence.")
                return SeedResult(database_seeded=True, dataset_status="failed")
            recording.status = "ready"
            recording.episode_count = 1
            recording.duration_seconds = episode.duration_seconds
            recording.recording_metadata = {
                **recording.recording_metadata,
                "episodes": [_episode_metadata(episode)],
            }
            db.commit()

    return _upload_sample_recording(
        engine=engine,
        uploader=uploader,
        namespace=namespace,
        repo_id=repo_id,
        token=token,
        output=output,
    )


def _upload_sample_recording(
    *,
    engine: Engine,
    uploader: HFDatasetUploader,
    namespace: str,
    repo_id: str,
    token: str,
    output: Callable[[str], None],
) -> SeedResult:
    with Session(engine) as db:
        recording = db.scalar(
            select(Recording)
            .where(Recording.id == _SAMPLE_RECORDING_ID)
            .with_for_update()
        )
        if recording is None:
            output("Sample dataset recording state is unavailable.")
            return SeedResult(database_seeded=True, dataset_status="failed")
        if recording.status == "uploaded" and recording.hf_repo_id == repo_id:
            output(f"Sample dataset already uploaded as {repo_id}.")
            return SeedResult(database_seeded=True, dataset_status="already_uploaded")
        if (
            recording.status == "uploading"
            and _upload_is_recent(recording)
            and not _same_host_attempt(recording)
        ):
            output("Sample dataset upload is active on another ctrl-pi process.")
            return SeedResult(database_seeded=True, dataset_status="failed")
        previous_status = recording.status
        previous_metadata = copy.deepcopy(recording.recording_metadata)
        source = RecordingUploadSource(
            recording_id=str(recording.id),
            task=recording.task,
            episode_count=recording.episode_count,
            metadata=copy.deepcopy(recording.recording_metadata),
        )
        recording_id = str(recording.id)
        started_at = datetime.now(UTC).isoformat()
        try:
            uploader.reserve(recording_id, repo_id)
        except UploadConflictError:
            output("Sample dataset upload is already active.")
            return SeedResult(database_seeded=True, dataset_status="failed")

        try:
            recording.status = "uploading"
            recording.recording_metadata = {
                **recording.recording_metadata,
                "upload": {
                    "status": "uploading",
                    "repo_id": repo_id,
                    "owner_recording_id": str(recording.id),
                    "remote_repo_created": False,
                    "private": True,
                    "started_at": started_at,
                    "seed_host": socket.gethostname(),
                },
            }
            db.commit()

            try:
                result = _upload_with_marker_adoption(
                    uploader=uploader,
                    source=source,
                    namespace=namespace,
                    repo_id=repo_id,
                    token=token,
                )
            except UploadConflictError:
                current = _locked_seed_recording(db)
                if current is not None and _owns_upload_attempt(current, started_at):
                    previous_metadata.pop("upload", None)
                    current.status = previous_status
                    current.recording_metadata = previous_metadata
                    db.commit()
                output("Sample dataset upload skipped: the target repo is unrelated.")
                return SeedResult(database_seeded=True, dataset_status="failed")
            except HubUploadError as error:
                _persist_seed_upload_failure(
                    db,
                    repo_id=repo_id,
                    started_at=started_at,
                    remote_repo_created=error.remote_repo_created,
                )
                output("Sample dataset upload failed; retry `make seed` safely.")
                return SeedResult(database_seeded=True, dataset_status="failed")
            except (HubAuthenticationError, DatasetConversionError):
                _persist_seed_upload_failure(
                    db,
                    repo_id=repo_id,
                    started_at=started_at,
                    remote_repo_created=False,
                )
                output("Sample dataset upload failed; verify Hugging Face configuration.")
                return SeedResult(database_seeded=True, dataset_status="failed")
            except Exception:
                _persist_seed_upload_failure(
                    db,
                    repo_id=repo_id,
                    started_at=started_at,
                    remote_repo_created=False,
                )
                output("Sample dataset upload failed without exposing credential details.")
                return SeedResult(database_seeded=True, dataset_status="failed")

            persistence = _persist_seed_upload_success(
                db,
                result=result,
                started_at=started_at,
            )
            if persistence == "already_uploaded":
                output(f"Sample dataset already uploaded as {repo_id}.")
                return SeedResult(
                    database_seeded=True, dataset_status="already_uploaded"
                )
            if persistence == "superseded":
                output("Sample dataset upload result was superseded by another process.")
                return SeedResult(database_seeded=True, dataset_status="failed")
        finally:
            uploader.release(recording_id, repo_id)

    output(f"Uploaded private sample dataset to {repo_id}.")
    return SeedResult(database_seeded=True, dataset_status="uploaded")


def _upload_with_marker_adoption(
    *,
    uploader: HFDatasetUploader,
    source: RecordingUploadSource,
    namespace: str,
    repo_id: str,
    token: str,
) -> DatasetUploadResult:
    try:
        return uploader.upload(
            source,
            namespace,
            _SAMPLE_REPO_NAME,
            True,
            token,
        )
    except UploadConflictError:
        if not uploader.repository_owned_by(
            repo_id=repo_id,
            recording_id=source.recording_id,
            token=token,
        ):
            raise
    adopted_source = RecordingUploadSource(
        recording_id=source.recording_id,
        task=source.task,
        episode_count=source.episode_count,
        metadata={
            **source.metadata,
            "upload": {
                "repo_id": repo_id,
                "owner_recording_id": source.recording_id,
                "remote_repo_created": True,
            },
        },
    )
    return uploader.upload(
        adopted_source,
        namespace,
        _SAMPLE_REPO_NAME,
        True,
        token,
    )


def _locked_seed_recording(db: Session) -> Recording | None:
    db.expire_all()
    return db.scalar(
        select(Recording)
        .where(Recording.id == _SAMPLE_RECORDING_ID)
        .with_for_update()
    )


def _persist_seed_upload_failure(
    db: Session,
    *,
    repo_id: str,
    started_at: str,
    remote_repo_created: bool,
) -> None:
    recording = _locked_seed_recording(db)
    if recording is None or recording.status == "uploaded":
        return
    if not _owns_upload_attempt(recording, started_at):
        return
    _mark_seed_upload_failed(
        recording,
        repo_id=repo_id,
        started_at=started_at,
        remote_repo_created=remote_repo_created,
    )
    db.commit()


def _persist_seed_upload_success(
    db: Session,
    *,
    result: DatasetUploadResult,
    started_at: str,
) -> Literal["persisted", "already_uploaded", "superseded"]:
    for attempt in range(2):
        recording = _locked_seed_recording(db)
        if recording is None:
            raise SQLAlchemyError("seed recording disappeared")
        if recording.status == "uploaded" and recording.hf_repo_id == result.repo_id:
            return "already_uploaded"
        if not _owns_upload_attempt(recording, started_at):
            return "superseded"
        recording.status = "uploaded"
        recording.hf_repo_id = result.repo_id
        recording.recording_metadata = {
            **recording.recording_metadata,
            "upload": {
                "status": "uploaded",
                "repo_id": result.repo_id,
                "owner_recording_id": str(recording.id),
                "remote_repo_created": True,
                "repo_url": result.repo_url,
                "revision": result.revision,
                "private": True,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "total_frames": result.total_frames,
                "fps": result.fps,
                "lerobot_version": "0.4.4",
            },
        }
        try:
            db.commit()
            return "persisted"
        except SQLAlchemyError:
            db.rollback()
            if attempt == 1:
                raise
    return "superseded"


def _seed_control_plane(
    db: Session,
    driver: MockYAMDriver,
    namespace: str | None,
) -> dict[str, Robot]:
    robots: dict[str, Robot] = {}
    for arm in driver.list_arms():
        arm_config = {
            "_ctrl_pi_seed": 1,
            "pair_id": arm.pair_id,
            "group_id": arm.group_id,
            "side": arm.side,
            "transport_kind": arm.transport_kind,
            "stable_identity": arm.stable_identity,
            "end_effector_kind": arm.end_effector_kind,
        }
        robot = db.scalar(select(Robot).where(Robot.driver_id == arm.id))
        if robot is None:
            robot = Robot(
                driver_id=arm.id,
                name=arm.name,
                role=arm.role,
                driver=arm.driver,
                can_interface=None,
                enabled=arm.connected,
                config=arm_config,
            )
            db.add(robot)
        elif (robot.config or {}).get("_ctrl_pi_seed") == 1:
            robot.name = arm.name
            robot.role = arm.role
            robot.driver = arm.driver
            robot.can_interface = None
            robot.enabled = arm.connected
            robot.config = arm_config
        db.flush()
        robots[arm.id] = robot

    namespace = (namespace or "example").strip() or "example"
    definitions = [
        {
            "id": _RUN_IDS[0],
            "name": "ACT block pickup baseline",
            "status": "completed",
            "current_step": 1000,
            "dataset_repo": f"{namespace}/ctrl-pi-yam-sample",
            "base_model": "lerobot/act_aloha_sim_transfer_cube_human",
            "runtime": "lerobot",
            "framework": "pytorch",
            "config": {"batch_size": 32, "learning_rate": 0.0001},
            "metrics": {
                "train/loss": [
                    {"step": 0, "value": 1.2},
                    {"step": 250, "value": 0.72},
                    {"step": 500, "value": 0.43},
                    {"step": 750, "value": 0.29},
                    {"step": 1000, "value": 0.21},
                ],
                "eval/success_rate": [
                    {"step": 250, "value": 0.3},
                    {"step": 500, "value": 0.55},
                    {"step": 750, "value": 0.72},
                    {"step": 1000, "value": 0.84},
                ],
            },
        },
        {
            "id": _RUN_IDS[1],
            "name": "SmolVLA placement experiment",
            "status": "running",
            "current_step": 600,
            "dataset_repo": f"{namespace}/ctrl-pi-yam-sample",
            "base_model": "lerobot/smolvla_base",
            "runtime": "lerobot",
            "framework": "pytorch",
            "config": {"batch_size": 16, "learning_rate": 0.00005},
            "metrics": {
                "train/loss": [
                    {"step": 0, "value": 1.65},
                    {"step": 200, "value": 1.08},
                    {"step": 400, "value": 0.81},
                    {"step": 600, "value": 0.64},
                ]
            },
        },
    ]
    for definition in definitions:
        run = db.get(TrainingRun, definition["id"])
        if run is None:
            run = TrainingRun(id=definition["id"], name=definition["name"])
            db.add(run)
        for field in (
            "name",
            "status",
            "current_step",
            "dataset_repo",
            "base_model",
            "runtime",
            "framework",
            "config",
            "metrics",
        ):
            setattr(run, field, copy.deepcopy(definition[field]))
        run.output_model_repo = None
        run.checkpoint_revision = None
        run.checkpoints = []
    return robots


def _sample_recording(
    db: Session,
    *,
    leader_id: uuid.UUID,
    follower_id: uuid.UUID,
) -> Recording:
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == _SAMPLE_RECORDING_ID)
        .with_for_update()
    )
    if recording is None:
        recording = Recording(
            id=_SAMPLE_RECORDING_ID,
            name="ctrl-pi sample demonstration",
            task="Move the mock block into the tray",
            status="draft",
            episode_count=0,
            duration_seconds=0.0,
            recording_metadata={"seed": {"schema": 1}},
        )
        db.add(recording)
    recording.leader_robot_id = leader_id
    recording.follower_robot_id = follower_id
    db.flush()
    return recording


def _upload_target(recording: Recording) -> str | None:
    upload = (recording.recording_metadata or {}).get("upload")
    if not isinstance(upload, dict):
        return None
    value = upload.get("repo_id")
    return value if isinstance(value, str) and value else None


def _upload_metadata(recording: Recording) -> dict[str, Any]:
    upload = (recording.recording_metadata or {}).get("upload")
    return upload if isinstance(upload, dict) else {}


def _same_host_attempt(recording: Recording) -> bool:
    return _upload_metadata(recording).get("seed_host") == socket.gethostname()


def _upload_is_recent(recording: Recording) -> bool:
    started_at = _upload_metadata(recording).get("started_at")
    if not isinstance(started_at, str):
        return False
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
    except ValueError:
        return False
    age = (datetime.now(UTC) - started.astimezone(UTC)).total_seconds()
    return 0 <= age < 30 * 60


def _owns_upload_attempt(recording: Recording, started_at: str) -> bool:
    upload = _upload_metadata(recording)
    return (
        recording.status == "uploading"
        and upload.get("started_at") == started_at
        and upload.get("seed_host") == socket.gethostname()
    )


def _has_finalized_episode(recording: Recording, staging_dir: Path) -> bool:
    episodes = (recording.recording_metadata or {}).get("episodes")
    if not isinstance(episodes, list) or len(episodes) != recording.episode_count:
        return False
    root = staging_dir.resolve()
    for episode in episodes:
        if not isinstance(episode, dict) or not isinstance(episode.get("artifact_key"), str):
            return False
        relative = Path(episode["artifact_key"])
        directory = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not directory.is_relative_to(root)
            or not (directory / "video.mp4").is_file()
            or not (directory / "samples.jsonl").is_file()
        ):
            return False
    return bool(episodes)


async def _record_sample_episode(
    *,
    recording_id: str,
    driver: MockYAMDriver,
    staging_dir: Path,
    fps: int,
    duration_seconds: float,
) -> EpisodeResult:
    manager = RecordingManager(
        driver=driver,
        camera=MockCamera(width=96, height=64),
        staging_dir=staging_dir,
        sync_duration_seconds=0.05,
        sync_steps=5,
    )
    await manager.startup()
    try:
        follower_writes = driver.action_counts["yam-follower"]
        teleop = await manager.start_teleop(
            recording_id,
            "yam-leader",
            "yam-follower",
            episode_count=0,
            status="draft",
        )
        if teleop.sync_enabled or teleop.sync_in_progress:
            raise RuntimeError("mock seed teleoperation did not start observation-only")
        await asyncio.sleep(2.0 / manager.teleop_frequency_hz)
        if driver.action_counts["yam-follower"] != follower_writes:
            raise RuntimeError("mock seed teleoperation wrote before sync enable")
        syncing = await manager.enable_sync(
            recording_id,
            acknowledge_slow_sync_motion=True,
        )
        if not syncing.sync_in_progress or syncing.sync_enabled:
            raise RuntimeError("mock seed pair synchronization did not start")
        deadline = asyncio.get_running_loop().time() + 2.0
        while True:
            state = await manager.state(
                recording_id,
                "yam-leader",
                "yam-follower",
                episode_count=0,
                status="teleop",
            )
            if state.sync_enabled and not state.sync_in_progress:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("mock seed pair synchronization did not complete")
            await asyncio.sleep(0.005)
        await manager.start_episode(
            recording_id,
            fps=fps,
            metadata={"operator": "ctrl-pi seed"},
        )
        await asyncio.sleep(max(duration_seconds, 2.0 / fps))
        _, result = await manager.stop_episode(
            recording_id,
            success=True,
            notes="Generated by make seed",
        )
        await manager.confirm_episode_persisted(recording_id, 1, "teleop")
        await manager.disable_sync(recording_id)
        await manager.stop_teleop(recording_id)
        return result
    finally:
        await manager.shutdown()


def _episode_metadata(result: EpisodeResult) -> dict[str, Any]:
    return {
        "index": result.index,
        "duration_seconds": result.duration_seconds,
        "sample_count": result.sample_count,
        "success": result.success,
        "notes": result.notes,
        "metadata": result.metadata,
        "artifact_key": result.artifact_key,
        "started_at": result.started_at.isoformat(),
        "ended_at": result.ended_at.isoformat(),
        "fps": result.fps,
    }


def _mark_seed_upload_failed(
    recording: Recording,
    *,
    repo_id: str,
    started_at: str,
    remote_repo_created: bool,
) -> None:
    recording.status = "failed"
    recording.recording_metadata = {
        **recording.recording_metadata,
        "upload": {
            "status": "failed",
            "repo_id": repo_id,
            "owner_recording_id": str(recording.id),
            "remote_repo_created": remote_repo_created,
            "private": True,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "error": "Dataset conversion or Hub transfer failed.",
        },
    }


def main() -> int:
    result = seed()
    if not result.database_seeded or result.dataset_status == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
