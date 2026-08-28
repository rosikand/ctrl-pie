from __future__ import annotations

import asyncio
import copy
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.api.arms import get_yam_driver
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import get_db
from ctrl_pi.drivers.yam import ArmNotFoundError, ArmTelemetry, YAMDriver
from ctrl_pi.hf import (
    DatasetConversionError,
    HFDatasetUploader,
    HubAuthenticationError,
    HubUploadError,
    RecordingUploadSource,
    UploadConflictError,
)
from ctrl_pi.models import AppSetting, Recording, Robot
from ctrl_pi.recording import (
    EpisodeResult,
    RecordingConflictError,
    RecordingManager,
    RecordingRuntimeError,
    RecordingStateSnapshot,
)

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

RecordingStatus = Literal[
    "draft", "teleop", "recording", "ready", "uploading", "uploaded", "failed"
]


class RecordingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    task: str = Field(min_length=1, max_length=2000)
    leader_robot_id: str = Field(min_length=1, max_length=120)
    follower_robot_id: str = Field(min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "task", "leader_robot_id", "follower_robot_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("metadata")
    @classmethod
    def reserve_episode_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {"episodes", "upload"}.intersection(value)
        if reserved:
            key = sorted(reserved)[0]
            raise ValueError(f"metadata key '{key}' is reserved")
        return value


class RecordingRead(BaseModel):
    id: str
    name: str
    task: str
    status: RecordingStatus
    leader_robot_id: str
    follower_robot_id: str
    episode_count: int
    duration_seconds: float
    hf_repo_id: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class RecordingsResponse(BaseModel):
    recordings: list[RecordingRead]


class RecordingState(BaseModel):
    recording_id: str
    teleop_active: bool
    episode_active: bool
    current_episode_index: int | None
    episode_duration_seconds: float
    episode_count: int
    status: RecordingStatus


class EpisodeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("operator", "notes")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class EpisodeStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: EpisodeMetadata | None = None


class EpisodeStop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("notes")
    @classmethod
    def strip_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class RecordingUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_name: str = Field(min_length=1, max_length=96)
    private: bool = True

    @field_validator("repo_name")
    @classmethod
    def validate_repo_name(cls, value: str) -> str:
        value = value.strip()
        if (
            not re.fullmatch(r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?", value)
            or ".." in value
            or "--" in value
        ):
            raise ValueError("repo_name must be a valid Hugging Face repository slug")
        return value


class RecordingUploadResponse(BaseModel):
    recording: RecordingRead
    repo_id: str
    repo_url: str
    revision: str | None


def get_recording_manager(request: Request) -> RecordingManager:
    return request.app.state.recording_manager


def get_hf_uploader(request: Request) -> HFDatasetUploader:
    return request.app.state.hf_uploader


def _recording_or_404(db: Session, recording_id: uuid.UUID) -> Recording:
    recording = db.get(Recording, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording session was not found.")
    return recording


def _robot_driver_id(db: Session, robot_id: uuid.UUID | None) -> str:
    if robot_id is None:
        return "unassigned"
    robot = db.get(Robot, robot_id)
    return "unassigned" if robot is None else robot.driver_id


def _recording_read(db: Session, recording: Recording) -> RecordingRead:
    return RecordingRead(
        id=str(recording.id),
        name=recording.name,
        task=recording.task,
        status=recording.status,
        leader_robot_id=_robot_driver_id(db, recording.leader_robot_id),
        follower_robot_id=_robot_driver_id(db, recording.follower_robot_id),
        episode_count=recording.episode_count,
        duration_seconds=recording.duration_seconds,
        hf_repo_id=recording.hf_repo_id,
        metadata=recording.recording_metadata,
        created_at=recording.created_at,
        updated_at=recording.updated_at,
    )


def _upsert_robot(db: Session, arm: ArmTelemetry) -> Robot:
    robot = db.scalar(select(Robot).where(Robot.driver_id == arm.id))
    if robot is None:
        robot = Robot(
            driver_id=arm.id,
            name=arm.name,
            role=arm.role,
            driver=arm.driver,
            can_interface=arm.can.interface,
            enabled=arm.connected,
            config={},
        )
        db.add(robot)
    else:
        robot.name = arm.name
        robot.role = arm.role
        robot.driver = arm.driver
        robot.can_interface = arm.can.interface
        robot.enabled = arm.connected
    db.flush()
    return robot


def _runtime_arguments(db: Session, recording: Recording) -> dict[str, Any]:
    return {
        "recording_id": str(recording.id),
        "leader_robot_id": _robot_driver_id(db, recording.leader_robot_id),
        "follower_robot_id": _robot_driver_id(db, recording.follower_robot_id),
        "episode_count": recording.episode_count,
        "status": recording.status,
    }


def _state(snapshot: RecordingStateSnapshot) -> RecordingState:
    return RecordingState(**snapshot.__dict__)


def _conflict(error: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(error))


def _reconcile_stale_upload(
    db: Session, recording: Recording, uploader: HFDatasetUploader
) -> bool:
    if recording.status != "uploading" or uploader.is_active(str(recording.id)):
        return False
    previous = recording.recording_metadata.get("upload", {})
    previous = previous if isinstance(previous, dict) else {}
    recording.status = "failed"
    recording.recording_metadata = {
        **recording.recording_metadata,
        "upload": {
            **previous,
            "status": "failed",
            "finished_at": datetime.now(UTC).isoformat(),
            "error": "Upload was interrupted before completion.",
        },
    }
    return True


def _has_upload_target(recording: Recording) -> bool:
    upload = recording.recording_metadata.get("upload", {})
    return isinstance(upload, dict) and bool(upload.get("repo_id"))


def _reject_recording_mutation(recording: Recording) -> None:
    if recording.status in {"uploading", "uploaded"} or _has_upload_target(recording):
        raise HTTPException(
            status_code=409,
            detail="Recordings with an upload target are immutable; retry the upload instead.",
        )


async def _wait_for_upload_task(task: asyncio.Task[Any]) -> bool:
    """Keep a thread-backed upload owned even if its HTTP request is cancelled."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    return cancelled


@router.get("", response_model=RecordingsResponse)
def list_recordings(
    db: Session = Depends(get_db),
    uploader: HFDatasetUploader = Depends(get_hf_uploader),
) -> RecordingsResponse:
    recordings = db.scalars(select(Recording).order_by(Recording.created_at.desc())).all()
    changed = False
    for recording in recordings:
        changed = _reconcile_stale_upload(db, recording, uploader) or changed
    if changed:
        db.commit()
    return RecordingsResponse(recordings=[_recording_read(db, item) for item in recordings])


@router.post("", response_model=RecordingRead, status_code=201)
def create_recording(
    payload: RecordingCreate,
    db: Session = Depends(get_db),
    driver: YAMDriver = Depends(get_yam_driver),
) -> RecordingRead:
    if payload.leader_robot_id == payload.follower_robot_id:
        raise HTTPException(status_code=422, detail="Leader and follower must be different arms.")
    try:
        leader_arm = driver.get_arm(payload.leader_robot_id)
        follower_arm = driver.get_arm(payload.follower_robot_id)
    except ArmNotFoundError as error:
        raise HTTPException(status_code=422, detail=f"Unknown arm: {error.args[0]}") from error
    if leader_arm.role != "leader" or follower_arm.role != "follower":
        raise HTTPException(
            status_code=422,
            detail="Selected arms must have leader and follower roles respectively.",
        )
    if not leader_arm.connected or not follower_arm.connected:
        raise HTTPException(status_code=422, detail="Both selected arms must be connected.")

    leader_robot = _upsert_robot(db, leader_arm)
    follower_robot = _upsert_robot(db, follower_arm)
    recording = Recording(
        name=payload.name,
        task=payload.task,
        status="draft",
        leader_robot_id=leader_robot.id,
        follower_robot_id=follower_robot.id,
        episode_count=0,
        duration_seconds=0.0,
        recording_metadata={**payload.metadata, "episodes": []},
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return _recording_read(db, recording)


@router.get("/{recording_id}", response_model=RecordingRead)
def get_recording(
    recording_id: uuid.UUID,
    db: Session = Depends(get_db),
    uploader: HFDatasetUploader = Depends(get_hf_uploader),
) -> RecordingRead:
    recording = _recording_or_404(db, recording_id)
    if _reconcile_stale_upload(db, recording, uploader):
        db.commit()
    return _recording_read(db, recording)


@router.get("/{recording_id}/state", response_model=RecordingState)
async def recording_state(
    recording_id: uuid.UUID,
    db: Session = Depends(get_db),
    manager: RecordingManager = Depends(get_recording_manager),
    uploader: HFDatasetUploader = Depends(get_hf_uploader),
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
    if _reconcile_stale_upload(db, recording, uploader):
        db.commit()
    snapshot = await manager.state(**_runtime_arguments(db, recording))
    if recording.status != snapshot.status:
        recording.status = snapshot.status
        db.commit()
    return _state(snapshot)


@router.post("/{recording_id}/teleop/start", response_model=RecordingState)
async def start_teleop(
    recording_id: uuid.UUID,
    db: Session = Depends(get_db),
    manager: RecordingManager = Depends(get_recording_manager),
    uploader: HFDatasetUploader = Depends(get_hf_uploader),
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
    if _reconcile_stale_upload(db, recording, uploader):
        db.commit()
    _reject_recording_mutation(recording)
    try:
        snapshot = await manager.start_teleop(**_runtime_arguments(db, recording))
    except (RecordingConflictError, ArmNotFoundError) as error:
        raise _conflict(error) from error
    recording.status = snapshot.status
    db.commit()
    return _state(snapshot)


@router.post("/{recording_id}/teleop/stop", response_model=RecordingState)
async def stop_teleop(
    recording_id: uuid.UUID,
    db: Session = Depends(get_db),
    manager: RecordingManager = Depends(get_recording_manager),
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
    await manager.ensure_session(**_runtime_arguments(db, recording))
    try:
        snapshot = await manager.stop_teleop(str(recording.id))
    except RecordingConflictError as error:
        raise _conflict(error) from error
    recording.status = snapshot.status
    db.commit()
    return _state(snapshot)


@router.post("/{recording_id}/episodes/start", response_model=RecordingState)
async def start_episode(
    recording_id: uuid.UUID,
    payload: EpisodeStart | None = None,
    db: Session = Depends(get_db),
    manager: RecordingManager = Depends(get_recording_manager),
    config: AppConfig = Depends(get_config),
    uploader: HFDatasetUploader = Depends(get_hf_uploader),
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
    if _reconcile_stale_upload(db, recording, uploader):
        db.commit()
    _reject_recording_mutation(recording)
    await manager.ensure_session(**_runtime_arguments(db, recording))
    setting = db.get(AppSetting, "recording_fps")
    fps = config.recording_fps if setting is None else int(setting.value)
    metadata = {} if payload is None or payload.metadata is None else payload.metadata.model_dump(exclude_none=True)
    try:
        snapshot = await manager.start_episode(str(recording.id), fps=fps, metadata=metadata)
    except RecordingConflictError as error:
        raise _conflict(error) from error
    except (RecordingRuntimeError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    recording.status = snapshot.status
    db.commit()
    return _state(snapshot)


@router.post("/{recording_id}/episodes/stop", response_model=RecordingState)
async def stop_episode(
    recording_id: uuid.UUID,
    payload: EpisodeStop | None = None,
    db: Session = Depends(get_db),
    manager: RecordingManager = Depends(get_recording_manager),
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
    await manager.ensure_session(**_runtime_arguments(db, recording))
    payload = payload or EpisodeStop()
    try:
        snapshot, result = await manager.stop_episode(
            str(recording.id), success=payload.success, notes=payload.notes
        )
    except RecordingConflictError as error:
        raise _conflict(error) from error
    except (RecordingRuntimeError, OSError) as error:
        recording.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(error)) from error

    recording.status = snapshot.status
    recording.episode_count += 1
    recording.duration_seconds += result.duration_seconds
    recording.recording_metadata = _metadata_with_episode(
        recording.recording_metadata, result
    )
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        await manager.episode_persistence_failed(str(recording.id))
        raise
    await manager.confirm_episode_persisted(
        str(recording.id), recording.episode_count, recording.status
    )
    return _state(snapshot)


@router.post("/{recording_id}/upload", response_model=RecordingUploadResponse)
async def upload_recording(
    recording_id: uuid.UUID,
    payload: RecordingUpload,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
    manager: RecordingManager = Depends(get_recording_manager),
    uploader: HFDatasetUploader = Depends(get_hf_uploader),
) -> RecordingUploadResponse:
    recording = _recording_or_404(db, recording_id)
    if _reconcile_stale_upload(db, recording, uploader):
        db.commit()

    namespace = (config.hf_namespace or "").strip()
    if not namespace:
        raise HTTPException(status_code=503, detail="Set HF_NAMESPACE before uploading.")
    if config.hf_token is None or not config.hf_token.get_secret_value().strip():
        raise HTTPException(status_code=503, detail="Set HF_TOKEN before uploading.")
    token = config.hf_token.get_secret_value()
    try:
        repo_id = uploader.repo_id(namespace, payload.repo_name)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    previous_upload = recording.recording_metadata.get("upload", {})
    previous_repo_id = (
        previous_upload.get("repo_id") if isinstance(previous_upload, dict) else None
    )
    if previous_repo_id is not None and previous_repo_id != repo_id:
        raise HTTPException(
            status_code=409,
            detail="A failed upload may only retry its original repository target.",
        )

    runtime = await manager.state(**_runtime_arguments(db, recording))
    if runtime.teleop_active or runtime.episode_active:
        raise HTTPException(
            status_code=409,
            detail="Stop teleop and finalize the episode before uploading.",
        )
    if recording.status not in {"ready", "failed"}:
        raise HTTPException(
            status_code=409,
            detail="Only ready or failed recordings can be uploaded.",
        )
    if recording.episode_count < 1:
        raise HTTPException(status_code=409, detail="Record at least one episode before uploading.")

    source = RecordingUploadSource(
        recording_id=str(recording.id),
        task=recording.task,
        episode_count=recording.episode_count,
        metadata=copy.deepcopy(recording.recording_metadata),
    )
    previous_status = recording.status
    previous_metadata = copy.deepcopy(recording.recording_metadata)
    try:
        uploader.reserve(str(recording.id), repo_id)
    except UploadConflictError as error:
        raise _conflict(error) from error

    try:
        started_at = datetime.now(UTC).isoformat()
        recording.status = "uploading"
        recording.recording_metadata = {
            **recording.recording_metadata,
            "upload": {
                "status": "uploading",
                "repo_id": repo_id,
                "owner_recording_id": str(recording.id),
                "remote_repo_created": False,
                "private": payload.private,
                "started_at": started_at,
            },
        }
        db.commit()

        upload_task = asyncio.create_task(
            asyncio.to_thread(
                uploader.upload,
                source,
                namespace,
                payload.repo_name,
                payload.private,
                token,
            )
        )
        cancelled = await _wait_for_upload_task(upload_task)
        try:
            result = upload_task.result()
        except HubAuthenticationError as error:
            _mark_upload_failed(
                db, recording, repo_id, payload.private, started_at, False
            )
            if cancelled:
                raise asyncio.CancelledError from error
            raise HTTPException(status_code=403, detail=str(error)) from error
        except DatasetConversionError as error:
            _mark_upload_failed(
                db, recording, repo_id, payload.private, started_at, False
            )
            if cancelled:
                raise asyncio.CancelledError from error
            raise HTTPException(status_code=500, detail=str(error)) from error
        except UploadConflictError as error:
            _restore_after_upload_collision(
                db,
                recording,
                previous_status=previous_status,
                previous_metadata=previous_metadata,
            )
            if cancelled:
                raise asyncio.CancelledError from error
            raise HTTPException(status_code=409, detail=str(error)) from error
        except HubUploadError as error:
            _mark_upload_failed(
                db,
                recording,
                repo_id,
                payload.private,
                started_at,
                error.remote_repo_created,
            )
            if cancelled:
                raise asyncio.CancelledError from error
            raise HTTPException(status_code=502, detail=str(error)) from error
        except Exception as error:
            _mark_upload_failed(
                db, recording, repo_id, payload.private, started_at, False
            )
            if cancelled:
                raise asyncio.CancelledError from error
            raise HTTPException(
                status_code=502,
                detail="Dataset upload failed without exposing credential details.",
            ) from error

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
                "private": payload.private,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "total_frames": result.total_frames,
                "fps": result.fps,
                "lerobot_version": "0.4.4",
            },
        }
        db.commit()
        db.refresh(recording)
        if cancelled:
            raise asyncio.CancelledError
        return RecordingUploadResponse(
            recording=_recording_read(db, recording),
            repo_id=result.repo_id,
            repo_url=result.repo_url,
            revision=result.revision,
        )
    finally:
        uploader.release(str(recording.id), repo_id)


def _restore_after_upload_collision(
    db: Session,
    recording: Recording,
    *,
    previous_status: str,
    previous_metadata: dict[str, Any],
) -> None:
    previous_metadata.pop("upload", None)
    recording.status = previous_status
    recording.recording_metadata = previous_metadata
    db.commit()


def _mark_upload_failed(
    db: Session,
    recording: Recording,
    repo_id: str,
    private: bool,
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
            "private": private,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "error": "Dataset conversion or Hub transfer failed.",
        },
    }
    db.commit()


def _metadata_with_episode(
    current: dict[str, Any], result: EpisodeResult
) -> dict[str, Any]:
    stored_episodes = current.get("episodes", [])
    episodes = list(stored_episodes) if isinstance(stored_episodes, list) else []
    episodes.append(
        {
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
    )
    return {**current, "episodes": episodes}
