from __future__ import annotations

import uuid
from datetime import datetime
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
        if "episodes" in value:
            raise ValueError("metadata key 'episodes' is reserved")
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


def get_recording_manager(request: Request) -> RecordingManager:
    return request.app.state.recording_manager


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


@router.get("", response_model=RecordingsResponse)
def list_recordings(db: Session = Depends(get_db)) -> RecordingsResponse:
    recordings = db.scalars(select(Recording).order_by(Recording.created_at.desc())).all()
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
    recording_id: uuid.UUID, db: Session = Depends(get_db)
) -> RecordingRead:
    return _recording_read(db, _recording_or_404(db, recording_id))


@router.get("/{recording_id}/state", response_model=RecordingState)
async def recording_state(
    recording_id: uuid.UUID,
    db: Session = Depends(get_db),
    manager: RecordingManager = Depends(get_recording_manager),
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
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
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
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
) -> RecordingState:
    recording = _recording_or_404(db, recording_id)
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
        }
    )
    return {**current, "episodes": episodes}
