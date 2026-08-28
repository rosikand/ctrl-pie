from __future__ import annotations

import asyncio
import copy
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.camera import MockCamera
from ctrl_pi.deployments import (
    DeploymentNotFoundError,
    DeploymentRecord,
    DeploymentService,
    DeploymentStorageError,
)
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import (
    ArmAction,
    ArmNotFoundError,
    ArmTelemetry,
    YAMDriver,
)
from ctrl_pi.inference_transport import InferenceTransport
from ctrl_pi.models import Deployment, Recording, Robot
from ctrl_pi.recording import (
    EpisodeResult,
    RecordingConflictError,
    RecordingManager,
    RecordingRuntimeError,
)
from ctrl_pi.rig import RigLease
from ctrl_pi.robot_inference import (
    RobotInferenceConfigurationError,
    RobotInferenceConflictError,
    RobotInferenceLoop,
    RobotInferencePreparationError,
    RobotInferenceSnapshot,
)

SessionStatus = Literal["idle", "starting", "running", "stopping", "stopped", "failed"]
RecordingSessionStatus = Literal[
    "disabled", "starting", "recording", "finalizing", "ready", "failed"
]


class InferenceSessionError(RuntimeError):
    """A safe inference-orchestration error."""


class InferenceSessionConflictError(InferenceSessionError):
    pass


class InferenceSessionConfigurationError(InferenceSessionError):
    pass


class InferenceSessionRuntimeError(InferenceSessionError):
    pass


class InferenceSessionStorageError(InferenceSessionError):
    pass


class InferenceSessionUnavailableError(InferenceSessionError):
    pass


@dataclass(frozen=True)
class InferenceStartOptions:
    arm_id: str
    task: str
    record_session: bool = False
    recording_name: str | None = None
    recording_metadata: dict[str, str] | None = None
    # Internal deterministic smoke boundary. It is deliberately not exposed
    # by the public REST payload.
    max_steps: int | None = None

    def __post_init__(self) -> None:
        arm_id = self.arm_id.strip() if isinstance(self.arm_id, str) else ""
        task = self.task.strip() if isinstance(self.task, str) else ""
        if not arm_id or len(arm_id) > 120 or "\x00" in arm_id:
            raise InferenceSessionConfigurationError("The inference arm ID is invalid.")
        if not task or len(task) > 512 or "\x00" in task:
            raise InferenceSessionConfigurationError("The inference task is invalid.")
        if not isinstance(self.record_session, bool):
            raise InferenceSessionConfigurationError(
                "record_session must be a boolean."
            )
        name = self.recording_name
        if name is not None:
            name = name.strip() if isinstance(name, str) else ""
            if not name or len(name) > 160 or "\x00" in name:
                raise InferenceSessionConfigurationError(
                    "The recording name is invalid."
                )
        metadata = self.recording_metadata or {}
        if not isinstance(metadata, dict) or set(metadata) - {"operator", "notes"}:
            raise InferenceSessionConfigurationError(
                "Inference recording metadata is invalid."
            )
        normalized: dict[str, str] = {}
        for key, maximum in (("operator", 160), ("notes", 2000)):
            value = metadata.get(key)
            if value is None:
                continue
            value = value.strip() if isinstance(value, str) else ""
            if not value or len(value) > maximum or "\x00" in value:
                raise InferenceSessionConfigurationError(
                    "Inference recording metadata is invalid."
                )
            normalized[key] = value
        if (
            self.max_steps is not None
            and (
                isinstance(self.max_steps, bool)
                or not isinstance(self.max_steps, int)
                or not 1 <= self.max_steps <= 1_000_000
            )
        ):
            raise InferenceSessionConfigurationError(
                "The inference step limit is invalid."
            )
        object.__setattr__(self, "arm_id", arm_id)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "recording_name", name)
        object.__setattr__(self, "recording_metadata", normalized)


@dataclass(frozen=True)
class InferenceStopOptions:
    recording_success: bool = True
    recording_notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recording_success, bool):
            raise InferenceSessionConfigurationError(
                "recording_success must be a boolean."
            )
        notes = self.recording_notes
        if notes is not None:
            notes = notes.strip() if isinstance(notes, str) else ""
            if len(notes) > 2000 or "\x00" in notes:
                raise InferenceSessionConfigurationError(
                    "The inference recording notes are invalid."
                )
            notes = notes or None
        object.__setattr__(self, "recording_notes", notes)


@dataclass(frozen=True)
class InferenceRecordingSnapshot:
    enabled: bool
    status: RecordingSessionStatus
    recording_id: uuid.UUID | None
    episode_count: int
    duration_seconds: float
    hf_repo_id: str | None


@dataclass(frozen=True)
class InferenceSessionSnapshot:
    deployment: DeploymentRecord
    session_status: SessionStatus
    endpoint_healthy: bool
    teardown_verified: bool
    steps_executed: int
    requests_completed: int
    dropped_chunks: int
    queue_depth: int
    last_latency_ms: float | None
    average_latency_ms: float | None
    frequency_hz: float
    last_error: str | None
    session_started_at: datetime | None
    session_stopped_at: datetime | None
    recording: InferenceRecordingSnapshot


@dataclass
class _LiveSession:
    deployment_id: uuid.UUID
    session_id: uuid.UUID
    arm_id: str
    status: SessionStatus
    loop: RobotInferenceLoop | None = None
    endpoint_healthy: bool = False
    teardown_verified: bool = False
    last_error: str | None = None
    recording_id: uuid.UUID | None = None
    recording_status: RecordingSessionStatus = "disabled"
    deployment: DeploymentRecord | None = None
    supervisor_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[None] | None = None


TransportFactory = Callable[[DeploymentRecord], InferenceTransport]


class InferenceSessionManager:
    """Owns one process-local robot loop per durable provider deployment."""

    def __init__(
        self,
        *,
        deployment_service: DeploymentService,
        driver: YAMDriver,
        camera: MockCamera,
        rig_lease: RigLease,
        recording_manager: RecordingManager,
        transport_factory: TransportFactory,
        session_factory: Callable[[], Session] | None,
        recording_fps: int = 20,
    ) -> None:
        if not 1 <= recording_fps <= 60:
            raise ValueError("recording_fps must be between 1 and 60")
        self.deployment_service = deployment_service
        self.driver = driver
        self.camera = camera
        self.rig_lease = rig_lease
        self.recording_manager = recording_manager
        self.transport_factory = transport_factory
        self._session_factory = session_factory
        self.recording_fps = recording_fps
        self._sessions: dict[uuid.UUID, _LiveSession] = {}
        self._idle_state_cache: dict[
            uuid.UUID, tuple[float, InferenceSessionSnapshot]
        ] = {}
        self._lock = asyncio.Lock()
        self._shutting_down = False

    async def startup(self) -> None:
        """Never auto-resume motion; tear down unattended running providers."""

        async with self._lock:
            self._shutting_down = False
            self._idle_state_cache.clear()
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as db:
                running_ids = [
                    record.id
                    for record in self.deployment_service.list(db)
                    if record.status == "running"
                ]
        except Exception:
            return
        for deployment_id in running_ids:
            try:
                with self._session_factory() as db:
                    await self.deployment_service.stop(db, deployment_id)
            except Exception:
                # The deployment service leaves an explicit retryable failure;
                # startup remains available and never starts an arm.
                continue

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutting_down = True
            deployment_ids = [
                deployment_id
                for deployment_id, runtime in self._sessions.items()
                if runtime.status in {"starting", "running", "stopping"}
            ]
        for deployment_id in deployment_ids:
            try:
                await self._ensure_stopped(deployment_id, InferenceStopOptions())
            except Exception:
                continue
        if self._session_factory is None:
            return
        try:
            owned_active = {
                state.deployment_id
                for state in await asyncio.to_thread(
                    self.deployment_service.target.list_owned
                )
                if not state.stopped_verified
            }
            with self._session_factory() as db:
                unattended = [
                    record.id
                    for record in self.deployment_service.list(db)
                    if record.target_kind == self.deployment_service.target.kind
                    and (
                        record.status in {"running", "stopping"}
                        or (
                            record.status == "failed"
                            and record.id in owned_active
                        )
                    )
                ]
        except Exception:
            return
        for deployment_id in unattended:
            try:
                with self._session_factory() as db:
                    await self.deployment_service.stop(db, deployment_id)
            except Exception:
                continue

    async def active_resource_counts(self) -> tuple[int, int]:
        """Return active owned tasks and queued robot actions for teardown gates."""

        async with self._lock:
            sessions = list(self._sessions.values())
        task_count = 0
        queue_count = 0
        for live in sessions:
            seen: set[int] = set()
            for task in (live.supervisor_task, live.stop_task):
                if task is not None and not task.done() and id(task) not in seen:
                    seen.add(id(task))
                    task_count += 1
            if live.loop is not None:
                loop_tasks, queued = await live.loop.active_resource_counts()
                task_count += loop_tasks
                queue_count += queued
        return task_count, queue_count

    async def start(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        options: InferenceStartOptions,
        *,
        recording_fps: int | None = None,
    ) -> InferenceSessionSnapshot:
        if self._session_factory is None:
            raise InferenceSessionStorageError(
                "PostgreSQL session orchestration is not configured."
            )
        record = self.deployment_service.get(db, deployment_id)
        if record.status != "running":
            raise InferenceSessionConflictError(
                "Only a running deployment can start robot inference."
            )
        if (
            record.checkpoint_revision is None
            or re.fullmatch(r"[0-9a-f]{40}", record.checkpoint_revision) is None
        ):
            raise InferenceSessionConfigurationError(
                "The deployment does not have an immutable model revision."
            )
        try:
            arm = self.driver.get_arm(options.arm_id)
        except ArmNotFoundError:
            raise InferenceSessionConfigurationError(
                "The selected inference arm is unavailable."
            ) from None
        except Exception:
            raise InferenceSessionRuntimeError(
                "The selected inference arm could not be read safely."
            ) from None
        if not arm.connected or arm.role != "follower":
            raise InferenceSessionConfigurationError(
                "Inference requires a connected follower arm."
            )

        async with self._lock:
            if self._shutting_down:
                raise InferenceSessionConflictError(
                    "The inference service is shutting down."
                )
            existing = self._sessions.get(deployment_id)
            if existing is not None and existing.status in {
                "starting",
                "running",
                "stopping",
            }:
                raise InferenceSessionConflictError(
                    "An inference session is already active for this deployment."
                )
            if any(
                item.status in {"starting", "running", "stopping"}
                for key, item in self._sessions.items()
                if key != deployment_id
            ):
                raise InferenceSessionConflictError(
                    "Another inference session already controls the arm rig."
                )
            live = _LiveSession(
                deployment_id=deployment_id,
                session_id=uuid.uuid4(),
                arm_id=options.arm_id,
                status="starting",
                recording_status="starting" if options.record_session else "disabled",
            )
            self._sessions[deployment_id] = live
            self._idle_state_cache.pop(deployment_id, None)

        recording_id: uuid.UUID | None = None
        loop: RobotInferenceLoop | None = None

        async def observe_action(
            observation: ArmTelemetry,
            action: ArmAction,
            applied: ArmTelemetry,
        ) -> None:
            if recording_id is None:
                raise RecordingRuntimeError(
                    "inference recording could not start safely"
                )
            await self.recording_manager.capture_inference_action(
                str(recording_id), observation, action, applied
            )

        async def prepare_recording() -> None:
            if recording_id is None:
                return
            fps = self.recording_fps if recording_fps is None else recording_fps
            await self.recording_manager.start_inference_episode(
                str(recording_id),
                options.arm_id,
                0,
                "draft",
                fps,
                copy.deepcopy(options.recording_metadata or {}),
                inference_owner_id=str(live.session_id),
            )
            live.recording_status = "recording"

        try:
            recording_id = self._persist_session_start(
                db,
                record,
                arm,
                options,
            )
            live.recording_id = recording_id
            live.deployment = self.deployment_service.get(db, deployment_id)
            try:
                transport = self.transport_factory(record)
            except Exception:
                raise InferenceSessionUnavailableError(
                    "The inference transport is not configured safely."
                ) from None
            loop = RobotInferenceLoop(
                driver=self.driver,
                camera=self.camera,
                rig_lease=self.rig_lease,
                transport=transport,
                arm_id=options.arm_id,
                task=options.task,
                expected_runtime=record.runtime,  # type: ignore[arg-type]
                expected_model_repo=record.model_repo,
                expected_revision=record.checkpoint_revision,
                session_id=live.session_id,
                allow_opaque_mock_actions=isinstance(self.driver, MockYAMDriver),
                max_steps=options.max_steps,
                action_observer=observe_action if options.record_session else None,
                start_hook=prepare_recording if options.record_session else None,
            )
            live.loop = loop
            await loop.start()
            live.endpoint_healthy = True
            live.status = "running"
            live.supervisor_task = asyncio.create_task(
                self._supervise(deployment_id),
                name=f"ctrl-pi-inference-supervisor-{deployment_id}",
            )
            return self.state(db, deployment_id)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._cleanup_failed_start(deployment_id, loop, recording_id),
                name=f"ctrl-pi-inference-start-cleanup-{deployment_id}",
            )
            await self._settle_task(cleanup)
            raise
        except (RobotInferenceConflictError, RecordingConflictError) as error:
            await self._cleanup_failed_start(deployment_id, loop, recording_id)
            raise InferenceSessionConflictError(str(error)) from None
        except RobotInferencePreparationError:
            await self._cleanup_failed_start(deployment_id, loop, recording_id)
            raise InferenceSessionRuntimeError(
                "The inference recording could not start safely."
            ) from None
        except RobotInferenceConfigurationError:
            await self._cleanup_failed_start(deployment_id, loop, recording_id)
            raise InferenceSessionRuntimeError(
                "The deployed runtime identity could not be verified safely."
            ) from None
        except InferenceSessionError:
            await self._cleanup_failed_start(deployment_id, loop, recording_id)
            raise
        except (RecordingRuntimeError, OSError):
            await self._cleanup_failed_start(deployment_id, loop, recording_id)
            raise InferenceSessionRuntimeError(
                "The inference recording could not start safely."
            ) from None
        except Exception:
            await self._cleanup_failed_start(deployment_id, loop, recording_id)
            raise InferenceSessionRuntimeError(
                "The robot inference session could not start safely."
            ) from None

    def state(
        self, db: Session, deployment_id: uuid.UUID
    ) -> InferenceSessionSnapshot:
        deployment = self.deployment_service.get(db, deployment_id)
        live = self._sessions.get(deployment_id)
        recording = self._recording_snapshot(db, deployment.recording_id, live)
        if live is None:
            status: SessionStatus
            if deployment.status == "stopped":
                status = "stopped"
            elif deployment.status == "failed":
                status = "failed"
            else:
                status = "idle"
            return InferenceSessionSnapshot(
                deployment=deployment,
                session_status=status,
                endpoint_healthy=deployment.status == "running",
                teardown_verified=deployment.status == "stopped",
                steps_executed=0,
                requests_completed=0,
                dropped_chunks=0,
                queue_depth=0,
                last_latency_ms=None,
                average_latency_ms=None,
                frequency_hz=0.0,
                last_error=None,
                session_started_at=None,
                session_stopped_at=None,
                recording=recording,
            )
        loop_snapshot = live.loop.snapshot() if live.loop is not None else None
        return self._snapshot(deployment, live, loop_snapshot, recording)

    async def read(self, deployment_id: uuid.UUID) -> InferenceSessionSnapshot:
        terminal_supervisor: asyncio.Task[None] | None = None
        async with self._lock:
            live = self._sessions.get(deployment_id)
            if (
                live is not None
                and live.deployment is not None
                and live.status in {"starting", "running", "stopping"}
            ):
                loop_snapshot = live.loop.snapshot() if live.loop is not None else None
                return self._snapshot(
                    live.deployment,
                    live,
                    loop_snapshot,
                    InferenceRecordingSnapshot(
                        enabled=live.recording_status != "disabled",
                        status=live.recording_status,
                        recording_id=live.recording_id,
                        episode_count=0,
                        duration_seconds=0.0,
                        hf_repo_id=None,
                    ),
                )
            if (
                live is not None
                and live.status in {"stopped", "failed"}
                and live.supervisor_task is not None
                and not live.supervisor_task.done()
                and live.supervisor_task is not asyncio.current_task()
            ):
                terminal_supervisor = live.supervisor_task
            cached = self._idle_state_cache.get(deployment_id)
            if live is None and cached is not None and time.monotonic() - cached[0] < 2:
                return cached[1]
        if terminal_supervisor is not None:
            try:
                await asyncio.shield(terminal_supervisor)
            except Exception:
                pass
        if self._session_factory is None:
            raise InferenceSessionStorageError(
                "PostgreSQL session orchestration is not configured."
            )
        try:
            with self._session_factory() as db:
                snapshot = self.state(db, deployment_id)
            if snapshot.session_status == "idle":
                async with self._lock:
                    if self._sessions.get(deployment_id) is None:
                        self._idle_state_cache[deployment_id] = (
                            time.monotonic(),
                            snapshot,
                        )
            return snapshot
        except (DeploymentNotFoundError, DeploymentStorageError):
            raise
        except SQLAlchemyError:
            raise InferenceSessionStorageError(
                "PostgreSQL could not read inference session state."
            ) from None

    async def stop(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        options: InferenceStopOptions | None = None,
    ) -> InferenceSessionSnapshot:
        self.deployment_service.get(db, deployment_id)
        live = self._sessions.get(deployment_id)
        if live is not None and live.status == "starting":
            raise InferenceSessionConflictError(
                "The inference session is still starting; retry stop shortly."
            )
        if live is None:
            await self.deployment_service.stop(db, deployment_id)
            async with self._lock:
                self._idle_state_cache.pop(deployment_id, None)
            return self.state(db, deployment_id)
        if self._session_factory is None:
            live = self._sessions.get(deployment_id)
            if live is not None and live.status in {"starting", "running", "stopping"}:
                raise InferenceSessionStorageError(
                    "PostgreSQL session orchestration is not configured."
                )
            await self.deployment_service.stop(db, deployment_id)
            return self.state(db, deployment_id)
        await self._ensure_stopped(deployment_id, options or InferenceStopOptions())
        db.expire_all()
        return self.state(db, deployment_id)

    async def _ensure_stopped(
        self,
        deployment_id: uuid.UUID,
        options: InferenceStopOptions,
    ) -> None:
        async with self._lock:
            live = self._sessions.get(deployment_id)
            if live is None:
                live = _LiveSession(
                    deployment_id=deployment_id,
                    session_id=uuid.uuid4(),
                    arm_id="",
                    status="stopping",
                )
                self._sessions[deployment_id] = live
            task = live.stop_task
            if task is None or task.done():
                live.status = "stopping"
                task = asyncio.create_task(
                    self._stop_inner(live, options),
                    name=f"ctrl-pi-inference-stop-{deployment_id}",
                )
                live.stop_task = task
        await self._settle_task(task)

    async def _stop_inner(
        self,
        live: _LiveSession,
        options: InferenceStopOptions,
    ) -> None:
        local_error: str | None = (
            live.last_error if live.recording_status == "failed" else None
        )
        provider_error: Exception | None = None
        try:
            if live.loop is not None:
                try:
                    loop_snapshot = await live.loop.stop()
                    if loop_snapshot.last_error is not None:
                        local_error = loop_snapshot.last_error
                except Exception:
                    local_error = "The robot inference loop could not stop cleanly."
            if live.recording_id is not None and live.recording_status in {
                "starting",
                "recording",
            }:
                live.recording_status = "finalizing"
                try:
                    _, result = await self.recording_manager.stop_inference_episode(
                        str(live.recording_id),
                        success=options.recording_success and local_error is None,
                        notes=options.recording_notes,
                    )
                    await self._persist_episode_result(live.recording_id, result)
                    live.recording_status = "ready"
                except Exception:
                    live.recording_status = "failed"
                    local_error = local_error or (
                        "The inference recording could not be finalized safely."
                    )
                    try:
                        await self.recording_manager.episode_persistence_failed(
                            str(live.recording_id)
                        )
                    except Exception:
                        pass
                    self._mark_recording_failed(live.recording_id)
        finally:
            try:
                if self._session_factory is None:
                    raise InferenceSessionStorageError(
                        "PostgreSQL session orchestration is not configured."
                    )
                with self._session_factory() as provider_db:
                    stopped = await self.deployment_service.stop(
                        provider_db, live.deployment_id
                    )
                live.teardown_verified = stopped.status == "stopped"
                live.deployment = stopped
            except Exception as error:
                provider_error = error
                live.teardown_verified = False

        if provider_error is not None:
            live.status = "failed"
            live.endpoint_healthy = False
            live.last_error = "The compute target could not verify complete teardown."
            if isinstance(provider_error, DeploymentStorageError):
                raise InferenceSessionStorageError(
                    "PostgreSQL could not persist inference teardown state."
                ) from None
            raise InferenceSessionRuntimeError(live.last_error) from None
        live.endpoint_healthy = False
        live.status = "failed" if local_error is not None else "stopped"
        live.last_error = local_error

    async def _supervise(self, deployment_id: uuid.UUID) -> None:
        live = self._sessions.get(deployment_id)
        if live is None or live.loop is None:
            return
        try:
            snapshot = await live.loop.wait()
            if live.status == "running":
                await self._ensure_stopped(
                    deployment_id,
                    InferenceStopOptions(
                        recording_success=snapshot.last_error is None,
                    ),
                )
        except Exception:
            live.status = "failed"
            live.endpoint_healthy = False
            live.last_error = live.last_error or (
                "The inference supervisor stopped safely."
            )

    async def _cleanup_failed_start(
        self,
        deployment_id: uuid.UUID,
        loop: RobotInferenceLoop | None,
        recording_id: uuid.UUID | None,
    ) -> None:
        live = self._sessions.get(deployment_id)
        if loop is not None:
            try:
                await loop.shutdown()
            except Exception:
                pass
        if recording_id is not None:
            try:
                await self.recording_manager.abort_inference_episode(str(recording_id))
            except Exception:
                pass
            self._mark_recording_failed(recording_id)
        try:
            if self._session_factory is not None:
                with self._session_factory() as db:
                    await self.deployment_service.stop(db, deployment_id)
        except Exception:
            pass
        if live is not None:
            live.status = "failed"
            live.endpoint_healthy = False
            live.teardown_verified = self._deployment_stopped(deployment_id)
            live.recording_status = "failed" if recording_id else "disabled"
            live.last_error = "The robot inference session could not start safely."

    def _persist_session_start(
        self,
        db: Session,
        record: DeploymentRecord,
        arm: ArmTelemetry,
        options: InferenceStartOptions,
    ) -> uuid.UUID | None:
        try:
            deployment = db.scalar(
                select(Deployment)
                .where(Deployment.id == record.id)
                .with_for_update()
            )
            if deployment is None:
                raise DeploymentNotFoundError("Deployment was not found.")
            if deployment.status != "running":
                raise InferenceSessionConflictError(
                    "Only a running deployment can start robot inference."
                )
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
                db.flush()
            else:
                robot.role = arm.role
                robot.driver = arm.driver
                robot.can_interface = arm.can.interface
                robot.enabled = arm.connected
            recording_id: uuid.UUID | None = None
            if options.record_session:
                recording = Recording(
                    name=options.recording_name
                    or f"Inference — {record.name}"[:160],
                    task=options.task,
                    status="draft",
                    leader_robot_id=robot.id,
                    follower_robot_id=robot.id,
                    episode_count=0,
                    duration_seconds=0.0,
                    recording_metadata={
                        "source": "inference",
                        "deployment_id": str(record.id),
                        "runtime": record.runtime,
                        "model_repo": record.model_repo,
                        "revision": record.checkpoint_revision,
                        **copy.deepcopy(options.recording_metadata or {}),
                        "episodes": [],
                    },
                )
                db.add(recording)
                db.flush()
                recording_id = recording.id
            deployment.robot_id = robot.id
            deployment.record_session = options.record_session
            deployment.recording_id = recording_id
            db.commit()
            return recording_id
        except (InferenceSessionConflictError, DeploymentNotFoundError):
            db.rollback()
            raise
        except SQLAlchemyError:
            db.rollback()
            raise InferenceSessionStorageError(
                "PostgreSQL could not persist inference session state."
            ) from None

    async def _persist_episode_result(
        self, recording_id: uuid.UUID, result: EpisodeResult
    ) -> None:
        if self._session_factory is None:
            raise InferenceSessionStorageError(
                "PostgreSQL session orchestration is not configured."
            )
        with self._session_factory() as db:
            try:
                recording = db.scalar(
                    select(Recording)
                    .where(Recording.id == recording_id)
                    .with_for_update()
                )
                if recording is None:
                    raise InferenceSessionStorageError(
                        "Inference recording metadata is unavailable."
                    )
                stored = recording.recording_metadata.get("episodes", [])
                episodes = list(stored) if isinstance(stored, list) else []
                episodes.append(self._episode_metadata(result))
                recording.recording_metadata = {
                    **recording.recording_metadata,
                    "episodes": episodes,
                }
                recording.episode_count += 1
                recording.duration_seconds += result.duration_seconds
                recording.status = "ready"
                db.commit()
                episode_count = recording.episode_count
            except InferenceSessionStorageError:
                db.rollback()
                raise
            except SQLAlchemyError:
                db.rollback()
                raise InferenceSessionStorageError(
                    "PostgreSQL could not persist inference recording metadata."
                ) from None
        # Release the manager's completed-episode sentinel only after the DB
        # aggregate is durable.
        await self.recording_manager.confirm_episode_persisted(
            str(recording_id), episode_count, "ready"
        )

    def _mark_recording_failed(self, recording_id: uuid.UUID) -> None:
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as db:
                recording = db.get(Recording, recording_id)
                if recording is not None and recording.status != "uploaded":
                    recording.status = "failed"
                    db.commit()
        except SQLAlchemyError:
            return

    def _deployment_stopped(self, deployment_id: uuid.UUID) -> bool:
        if self._session_factory is None:
            return False
        try:
            with self._session_factory() as db:
                return self.deployment_service.get(db, deployment_id).status == "stopped"
        except Exception:
            return False

    def _recording_snapshot(
        self,
        db: Session,
        recording_id: uuid.UUID | None,
        live: _LiveSession | None,
    ) -> InferenceRecordingSnapshot:
        enabled = recording_id is not None or (
            live is not None and live.recording_status != "disabled"
        )
        if recording_id is None:
            return InferenceRecordingSnapshot(
                enabled=enabled,
                status=(live.recording_status if live is not None else "disabled"),
                recording_id=None,
                episode_count=0,
                duration_seconds=0.0,
                hf_repo_id=None,
            )
        try:
            recording = db.get(Recording, recording_id)
        except SQLAlchemyError:
            raise InferenceSessionStorageError(
                "PostgreSQL could not read inference recording state."
            ) from None
        if recording is None:
            return InferenceRecordingSnapshot(
                enabled=True,
                status="failed",
                recording_id=recording_id,
                episode_count=0,
                duration_seconds=0.0,
                hf_repo_id=None,
            )
        status: RecordingSessionStatus
        if live is not None:
            status = live.recording_status
        elif recording.status in {"ready", "uploaded"}:
            status = "ready"
        elif recording.status == "failed":
            status = "failed"
        else:
            status = "failed"
        return InferenceRecordingSnapshot(
            enabled=True,
            status=status,
            recording_id=recording.id,
            episode_count=recording.episode_count,
            duration_seconds=recording.duration_seconds,
            hf_repo_id=recording.hf_repo_id,
        )

    @staticmethod
    def _snapshot(
        deployment: DeploymentRecord,
        live: _LiveSession,
        loop: RobotInferenceSnapshot | None,
        recording: InferenceRecordingSnapshot,
    ) -> InferenceSessionSnapshot:
        return InferenceSessionSnapshot(
            deployment=deployment,
            session_status=live.status,
            endpoint_healthy=live.endpoint_healthy,
            teardown_verified=live.teardown_verified,
            steps_executed=0 if loop is None else loop.steps_executed,
            requests_completed=0 if loop is None else loop.requests_completed,
            dropped_chunks=0 if loop is None else loop.dropped_chunks,
            queue_depth=0 if loop is None else loop.queue_depth,
            last_latency_ms=None if loop is None else loop.last_latency_ms,
            average_latency_ms=None if loop is None else loop.average_latency_ms,
            frequency_hz=0.0 if loop is None else loop.frequency_hz,
            last_error=live.last_error or (None if loop is None else loop.last_error),
            session_started_at=None if loop is None else loop.started_at,
            session_stopped_at=None if loop is None else loop.stopped_at,
            recording=recording,
        )

    @staticmethod
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

    @staticmethod
    async def _settle_task(task: asyncio.Task[None]) -> None:
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                if task.done():
                    raise
                continue
