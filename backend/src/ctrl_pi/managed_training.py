from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeVar

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.managed_training_artifacts import (
    ManagedTrainingArtifactError,
    ManagedTrainingArtifactService,
)
from ctrl_pi.models import ManagedTrainingJob, TrainingRun
from ctrl_pi.training_compute import (
    ManagedTrainingComputeSize,
    ManagedTrainingConfigurationError as TargetConfigurationError,
    ManagedTrainingOwnershipError,
    ManagedTrainingProtocolError,
    ManagedTrainingSpec,
    ManagedTrainingTarget,
    ManagedTrainingTargetError,
    ManagedTrainingTransientError,
    TrainingCheckpointEvent,
    TrainingHandle,
    TrainingLogEvent,
    TrainingMetricEvent,
    TrainingPoll,
    TrainingResult,
    TrainingTargetKind,
    TrainingTargetState,
    training_app_name,
    training_gpu_spec,
    training_ownership_tag,
    validate_training_handle,
    validate_training_state,
)
from ctrl_pi.training_store import (
    TrainingStoreError,
    append_checkpoint,
    append_console_log,
    append_metrics,
)

ManagedTrainingStatus = Literal[
    "created",
    "launching",
    "running",
    "finalizing",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
]
ManagedTrainingOutcome = Literal["pending", "succeeded", "failed", "cancelled"]

_ACTIVE_STATUSES = {
    "created",
    "launching",
    "running",
    "finalizing",
    "cancelling",
}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_SAFE_REVISION = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?"
)
_SHA = re.compile(r"[0-9a-f]{40}")
_T = TypeVar("_T")


class ManagedTrainingError(RuntimeError):
    """A safe public managed-training lifecycle failure."""


class ManagedTrainingNotFoundError(ManagedTrainingError):
    pass


class ManagedTrainingConflictError(ManagedTrainingError):
    pass


class ManagedTrainingStorageError(ManagedTrainingError):
    pass


class ManagedTrainingConfigurationError(ManagedTrainingError):
    pass


@dataclass(frozen=True)
class ManagedTrainingLaunch:
    idempotency_key: uuid.UUID
    name: str
    dataset_repo: str
    dataset_revision: str | None
    base_model: str
    base_model_revision: str | None
    output_model_repo: str
    output_private: bool
    acknowledge_public_model_risk: bool
    acknowledge_compute_cost: bool
    runtime: Literal["lerobot"]
    compute_size: ManagedTrainingComputeSize
    max_steps: int
    batch_size: int
    log_every: int
    save_every: int
    seed: int
    num_workers: int
    timeout_minutes: int


@dataclass(frozen=True)
class ManagedTrainingRecord:
    id: uuid.UUID
    training_run_id: uuid.UUID
    idempotency_key: uuid.UUID
    request_hash: str
    status: ManagedTrainingStatus
    outcome: ManagedTrainingOutcome
    target_kind: TrainingTargetKind
    provider_state: str
    compute_size: str
    runtime: str
    dataset_repo: str
    requested_dataset_revision: str | None
    dataset_revision: str | None
    base_model: str
    requested_base_model_revision: str | None
    base_model_revision: str | None
    output_model_repo: str
    output_private: bool
    output_marker_revision: str | None
    output_revision: str | None
    max_steps: int
    batch_size: int
    log_every: int
    save_every: int
    seed: int
    num_workers: int
    timeout_seconds: int
    deadline_at: datetime
    provider_app_id: str | None
    provider_function_call_id: str | None
    last_event_sequence: int
    event_gap: bool
    launch_attempted_at: datetime | None
    provider_launch_started_at: datetime | None
    started_at: datetime | None
    execution_finished_at: datetime | None
    cancel_requested_at: datetime | None
    teardown_verified_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime

    @property
    def teardown_verified(self) -> bool:
        return self.teardown_verified_at is not None


@dataclass(frozen=True)
class ManagedTrainingPage:
    jobs: list[ManagedTrainingRecord]
    next_cursor: str | None


class ManagedTrainingManager:
    """Durably supervises managed compute without relaunching after crashes."""

    def __init__(
        self,
        target: ManagedTrainingTarget,
        artifact_service: ManagedTrainingArtifactService,
        *,
        session_factory: Callable[[], Session],
        target_factory: Callable[[TrainingTargetKind], ManagedTrainingTarget | None]
        | None = None,
        artifact_service_factory: Callable[
            [TrainingTargetKind], ManagedTrainingArtifactService | None
        ]
        | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        poll_interval_seconds: float = 1.0,
        teardown_timeout_seconds: float = 30.0,
    ) -> None:
        self.target = target
        self.artifact_service = artifact_service
        self.session_factory = session_factory
        self._target_factory = target_factory
        self._artifact_service_factory = artifact_service_factory
        self._now = now
        self._poll_interval = max(0.01, poll_interval_seconds)
        self._teardown_timeout = max(0.1, teardown_timeout_seconds)
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._reconciliation_task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._started = False

    async def startup(self) -> None:
        if self._started:
            return
        self._started = True
        self._shutdown.clear()
        try:
            reconciled = await self._reconcile_once()
        except Exception:
            reconciled = False
        if not reconciled:
            self._reconciliation_task = asyncio.create_task(
                self._retry_startup_reconciliation()
            )

    async def _reconcile_once(self) -> bool:
        jobs = self._active_job_records()
        active_ids = {job.id for job in jobs}
        kinds = {job.target_kind for job in jobs}
        kinds.add(self.target.kind)
        reconciled = True
        for kind in sorted(kinds):
            target = self._target_for(kind)
            if target is None:
                reconciled = False
                continue
            try:
                states = await asyncio.to_thread(target.list_owned)
            except Exception:
                reconciled = False
                for job in jobs:
                    if job.target_kind == kind:
                        self._note_retryable_error(
                            job.id,
                            "Managed training provider reconciliation is temporarily unavailable.",
                        )
                continue
            if self._shutdown.is_set():
                return False
            for state in states:
                try:
                    validate_training_state(state)
                except Exception:
                    reconciled = False
                    continue
                if state.job_id not in active_ids and not state.stopped_verified:
                    self._schedule_orphan_cleanup(target, state)
        if self._shutdown.is_set():
            return False
        for job in jobs:
            self._schedule(job.id, resume=True)
        return reconciled

    async def _retry_startup_reconciliation(self) -> None:
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._poll_interval
                )
            except TimeoutError:
                pass
            if self._shutdown.is_set():
                return
            try:
                if await self._reconcile_once():
                    return
            except Exception:
                continue

    async def shutdown(self) -> None:
        if not self._started:
            return
        self._shutdown.set()
        tasks = list(self._tasks.values())
        if self._reconciliation_task is not None:
            tasks.append(self._reconciliation_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._reconciliation_task = None
        # A user cancel, observed failure, or deadline that raced graceful
        # shutdown still gets one exact stop-and-inspect attempt. Healthy
        # pending/running executions remain remote and are reattached on boot.
        try:
            required_cleanup = self._active_job_records()
        except Exception:
            required_cleanup = []
        for job in required_cleanup:
            if job.outcome != "pending":
                await self._resume_job(job.id)
        self._started = False

    async def create(
        self, db: Session, launch: ManagedTrainingLaunch
    ) -> ManagedTrainingRecord:
        self._validate_launch(launch)
        request_hash = managed_training_request_hash(launch)
        existing = db.scalar(
            select(ManagedTrainingJob).where(
                ManagedTrainingJob.idempotency_key == launch.idempotency_key
            )
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                raise ManagedTrainingConflictError(
                    "The idempotency key was already used with a different request."
                )
            return _record(existing)
        active = db.scalar(
            select(ManagedTrainingJob.id)
            .where(ManagedTrainingJob.status.in_(_ACTIVE_STATUSES))
            .limit(1)
        )
        if active is not None:
            raise ManagedTrainingConflictError(
                "Only one managed training job may be active at a time."
            )

        now = _aware(self._now())
        run = TrainingRun(
            name=launch.name,
            status="created",
            current_step=0,
            dataset_repo=launch.dataset_repo,
            base_model=launch.base_model,
            runtime=launch.runtime,
            framework="lerobot",
            output_model_repo=launch.output_model_repo,
            checkpoint_revision=None,
            config={
                "managed": True,
                "compute_size": launch.compute_size,
                "max_steps": launch.max_steps,
                "batch_size": launch.batch_size,
                "log_every": launch.log_every,
                "save_every": launch.save_every,
                "seed": launch.seed,
                "num_workers": launch.num_workers,
                "timeout_minutes": launch.timeout_minutes,
                "output_private": launch.output_private,
            },
            metrics={},
            checkpoints=[],
            console_logs=[],
        )
        job = ManagedTrainingJob(
            idempotency_key=launch.idempotency_key,
            request_hash=request_hash,
            status="created",
            outcome="pending",
            target_kind=self.target.kind,
            provider_state="pending",
            compute_size=launch.compute_size,
            runtime=launch.runtime,
            requested_dataset_revision=launch.dataset_revision,
            dataset_revision=None,
            requested_base_model_revision=launch.base_model_revision,
            base_model_revision=None,
            dataset_repo=launch.dataset_repo,
            base_model=launch.base_model,
            output_model_repo=launch.output_model_repo,
            output_private=launch.output_private,
            output_marker_revision=None,
            output_revision=None,
            max_steps=launch.max_steps,
            batch_size=launch.batch_size,
            log_every=launch.log_every,
            save_every=launch.save_every,
            seed=launch.seed,
            num_workers=launch.num_workers,
            timeout_seconds=launch.timeout_minutes * 60,
            deadline_at=now + timedelta(minutes=launch.timeout_minutes),
            last_event_sequence=0,
            event_gap=False,
        )
        try:
            db.add(run)
            db.flush()
            job.training_run_id = run.id
            db.add(job)
            db.commit()
            db.refresh(job)
        except IntegrityError:
            db.rollback()
            existing = db.scalar(
                select(ManagedTrainingJob).where(
                    ManagedTrainingJob.idempotency_key == launch.idempotency_key
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise ManagedTrainingConflictError(
                        "The idempotency key was already used with a different request."
                    ) from None
                return _record(existing)
            active = db.scalar(
                select(ManagedTrainingJob.id)
                .where(ManagedTrainingJob.status.in_(_ACTIVE_STATUSES))
                .limit(1)
            )
            if active is not None:
                raise ManagedTrainingConflictError(
                    "Only one managed training job may be active at a time."
                ) from None
            raise ManagedTrainingConflictError(
                "The output model repository is already assigned to another job."
            ) from None
        except SQLAlchemyError:
            db.rollback()
            raise ManagedTrainingStorageError(
                "PostgreSQL could not create managed training lifecycle state."
            ) from None
        self._schedule(job.id, resume=False)
        return _record(job)

    def get(self, db: Session, job_id: uuid.UUID) -> ManagedTrainingRecord:
        job = db.get(ManagedTrainingJob, job_id)
        if job is None:
            raise ManagedTrainingNotFoundError(
                "Managed training job was not found."
            )
        return _record(job)

    def list(
        self,
        db: Session,
        *,
        limit: int,
        before_created_at: datetime | None = None,
        before_id: uuid.UUID | None = None,
    ) -> ManagedTrainingPage:
        if not 1 <= limit <= 100:
            raise ValueError("managed training list limit is invalid")
        statement = select(ManagedTrainingJob)
        if before_created_at is not None and before_id is not None:
            before_created_at = _aware(before_created_at)
            statement = statement.where(
                or_(
                    ManagedTrainingJob.created_at < before_created_at,
                    (
                        (ManagedTrainingJob.created_at == before_created_at)
                        & (ManagedTrainingJob.id < before_id)
                    ),
                )
            )
        rows = list(
            db.scalars(
                statement.order_by(
                    ManagedTrainingJob.created_at.desc(),
                    ManagedTrainingJob.id.desc(),
                ).limit(limit + 1)
            )
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = f"{_aware(last.created_at).isoformat()}|{last.id}"
        return ManagedTrainingPage(
            jobs=[_record(row) for row in rows], next_cursor=next_cursor
        )

    async def cancel(
        self, db: Session, job_id: uuid.UUID
    ) -> ManagedTrainingRecord:
        job = db.scalar(
            select(ManagedTrainingJob)
            .where(ManagedTrainingJob.id == job_id)
            .with_for_update()
        )
        if job is None:
            raise ManagedTrainingNotFoundError(
                "Managed training job was not found."
            )
        if job.status in _TERMINAL_STATUSES:
            return _record(job)
        now = _aware(self._now())
        if job.outcome == "pending":
            job.outcome = "cancelled"
            job.execution_finished_at = now
        job.cancel_requested_at = job.cancel_requested_at or now
        if job.outcome == "cancelled":
            job.status = "cancelling"
            job.provider_state = "stopping"
        self._commit(db)
        db.refresh(job)
        self._schedule(job.id, resume=True)
        return _record(job)

    def _schedule(self, job_id: uuid.UUID, *, resume: bool) -> None:
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            return
        coroutine = self._resume_job(job_id) if resume else self._launch_job(job_id)
        task = asyncio.create_task(coroutine)
        self._tasks[job_id] = task

        def done(completed: asyncio.Task[None]) -> None:
            if self._tasks.get(job_id) is completed:
                self._tasks.pop(job_id, None)
            if not self._shutdown.is_set():
                asyncio.create_task(self._reschedule_if_active(job_id))

        task.add_done_callback(done)

    async def _reschedule_if_active(self, job_id: uuid.UUID) -> None:
        while not self._shutdown.is_set():
            try:
                current = self._load_record(job_id)
            except Exception:
                await asyncio.sleep(self._poll_interval)
                continue
            if current is not None and current.status in _ACTIVE_STATUSES:
                self._schedule(job_id, resume=True)
            return

    def _schedule_orphan_cleanup(
        self, target: ManagedTrainingTarget, state: TrainingTargetState
    ) -> None:
        orphan_key = state.job_id
        current = self._tasks.get(orphan_key)
        if current is not None and not current.done():
            return
        self._tasks[orphan_key] = asyncio.create_task(
            self._cleanup_orphan(target, state)
        )

    async def _launch_job(self, job_id: uuid.UUID) -> None:
        handle: TrainingHandle | None = None
        launch_task: asyncio.Task[TrainingHandle] | None = None
        job: ManagedTrainingRecord | None = None
        target = self.target
        try:
            job = self._claim_launch(job_id)
            if job is None:
                await self._resume_job(job_id)
                return
            if self._shutdown.is_set():
                await self._cancel_unlaunched_job(job_id, target=target)
                return
            prepared = await asyncio.to_thread(
                self.artifact_service.prepare,
                job_id=job.id,
                request_hash=job.request_hash,
                dataset_repo=job.dataset_repo,
                dataset_revision=job.requested_dataset_revision,
                base_model=job.base_model,
                base_model_revision=job.requested_base_model_revision,
                output_model_repo=job.output_model_repo,
                output_private=job.output_private,
            )
            job = self._persist_prepared(job_id, prepared)
            if job.outcome != "pending":
                await self._cleanup_until_stopped(job_id, target=target)
                return
            if self._shutdown.is_set():
                await self._cancel_unlaunched_job(job_id, target=target)
                return
            remaining = math.ceil(
                (_aware(job.deadline_at) - _aware(self._now())).total_seconds()
            )
            if remaining <= 0:
                raise ManagedTrainingConfigurationError(
                    "Managed training preparation exceeded its hard deadline."
                )
            spec = ManagedTrainingSpec(
                job_id=job.id,
                request_hash=job.request_hash,
                app_name=training_app_name(job.id),
                ownership_tag=training_ownership_tag(job.id),
                dataset_repo=job.dataset_repo,
                dataset_revision=_required_sha(job.dataset_revision, "dataset"),
                base_model=job.base_model,
                base_model_revision=_required_sha(job.base_model_revision, "base model"),
                output_model_repo=job.output_model_repo,
                output_marker_revision=_required_sha(
                    job.output_marker_revision, "output marker"
                ),
                output_private=job.output_private,
                runtime="lerobot",
                max_steps=job.max_steps,
                batch_size=job.batch_size,
                log_every=job.log_every,
                save_every=job.save_every,
                seed=job.seed,
                num_workers=job.num_workers,
                compute_size=job.compute_size,
                timeout_seconds=min(remaining, 86_400),
                deadline_at=_aware(job.deadline_at),
            )
            job = self._mark_provider_launch_started(job_id)
            if job.outcome != "pending":
                await self._cleanup_until_stopped(job_id, target=target)
                return
            launch_task = asyncio.create_task(asyncio.to_thread(target.launch, spec))
            returned_handle = await asyncio.shield(launch_task)
            self._validate_handle_for_job(returned_handle, job)
            handle = returned_handle
            job = self._persist_handle(job_id, handle)
            if self._shutdown.is_set() and job.outcome == "pending":
                # Launch crossed the authorization boundary before shutdown and
                # now has durable exact ownership IDs. Preserve the accepted
                # remote job; startup reattaches without relaunching it.
                return
            if job.outcome != "pending":
                await self._cleanup_until_stopped(
                    job_id, target=target, preferred_handle=handle
                )
                return
            await self._monitor(job_id, target, handle)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._settle_interrupted_supervisor(
                    job_id,
                    target=target,
                    job=job,
                    handle=handle,
                    launch_task=launch_task,
                )
            )
            try:
                await self._settle_owned_task(cleanup)
            except Exception:
                if handle is not None:
                    await self._emergency_stop(target, handle)
                else:
                    await self._emergency_discover_stop(target, job_id)
            raise
        except Exception as error:
            await self._fail_and_cleanup(
                job_id,
                target=target,
                preferred_handle=handle,
                message=_safe_failure(error),
            )

    async def _cancel_unlaunched_job(
        self, job_id: uuid.UUID, *, target: ManagedTrainingTarget
    ) -> None:
        try:
            self._choose_outcome(
                job_id,
                "cancelled",
                "Managed training stopped before compute because ctrl-pi shut down.",
            )
        except Exception:
            # No provider mutation has happened on this path. The durable launch
            # claim prevents a later restart from spawning duplicate compute.
            return
        await self._cleanup_until_stopped(job_id, target=target)

    async def _settle_interrupted_supervisor(
        self,
        job_id: uuid.UUID,
        *,
        target: ManagedTrainingTarget,
        job: ManagedTrainingRecord | None,
        handle: TrainingHandle | None,
        launch_task: asyncio.Task[TrainingHandle] | None,
    ) -> None:
        safe_handle = handle
        handle_persisted = handle is not None
        if safe_handle is None and launch_task is not None:
            returned_handle: TrainingHandle | None = None
            try:
                returned_handle = await self._settle_owned_task(launch_task)
            except asyncio.CancelledError:
                returned_handle = None
            except Exception:
                returned_handle = None
            if returned_handle is not None and job is not None:
                try:
                    self._validate_handle_for_job(returned_handle, job)
                except Exception:
                    # Never stop a returned handle whose identity is not the exact
                    # durable job. Deterministic discovery below targets only ours.
                    returned_handle = None
                else:
                    safe_handle = returned_handle
                    try:
                        self._persist_handle(job_id, safe_handle)
                    except Exception:
                        # The validated provider identity is still sufficient for
                        # an emergency exact stop if PostgreSQL is unavailable.
                        pass
                    else:
                        handle_persisted = True
        if self._shutdown.is_set() and safe_handle is not None and handle_persisted:
            try:
                current = self._load_record(job_id)
            except Exception:
                current = None
            if current is not None and current.outcome == "pending":
                # Graceful process shutdown is not user cancellation. A durably
                # owned job remains bounded by its provider/deadline and resumes
                # supervision on the next boot.
                return
        try:
            self._choose_outcome(
                job_id,
                "cancelled",
                "Managed training stopped because its supervisor was interrupted.",
            )
        except Exception:
            if safe_handle is not None:
                await self._emergency_stop(target, safe_handle)
            else:
                await self._emergency_discover_stop(target, job_id)
            return
        await self._cleanup_until_stopped(
            job_id,
            target=target,
            preferred_handle=safe_handle,
        )

    @staticmethod
    async def _settle_owned_task(
        task: asyncio.Task[_T],
    ) -> _T:
        """Resolve a resource-creating task despite cancellation of its caller."""

        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                if task.done():
                    return task.result()
                continue

    async def _resume_job(self, job_id: uuid.UUID) -> None:
        job = self._load_record(job_id)
        if job is None or job.status in _TERMINAL_STATUSES:
            return
        if job.outcome == "pending" and _aware(self._now()) >= _aware(job.deadline_at):
            self._choose_outcome(
                job_id,
                "cancelled",
                "Managed training reached its hard deadline.",
            )
            refreshed = self._load_record(job_id)
            if refreshed is None:
                return
            job = refreshed
        target = self._target_for(job.target_kind)
        if target is None:
            self._note_retryable_error(
                job_id, "The original managed training provider is unavailable."
            )
            return
        if job.status in {"finalizing", "cancelling"} or job.outcome != "pending":
            await self._cleanup_until_stopped(job_id, target=target)
            return
        if job.provider_app_id is not None:
            handle = self._handle_from_record(job)
            try:
                state = await asyncio.to_thread(target.inspect, handle)
                self._validate_state_for_handle(state, handle)
            except (
                ManagedTrainingOwnershipError,
                ManagedTrainingProtocolError,
                TargetConfigurationError,
                ManagedTrainingConfigurationError,
                ValueError,
            ) as error:
                await self._fail_and_cleanup(
                    job_id,
                    target=target,
                    preferred_handle=handle,
                    message=_safe_failure(error),
                )
                return
            except Exception:
                self._note_retryable_error(
                    job_id,
                    "Managed training reattachment is temporarily unavailable.",
                )
                await asyncio.sleep(self._poll_interval)
                return
            self._persist_recovered_handle(job_id, handle, state)
            await self._monitor(job_id, target, handle)
            return
        else:
            try:
                state = await self._discover_state(target, job)
            except Exception:
                self._note_retryable_error(
                    job_id,
                    "Managed training discovery is temporarily unavailable.",
                )
                await asyncio.sleep(self._poll_interval)
                return
        if state is None:
            if (
                job.provider_launch_started_at is not None
                and _aware(self._now()) < _aware(job.deadline_at)
            ):
                self._note_retryable_error(
                    job_id,
                    "Managed training provider launch is still being reconciled.",
                )
                await asyncio.sleep(self._poll_interval)
                return
            await self._fail_and_cleanup(
                job_id,
                target=target,
                message=(
                    "Managed training stopped because its exact provider call could not "
                    "be reattached."
                ),
            )
            return
        if job.provider_launch_started_at is not None:
            await self._fail_and_cleanup(
                job_id,
                target=target,
                preferred_handle=state.handle(request_hash=job.request_hash),
                message=(
                    "Managed training stopped because provider launch completed "
                    "after its durable handle was lost."
                ),
            )
            return
        if state.provider_function_call_id is None:
            await self._fail_and_cleanup(
                job_id,
                target=target,
                preferred_handle=state.handle(request_hash=job.request_hash),
                message=(
                    "Managed training stopped because its exact FunctionCall could not "
                    "be reattached."
                ),
            )
            return
        handle = state.handle(request_hash=job.request_hash)
        self._persist_recovered_handle(job_id, handle, state)
        await self._monitor(job_id, target, handle)

    async def _monitor(
        self,
        job_id: uuid.UUID,
        target: ManagedTrainingTarget,
        handle: TrainingHandle,
    ) -> None:
        while not self._shutdown.is_set():
            job = self._load_record(job_id)
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            if job.outcome != "pending":
                await self._cleanup_until_stopped(
                    job_id, target=target, preferred_handle=handle
                )
                return
            if _aware(self._now()) >= _aware(job.deadline_at):
                self._choose_outcome(
                    job_id,
                    "cancelled",
                    "Managed training reached its hard deadline.",
                )
                await self._cleanup_until_stopped(
                    job_id, target=target, preferred_handle=handle
                )
                return
            try:
                poll = await asyncio.to_thread(
                    target.poll,
                    handle,
                    after_sequence=job.last_event_sequence,
                    limit=200,
                )
                self._validate_poll_identity(poll, handle)
                terminal = await self._ingest_poll(job, poll)
            except (
                ManagedTrainingOwnershipError,
                ManagedTrainingProtocolError,
                TargetConfigurationError,
                ManagedTrainingConfigurationError,
                ValueError,
            ) as error:
                await self._fail_and_cleanup(
                    job_id,
                    target=target,
                    preferred_handle=handle,
                    message=_safe_failure(error),
                )
                return
            except Exception:
                self._note_retryable_error(
                    job_id,
                    "Managed training provider status is temporarily unavailable.",
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=self._poll_interval
                    )
                except TimeoutError:
                    pass
                continue
            if terminal is not None:
                outcome, message = terminal
                self._choose_outcome(job_id, outcome, message)
                await self._cleanup_until_stopped(
                    job_id, target=target, preferred_handle=handle
                )
                return
            if poll.has_more:
                continue
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._poll_interval
                )
            except TimeoutError:
                pass

    async def _ingest_poll(
        self, job: ManagedTrainingRecord, poll: TrainingPoll
    ) -> tuple[Literal["succeeded", "failed", "cancelled"], str | None] | None:
        events = [event for event in poll.events if event.sequence > job.last_event_sequence]
        expected = job.last_event_sequence + 1
        gap = poll.truncated
        for event in events:
            if event.sequence != expected:
                gap = True
            expected = event.sequence + 1
            if event.step is not None and event.step > job.max_steps:
                raise ManagedTrainingProtocolError(
                    "Managed training emitted a step beyond the configured maximum."
                )
            if isinstance(event, TrainingCheckpointEvent):
                if event.repo_id != job.output_model_repo:
                    raise ManagedTrainingOwnershipError(
                        "Managed training emitted a checkpoint for the wrong repository."
                    )
                artifact_service = self._artifact_for(job.target_kind)
                if artifact_service is None:
                    raise ManagedTrainingConfigurationError(
                        "The managed training artifact verifier is unavailable."
                    )
                await asyncio.to_thread(
                    artifact_service.verify_output_revision,
                    job_id=job.id,
                    request_hash=job.request_hash,
                    output_model_repo=job.output_model_repo,
                    output_private=job.output_private,
                    revision=event.revision,
                    # Every persisted managed checkpoint must be loadable by the
                    # existing offline inference runtime, not only the final one.
                    require_deployable_root=True,
                )
        if poll.next_sequence > (events[-1].sequence if events else job.last_event_sequence):
            gap = True
        self._persist_events(job.id, job.last_event_sequence, events, poll, gap)

        state = poll.state.execution_state
        if state == "succeeded":
            if poll.has_more:
                return None
            refreshed = self._load_record(job.id)
            if refreshed is None:
                raise ManagedTrainingNotFoundError("Managed training job was not found.")
            result = poll.result
            if result is None:
                return "failed", "Managed training returned no final model result."
            await self._verify_and_persist_result(refreshed, result)
            return "succeeded", None
        if state == "failed":
            return "failed", "Managed training failed in the compute provider."
        if state == "cancelled":
            return "cancelled", None
        if state == "unknown":
            return None
        if state not in {"pending", "running"}:
            return "failed", "Managed training provider state is unavailable."
        return None

    async def _verify_and_persist_result(
        self, job: ManagedTrainingRecord, result: TrainingResult
    ) -> None:
        if (
            result.job_id != job.id
            or result.request_hash != job.request_hash
            or result.output_model_repo != job.output_model_repo
            or result.step != job.max_steps
        ):
            raise ManagedTrainingProtocolError(
                "Managed training returned an invalid final result identity."
            )
        artifact_service = self._artifact_for(job.target_kind)
        if artifact_service is None:
            raise ManagedTrainingConfigurationError(
                "The managed training artifact verifier is unavailable."
            )
        revision = await asyncio.to_thread(
            artifact_service.verify_output_revision,
            job_id=job.id,
            request_hash=job.request_hash,
            output_model_repo=job.output_model_repo,
            output_private=job.output_private,
            revision=result.revision,
            require_deployable_root=True,
        )
        with self.session_factory() as db:
            row = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job.id)
                .with_for_update()
            )
            if row is None:
                raise ManagedTrainingNotFoundError(
                    "Managed training job was not found."
                )
            run = db.scalar(
                select(TrainingRun)
                .where(TrainingRun.id == row.training_run_id)
                .with_for_update()
            )
            if run is None:
                raise ManagedTrainingStorageError(
                    "Managed training run state is unavailable."
                )
            append_checkpoint(
                run,
                repo_id=row.output_model_repo,
                revision=revision,
                step=result.step,
            )
            row.output_revision = revision
            self._commit(db)

    async def _cleanup_until_stopped(
        self,
        job_id: uuid.UUID,
        *,
        target: ManagedTrainingTarget,
        preferred_handle: TrainingHandle | None = None,
    ) -> None:
        handle = preferred_handle
        attempted_cleanup = False
        while True:
            job = self._load_record(job_id)
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            if handle is None:
                try:
                    if job.provider_app_id is not None:
                        handle = self._handle_from_record(job)
                    else:
                        state = await self._discover_state(target, job)
                        if state is None:
                            if (
                                job.provider_launch_started_at is not None
                                and _aware(self._now()) < _aware(job.deadline_at)
                            ):
                                self._set_cleanup_pending_error(
                                    job_id,
                                    "Managed training provider launch settlement is still pending.",
                                )
                                if self._shutdown.is_set():
                                    return
                                try:
                                    await asyncio.wait_for(
                                        self._shutdown.wait(),
                                        timeout=self._poll_interval,
                                    )
                                except TimeoutError:
                                    pass
                                continue
                            self._mark_terminal(job_id, provider_state="stopped")
                            return
                        handle = state.handle(request_hash=job.request_hash)
                        self._persist_recovered_handle(job_id, handle, state)
                except Exception:
                    self._set_cleanup_pending_error(
                        job_id, "Managed training teardown is not yet verified."
                    )
                    if self._shutdown.is_set() and attempted_cleanup:
                        return
                    attempted_cleanup = True
                    try:
                        await asyncio.wait_for(
                            self._shutdown.wait(), timeout=self._poll_interval
                        )
                    except TimeoutError:
                        pass
                    continue
            try:
                attempted_cleanup = True
                if job.outcome == "cancelled":
                    try:
                        await asyncio.to_thread(target.cancel, handle)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Cancellation is advisory. Always attempt the exact owned App
                        # stop as the independent teardown backstop.
                        pass
                try:
                    await asyncio.to_thread(target.stop, handle)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The provider operation may have succeeded despite a transport
                    # failure. Authoritative inspection below decides whether the
                    # resource is actually stopped.
                    pass
                deadline = asyncio.get_running_loop().time() + self._teardown_timeout
                while True:
                    state = await asyncio.to_thread(target.inspect, handle)
                    self._validate_state_for_handle(state, handle)
                    if state.stopped_verified:
                        self._mark_terminal(job_id, provider_state="stopped")
                        return
                    if asyncio.get_running_loop().time() >= deadline:
                        raise ManagedTrainingError(
                            "Managed training teardown could not be verified."
                        )
                    await asyncio.sleep(min(self._poll_interval, 0.25))
            except asyncio.CancelledError:
                raise
            except Exception:
                self._set_cleanup_pending_error(
                    job_id, "Managed training teardown is not yet verified."
                )
                if self._shutdown.is_set():
                    return
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=self._poll_interval
                    )
                except TimeoutError:
                    pass

    async def _fail_and_cleanup(
        self,
        job_id: uuid.UUID,
        *,
        target: ManagedTrainingTarget,
        message: str,
        preferred_handle: TrainingHandle | None = None,
    ) -> None:
        persistence_failed = False
        try:
            self._choose_outcome(job_id, "failed", message)
        except Exception:
            persistence_failed = True
        finally:
            if persistence_failed:
                if preferred_handle is not None:
                    await self._emergency_stop(target, preferred_handle)
                else:
                    await self._emergency_discover_stop(target, job_id)
            else:
                await self._cleanup_until_stopped(
                    job_id, target=target, preferred_handle=preferred_handle
                )

    async def _discover_state(
        self, target: ManagedTrainingTarget, job: ManagedTrainingRecord
    ) -> TrainingTargetState | None:
        matches: list[TrainingTargetState] = []
        for attempt in range(2):
            states = await asyncio.to_thread(target.list_owned)
            matches = []
            for state in states:
                validate_training_state(state)
                if state.job_id == job.id:
                    if (
                        state.app_name != training_app_name(job.id)
                        or state.ownership_tag != training_ownership_tag(job.id)
                    ):
                        raise ManagedTrainingOwnershipError(
                            "Managed training provider ownership could not be verified."
                        )
                    matches.append(state)
            if len(matches) > 1:
                raise ManagedTrainingOwnershipError(
                    "Multiple managed training resources claimed the same job."
                )
            if matches:
                break
            if attempt == 0:
                await asyncio.sleep(self._poll_interval)
        if not matches:
            return None
        state = matches[0]
        if job.provider_app_id is not None and state.provider_app_id != job.provider_app_id:
            raise ManagedTrainingOwnershipError(
                "Managed training provider App identity changed."
            )
        if (
            job.provider_function_call_id is not None
            and state.provider_function_call_id is not None
            and state.provider_function_call_id != job.provider_function_call_id
        ):
            raise ManagedTrainingOwnershipError(
                "Managed training FunctionCall identity changed."
            )
        return state

    async def _cleanup_orphan(
        self, target: ManagedTrainingTarget, state: TrainingTargetState
    ) -> None:
        handle = state.handle()
        attempted = False
        while True:
            try:
                attempted = True
                await asyncio.to_thread(target.stop, handle)
                verified = await asyncio.to_thread(target.inspect, handle)
                self._validate_state_for_handle(verified, handle)
                if verified.stopped_verified:
                    return
            except Exception:
                pass
            if self._shutdown.is_set() and attempted:
                return
            await asyncio.sleep(self._poll_interval)

    async def _emergency_stop(
        self, target: ManagedTrainingTarget, handle: TrainingHandle
    ) -> None:
        try:
            await asyncio.to_thread(target.stop, handle)
            state = await asyncio.to_thread(target.inspect, handle)
            self._validate_state_for_handle(state, handle)
        except Exception:
            return

    async def _emergency_discover_stop(
        self, target: ManagedTrainingTarget, job_id: uuid.UUID
    ) -> None:
        try:
            states = await asyncio.to_thread(target.list_owned)
            matches = [state for state in states if state.job_id == job_id]
            if len(matches) != 1:
                return
            state = matches[0]
            validate_training_state(state)
            if (
                state.app_name != training_app_name(job_id)
                or state.ownership_tag != training_ownership_tag(job_id)
            ):
                return
            await self._emergency_stop(target, state.handle())
        except Exception:
            return

    def _claim_launch(self, job_id: uuid.UUID) -> ManagedTrainingRecord | None:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None or job.status != "created" or job.launch_attempted_at is not None:
                return None
            now = _aware(self._now())
            job.status = "launching"
            job.launch_attempted_at = now
            run = db.get(TrainingRun, job.training_run_id)
            if run is None:
                raise ManagedTrainingStorageError(
                    "Managed training run state is unavailable."
                )
            run.status = "running"
            self._commit(db)
            db.refresh(job)
            return _record(job)

    def _persist_prepared(self, job_id: uuid.UUID, prepared: object) -> ManagedTrainingRecord:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None:
                raise ManagedTrainingNotFoundError("Managed training job was not found.")
            for field in (
                "dataset_repo",
                "dataset_revision",
                "base_model",
                "base_model_revision",
                "output_model_repo",
                "output_marker_revision",
                "output_private",
            ):
                setattr(job, field, getattr(prepared, field))
            self._commit(db)
            db.refresh(job)
            return _record(job)

    def _persist_handle(
        self, job_id: uuid.UUID, handle: TrainingHandle
    ) -> ManagedTrainingRecord:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None:
                raise ManagedTrainingNotFoundError("Managed training job was not found.")
            job.provider_app_id = handle.provider_app_id
            job.provider_function_call_id = handle.provider_function_call_id
            if job.outcome == "pending":
                job.status = "running"
                job.provider_state = "running"
                job.started_at = job.started_at or _aware(self._now())
            self._commit(db)
            db.refresh(job)
            return _record(job)

    def _mark_provider_launch_started(
        self, job_id: uuid.UUID
    ) -> ManagedTrainingRecord:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None:
                raise ManagedTrainingNotFoundError(
                    "Managed training job was not found."
                )
            if job.provider_launch_started_at is None:
                job.provider_launch_started_at = _aware(self._now())
            self._commit(db)
            db.refresh(job)
            return _record(job)

    def _persist_recovered_handle(
        self,
        job_id: uuid.UUID,
        handle: TrainingHandle,
        state: TrainingTargetState,
    ) -> None:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None:
                return
            if (
                job.provider_app_id is not None
                and job.provider_app_id != handle.provider_app_id
            ):
                raise ManagedTrainingConflictError(
                    "Managed training provider App identity changed."
                )
            if (
                job.provider_function_call_id is not None
                and handle.provider_function_call_id is not None
                and job.provider_function_call_id != handle.provider_function_call_id
            ):
                raise ManagedTrainingConflictError(
                    "Managed training FunctionCall identity changed."
                )
            job.provider_app_id = job.provider_app_id or handle.provider_app_id
            job.provider_function_call_id = (
                job.provider_function_call_id or handle.provider_function_call_id
            )
            job.provider_state = state.execution_state
            if job.status in {"created", "launching"} and job.outcome == "pending":
                job.status = "running"
                job.started_at = job.started_at or _aware(self._now())
            self._commit(db)

    def _persist_events(
        self,
        job_id: uuid.UUID,
        prior_sequence: int,
        events: list[object],
        poll: TrainingPoll,
        gap: bool,
    ) -> None:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None:
                raise ManagedTrainingNotFoundError("Managed training job was not found.")
            if job.last_event_sequence != prior_sequence:
                raise ManagedTrainingConflictError(
                    "Managed training event ingestion raced another supervisor."
                )
            run = db.scalar(
                select(TrainingRun)
                .where(TrainingRun.id == job.training_run_id)
                .with_for_update()
            )
            if run is None:
                raise ManagedTrainingStorageError(
                    "Managed training run state is unavailable."
                )
            for event in events:
                if isinstance(event, TrainingLogEvent):
                    append_console_log(
                        run,
                        source=event.source,
                        line=event.line,
                        step=event.step,
                    )
                elif isinstance(event, TrainingMetricEvent):
                    append_metrics(run, step=event.step, metrics=event.metrics)
                elif isinstance(event, TrainingCheckpointEvent):
                    append_checkpoint(
                        run,
                        repo_id=event.repo_id,
                        revision=event.revision,
                        step=event.step,
                    )
                else:
                    raise ManagedTrainingProtocolError(
                        "Managed training emitted an unsupported event."
                    )
            job.last_event_sequence = max(
                prior_sequence,
                poll.next_sequence,
                events[-1].sequence if events else prior_sequence,
            )
            job.event_gap = job.event_gap or gap
            job.provider_state = poll.state.execution_state
            if job.outcome == "pending":
                job.last_error = None
            self._commit(db)

    def _choose_outcome(
        self,
        job_id: uuid.UUID,
        outcome: Literal["succeeded", "failed", "cancelled"],
        message: str | None,
    ) -> None:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            if job.outcome == "pending":
                job.outcome = outcome
                job.execution_finished_at = _aware(self._now())
                job.last_error = _bounded_error(message) if message else None
            if job.outcome == "cancelled":
                job.status = "cancelling"
                job.provider_state = "stopping"
            else:
                job.status = "finalizing"
            self._commit(db)

    def _mark_terminal(self, job_id: uuid.UUID, *, provider_state: str) -> None:
        with self.session_factory() as db:
            job = db.scalar(
                select(ManagedTrainingJob)
                .where(ManagedTrainingJob.id == job_id)
                .with_for_update()
            )
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            if job.outcome == "pending":
                job.outcome = "failed"
                job.last_error = "Managed training ended without an execution result."
                job.execution_finished_at = _aware(self._now())
            status_by_outcome = {
                "succeeded": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }
            job.status = status_by_outcome[job.outcome]
            job.provider_state = provider_state
            job.teardown_verified_at = _aware(self._now())
            run = db.get(TrainingRun, job.training_run_id)
            if run is None:
                raise ManagedTrainingStorageError(
                    "Managed training run state is unavailable."
                )
            run.status = job.status
            self._commit(db)

    def _note_retryable_error(self, job_id: uuid.UUID, message: str) -> None:
        with self.session_factory() as db:
            job = db.get(ManagedTrainingJob, job_id)
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            job.last_error = _bounded_error(message)
            self._commit(db)

    def _set_cleanup_pending_error(self, job_id: uuid.UUID, message: str) -> None:
        with self.session_factory() as db:
            job = db.get(ManagedTrainingJob, job_id)
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            job.last_error = _bounded_error(message)
            if job.outcome == "pending":
                job.outcome = "failed"
                job.execution_finished_at = _aware(self._now())
            job.status = "cancelling" if job.outcome == "cancelled" else "finalizing"
            self._commit(db)

    def _active_job_records(self) -> list[ManagedTrainingRecord]:
        with self.session_factory() as db:
            try:
                with db.no_autoflush:
                    rows = db.scalars(
                        select(ManagedTrainingJob).where(
                            ManagedTrainingJob.status.in_(_ACTIVE_STATUSES)
                        )
                    ).all()
            except SQLAlchemyError:
                db.rollback()
                raise ManagedTrainingStorageError(
                    "PostgreSQL could not read managed training lifecycle state."
                ) from None
            return [_record(row) for row in rows]

    def _load_record(self, job_id: uuid.UUID) -> ManagedTrainingRecord | None:
        with self.session_factory() as db:
            job = db.get(ManagedTrainingJob, job_id)
            return None if job is None else _record(job)

    def _target_for(self, kind: TrainingTargetKind) -> ManagedTrainingTarget | None:
        if kind == self.target.kind:
            return self.target
        return None if self._target_factory is None else self._target_factory(kind)

    def _artifact_for(
        self, kind: TrainingTargetKind
    ) -> ManagedTrainingArtifactService | None:
        if kind == self.target.kind:
            return self.artifact_service
        return (
            None
            if self._artifact_service_factory is None
            else self._artifact_service_factory(kind)
        )

    @staticmethod
    def _handle_from_record(job: ManagedTrainingRecord) -> TrainingHandle:
        if job.provider_app_id is None:
            raise ManagedTrainingStorageError(
                "Managed training provider App identity is unavailable."
            )
        return TrainingHandle(
            job_id=job.id,
            provider_app_id=job.provider_app_id,
            provider_function_call_id=job.provider_function_call_id,
            app_name=training_app_name(job.id),
            ownership_tag=training_ownership_tag(job.id),
            request_hash=job.request_hash,
        )

    @staticmethod
    def _validate_handle_for_job(
        handle: TrainingHandle, job: ManagedTrainingRecord
    ) -> None:
        validate_training_handle(handle)
        if (
            handle.job_id != job.id
            or handle.app_name != training_app_name(job.id)
            or handle.ownership_tag != training_ownership_tag(job.id)
            or handle.request_hash != job.request_hash
        ):
            raise ManagedTrainingOwnershipError(
                "Managed training provider returned the wrong resource identity."
            )

    @staticmethod
    def _validate_state_for_handle(
        state: TrainingTargetState, handle: TrainingHandle
    ) -> None:
        validate_training_state(state)
        if (
            state.job_id != handle.job_id
            or state.provider_app_id != handle.provider_app_id
            or (
                handle.provider_function_call_id is not None
                and state.provider_function_call_id
                != handle.provider_function_call_id
            )
            or state.app_name != handle.app_name
            or state.ownership_tag != handle.ownership_tag
        ):
            raise ManagedTrainingOwnershipError(
                "Managed training provider state identity changed."
            )

    @classmethod
    def _validate_poll_identity(
        cls, poll: TrainingPoll, handle: TrainingHandle
    ) -> None:
        cls._validate_state_for_handle(poll.state, handle)
        if poll.result is not None and (
            poll.result.job_id != handle.job_id
            or poll.result.request_hash != handle.request_hash
        ):
            raise ManagedTrainingProtocolError(
                "Managed training result identity changed."
            )

    @staticmethod
    def _validate_launch(launch: ManagedTrainingLaunch) -> None:
        if not launch.name.strip() or launch.name.strip() != launch.name or len(launch.name) > 160:
            raise ValueError("managed training name is invalid")
        for repo_id in (
            launch.dataset_repo,
            launch.base_model,
            launch.output_model_repo,
        ):
            if not 3 <= len(repo_id) <= 255 or repo_id.count("/") != 1:
                raise ValueError("managed training repository ID is invalid")
            try:
                from huggingface_hub.utils import HFValidationError, validate_repo_id

                validate_repo_id(repo_id)
            except HFValidationError:
                raise ValueError("managed training repository ID is invalid") from None
        for revision in (launch.dataset_revision, launch.base_model_revision):
            if revision is not None and (
                _SAFE_REVISION.fullmatch(revision) is None
                or ".." in revision
                or "//" in revision
                or "--" in revision
            ):
                raise ValueError("managed training revision is invalid")
        if not launch.acknowledge_compute_cost:
            raise ValueError("managed training compute cost must be acknowledged")
        if not launch.output_private and not launch.acknowledge_public_model_risk:
            raise ValueError("public model risk must be acknowledged")
        if launch.runtime != "lerobot":
            raise ValueError("managed training runtime is unsupported")
        training_gpu_spec(launch.compute_size)
        if not 1 <= launch.max_steps <= 2_147_483_647:
            raise ValueError("max_steps is invalid")
        if not 1 <= launch.batch_size <= 4_096:
            raise ValueError("batch_size is invalid")
        if (
            not 1 <= launch.log_every <= launch.max_steps
            or math.ceil(launch.max_steps / launch.log_every) * 64 > 10_000
        ):
            raise ValueError("log_every exceeds managed metric bounds")
        if (
            not 1 <= launch.save_every <= launch.max_steps
            or math.ceil(launch.max_steps / launch.save_every) > 512
        ):
            raise ValueError("save_every exceeds managed checkpoint bounds")
        if not 0 <= launch.seed <= 2_147_483_647:
            raise ValueError("seed is invalid")
        if not 0 <= launch.num_workers <= 64:
            raise ValueError("num_workers is invalid")
        if not 1 <= launch.timeout_minutes <= 1_440:
            raise ValueError("timeout_minutes is invalid")

    @staticmethod
    def _commit(db: Session) -> None:
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise ManagedTrainingStorageError(
                "PostgreSQL could not persist managed training lifecycle state."
            ) from None


def managed_training_request_hash(launch: ManagedTrainingLaunch) -> str:
    payload = asdict(launch)
    payload.pop("idempotency_key")
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_managed_training_cursor(value: str) -> tuple[datetime, uuid.UUID]:
    if not value or len(value) > 128 or "|" not in value:
        raise ValueError("managed training cursor is invalid")
    timestamp_raw, id_raw = value.rsplit("|", 1)
    try:
        timestamp = datetime.fromisoformat(timestamp_raw)
        identifier = uuid.UUID(id_raw)
    except ValueError:
        raise ValueError("managed training cursor is invalid") from None
    return _aware(timestamp), identifier


def _record(job: ManagedTrainingJob) -> ManagedTrainingRecord:
    return ManagedTrainingRecord(
        id=job.id,
        training_run_id=job.training_run_id,
        idempotency_key=job.idempotency_key,
        request_hash=job.request_hash,
        status=job.status,
        outcome=job.outcome,
        target_kind=job.target_kind,
        provider_state=job.provider_state,
        compute_size=job.compute_size,
        runtime=job.runtime,
        dataset_repo=job.dataset_repo,
        requested_dataset_revision=job.requested_dataset_revision,
        dataset_revision=job.dataset_revision,
        base_model=job.base_model,
        requested_base_model_revision=job.requested_base_model_revision,
        base_model_revision=job.base_model_revision,
        output_model_repo=job.output_model_repo,
        output_private=job.output_private,
        output_marker_revision=job.output_marker_revision,
        output_revision=job.output_revision,
        max_steps=job.max_steps,
        batch_size=job.batch_size,
        log_every=job.log_every,
        save_every=job.save_every,
        seed=job.seed,
        num_workers=job.num_workers,
        timeout_seconds=job.timeout_seconds,
        deadline_at=_aware(job.deadline_at),
        provider_app_id=job.provider_app_id,
        provider_function_call_id=job.provider_function_call_id,
        last_event_sequence=job.last_event_sequence,
        event_gap=job.event_gap,
        launch_attempted_at=_optional_aware(job.launch_attempted_at),
        provider_launch_started_at=_optional_aware(job.provider_launch_started_at),
        started_at=_optional_aware(job.started_at),
        execution_finished_at=_optional_aware(job.execution_finished_at),
        cancel_requested_at=_optional_aware(job.cancel_requested_at),
        teardown_verified_at=_optional_aware(job.teardown_verified_at),
        last_error=job.last_error,
        created_at=_aware(job.created_at),
        updated_at=_aware(job.updated_at),
    )


def _required_sha(value: str | None, label: str) -> str:
    if value is None or _SHA.fullmatch(value) is None:
        raise ManagedTrainingStorageError(
            f"Managed training {label} identity is unavailable."
        )
    return value


def _safe_failure(error: Exception) -> str:
    if isinstance(error, ManagedTrainingArtifactError):
        return _bounded_error(str(error))
    if isinstance(error, ManagedTrainingConfigurationError):
        return _bounded_error(str(error))
    if isinstance(error, TrainingStoreError):
        return _bounded_error(str(error))
    if isinstance(error, ManagedTrainingTargetError):
        return "Managed training provider operation failed."
    if isinstance(error, ManagedTrainingError):
        return _bounded_error(str(error))
    return "Managed training failed safely."


def _bounded_error(value: str | None) -> str:
    normalized = " ".join((value or "Managed training failed safely.").split())
    return normalized[:240]


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _optional_aware(value: datetime | None) -> datetime | None:
    return None if value is None else _aware(value)


__all__ = [
    "ManagedTrainingConfigurationError",
    "ManagedTrainingConflictError",
    "ManagedTrainingError",
    "ManagedTrainingLaunch",
    "ManagedTrainingManager",
    "ManagedTrainingNotFoundError",
    "ManagedTrainingOutcome",
    "ManagedTrainingPage",
    "ManagedTrainingRecord",
    "ManagedTrainingStatus",
    "ManagedTrainingStorageError",
    "managed_training_request_hash",
    "parse_managed_training_cursor",
]
