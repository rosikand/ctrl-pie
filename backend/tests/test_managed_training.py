from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from ctrl_pi.db import Base, get_db
from ctrl_pi.main import create_app
from ctrl_pi.managed_training import (
    ManagedTrainingConflictError,
    ManagedTrainingLaunch,
    ManagedTrainingManager,
)
from ctrl_pi.managed_training_artifacts import (
    ManagedTrainingArtifactVerificationError,
    StubManagedTrainingArtifactService,
)
from ctrl_pi.models import ManagedTrainingJob, TrainingRun
from ctrl_pi.training_compute import (
    ManagedTrainingSpec,
    ManagedTrainingTargetError,
    TrainingHandle,
    TrainingLogEvent,
    TrainingMetricEvent,
    TrainingPoll,
    TrainingTargetState,
    training_app_name,
    training_ownership_tag,
)
from ctrl_pi.training_compute_stub import StubManagedTrainingTarget


def _factory():
    database_name = f"ctrl-pi-managed-training-{uuid.uuid4().hex}"
    engine = create_engine(
        (
            f"sqlite+pysqlite:///file:{database_name}"
            "?mode=memory&cache=shared&uri=true"
        ),
        connect_args={"check_same_thread": False, "timeout": 5.0},
    )

    @event.listens_for(engine, "connect")
    def configure_test_connection(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout = 5000")
        # Shared-cache SQLite otherwise rejects a concurrent status read while
        # the worker-thread event transaction is committing. PostgreSQL uses
        # MVCC for this production path, so let the test observer avoid taking
        # a SQLite table read lock instead of sharing one unsafe connection.
        cursor.execute("PRAGMA read_uncommitted = 1")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _launch(**overrides: Any) -> ManagedTrainingLaunch:
    values: dict[str, Any] = {
        "idempotency_key": uuid.uuid4(),
        "name": "Managed ACT training",
        "dataset_repo": "acme/dataset",
        "dataset_revision": None,
        "base_model": "acme/base-model",
        "base_model_revision": None,
        "output_model_repo": "acme/output-model",
        "output_private": True,
        "acknowledge_public_model_risk": False,
        "acknowledge_compute_cost": True,
        "runtime": "lerobot",
        "compute_size": "Modal: A10G",
        "max_steps": 100,
        "batch_size": 8,
        "log_every": 1,
        "save_every": 10,
        "seed": 42,
        "num_workers": 0,
        "timeout_minutes": 1,
    }
    values.update(overrides)
    return ManagedTrainingLaunch(**values)


async def _wait_status(factory, job_id: uuid.UUID, statuses: set[str]):
    for _ in range(300):
        try:
            with factory() as db:
                job = db.get(ManagedTrainingJob, job_id)
                if job is not None and job.status in statuses:
                    return job
        except OperationalError as error:
            if "locked" not in str(error).casefold():
                raise
        await asyncio.sleep(0.01)
    raise AssertionError(f"managed training did not reach {statuses}")


class HoldingTarget:
    kind = "stub"

    def __init__(self) -> None:
        self.states: dict[uuid.UUID, TrainingTargetState] = {}
        self.request_hashes: dict[uuid.UUID, str] = {}
        self.stop_handles: list[TrainingHandle] = []
        self.inspect_handles: list[TrainingHandle] = []
        self.list_app_only = False
        self.stop_failures = 0
        self.inspect_failures = 0
        self.list_calls = 0

    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle:
        handle = TrainingHandle(
            job_id=spec.job_id,
            provider_app_id=f"hold-{spec.job_id.hex}",
            provider_function_call_id=f"call-{spec.job_id.hex}",
            app_name=spec.app_name,
            ownership_tag=spec.ownership_tag,
            request_hash=spec.request_hash,
        )
        self.request_hashes[spec.job_id] = spec.request_hash
        self.states[spec.job_id] = self._state(handle, "running", "running", 1)
        return handle

    def poll(
        self, handle: TrainingHandle, *, after_sequence: int, limit: int
    ) -> TrainingPoll:
        return TrainingPoll(
            state=self.inspect(handle),
            events=(),
            next_sequence=after_sequence,
            truncated=False,
            has_more=False,
        )

    def inspect(self, handle: TrainingHandle) -> TrainingTargetState:
        self.inspect_handles.append(handle)
        if self.inspect_failures:
            self.inspect_failures -= 1
            raise ManagedTrainingTargetError("temporary inspect outage")
        state = self.states.get(handle.job_id)
        if state is None:
            return self._state(handle, "unknown", "unknown", 0, exists=False)
        if (
            state.provider_app_id != handle.provider_app_id
            or (
                handle.provider_function_call_id is not None
                and state.provider_function_call_id
                != handle.provider_function_call_id
            )
        ):
            raise ManagedTrainingTargetError("identity mismatch")
        return state

    def cancel(self, handle: TrainingHandle) -> None:
        state = self.inspect(handle)
        self.states[handle.job_id] = replace(
            state,
            execution_state="cancelled",
            resource_lifecycle="stopping",
            running_tasks=0,
        )

    def stop(self, handle: TrainingHandle) -> None:
        self.stop_handles.append(handle)
        if self.stop_failures:
            self.stop_failures -= 1
            raise ManagedTrainingTargetError("temporary stop failure")
        state = self.inspect(handle)
        execution = (
            state.execution_state
            if state.execution_state not in {"pending", "running"}
            else "cancelled"
        )
        self.states[handle.job_id] = replace(
            state,
            execution_state=execution,
            resource_lifecycle="stopped",
            running_tasks=0,
        )

    def list_owned(self) -> list[TrainingTargetState]:
        self.list_calls += 1
        states = list(self.states.values())
        if self.list_app_only:
            return [replace(state, provider_function_call_id=None) for state in states]
        return states

    @staticmethod
    def _state(
        handle: TrainingHandle,
        resource_lifecycle: str,
        execution_state: str,
        running_tasks: int,
        *,
        exists: bool = True,
    ) -> TrainingTargetState:
        return TrainingTargetState(
            job_id=handle.job_id,
            provider_app_id=handle.provider_app_id,
            provider_function_call_id=handle.provider_function_call_id,
            app_name=handle.app_name,
            ownership_tag=handle.ownership_tag,
            exists=exists,
            resource_lifecycle=resource_lifecycle,
            execution_state=execution_state,
            running_tasks=running_tasks,
        )


class GapTarget(StubManagedTrainingTarget):
    def poll(self, *args: Any, **kwargs: Any) -> TrainingPoll:
        return replace(super().poll(*args, **kwargs), truncated=True)


class UnknownOnceTarget(StubManagedTrainingTarget):
    def __init__(self) -> None:
        super().__init__()
        self.unknown_returned = False

    def poll(
        self, handle: TrainingHandle, *, after_sequence: int, limit: int
    ) -> TrainingPoll:
        if not self.unknown_returned:
            self.unknown_returned = True
            return TrainingPoll(
                state=replace(self.inspect(handle), execution_state="unknown"),
                events=(),
                next_sequence=after_sequence,
                truncated=False,
                has_more=False,
            )
        return super().poll(handle, after_sequence=after_sequence, limit=limit)


class PartialResultTarget(StubManagedTrainingTarget):
    def poll(self, *args: Any, **kwargs: Any) -> TrainingPoll:
        poll = super().poll(*args, **kwargs)
        assert poll.result is not None
        return replace(poll, result=replace(poll.result, step=poll.result.step - 1))


class WrongHandleTarget(HoldingTarget):
    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle:
        super().launch(spec)
        foreign_id = uuid.uuid4()
        return TrainingHandle(
            job_id=foreign_id,
            provider_app_id=f"hold-{foreign_id.hex}",
            provider_function_call_id=f"call-{foreign_id.hex}",
            app_name=training_app_name(foreign_id),
            ownership_tag=training_ownership_tag(foreign_id),
            request_hash=spec.request_hash,
        )


class CountingTarget(HoldingTarget):
    def __init__(self) -> None:
        super().__init__()
        self.launch_calls = 0

    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle:
        self.launch_calls += 1
        return super().launch(spec)


class BlockingLaunchTarget(CountingTarget):
    def __init__(self) -> None:
        super().__init__()
        self.launch_started = threading.Event()
        self.release_launch = threading.Event()

    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle:
        self.launch_started.set()
        if not self.release_launch.wait(timeout=5):
            raise ManagedTrainingTargetError("test launch was not released")
        return super().launch(spec)


class DelayedAppearanceTarget(HoldingTarget):
    def __init__(self) -> None:
        super().__init__()
        self.visible = threading.Event()

    def list_owned(self) -> list[TrainingTargetState]:
        self.list_calls += 1
        if not self.visible.is_set():
            return []
        return list(self.states.values())


class BlockingArtifactService(StubManagedTrainingArtifactService):
    def __init__(self) -> None:
        self.prepare_started = threading.Event()
        self.release_prepare = threading.Event()

    def prepare(self, **kwargs: Any):
        self.prepare_started.set()
        if not self.release_prepare.wait(timeout=5):
            raise RuntimeError("test preparation was not released")
        return super().prepare(**kwargs)


class FlakySessionFactory:
    def __init__(self, delegate, failures: int) -> None:
        self.delegate = delegate
        self.failures = failures
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary database outage")
        return self.delegate()


class StartupFailingManager:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def startup(self) -> None:
        raise RuntimeError("managed training startup failed")

    async def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "event_gap"),
    [(StubManagedTrainingTarget(), False), (GapTarget(), True), (UnknownOnceTarget(), False)],
)
async def test_supervisor_completes_exact_artifact_and_surfaces_event_gap(
    target, event_gap: bool
) -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"completed", "failed"})
    assert job.status == "completed"
    assert job.outcome == "succeeded"
    assert job.event_gap is event_gap
    assert job.output_revision is not None
    assert job.teardown_verified_at is not None
    with factory() as db:
        run = db.get(TrainingRun, job.training_run_id)
        assert run is not None
        assert run.current_step == 100
        assert run.status == "completed"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_partial_result_fails_and_is_torn_down() -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        PartialResultTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"failed"})
    assert job.outcome == "failed"
    assert job.teardown_verified_at is not None
    assert job.output_revision is None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_restart_uses_persisted_call_id_not_app_only_listing() -> None:
    _engine, factory = _factory()
    target = HoldingTarget()
    artifacts = StubManagedTrainingArtifactService()
    first = ManagedTrainingManager(
        target, artifacts, session_factory=factory, poll_interval_seconds=0.01
    )
    await first.startup()
    with factory() as db:
        record = await first.create(db, _launch())
    await _wait_status(factory, record.id, {"running"})
    await first.shutdown()
    target.list_app_only = True

    second = ManagedTrainingManager(
        target, artifacts, session_factory=factory, poll_interval_seconds=0.01
    )
    await second.startup()
    await asyncio.sleep(0.05)
    expected_call = f"call-{record.id.hex}"
    assert any(
        handle.provider_function_call_id == expected_call
        for handle in target.inspect_handles
    )
    with factory() as db:
        current = db.get(ManagedTrainingJob, record.id)
        assert current is not None
        assert current.provider_function_call_id == expected_call
        await second.cancel(db, record.id)
    job = await _wait_status(factory, record.id, {"cancelled"})
    assert job.teardown_verified_at is not None
    await second.shutdown()


@pytest.mark.asyncio
async def test_app_without_reattachable_call_is_failed_and_stopped() -> None:
    _engine, factory = _factory()
    target = HoldingTarget()
    artifacts = StubManagedTrainingArtifactService()
    first = ManagedTrainingManager(
        target, artifacts, session_factory=factory, poll_interval_seconds=0.01
    )
    await first.startup()
    with factory() as db:
        record = await first.create(db, _launch())
    await _wait_status(factory, record.id, {"running"})
    await first.shutdown()
    with factory() as db:
        row = db.get(ManagedTrainingJob, record.id)
        assert row is not None
        row.status = "launching"
        row.provider_app_id = None
        row.provider_function_call_id = None
        db.commit()
    target.list_app_only = True
    second = ManagedTrainingManager(
        target, artifacts, session_factory=factory, poll_interval_seconds=0.01
    )
    await second.startup()
    job = await _wait_status(factory, record.id, {"failed"})
    assert job.teardown_verified_at is not None
    assert target.states[record.id].resource_lifecycle == "stopped"
    await second.shutdown()


@pytest.mark.asyncio
async def test_one_active_job_cap_and_idempotent_replay() -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        HoldingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    first_payload = _launch()
    with factory() as db:
        first = await manager.create(db, first_payload)
    await _wait_status(factory, first.id, {"running"})
    with factory() as db:
        replay = await manager.create(db, first_payload)
        assert replay.id == first.id
    with factory() as db:
        with pytest.raises(ManagedTrainingConflictError, match="Only one"):
            await manager.create(
                db,
                _launch(output_model_repo="acme/second-output"),
            )
    with factory() as db:
        await manager.cancel(db, first.id)
    await _wait_status(factory, first.id, {"cancelled"})
    await manager.shutdown()


@pytest.mark.asyncio
async def test_wrong_returned_handle_never_stops_foreign_job() -> None:
    _engine, factory = _factory()
    target = WrongHandleTarget()
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"failed"})
    assert job.teardown_verified_at is not None
    assert target.stop_handles
    assert all(handle.job_id == record.id for handle in target.stop_handles)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_restart_deadline_is_persisted_before_inspect_outage() -> None:
    _engine, factory = _factory()
    target = HoldingTarget()
    artifacts = StubManagedTrainingArtifactService()
    first = ManagedTrainingManager(
        target, artifacts, session_factory=factory, poll_interval_seconds=0.01
    )
    await first.startup()
    with factory() as db:
        record = await first.create(db, _launch())
    await _wait_status(factory, record.id, {"running"})
    await first.shutdown()
    with factory() as db:
        row = db.get(ManagedTrainingJob, record.id)
        assert row is not None
        row.deadline_at = datetime(2026, 8, 28, tzinfo=UTC)
        db.commit()
    target.inspect_failures = 100
    second = ManagedTrainingManager(
        target, artifacts, session_factory=factory, poll_interval_seconds=0.01
    )
    await second.startup()
    for _ in range(200):
        with factory() as db:
            current = db.get(ManagedTrainingJob, record.id)
            assert current is not None
            if current.outcome == "cancelled":
                break
        await asyncio.sleep(0.01)
    assert current.outcome == "cancelled"
    assert current.status == "cancelling"
    assert target.stop_handles
    await second.shutdown()


@pytest.mark.asyncio
async def test_orphan_cleanup_retries_beyond_three_failures() -> None:
    _engine, factory = _factory()
    target = HoldingTarget()
    job_id = uuid.uuid4()
    handle = TrainingHandle(
        job_id=job_id,
        provider_app_id=f"hold-{job_id.hex}",
        provider_function_call_id=f"call-{job_id.hex}",
        app_name=training_app_name(job_id),
        ownership_tag=training_ownership_tag(job_id),
    )
    target.states[job_id] = target._state(handle, "running", "running", 1)
    target.stop_failures = 4
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    for _ in range(200):
        if target.states[job_id].resource_lifecycle == "stopped":
            break
        await asyncio.sleep(0.01)
    assert target.states[job_id].resource_lifecycle == "stopped"
    assert len(target.stop_handles) >= 5
    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_during_artifact_preparation_never_launches_compute() -> None:
    _engine, factory = _factory()
    target = CountingTarget()
    artifacts = BlockingArtifactService()
    manager = ManagedTrainingManager(
        target,
        artifacts,
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    assert await asyncio.to_thread(artifacts.prepare_started.wait, 1)

    shutdown = asyncio.create_task(manager.shutdown())
    while not manager._shutdown.is_set():
        await asyncio.sleep(0)
    artifacts.release_prepare.set()
    await asyncio.wait_for(shutdown, timeout=2)

    assert target.launch_calls == 0
    with factory() as db:
        job = db.get(ManagedTrainingJob, record.id)
        assert job is not None
        assert job.status == "cancelled"
        assert job.teardown_verified_at is not None


@pytest.mark.asyncio
async def test_shutdown_racing_provider_launch_persists_and_reattaches_exact_handle() -> None:
    _engine, factory = _factory()
    target = BlockingLaunchTarget()
    artifacts = StubManagedTrainingArtifactService()
    manager = ManagedTrainingManager(
        target,
        artifacts,
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    assert await asyncio.to_thread(target.launch_started.wait, 1)

    shutdown = asyncio.create_task(manager.shutdown())
    while not manager._shutdown.is_set():
        await asyncio.sleep(0)
    target.release_launch.set()
    await asyncio.wait_for(shutdown, timeout=2)

    assert target.launch_calls == 1
    assert target.stop_handles == []
    with factory() as db:
        job = db.get(ManagedTrainingJob, record.id)
        assert job is not None
        assert job.status == "running"
        assert job.outcome == "pending"
        assert job.provider_app_id == f"hold-{record.id.hex}"
        assert job.provider_function_call_id == f"call-{record.id.hex}"
        assert job.teardown_verified_at is None

    restarted = ManagedTrainingManager(
        target,
        artifacts,
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await restarted.startup()
    await asyncio.sleep(0.05)
    assert any(handle.job_id == record.id for handle in target.inspect_handles)
    assert target.stop_handles == []
    with factory() as db:
        await restarted.cancel(db, record.id)
    terminal = await _wait_status(factory, record.id, {"cancelled"})
    assert terminal.teardown_verified_at is not None
    await restarted.shutdown()


@pytest.mark.asyncio
async def test_cancellation_mid_launch_settles_and_stops_exact_provider_handle() -> None:
    _engine, factory = _factory()
    target = BlockingLaunchTarget()
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    assert await asyncio.to_thread(target.launch_started.wait, 1)
    supervisor = manager._tasks[record.id]
    supervisor.cancel()
    target.release_launch.set()
    with pytest.raises(asyncio.CancelledError):
        await supervisor

    job = await _wait_status(factory, record.id, {"cancelled"})
    assert job.teardown_verified_at is not None
    assert target.stop_handles
    assert all(handle.job_id == record.id for handle in target.stop_handles)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_lifespan_cancellation_mid_launch_preserves_durable_exact_handle() -> None:
    _engine, factory = _factory()
    target = BlockingLaunchTarget()
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    assert await asyncio.to_thread(target.launch_started.wait, 1)

    supervisor = manager._tasks[record.id]
    manager._shutdown.set()
    supervisor.cancel()
    target.release_launch.set()
    with pytest.raises(asyncio.CancelledError):
        await supervisor

    with factory() as db:
        job = db.get(ManagedTrainingJob, record.id)
        assert job is not None
        assert job.status == "running"
        assert job.outcome == "pending"
        assert job.provider_app_id == f"hold-{record.id.hex}"
        assert job.provider_function_call_id == f"call-{record.id.hex}"
        assert job.teardown_verified_at is None
    assert target.stop_handles == []
    await manager.shutdown()


@pytest.mark.asyncio
async def test_startup_database_outage_is_nonfatal_and_reconciles_after_recovery() -> None:
    _engine, factory = _factory()
    flaky_factory = FlakySessionFactory(factory, failures=2)
    target = HoldingTarget()
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=flaky_factory,
        poll_interval_seconds=0.01,
    )

    await manager.startup()
    assert target.list_calls == 0
    for _ in range(200):
        if target.list_calls:
            break
        await asyncio.sleep(0.01)
    assert flaky_factory.calls >= 3
    assert target.list_calls >= 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_late_provider_app_is_stopped_before_ambiguous_launch_terminalizes() -> None:
    _engine, factory = _factory()
    target = DelayedAppearanceTarget()
    job_id = uuid.uuid4()
    request_hash = "a" * 64
    now = datetime.now(UTC)
    with factory() as db:
        run = TrainingRun(name="ambiguous launch", status="running")
        db.add(run)
        db.flush()
        db.add(
            ManagedTrainingJob(
                id=job_id,
                training_run_id=run.id,
                idempotency_key=uuid.uuid4(),
                request_hash=request_hash,
                status="launching",
                outcome="pending",
                target_kind="stub",
                provider_state="pending",
                compute_size="Modal: A10G",
                runtime="lerobot",
                dataset_repo="acme/data",
                dataset_revision="b" * 40,
                base_model="acme/base",
                base_model_revision="c" * 40,
                output_model_repo="acme/late-output",
                output_private=True,
                output_marker_revision="d" * 40,
                max_steps=100,
                batch_size=8,
                log_every=1,
                save_every=10,
                seed=42,
                num_workers=0,
                timeout_seconds=60,
                deadline_at=now + timedelta(minutes=1),
                launch_attempted_at=now,
                provider_launch_started_at=now,
            )
        )
        db.commit()
    handle = TrainingHandle(
        job_id=job_id,
        provider_app_id=f"hold-{job_id.hex}",
        provider_function_call_id=f"call-{job_id.hex}",
        app_name=training_app_name(job_id),
        ownership_tag=training_ownership_tag(job_id),
        request_hash=request_hash,
    )
    target.states[job_id] = target._state(handle, "running", "running", 1)
    manager = ManagedTrainingManager(
        target,
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )

    await manager.startup()
    await asyncio.sleep(0.08)
    with factory() as db:
        unresolved = db.get(ManagedTrainingJob, job_id)
        assert unresolved is not None
        assert unresolved.status == "launching"
        assert unresolved.teardown_verified_at is None
    assert target.list_calls >= 3

    target.visible.set()
    terminal = await _wait_status(factory, job_id, {"failed"})
    assert terminal.teardown_verified_at is not None
    assert target.stop_handles
    assert all(item.job_id == job_id for item in target.stop_handles)
    assert target.states[job_id].resource_lifecycle == "stopped"
    await manager.shutdown()


def test_database_rejects_second_active_row_even_without_service_check() -> None:
    _engine, factory = _factory()
    with factory() as db:
        for index in range(2):
            run = TrainingRun(name=f"run-{index}", status="created")
            db.add(run)
            db.flush()
            db.add(
                ManagedTrainingJob(
                    training_run_id=run.id,
                    idempotency_key=uuid.uuid4(),
                    request_hash="a" * 64,
                    status="created",
                    outcome="pending",
                    target_kind="stub",
                    provider_state="pending",
                    compute_size="Modal: A10G",
                    runtime="lerobot",
                    dataset_repo="acme/data",
                    base_model="acme/base",
                    output_model_repo=f"acme/out-{index}",
                    output_private=True,
                    max_steps=1,
                    batch_size=1,
                    log_every=1,
                    save_every=1,
                    seed=0,
                    num_workers=0,
                    timeout_seconds=60,
                    deadline_at=datetime(2026, 8, 29, 5, tzinfo=UTC),
                )
            )
            if index == 0:
                db.flush()
        with pytest.raises(IntegrityError):
            db.flush()


def test_stopped_proof_rejects_live_execution_state() -> None:
    job_id = uuid.uuid4()
    state = TrainingTargetState(
        job_id=job_id,
        provider_app_id=f"app-{job_id.hex}",
        provider_function_call_id=f"call-{job_id.hex}",
        app_name=training_app_name(job_id),
        ownership_tag=training_ownership_tag(job_id),
        exists=False,
        resource_lifecycle="unknown",
        execution_state="running",
        running_tasks=0,
    )
    assert state.stopped_verified is False


def _api_payload(key: uuid.UUID, **overrides: Any) -> dict[str, Any]:
    payload = {
        "idempotency_key": str(key),
        "name": "Managed SDK training",
        "dataset_repo": "acme/data",
        "base_model": "acme/base",
        "output_model_repo": "acme/output",
        "compute_size": "Modal: A10G",
        "max_steps": 100,
        "log_every": 1,
        "save_every": 10,
        "acknowledge_compute_cost": True,
    }
    payload.update(overrides)
    return payload


def test_managed_training_rest_surface_and_managed_run_protection() -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        StubManagedTrainingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    app = create_app(managed_training_manager=manager)

    def database():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    key = uuid.uuid4()
    with TestClient(app) as client:
        launched = client.post("/api/trainer/jobs", json=_api_payload(key))
        assert launched.status_code == 202, launched.text
        job_id = launched.json()["id"]
        run_id = launched.json()["training_run_id"]
        replay = client.post("/api/trainer/jobs", json=_api_payload(key))
        assert replay.status_code == 202
        assert replay.json()["id"] == job_id
        for _ in range(200):
            detail = client.get(f"/api/trainer/jobs/{job_id}")
            if detail.json()["status"] == "completed":
                break
            time.sleep(0.01)
        assert detail.status_code == 200
        assert detail.json()["teardown_verified"] is True
        assert detail.json()["target_kind"] == "stub"
        listed = client.get("/api/trainer/jobs", params={"limit": 1})
        logs = client.get(f"/api/trainer/jobs/{job_id}/logs")
        metrics = client.get(f"/api/trainer/jobs/{job_id}/metrics")
        checkpoints = client.get(f"/api/trainer/jobs/{job_id}/checkpoints")
        run = client.get(f"/api/trainer/runs/{run_id}")
        assert listed.status_code == logs.status_code == metrics.status_code == 200
        assert checkpoints.status_code == run.status_code == 200
        assert run.json()["managed_job"]["id"] == job_id
        for method, suffix, body in (
            (client.patch, "", {"status": "failed"}),
            (client.post, "/metrics", {"step": 1, "metrics": {"loss": 1.0}}),
            (
                client.post,
                "/checkpoints",
                {"repo_id": "acme/output", "revision": "a" * 40, "step": 1},
            ),
            (client.post, "/logs", {"line": "external mutation"}),
        ):
            response = method(f"/api/trainer/runs/{run_id}{suffix}", json=body)
            assert response.status_code == 409
        assert client.post(f"/api/trainer/jobs/{job_id}/cancel").status_code == 202


def test_managed_training_rest_requires_strict_explicit_consent() -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        StubManagedTrainingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
    )
    app = create_app(managed_training_manager=manager)

    def database():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        missing = client.post(
            "/api/trainer/jobs",
            json=_api_payload(uuid.uuid4(), acknowledge_compute_cost=False),
        )
        coercion = client.post(
            "/api/trainer/jobs",
            json=_api_payload(uuid.uuid4(), acknowledge_compute_cost=1),
        )
        public = client.post(
            "/api/trainer/jobs",
            json=_api_payload(
                uuid.uuid4(),
                output_private=False,
                acknowledge_public_model_risk=False,
            ),
        )
    assert missing.status_code == coercion.status_code == public.status_code == 422


def test_lifespan_rolls_back_partially_started_managed_training_manager() -> None:
    manager = StartupFailingManager()
    app = create_app(managed_training_manager=manager)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="managed training startup failed"):
        with TestClient(app):
            pass

    assert manager.shutdown_called is True


class RootRequirementRecordingArtifacts(StubManagedTrainingArtifactService):
    """Record how each revision verification treats the repository root."""

    def __init__(self) -> None:
        self.deployable_root_calls: list[bool] = []

    def verify_output_revision(self, **kwargs: Any) -> str:
        self.deployable_root_calls.append(bool(kwargs["require_deployable_root"]))
        return super().verify_output_revision(**kwargs)


class InvalidFinalArtifacts(StubManagedTrainingArtifactService):
    def verify_output_revision(self, **kwargs: Any) -> str:
        if kwargs["require_deployable_root"]:
            raise ManagedTrainingArtifactVerificationError(
                "The immutable output model lacks a deployable root checkpoint."
            )
        return super().verify_output_revision(**kwargs)


class EventPersistenceThreadRecordingManager(ManagedTrainingManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.event_persistence_threads: list[int] = []

    def _persist_events(self, *args: Any, **kwargs: Any) -> None:
        self.event_persistence_threads.append(threading.get_ident())
        super()._persist_events(*args, **kwargs)


async def test_intermediate_checkpoint_events_do_not_require_a_deployable_root() -> None:
    """The worker uploads non-final checkpoints under checkpoints/<step>/, so
    demanding root-deployable files for every checkpoint event would fail every
    intermediate save and wedge event ingestion until the hard deadline."""

    _engine, factory = _factory()
    artifacts = RootRequirementRecordingArtifacts()
    manager = ManagedTrainingManager(
        StubManagedTrainingTarget(),
        artifacts,
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"completed", "failed"})
    assert job.status == "completed"
    # The default launch saves an intermediate checkpoint (save_every=10 <
    # max_steps=100): its event verification must not demand the root layout;
    # only the final-result verification may.
    assert artifacts.deployable_root_calls == [False, True]
    await manager.shutdown()


async def test_event_persistence_runs_off_the_async_supervisor_loop() -> None:
    _engine, factory = _factory()
    loop_thread = threading.get_ident()
    manager = EventPersistenceThreadRecordingManager(
        StubManagedTrainingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"completed", "failed"})

    assert job.status == "completed"
    assert manager.event_persistence_threads
    assert all(item != loop_thread for item in manager.event_persistence_threads)
    await manager.shutdown()


async def test_immutable_final_artifact_failure_stops_compute_without_deadline_spin() -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        StubManagedTrainingTarget(),
        InvalidFinalArtifacts(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())

    job = await _wait_status(factory, record.id, {"completed", "failed"})

    assert job.status == "failed"
    assert job.outcome == "failed"
    assert job.teardown_verified_at is not None
    assert job.last_error is not None
    assert "deployable root checkpoint" in job.last_error
    await manager.shutdown()


class SecretLikeLineTarget(StubManagedTrainingTarget):
    def poll(
        self, handle: TrainingHandle, *, after_sequence: int, limit: int
    ) -> TrainingPoll:
        poll = super().poll(handle, after_sequence=after_sequence, limit=limit)
        events = tuple(
            replace(event, line="password: hunter2")
            if isinstance(event, TrainingLogEvent) and event.sequence == 1
            else event
            for event in poll.events
        )
        return replace(poll, events=events)


async def test_secret_like_provider_log_line_is_redacted_without_failing_the_job() -> None:
    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        SecretLikeLineTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"completed", "failed"})
    assert job.status == "completed"
    with factory() as db:
        run = db.get(TrainingRun, job.training_run_id)
        assert run is not None
        lines = [entry["line"] for entry in run.console_logs]
    assert not any("hunter2" in line for line in lines)
    assert any("redacted one provider console line" in line for line in lines)
    await manager.shutdown()


class MetricFloodTarget(StubManagedTrainingTarget):
    """Emit more metric names than the bounded run store permits."""

    def poll(
        self, handle: TrainingHandle, *, after_sequence: int, limit: int
    ) -> TrainingPoll:
        poll = super().poll(handle, after_sequence=after_sequence, limit=limit)
        flood = tuple(
            TrainingMetricEvent(
                sequence=index + 1,
                step=index + 1,
                metrics={f"metric{index}x{name}": 0.5 for name in range(64)},
            )
            for index in range(3)
        )
        return replace(poll, events=flood, next_sequence=3, has_more=False)


async def test_storage_limit_violation_fails_the_job_instead_of_spinning() -> None:
    """A deterministic bounded-store violation cannot heal by re-polling the
    identical event page; it must fail the job promptly rather than burn the
    provider budget until the hard deadline."""

    _engine, factory = _factory()
    manager = ManagedTrainingManager(
        MetricFloodTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await manager.startup()
    with factory() as db:
        record = await manager.create(db, _launch())
    job = await _wait_status(factory, record.id, {"completed", "failed"})
    assert job.status == "failed"
    assert job.outcome == "failed"
    assert job.teardown_verified_at is not None
    await manager.shutdown()


async def test_unavailable_original_target_delays_before_supervisor_reschedule() -> None:
    """An active row can outlive a backend whose original adapter is absent.

    The supervisor done-callback immediately schedules another resume attempt,
    so the unavailable-target branch itself must apply the configured delay.
    """

    _engine, factory = _factory()
    first = ManagedTrainingManager(
        HoldingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.01,
    )
    await first.startup()
    with factory() as db:
        record = await first.create(db, _launch())
    await _wait_status(factory, record.id, {"running"})
    await first.shutdown()
    with factory() as db:
        row = db.get(ManagedTrainingJob, record.id)
        assert row is not None
        row.target_kind = "modal"
        db.commit()

    unavailable = ManagedTrainingManager(
        StubManagedTrainingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=factory,
        poll_interval_seconds=0.05,
    )
    started = time.monotonic()
    await unavailable._resume_job(record.id)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.04
    with factory() as db:
        row = db.get(ManagedTrainingJob, record.id)
        assert row is not None
        assert row.status == "running"
        assert row.last_error == (
            "The original managed training provider is unavailable."
        )


class StatementCapturingSession:
    def __init__(self, statements: list[Any]) -> None:
        self._statements = statements

    def __enter__(self) -> StatementCapturingSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def scalar(self, statement: Any) -> None:
        self._statements.append(statement)
        return None


def test_retryable_and_cleanup_error_writes_request_row_locks() -> None:
    """PostgreSQL serializes competing supervisor/cancel status writers."""

    statements: list[Any] = []
    manager = ManagedTrainingManager(
        StubManagedTrainingTarget(),
        StubManagedTrainingArtifactService(),
        session_factory=lambda: StatementCapturingSession(statements),  # type: ignore[arg-type]
    )
    job_id = uuid.uuid4()

    manager._note_retryable_error(job_id, "temporary observation outage")
    manager._set_cleanup_pending_error(job_id, "cleanup still pending")

    assert len(statements) == 2
    assert all(
        "FOR UPDATE" in str(statement.compile(dialect=postgresql.dialect()))
        for statement in statements
    )
