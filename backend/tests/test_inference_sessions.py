from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Generator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ctrl_pi.api.recordings import reconcile_recordings_startup
from ctrl_pi.camera import CameraFrame, MockCamera
from ctrl_pi.compute import ComputeConfigurationError, ComputeTargetError
from ctrl_pi.compute_stub import StubComputeTarget
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import Base, get_db
from ctrl_pi.deployments import DeploymentProviderError, DeploymentService
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.inference_runtime import (
    MOCK_MODEL_REPO,
    MOCK_MODEL_REVISION,
    RuntimeLoadSpec,
    StubInferenceRuntime,
)
from ctrl_pi.inference_sessions import (
    InferenceSessionConflictError,
    InferenceSessionManager,
    InferenceSessionRuntimeError,
    InferenceSessionUnavailableError,
    InferenceStartOptions,
    InferenceStopOptions,
)
from ctrl_pi.inference_transport import InProcessInferenceTransport
from ctrl_pi.main import create_app
from ctrl_pi.models import Deployment, InferenceEndpoint, Recording
from ctrl_pi.recording import EPISODE_MANIFEST_FILENAME, RecordingManager
from ctrl_pi.rig import RigLease


class _FakeVideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: int) -> None:
        del width, height, fps
        self.output_path = output_path
        self.output_path.write_bytes(b"")
        self.closed = False

    def write(self, frame: CameraFrame) -> None:
        assert not self.closed and frame.rgb
        with self.output_path.open("ab") as output:
            output.write(b"f")

    def close(self) -> None:
        self.closed = True
        if self.output_path.stat().st_size == 0:
            raise RuntimeError("test writer received no frame")

    def terminate(self) -> None:
        self.closed = True


class _ModalStubComputeTarget(StubComputeTarget):
    """A network-free Modal-shaped lifecycle target for routing tests."""

    @property
    def kind(self):
        return "modal"


class _UnavailableModalTarget(_ModalStubComputeTarget):
    def __init__(self) -> None:
        super().__init__()
        self.stop_attempts = 0

    def stop(self, handle) -> None:
        del handle
        self.stop_attempts += 1
        raise ComputeConfigurationError("missing Modal credentials")


@pytest.fixture
def inference_stack(tmp_path: Path):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    target = StubComputeTarget()
    service = DeploymentService(
        target,
        session_factory=factory,
        nonce_factory=lambda: "m11-test-deployment",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    driver = MockYAMDriver()
    camera = MockCamera(width=64, height=48)
    rig = RigLease()
    recording_manager = RecordingManager(
        driver=driver,
        camera=camera,
        staging_dir=tmp_path / "recordings",
        rig_lease=rig,
    )

    def transport_factory(record):
        runtime = StubInferenceRuntime(runtime=record.runtime)
        runtime.load(
            RuntimeLoadSpec(
                model_repo=record.model_repo,
                revision=record.checkpoint_revision,
                local_model_path=None,
                device="cpu",
                actions_per_chunk=8,
            )
        )
        return InProcessInferenceTransport(runtime)

    manager = InferenceSessionManager(
        deployment_service=service,
        driver=driver,
        camera=camera,
        rig_lease=rig,
        recording_manager=recording_manager,
        transport_factory=transport_factory,
        session_factory=factory,
        recording_fps=50,
        # Most unit tests drive reconciliation explicitly. Dedicated watchdog
        # tests enable the background loop and always shut it down.
        watchdog_interval_seconds=None,
    )
    return (
        engine,
        factory,
        target,
        service,
        driver,
        recording_manager,
        manager,
    )


def _manager_for_service(
    inference_stack,
    service: DeploymentService,
    cleanup_service_factory,
) -> InferenceSessionManager:
    _, factory, _, _, driver, recording_manager, existing = inference_stack
    return InferenceSessionManager(
        deployment_service=service,
        driver=driver,
        camera=existing.camera,
        rig_lease=existing.rig_lease,
        recording_manager=recording_manager,
        transport_factory=existing.transport_factory,
        session_factory=factory,
        cleanup_service_factory=cleanup_service_factory,
        recording_fps=50,
        watchdog_interval_seconds=None,
    )


async def _deploy(service: DeploymentService, factory) -> uuid.UUID:
    with factory() as db:
        record = await service.deploy(
            db,
            name="M11 mock policy",
            model_repo=MOCK_MODEL_REPO,
            checkpoint_revision=MOCK_MODEL_REVISION,
            runtime="stub",
            compute_size="CPU",
            timeout_seconds=60,
        )
        return record.id


async def _terminal(
    manager: InferenceSessionManager,
    deployment_id: uuid.UUID,
    *,
    timeout: float = 5,
):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        snapshot = await manager.read(deployment_id)
        if snapshot.session_status in {"stopped", "failed"}:
            return snapshot
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("inference test did not reach terminal state")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_deploy_state_start_exact_steps_and_verified_teardown(
    inference_stack,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    await manager.recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)

    idle = await manager.read(deployment_id)
    assert idle.session_status == "idle"
    assert idle.endpoint_healthy is True

    with factory() as db:
        started = await manager.start(
            db,
            deployment_id,
            InferenceStartOptions(
                arm_id="yam-follower",
                task="Move deterministically",
                max_steps=5,
            ),
        )
    assert started.session_status == "running"
    terminal = await _terminal(manager, deployment_id)
    assert terminal.steps_executed == 5
    assert terminal.session_status == terminal.deployment.status == "stopped"
    assert terminal.teardown_verified is True
    assert terminal.queue_depth == 0
    assert all(state.stopped_verified for state in target.list_owned())
    with factory() as db:
        durable = db.get(Deployment, deployment_id)
        assert durable.latency_ms is None and durable.frequency_hz is None
    await asyncio.sleep(0)
    assert await manager.active_resource_counts() == (0, 0)
    assert manager.rig_lease.current() is None


@pytest.mark.asyncio
async def test_passive_recording_captures_first_applied_action_and_finalizes(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, _, service, _, recording_manager, manager = inference_stack
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    await recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)

    with factory() as db:
        await manager.start(
            db,
            deployment_id,
            InferenceStartOptions(
                arm_id="yam-follower",
                task="Record applied policy actions",
                record_session=True,
                recording_name="Inference capture",
                recording_metadata={"operator": "mock operator"},
                max_steps=6,
            ),
            recording_fps=50,
        )
    terminal = await _terminal(manager, deployment_id)

    assert terminal.steps_executed == 6
    assert terminal.recording.status == "ready"
    assert terminal.recording.episode_count == 1
    assert terminal.recording.recording_id is not None
    with factory() as db:
        recording = db.get(Recording, terminal.recording.recording_id)
        assert recording is not None
        assert recording.status == "ready"
        assert recording.episode_count == 1
        episode = recording.recording_metadata["episodes"][0]
    artifact = recording_manager.staging_dir / episode["artifact_key"]
    samples = [
        json.loads(line)
        for line in (artifact / "samples.jsonl").read_text().splitlines()
    ]
    assert len(samples) == episode["sample_count"]
    assert samples and samples[0]["frame_index"] == 0
    action = samples[0]["action"]["joint_positions_radians"]
    observation = {
        joint["name"]: joint["position_radians"]
        for joint in samples[0]["observation"]["joints"]
    }
    assert action == observation
    assert (artifact / "video.mp4").read_bytes()
    assert await recording_manager.active_resource_counts() == (0, 0)


@pytest.mark.asyncio
async def test_rig_conflict_stops_provider_without_arm_motion(inference_stack) -> None:
    _, factory, target, service, driver, recording_manager, manager = inference_stack
    await recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)
    before = driver.get_arm("yam-follower")
    lease = manager.rig_lease.acquire("manual", "test-jog")
    try:
        with factory() as db:
            with pytest.raises(InferenceSessionConflictError):
                await manager.start(
                    db,
                    deployment_id,
                    InferenceStartOptions(
                        arm_id="yam-follower",
                        task="Must not move",
                    ),
                )
    finally:
        manager.rig_lease.release(lease)
    after = driver.get_arm("yam-follower")
    assert [joint.position_radians for joint in before.joints] == [
        joint.position_radians for joint in after.joints
    ]
    assert all(state.stopped_verified for state in target.list_owned())


@pytest.mark.asyncio
async def test_restart_never_resumes_actions_and_tears_down_running_provider(
    inference_stack,
) -> None:
    _, factory, target, service, driver, recording_manager, manager = inference_stack
    await recording_manager.startup()
    deployment_id = await _deploy(service, factory)
    before = driver.get_arm("yam-follower")

    await manager.startup()

    with factory() as db:
        deployment = db.get(Deployment, deployment_id)
        assert deployment.status == "stopped"
    after = driver.get_arm("yam-follower")
    assert [joint.position_radians for joint in before.joints] == [
        joint.position_radians for joint in after.joints
    ]
    assert all(state.stopped_verified for state in target.list_owned())
    assert await manager.active_resource_counts() == (0, 0)


@pytest.mark.asyncio
async def test_startup_database_outage_retries_unexpired_cleanup_without_motion(
    inference_stack,
) -> None:
    _, factory, target, service, driver, _, existing = inference_stack
    deployment_id = await _deploy(service, factory)
    before = driver.get_arm("yam-follower")
    attempts = 0

    def flaky_factory():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SQLAlchemyError("transient startup database outage")
        return factory()

    manager = InferenceSessionManager(
        deployment_service=service,
        driver=driver,
        camera=existing.camera,
        rig_lease=existing.rig_lease,
        recording_manager=existing.recording_manager,
        transport_factory=existing.transport_factory,
        session_factory=flaky_factory,
        recording_fps=50,
        watchdog_interval_seconds=None,
    )

    await manager.startup()
    with factory() as db:
        assert service.get(db, deployment_id).status == "running"
        with pytest.raises(
            InferenceSessionUnavailableError,
            match="Startup deployment reconciliation is still pending",
        ):
            await manager.start(
                db,
                deployment_id,
                InferenceStartOptions(
                    arm_id="yam-follower",
                    task="Must not move before restart cleanup",
                ),
            )

    await manager._watchdog_once()

    with factory() as db:
        assert service.get(db, deployment_id).status == "stopped"
    after = driver.get_arm("yam-follower")
    assert [joint.position_radians for joint in before.joints] == [
        joint.position_radians for joint in after.joints
    ]
    assert all(state.stopped_verified for state in target.list_owned())
    assert await manager.active_resource_counts() == (0, 0)


@pytest.mark.asyncio
async def test_watchdog_expires_an_idle_deployment_from_its_persisted_timeout(
    inference_stack,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    deployment_id = await _deploy(service, factory)
    with factory() as db:
        record = service.get(db, deployment_id)
        assert record.timeout_seconds == 60
        assert record.started_at is not None
        expires_after = record.started_at + timedelta(seconds=61)
    manager._now = lambda: expires_after

    await manager._watchdog_once()

    with factory() as db:
        assert service.get(db, deployment_id).status == "stopped"
    assert all(state.stopped_verified for state in target.list_owned())


@pytest.mark.asyncio
async def test_background_watchdog_enforces_deadline_for_deployments_created_later(
    inference_stack,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    manager._watchdog_interval_seconds = 0.01
    await manager.startup()
    try:
        deployment_id = await _deploy(service, factory)
        with factory() as db:
            record = service.get(db, deployment_id)
        assert record.started_at is not None
        manager._now = lambda: record.started_at + timedelta(seconds=61)

        deadline = asyncio.get_running_loop().time() + 1
        while True:
            with factory() as db:
                if service.get(db, deployment_id).status == "stopped":
                    break
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("deployment watchdog did not enforce the deadline")
            await asyncio.sleep(0.01)
        assert all(state.stopped_verified for state in target.list_owned())
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_watchdog_stops_live_motion_before_provider_teardown(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, _, recording_manager, manager = inference_stack
    await recording_manager.startup()
    deployment_id = await _deploy(service, factory)
    with factory() as db:
        started = await manager.start(
            db,
            deployment_id,
            InferenceStartOptions(
                arm_id="yam-follower",
                task="Stop at the durable deployment deadline",
            ),
        )
    assert started.deployment.started_at is not None
    manager._now = lambda: started.deployment.started_at + timedelta(seconds=61)
    original_stop = target.stop
    stop_observations: list[bool] = []

    def assert_local_loop_stopped(handle) -> None:
        stop_observations.append(manager.rig_lease.current() is None)
        original_stop(handle)

    monkeypatch.setattr(target, "stop", assert_local_loop_stopped)
    await manager._watchdog_once()

    terminal = await manager.read(deployment_id)
    assert terminal.session_status == terminal.deployment.status == "stopped"
    assert terminal.teardown_verified is True
    assert stop_observations == [True]
    assert await manager.active_resource_counts() == (0, 0)


@pytest.mark.asyncio
async def test_watchdog_retries_a_failed_expired_teardown(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    deployment_id = await _deploy(service, factory)
    with factory() as db:
        record = service.get(db, deployment_id)
        assert record.started_at is not None
        manager._now = lambda: record.started_at + timedelta(seconds=61)
    original_stop = target.stop
    attempts = 0

    def fail_once(handle) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ComputeTargetError("raw provider failure")
        original_stop(handle)

    monkeypatch.setattr(target, "stop", fail_once)
    await manager._watchdog_once()
    with factory() as db:
        assert service.get(db, deployment_id).status == "failed"
    assert any(not state.stopped_verified for state in target.list_owned())

    await manager._watchdog_once()
    with factory() as db:
        assert service.get(db, deployment_id).status == "stopped"
    assert attempts == 2
    assert all(state.stopped_verified for state in target.list_owned())


@pytest.mark.asyncio
async def test_shutdown_retries_failed_owned_cleanup_to_verified_stop(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    deployment_id = await _deploy(service, factory)
    original_stop = target.stop
    attempts = 0

    def fail_once(handle) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ComputeTargetError("first provider stop failed")
        original_stop(handle)

    monkeypatch.setattr(target, "stop", fail_once)
    with factory() as db:
        with pytest.raises(DeploymentProviderError):
            await service.stop(db, deployment_id)
    with factory() as db:
        assert service.get(db, deployment_id).status == "failed"
    assert any(not state.stopped_verified for state in target.list_owned())

    await manager.shutdown()

    with factory() as db:
        assert service.get(db, deployment_id).status == "stopped"
    assert attempts == 2
    assert all(state.stopped_verified for state in target.list_owned())


@pytest.mark.asyncio
async def test_watchdog_cancels_a_blocked_start_and_tears_down_provider_boundedly(
    inference_stack,
) -> None:
    _, factory, target, service, _, recording_manager, manager = inference_stack
    await recording_manager.startup()
    deployment_id = await _deploy(service, factory)
    with factory() as db:
        record = service.get(db, deployment_id)
    assert record.started_at is not None

    entered = threading.Event()
    release = threading.Event()
    original_factory = manager.transport_factory

    class SlowStartTransport:
        def __init__(self, delegate) -> None:
            self.delegate = delegate

        def describe(self):
            entered.set()
            assert release.wait(timeout=3)
            return self.delegate.describe()

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    manager.transport_factory = lambda deployment: SlowStartTransport(
        original_factory(deployment)
    )

    async def start_blocked():
        with factory() as db:
            return await manager.start(
                db,
                deployment_id,
                InferenceStartOptions(
                    arm_id="yam-follower",
                    task="Never outlive the provider deadline",
                ),
            )

    start_task = asyncio.create_task(start_blocked())
    assert await asyncio.to_thread(entered.wait, 1)
    manager._now = lambda: record.started_at + timedelta(seconds=61)
    before = asyncio.get_running_loop().time()
    try:
        await manager._watchdog_once()
        assert asyncio.get_running_loop().time() - before < 1
        with factory() as db:
            assert service.get(db, deployment_id).status == "stopped"
        assert all(state.stopped_verified for state in target.list_owned())
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert manager.rig_lease.current() is None
    assert await manager.active_resource_counts() == (0, 0)


@pytest.mark.asyncio
async def test_watchdog_requires_provider_absence_proof_for_prehandle_failures(
    inference_stack,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    endpoint = InferenceEndpoint(
        name="failed before provider handle",
        runtime="stub",
        status="failed",
    )
    with factory() as db:
        db.add(endpoint)
        db.flush()
        deployment = Deployment(
            endpoint_id=endpoint.id,
            model_repo=MOCK_MODEL_REPO,
            checkpoint_revision=MOCK_MODEL_REVISION,
            runtime="stub",
            compute_size="CPU",
            target_kind="stub",
            timeout_seconds=60,
            status="failed",
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    assert target.list_owned() == []

    await manager._watchdog_once()

    with factory() as db:
        stopped = service.get(db, deployment_id)
    assert stopped.status == "stopped"
    assert stopped.provider_app_id is None


@pytest.mark.asyncio
async def test_restart_cleans_modal_row_through_modal_target_in_mock_mode(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, stub_target, stub_service, _, _, _ = inference_stack
    modal_target = _ModalStubComputeTarget()
    modal_service = DeploymentService(
        modal_target,
        session_factory=factory,
        nonce_factory=lambda: "modal-restart-cleanup",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    deployment_id = await _deploy(modal_service, factory)
    wrong_stop_calls = 0
    original_stub_stop = stub_target.stop

    def counted_wrong_stop(handle):
        nonlocal wrong_stop_calls
        wrong_stop_calls += 1
        return original_stub_stop(handle)

    monkeypatch.setattr(stub_target, "stop", counted_wrong_stop)
    manager = _manager_for_service(
        inference_stack,
        stub_service,
        lambda kind: modal_service if kind == "modal" else stub_service,
    )

    await manager.startup()

    with factory() as db:
        assert stub_service.get(db, deployment_id).status == "stopped"
    assert wrong_stop_calls == 0
    assert all(state.stopped_verified for state in modal_target.list_owned())


@pytest.mark.asyncio
async def test_watchdog_cleans_stub_deadline_through_stub_target_in_modal_mode(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, stub_target, stub_service, _, _, _ = inference_stack
    deployment_id = await _deploy(stub_service, factory)
    with factory() as db:
        record = stub_service.get(db, deployment_id)
    assert record.started_at is not None

    modal_target = _ModalStubComputeTarget()
    modal_service = DeploymentService(
        modal_target,
        session_factory=factory,
        nonce_factory=lambda: "modal-current-target",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    wrong_stop_calls = 0
    original_modal_stop = modal_target.stop

    def counted_wrong_stop(handle):
        nonlocal wrong_stop_calls
        wrong_stop_calls += 1
        return original_modal_stop(handle)

    monkeypatch.setattr(modal_target, "stop", counted_wrong_stop)
    manager = _manager_for_service(
        inference_stack,
        modal_service,
        lambda kind: stub_service if kind == "stub" else modal_service,
    )
    manager._now = lambda: record.started_at + timedelta(seconds=61)

    await manager._watchdog_once()

    with factory() as db:
        assert stub_service.get(db, deployment_id).status == "stopped"
    assert wrong_stop_calls == 0
    assert all(state.stopped_verified for state in stub_target.list_owned())


@pytest.mark.asyncio
async def test_missing_modal_cleanup_credentials_remain_failed_and_retryable(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, stub_target, stub_service, _, _, _ = inference_stack
    provider_target = _ModalStubComputeTarget()
    provider_service = DeploymentService(
        provider_target,
        session_factory=factory,
        nonce_factory=lambda: "modal-provider",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    deployment_id = await _deploy(provider_service, factory)
    with factory() as db:
        record = provider_service.get(db, deployment_id)
    assert record.started_at is not None

    unavailable_target = _UnavailableModalTarget()
    unavailable_service = DeploymentService(
        unavailable_target,
        session_factory=factory,
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    wrong_stop_calls = 0
    original_stub_stop = stub_target.stop

    def counted_wrong_stop(handle):
        nonlocal wrong_stop_calls
        wrong_stop_calls += 1
        return original_stub_stop(handle)

    monkeypatch.setattr(stub_target, "stop", counted_wrong_stop)
    manager = _manager_for_service(
        inference_stack,
        stub_service,
        lambda kind: unavailable_service if kind == "modal" else stub_service,
    )
    manager._now = lambda: record.started_at + timedelta(seconds=61)

    await manager._watchdog_once()
    await manager._watchdog_once()

    with factory() as db:
        failed = stub_service.get(db, deployment_id)
    assert failed.status == "failed"
    assert failed.stopped_at is None
    assert unavailable_target.stop_attempts == 2
    assert wrong_stop_calls == 0
    assert any(not state.stopped_verified for state in provider_target.list_owned())


@pytest.mark.asyncio
async def test_idle_health_is_provider_backed_cached_and_detects_disappearance(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, _, _, manager = inference_stack
    ticks = [0.0]
    manager._monotonic = lambda: ticks[0]
    manager._idle_health_cache_seconds = 2
    inspect_calls = 0
    original_inspect = target.inspect

    def counted_inspect(handle):
        nonlocal inspect_calls
        inspect_calls += 1
        return original_inspect(handle)

    monkeypatch.setattr(target, "inspect", counted_inspect)
    deployment_id = await _deploy(service, factory)
    health_calls = 0

    def forbidden_idle_health(handle, nonce):
        del handle, nonce
        nonlocal health_calls
        health_calls += 1
        raise AssertionError("idle polling must not call the runtime endpoint")

    monkeypatch.setattr(target, "health", forbidden_idle_health)

    assert (await manager.read(deployment_id)).endpoint_healthy is True
    first_probe_calls = inspect_calls
    assert first_probe_calls >= 1
    assert (await manager.read(deployment_id)).endpoint_healthy is True
    assert inspect_calls == first_probe_calls
    state = target.list_owned()[0]
    target.stop(state.handle())
    assert (await manager.read(deployment_id)).endpoint_healthy is True

    ticks[0] = 3
    disappeared = await manager.read(deployment_id)
    assert disappeared.deployment.status == "running"
    assert disappeared.session_status == "idle"
    assert disappeared.endpoint_healthy is False
    assert health_calls == 0


@pytest.mark.asyncio
async def test_idle_health_probe_has_a_bounded_response_time(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, _, _, _ = inference_stack
    deployment_id = await _deploy(service, factory)
    with factory() as db:
        record = service.get(db, deployment_id)
    original_inspect = target.inspect
    entered = threading.Event()
    release = threading.Event()
    health_calls = 0

    def slow_inspect(handle):
        entered.set()
        assert release.wait(timeout=1)
        return original_inspect(handle)

    def forbidden_late_health(handle, nonce):
        del handle, nonce
        nonlocal health_calls
        health_calls += 1
        raise AssertionError("timed-out inspection continued into runtime health")

    monkeypatch.setattr(target, "inspect", slow_inspect)
    monkeypatch.setattr(target, "health", forbidden_late_health)
    before = asyncio.get_running_loop().time()
    assert (
        await service.endpoint_healthy(record, timeout_seconds=0.01)
        is False
    )
    assert asyncio.get_running_loop().time() - before < 0.1
    assert entered.is_set()
    release.set()
    for _ in range(100):
        if not service._readiness_probes:
            break
        await asyncio.sleep(0.005)
    assert not service._readiness_probes
    assert health_calls == 0


@pytest.mark.asyncio
async def test_duplicate_start_conflicts_and_stop_joins_shared_loop(
    inference_stack,
) -> None:
    _, factory, _, service, _, recording_manager, manager = inference_stack
    await recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)
    with factory() as db:
        await manager.start(
            db,
            deployment_id,
            InferenceStartOptions(arm_id="yam-follower", task="One owner"),
        )
    with factory() as db:
        with pytest.raises(InferenceSessionConflictError):
            await manager.start(
                db,
                deployment_id,
                InferenceStartOptions(arm_id="yam-follower", task="Second owner"),
            )
    with factory() as db:
        stopped = await manager.stop(db, deployment_id, InferenceStopOptions())
    assert stopped.session_status == "stopped"
    assert stopped.teardown_verified is True
    assert manager.rig_lease.current() is None


@pytest.mark.asyncio
async def test_stale_runtime_identity_writes_no_action_and_tears_down_provider(
    inference_stack,
) -> None:
    _, factory, target, service, driver, recording_manager, manager = inference_stack
    await recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)
    before = driver.get_arm("yam-follower")

    def stale_transport(record):
        runtime = StubInferenceRuntime(runtime="stub")
        runtime.load(
            RuntimeLoadSpec(
                model_repo=record.model_repo,
                revision="f" * 40,
                local_model_path=None,
                device="cpu",
                actions_per_chunk=4,
            )
        )
        return InProcessInferenceTransport(runtime)

    manager.transport_factory = stale_transport
    with factory() as db:
        with pytest.raises(InferenceSessionRuntimeError):
            await manager.start(
                db,
                deployment_id,
                InferenceStartOptions(
                    arm_id="yam-follower",
                    task="Reject stale endpoint",
                ),
            )
    after = driver.get_arm("yam-follower")
    assert [joint.position_radians for joint in before.joints] == [
        joint.position_radians for joint in after.joints
    ]
    assert all(state.stopped_verified for state in target.list_owned())
    assert manager.rig_lease.current() is None


@pytest.mark.asyncio
async def test_loop_failure_precedes_verified_provider_stop(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, driver, recording_manager, manager = inference_stack
    await recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)
    order: list[str] = []

    def fail_action(_arm_id, _action):
        order.append("driver_failure")
        raise RuntimeError("raw driver detail")

    original_stop = target.stop

    def ordered_stop(handle):
        order.append("provider_stop")
        return original_stop(handle)

    monkeypatch.setattr(driver, "apply_action", fail_action)
    monkeypatch.setattr(target, "stop", ordered_stop)
    with factory() as db:
        await manager.start(
            db,
            deployment_id,
            InferenceStartOptions(arm_id="yam-follower", task="Fail safely"),
        )
    terminal = await _terminal(manager, deployment_id)
    assert order == ["driver_failure", "provider_stop"]
    assert terminal.session_status == "failed"
    assert terminal.deployment.status == "stopped"
    assert terminal.teardown_verified is True
    assert terminal.last_error == "The robot inference loop stopped safely."


@pytest.mark.asyncio
async def test_recording_persistence_failure_still_tears_down_every_resource(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, target, service, _, recording_manager, manager = inference_stack
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    await recording_manager.startup()
    await manager.startup()
    deployment_id = await _deploy(service, factory)

    async def fail_persistence(_recording_id, _result):
        raise RuntimeError("raw database detail")

    monkeypatch.setattr(manager, "_persist_episode_result", fail_persistence)
    with factory() as db:
        started = await manager.start(
            db,
            deployment_id,
            InferenceStartOptions(
                arm_id="yam-follower",
                task="Fail recording persistence safely",
                record_session=True,
                max_steps=3,
            ),
            recording_fps=50,
        )
    recording_id = started.recording.recording_id
    assert recording_id is not None
    terminal = await _terminal(manager, deployment_id)
    assert terminal.session_status == "failed"
    assert terminal.deployment.status == "stopped"
    assert terminal.teardown_verified is True
    assert terminal.recording.status == "failed"
    assert "raw database detail" not in (terminal.last_error or "")
    with factory() as db:
        assert db.get(Recording, recording_id).status == "failed"
    assert await recording_manager.active_resource_counts() == (0, 0)
    assert all(state.stopped_verified for state in target.list_owned())

    episode_directory = (
        recording_manager.staging_dir
        / str(recording_id)
        / "episode_000000"
    )
    assert (episode_directory / EPISODE_MANIFEST_FILENAME).is_file()
    restarted = RecordingManager(
        driver=manager.driver,
        camera=manager.camera,
        staging_dir=recording_manager.staging_dir,
    )
    await restarted.startup()
    await reconcile_recordings_startup(factory, restarted)
    await reconcile_recordings_startup(factory, restarted)
    with factory() as db:
        recovered = db.get(Recording, recording_id)
        assert recovered is not None
        assert recovered.status == "ready"
        assert recovered.episode_count == 1
        assert len(recovered.recording_metadata["episodes"]) == 1


@pytest.mark.asyncio
async def test_inference_post_commit_confirmation_is_cancellation_safe(
    inference_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory, _, _, _, recording_manager, manager = inference_stack
    monkeypatch.setattr("ctrl_pi.recording.FFmpegVideoWriter", _FakeVideoWriter)
    recording_id = uuid.uuid4()
    with factory() as db:
        db.add(
            Recording(
                id=recording_id,
                name="Inference cancellation",
                task="Persist once",
                status="recording",
                episode_count=0,
                duration_seconds=0,
                recording_metadata={"source": "inference", "episodes": []},
            )
        )
        db.commit()

    await recording_manager.startup()
    await recording_manager.start_teleop(
        str(recording_id), "yam-leader", "yam-follower", 0, "draft"
    )
    await recording_manager.start_episode(str(recording_id), fps=10, metadata={})
    _, result = await recording_manager.stop_episode(
        str(recording_id), success=True, notes=None
    )

    entered = asyncio.Event()
    release = asyncio.Event()
    original_confirm = recording_manager.confirm_episode_persisted

    async def blocked_confirm(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_confirm(*args, **kwargs)

    monkeypatch.setattr(recording_manager, "confirm_episode_persisted", blocked_confirm)
    persistence = asyncio.create_task(
        manager._persist_episode_result(recording_id, result)
    )
    await entered.wait()
    persistence.cancel()
    await asyncio.sleep(0)
    assert persistence.done() is False
    release.set()
    await persistence

    with factory() as db:
        stored = db.get(Recording, recording_id)
        assert stored is not None
        assert stored.status == "ready"
        assert stored.episode_count == 1
        assert len(stored.recording_metadata["episodes"]) == 1
    state = await recording_manager.state(
        str(recording_id), "yam-leader", "yam-follower", 1, "ready"
    )
    assert state.episode_active is False
    await recording_manager.stop_teleop(str(recording_id))


def test_rest_list_idle_health_start_stream_stop_and_secret_redaction(
    inference_stack,
) -> None:
    _, factory, _, service, driver, recording_manager, manager = inference_stack
    app = create_app(
        yam_driver=driver,
        recording_manager=recording_manager,
        deployment_service=service,
        inference_session_manager=manager,
    )

    def database() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        created = client.post(
            "/api/inference/deployments",
            json={
                "name": "API mock policy",
                "model_repo": MOCK_MODEL_REPO,
                "checkpoint_revision": MOCK_MODEL_REVISION,
                "runtime": "stub",
                "compute_size": "CPU",
            },
        )
        assert created.status_code == 201, created.text
        deployment_id = created.json()["id"]
        listed = client.get("/api/inference/deployments")
        assert [item["id"] for item in listed.json()["deployments"]] == [
            deployment_id
        ]
        idle = client.get(f"/api/inference/deployments/{deployment_id}/state")
        assert idle.json()["session_status"] == "idle"
        assert idle.json()["endpoint_healthy"] is True

        leaked = "ws-secret-must-not-echo"
        rejected = client.post(
            f"/api/inference/deployments/{deployment_id}/start",
            json={
                "arm_id": "yam-follower",
                "task": "API loop",
                "modal_proxy_token_secret": leaked,
            },
        )
        assert rejected.status_code == 422 and leaked not in rejected.text

        started = client.post(
            f"/api/inference/deployments/{deployment_id}/start",
            json={"arm_id": "yam-follower", "task": "API loop"},
        )
        assert started.status_code == 200, started.text
        assert started.json()["session_status"] == "running"
        with client.websocket_connect(
            f"/api/inference/deployments/{deployment_id}/stream"
        ) as socket:
            frame = socket.receive_json()
            assert frame["type"] == "inference_state"
            assert frame["state"]["id"] == deployment_id
        stopped = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["status"] == "stopped"
        assert stopped.json()["teardown_verified"] is True


def test_mock_lerobot_deploy_requires_sha_and_never_calls_resolver(
    inference_stack,
) -> None:
    _, factory, _, _, _, recording_manager, manager = inference_stack

    class ExplodingResolver:
        def resolve(self, **_kwargs):
            raise AssertionError("mock deploy must not call Hugging Face")

    target = StubComputeTarget()
    service = DeploymentService(
        target,
        model_revision_resolver=ExplodingResolver(),
        session_factory=factory,
        nonce_factory=lambda: "mock-runtime",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    local_manager = InferenceSessionManager(
        deployment_service=service,
        driver=manager.driver,
        camera=manager.camera,
        rig_lease=manager.rig_lease,
        recording_manager=recording_manager,
        transport_factory=manager.transport_factory,
        session_factory=factory,
    )
    app = create_app(
        yam_driver=manager.driver,
        recording_manager=recording_manager,
        deployment_service=service,
        inference_session_manager=local_manager,
    )

    def database() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        accepted = client.post(
            "/api/inference/deployments",
            json={
                "name": "Offline LeRobot",
                "model_repo": MOCK_MODEL_REPO,
                "checkpoint_revision": MOCK_MODEL_REVISION,
                "runtime": "lerobot",
                "compute_size": "Modal: A10G",
            },
        )
        rejected = client.post(
            "/api/inference/deployments",
            json={
                "name": "Mutable mock",
                "model_repo": MOCK_MODEL_REPO,
                "checkpoint_revision": "main",
                "runtime": "lerobot",
                "compute_size": "Modal: A10G",
            },
        )
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["checkpoint_revision"] == MOCK_MODEL_REVISION
    assert rejected.status_code == 503


def test_missing_runtime_proxy_configuration_is_503_and_redacted(
    inference_stack,
) -> None:
    _, factory, _, service, driver, recording_manager, manager = inference_stack
    secret = "ws-never-echo-this-value"

    def unavailable_transport(_record):
        raise ValueError(secret)

    manager.transport_factory = unavailable_transport
    app = create_app(
        yam_driver=driver,
        recording_manager=recording_manager,
        deployment_service=service,
        inference_session_manager=manager,
    )

    def database() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        created = client.post(
            "/api/inference/deployments",
            json={
                "name": "Missing proxy",
                "model_repo": MOCK_MODEL_REPO,
                "checkpoint_revision": MOCK_MODEL_REVISION,
                "runtime": "stub",
                "compute_size": "CPU",
            },
        )
        response = client.post(
            f"/api/inference/deployments/{created.json()['id']}/start",
            json={"arm_id": "yam-follower", "task": "Do not leak credentials"},
        )
    assert response.status_code == 503
    assert secret not in response.text


def test_real_openpi_is_rejected_before_provider_mutation(inference_stack) -> None:
    _, factory, target, service, driver, recording_manager, manager = inference_stack
    app = create_app(
        yam_driver=driver,
        recording_manager=recording_manager,
        deployment_service=service,
        inference_session_manager=manager,
    )

    def database() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    app.dependency_overrides[get_config] = lambda: AppConfig(
        _env_file=None,
        ctrl_pi_mock_mode=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/inference/deployments",
            json={
                "name": "Unavailable OpenPI",
                "model_repo": MOCK_MODEL_REPO,
                "checkpoint_revision": MOCK_MODEL_REVISION,
                "runtime": "openpi",
                "compute_size": "Modal: A10G",
            },
        )
    assert response.status_code == 422
    assert target.list_owned() == []
