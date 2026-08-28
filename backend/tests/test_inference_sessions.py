from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ctrl_pi.camera import CameraFrame, MockCamera
from ctrl_pi.compute_stub import StubComputeTarget
from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.db import Base, get_db
from ctrl_pi.deployments import DeploymentService
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
    InferenceStartOptions,
    InferenceStopOptions,
)
from ctrl_pi.inference_transport import InProcessInferenceTransport
from ctrl_pi.main import create_app
from ctrl_pi.models import Deployment, Recording
from ctrl_pi.recording import RecordingManager
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
