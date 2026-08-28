from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Generator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ctrl_pi.compute import (
    ComputeConfigurationError,
    ComputeOwnershipError,
    ComputeTargetError,
    DeploymentHandle,
    DeploymentSpec,
    HealthResult,
    TargetState,
    deployment_app_name,
    deployment_ownership_tag,
)
from ctrl_pi.db import Base, get_db
from ctrl_pi.deployments import (
    DeploymentConflictError,
    DeploymentProviderError,
    DeploymentService,
    DeploymentStorageError,
)
from ctrl_pi.main import create_app
from ctrl_pi.models import AppSetting, Deployment, InferenceEndpoint


class FakeTarget:
    kind = "stub"

    def __init__(self) -> None:
        self.states: dict[uuid.UUID, TargetState] = {}
        self.specs: list[DeploymentSpec] = []
        self.deploy_error: str | None = None
        self.health_error: Exception | None = None
        self.health_echo_matches = True
        self.deploy_without_url = False
        self.stop_failures = 0
        self.stop_leaves_running = False
        self.omit_owned = False
        self.on_deploy = None
        self.deploy_started: threading.Event | None = None
        self.deploy_release: threading.Event | None = None
        self.stop_started: threading.Event | None = None
        self.stop_release: threading.Event | None = None
        self.deploy_calls = 0
        self.health_calls = 0
        self.inspect_calls = 0
        self.stop_calls = 0
        self.list_calls = 0

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        self.deploy_calls += 1
        self.specs.append(spec)
        if self.on_deploy is not None:
            self.on_deploy(spec)
        if self.deploy_started is not None:
            self.deploy_started.set()
        if self.deploy_release is not None:
            assert self.deploy_release.wait(timeout=3)
        if self.deploy_error == "before":
            raise ComputeTargetError("provider secret must not leak")
        handle = self._handle(spec.deployment_id)
        if self.deploy_without_url:
            handle = replace(handle, endpoint_url=None)
        self.states[spec.deployment_id] = self._state(handle, "running", 1)
        if self.deploy_error == "after":
            raise ComputeTargetError("provider failed after app creation")
        if self.deploy_error == "config":
            raise ComputeConfigurationError("raw credential path")
        return handle

    def health(self, handle: DeploymentHandle, nonce: str) -> HealthResult:
        self.health_calls += 1
        if self.health_error is not None:
            raise self.health_error
        return HealthResult(
            healthy=True,
            echo=nonce if self.health_echo_matches else "wrong-nonce",
        )

    def inspect(self, handle: DeploymentHandle) -> TargetState:
        self.inspect_calls += 1
        state = self.states.get(handle.deployment_id)
        if state is None:
            return self._state(handle, "unknown", 0, exists=False)
        if state.provider_app_id != handle.provider_app_id:
            raise ComputeOwnershipError("provider identity mismatch")
        return state

    def stop(self, handle: DeploymentHandle) -> None:
        self.stop_calls += 1
        if self.stop_started is not None:
            self.stop_started.set()
        if self.stop_release is not None:
            assert self.stop_release.wait(timeout=3)
        if self.stop_failures:
            self.stop_failures -= 1
            raise ComputeTargetError("provider stop secret")
        if not self.stop_leaves_running:
            self.states[handle.deployment_id] = self._state(handle, "stopped", 0)

    def list_owned(self) -> list[TargetState]:
        self.list_calls += 1
        return [] if self.omit_owned else list(self.states.values())

    @staticmethod
    def _handle(deployment_id: uuid.UUID) -> DeploymentHandle:
        return DeploymentHandle(
            deployment_id=deployment_id,
            provider_app_id=f"fake-{deployment_id.hex}",
            app_name=deployment_app_name(deployment_id),
            ownership_tag=deployment_ownership_tag(deployment_id),
            endpoint_url=f"https://compute.invalid/{deployment_id}",
        )

    @staticmethod
    def _state(
        handle: DeploymentHandle,
        lifecycle: str,
        tasks: int,
        *,
        exists: bool = True,
    ) -> TargetState:
        return TargetState(
            deployment_id=handle.deployment_id,
            provider_app_id=handle.provider_app_id,
            app_name=handle.app_name,
            ownership_tag=handle.ownership_tag,
            exists=exists,
            lifecycle=lifecycle,
            running_tasks=tasks,
            endpoint_url=handle.endpoint_url,
        )


@pytest.fixture
def deployment_app(tmp_path: Path):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    target = FakeTarget()
    service = DeploymentService(
        target,
        session_factory=factory,
        nonce_factory=lambda: "fixed-deployment-nonce",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )
    app = create_app(deployment_service=service)

    def database() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    return app, engine, factory, target, service


def _deploy(client: TestClient, **overrides):
    payload = {
        "name": "M9 CPU probe",
        "model_repo": "acme/stub-policy",
        "checkpoint_revision": "a" * 40,
        "runtime": "stub",
        "compute_size": "CPU",
    }
    payload.update(overrides)
    return client.post("/api/inference/deployments", json=payload)


def test_api_stub_success_persists_atomic_state_and_idempotent_stop(
    deployment_app,
) -> None:
    app, engine, _, target, _ = deployment_app
    observed_statuses: list[tuple[str, str]] = []

    def observe_deploy(spec: DeploymentSpec) -> None:
        with Session(engine) as db:
            deployment = db.get(Deployment, spec.deployment_id)
            endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
            observed_statuses.append((deployment.status, endpoint.status))

    target.on_deploy = observe_deploy
    with Session(engine) as db:
        db.add(AppSetting(key="modal_timeout_minutes", value=7))
        db.commit()

    with TestClient(app) as client:
        created = _deploy(client)
        deployment_id = created.json()["id"]
        detail = client.get(f"/api/inference/deployments/{deployment_id}")
        stopped = client.post(f"/api/inference/deployments/{deployment_id}/stop")
        stopped_again = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )

    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["status"] == "running"
    assert payload["target_kind"] == "stub"
    assert payload["name"] == "M9 CPU probe"
    assert payload["runtime"] == "stub" and payload["compute_size"] == "CPU"
    assert payload["provider_app_id"].startswith("fake-")
    assert payload["endpoint_url"].startswith("https://compute.invalid/")
    assert payload["started_at"] is not None and payload["stopped_at"] is None
    assert detail.json() == payload
    assert stopped.status_code == stopped_again.status_code == 200
    assert stopped.json()["status"] == stopped_again.json()["status"] == "stopped"
    assert stopped.json()["provider_app_id"] == payload["provider_app_id"]
    assert stopped.json()["stopped_at"] is not None
    assert observed_statuses == [("deploying", "deploying")]
    assert target.specs[0].resources.timeout_seconds == 7 * 60
    assert target.specs[0].resources.min_containers == 0
    assert target.specs[0].resources.buffer_containers == 0
    assert target.specs[0].resources.max_containers == 1
    assert target.specs[0].resources.scaledown_window_seconds <= 60
    assert target.stop_calls == 1
    with Session(engine) as db:
        deployment = db.get(Deployment, uuid.UUID(deployment_id))
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "stopped"


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime": "lerobot"},
        {"compute_size": "Modal: A10G"},
        {"model_repo": "https://huggingface.co/acme/model"},
        {"checkpoint_revision": "../main"},
        {"unexpected_token": "must-not-echo"},
    ],
)
def test_api_validation_is_sanitized(deployment_app, overrides) -> None:
    app, _, _, _, _ = deployment_app
    with TestClient(app) as client:
        response = _deploy(client, **overrides)

    assert response.status_code == 422
    assert "must-not-echo" not in response.text


def test_unknown_and_invalid_transition_statuses(deployment_app) -> None:
    app, engine, _, _, _ = deployment_app
    unknown = uuid.uuid4()
    with TestClient(app) as client:
        missing = client.get(f"/api/inference/deployments/{unknown}")
        created = _deploy(client)
        deployment_id = uuid.UUID(created.json()["id"])
        with Session(engine) as db:
            deployment = db.get(Deployment, deployment_id)
            endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
            deployment.status = endpoint.status = "deploying"
            db.commit()
        conflict = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )

    assert missing.status_code == 404
    assert conflict.status_code == 409


def test_stop_rejects_persisted_target_kind_mismatch_before_mutation(
    deployment_app,
) -> None:
    app, engine, _, target, _ = deployment_app
    with TestClient(app) as client:
        created = _deploy(client)
        deployment_id = uuid.UUID(created.json()["id"])
        with Session(engine) as db:
            deployment = db.get(Deployment, deployment_id)
            deployment.target_kind = "modal"
            db.commit()
        detail = client.get(f"/api/inference/deployments/{deployment_id}")
        stop_calls = target.stop_calls
        rejected = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )

    assert detail.status_code == 200
    assert detail.json()["target_kind"] == "modal"
    assert rejected.status_code == 503
    assert target.stop_calls == stop_calls
    with Session(engine) as db:
        deployment = db.get(Deployment, deployment_id)
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "running"


@pytest.mark.parametrize("failure", ["before", "after"])
def test_deploy_failure_is_sanitized_and_compensates_discovered_resource(
    deployment_app, failure: str
) -> None:
    app, engine, _, target, _ = deployment_app
    target.deploy_error = failure
    with TestClient(app) as client:
        response = _deploy(client)

    assert response.status_code == 502
    assert "secret" not in response.text
    with Session(engine) as db:
        deployment = db.scalar(select(Deployment))
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "failed"
        if failure == "before":
            assert endpoint.provider_app_id is None
            assert target.stop_calls == 0
        else:
            assert endpoint.provider_app_id is not None
            assert target.stop_calls == 1
            assert target.states[deployment.id].stopped_verified is True


def test_health_nonce_failure_never_marks_running_and_cleans_up(deployment_app) -> None:
    app, engine, _, target, _ = deployment_app
    target.health_echo_matches = False
    with TestClient(app) as client:
        response = _deploy(client)

    assert response.status_code == 502
    with Session(engine) as db:
        deployment = db.scalar(select(Deployment))
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "failed"
        assert deployment.started_at is None
        assert endpoint.provider_app_id is not None
    assert target.stop_calls == 1


def test_provider_identity_is_retained_when_endpoint_url_was_never_published(
    deployment_app,
) -> None:
    app, engine, _, target, _ = deployment_app
    target.deploy_without_url = True
    with TestClient(app) as client:
        response = _deploy(client)

    assert response.status_code == 502
    assert target.health_calls == 0
    assert target.stop_calls == 1
    with Session(engine) as db:
        deployment = db.scalar(select(Deployment))
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "failed"
        assert endpoint.provider_app_id == f"fake-{deployment.id.hex}"
        assert endpoint.endpoint_url is None
        assert target.states[deployment.id].stopped_verified is True


def test_compute_configuration_failure_is_503_and_cleans_ambiguous_resource(
    deployment_app,
) -> None:
    app, engine, _, target, _ = deployment_app
    target.deploy_error = "config"
    with TestClient(app) as client:
        response = _deploy(client)

    assert response.status_code == 503
    assert "credential" not in response.text
    assert target.stop_calls == 1
    with Session(engine) as db:
        deployment = db.scalar(select(Deployment))
        assert deployment.status == "failed"


def test_initial_database_flush_failure_is_sanitized_before_provider_call(
    deployment_app, monkeypatch
) -> None:
    app, _, _, target, _ = deployment_app

    def fail_flush(self, *args, **kwargs):
        raise SQLAlchemyError("postgresql://secret-host/private")

    with TestClient(app) as client:
        monkeypatch.setattr(Session, "flush", fail_flush)
        response = _deploy(client)

    assert response.status_code == 503
    assert "secret-host" not in response.text
    assert target.deploy_calls == 0


def test_db_failure_after_handle_runs_compensating_teardown(
    deployment_app, monkeypatch
) -> None:
    app, engine, _, target, service = deployment_app

    def fail_persist(db, deployment_id, handle):
        raise DeploymentStorageError("safe database failure")

    monkeypatch.setattr(service, "_persist_handle", fail_persist)
    with TestClient(app) as client:
        response = _deploy(client)

    assert response.status_code == 503
    assert target.stop_calls == 1
    with Session(engine) as db:
        deployment = db.scalar(select(Deployment))
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "failed"
        assert endpoint.provider_app_id is not None


def test_stop_failure_is_retryable_and_verification_timeout_never_lies(
    deployment_app,
) -> None:
    app, engine, _, target, _ = deployment_app
    with TestClient(app) as client:
        deployment_id = _deploy(client).json()["id"]
        target.stop_failures = 1
        failed_stop = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )
        retried = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )

        second_id = _deploy(client, name="timeout probe").json()["id"]
        target.stop_leaves_running = True
        timeout = client.post(f"/api/inference/deployments/{second_id}/stop")

    assert failed_stop.status_code == 502
    assert retried.status_code == 200 and retried.json()["status"] == "stopped"
    assert timeout.status_code == 502
    with Session(engine) as db:
        timed_out = db.get(Deployment, uuid.UUID(second_id))
        endpoint = db.get(InferenceEndpoint, timed_out.endpoint_id)
        assert timed_out.status == endpoint.status == "failed"
        assert timed_out.stopped_at is None


def test_stop_retries_from_stopping_after_terminal_db_failure(
    deployment_app, monkeypatch
) -> None:
    app, engine, _, _, service = deployment_app
    with TestClient(app) as client:
        deployment_id = _deploy(client).json()["id"]
        original = service._set_stopped
        calls = 0

        def fail_once(db, target_id, *, handle=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise DeploymentStorageError("terminal commit failed")
            return original(db, target_id, handle=handle)

        monkeypatch.setattr(service, "_set_stopped", fail_once)
        failed = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )
        retried = client.post(
            f"/api/inference/deployments/{deployment_id}/stop"
        )

    assert failed.status_code == 503
    assert retried.status_code == 200
    assert retried.json()["status"] == "stopped"
    with Session(engine) as db:
        assert db.get(Deployment, uuid.UUID(deployment_id)).status == "stopped"


@pytest.mark.asyncio
async def test_stop_cleans_resource_that_appears_after_first_enumeration(
    deployment_app, monkeypatch
) -> None:
    _, engine, factory, target, service = deployment_app
    deployment_id = uuid.uuid4()
    handle = target._handle(deployment_id)
    target.states[deployment_id] = target._state(handle, "running", 1)
    with Session(engine) as db:
        endpoint = InferenceEndpoint(
            name="late provider resource",
            runtime="stub",
            status="failed",
        )
        db.add(endpoint)
        db.flush()
        db.add(
            Deployment(
                id=deployment_id,
                endpoint_id=endpoint.id,
                model_repo="acme/model",
                runtime="stub",
                compute_size="CPU",
                status="failed",
            )
        )
        db.commit()

    list_calls = 0

    def delayed_owned_resource() -> list[TargetState]:
        nonlocal list_calls
        list_calls += 1
        if list_calls == 1:
            return []
        return list(target.states.values())

    monkeypatch.setattr(target, "list_owned", delayed_owned_resource)
    with factory() as db:
        stopped = await service.stop(db, deployment_id)

    assert list_calls == 2
    assert target.stop_calls == 1
    assert stopped.status == "stopped"
    assert target.states[deployment_id].stopped_verified is True


@pytest.mark.asyncio
async def test_deploy_and_stop_cancellation_finish_cleanup(
    deployment_app,
) -> None:
    _, engine, factory, target, service = deployment_app
    target.deploy_started = threading.Event()
    target.deploy_release = threading.Event()
    with factory() as db:
        task = asyncio.create_task(
            service.deploy(
                db,
                name="cancelled deploy",
                model_repo="acme/model",
                checkpoint_revision=None,
                runtime="stub",
                compute_size="CPU",
                timeout_seconds=60,
            )
        )
        assert await asyncio.to_thread(target.deploy_started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        target.deploy_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    with Session(engine) as db:
        deployment = db.scalar(select(Deployment))
        assert deployment.status == "stopped"
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert endpoint.provider_app_id is not None
        cancelled_state = target.states[deployment.id]
        assert cancelled_state.stopped_verified is True
        assert cancelled_state.running_tasks == 0
        assert not [
            state
            for state in target.list_owned()
            if state.deployment_id == deployment.id and not state.stopped_verified
        ]

    target.stop_started = threading.Event()
    target.stop_release = threading.Event()
    # Start another deployment without a deploy block.
    target.deploy_started = None
    target.deploy_release = None
    with factory() as db:
        record = await service.deploy(
            db,
            name="cancelled stop",
            model_repo="acme/model",
            checkpoint_revision=None,
            runtime="stub",
            compute_size="CPU",
            timeout_seconds=60,
        )
    with factory() as db:
        stop_task = asyncio.create_task(service.stop(db, record.id))
        assert await asyncio.to_thread(target.stop_started.wait, 2)
        stop_task.cancel()
        await asyncio.sleep(0)
        target.stop_release.set()
        with pytest.raises(asyncio.CancelledError):
            await stop_task
    with Session(engine) as db:
        assert db.get(Deployment, record.id).status == "stopped"


@pytest.mark.asyncio
async def test_concurrent_stop_returns_conflict_without_blocking_first_stop(
    deployment_app,
) -> None:
    _, _, factory, target, service = deployment_app
    with factory() as db:
        record = await service.deploy(
            db,
            name="concurrent stop",
            model_repo="acme/model",
            checkpoint_revision=None,
            runtime="stub",
            compute_size="CPU",
            timeout_seconds=60,
        )
    target.stop_started = threading.Event()
    target.stop_release = threading.Event()
    with factory() as first_db, factory() as second_db:
        first = asyncio.create_task(service.stop(first_db, record.id))
        assert await asyncio.to_thread(target.stop_started.wait, 2)
        with pytest.raises(DeploymentConflictError):
            await service.stop(second_db, record.id)
        target.stop_release.set()
        stopped = await first

    assert stopped.status == "stopped"
    assert target.stop_calls == 1


@pytest.mark.asyncio
async def test_startup_reconciliation_inspects_persisted_handle_despite_list_omission(
    deployment_app,
) -> None:
    _, engine, factory, target, service = deployment_app
    deployment_id = uuid.uuid4()
    handle = target._handle(deployment_id)
    target.states[deployment_id] = target._state(handle, "running", 1)
    target.omit_owned = True
    with Session(engine) as db:
        endpoint = InferenceEndpoint(
            name="interrupted",
            runtime="stub",
            status="running",
            endpoint_url=handle.endpoint_url,
            provider_app_id=handle.provider_app_id,
        )
        db.add(endpoint)
        db.flush()
        db.add(
            Deployment(
                id=deployment_id,
                endpoint_id=endpoint.id,
                model_repo="acme/model",
                runtime="stub",
                compute_size="CPU",
                status="running",
            )
        )
        db.commit()

    await service.reconcile_startup()

    assert target.inspect_calls >= 1
    assert target.health_calls >= 1
    with Session(engine) as db:
        deployment = db.get(Deployment, deployment_id)
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "running"
        assert endpoint.last_heartbeat_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial", "expected"),
    [("deploying", "failed"), ("running", "failed"), ("stopping", "stopped")],
)
async def test_startup_requires_two_authoritative_absence_checks(
    deployment_app, initial: str, expected: str
) -> None:
    _, engine, _, target, service = deployment_app
    deployment_id = uuid.uuid4()
    target.omit_owned = True
    with Session(engine) as db:
        endpoint = InferenceEndpoint(
            name=f"missing {initial}", runtime="stub", status=initial
        )
        db.add(endpoint)
        db.flush()
        db.add(
            Deployment(
                id=deployment_id,
                endpoint_id=endpoint.id,
                model_repo="acme/model",
                runtime="stub",
                compute_size="CPU",
                status=initial,
            )
        )
        db.commit()

    await service.reconcile_startup()

    assert target.list_calls == 2
    with Session(engine) as db:
        deployment = db.get(Deployment, deployment_id)
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial", "expected"),
    [("stopping", "stopped"), ("running", "failed")],
)
async def test_reconcile_accepts_proven_stopped_state_without_endpoint_url(
    deployment_app, initial: str, expected: str
) -> None:
    _, engine, _, target, service = deployment_app
    deployment_id = uuid.uuid4()
    handle = target._handle(deployment_id)
    target.states[deployment_id] = replace(
        target._state(handle, "stopped", 0), endpoint_url=None
    )
    with Session(engine) as db:
        endpoint = InferenceEndpoint(
            name="stopped history", runtime="stub", status=initial
        )
        db.add(endpoint)
        db.flush()
        db.add(
            Deployment(
                id=deployment_id,
                endpoint_id=endpoint.id,
                model_repo="acme/model",
                runtime="stub",
                compute_size="CPU",
                status=initial,
            )
        )
        db.commit()

    await service.reconcile_startup()

    with Session(engine) as db:
        deployment = db.get(Deployment, deployment_id)
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == expected
        assert endpoint.provider_app_id == handle.provider_app_id
        assert endpoint.endpoint_url is None


@pytest.mark.asyncio
async def test_startup_cleans_late_visible_resource_for_failed_deployment(
    deployment_app,
) -> None:
    _, engine, _, target, service = deployment_app
    deployment_id = uuid.uuid4()
    handle = target._handle(deployment_id)
    target.states[deployment_id] = target._state(handle, "running", 1)
    with Session(engine) as db:
        endpoint = InferenceEndpoint(
            name="late orphan", runtime="stub", status="failed"
        )
        db.add(endpoint)
        db.flush()
        db.add(
            Deployment(
                id=deployment_id,
                endpoint_id=endpoint.id,
                model_repo="acme/model",
                runtime="stub",
                compute_size="CPU",
                status="failed",
            )
        )
        db.commit()

    await service.reconcile_startup()

    assert target.stop_calls == 1
    assert target.states[deployment_id].stopped_verified is True
    with Session(engine) as db:
        deployment = db.get(Deployment, deployment_id)
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "failed"
        assert endpoint.provider_app_id == handle.provider_app_id


@pytest.mark.asyncio
async def test_startup_leaves_other_target_kind_resources_untouched(
    deployment_app,
) -> None:
    _, engine, _, target, service = deployment_app
    deployment_id = uuid.uuid4()
    handle = target._handle(deployment_id)
    target.states[deployment_id] = target._state(handle, "running", 1)
    with Session(engine) as db:
        endpoint = InferenceEndpoint(
            name="modal-owned", runtime="stub", status="running"
        )
        db.add(endpoint)
        db.flush()
        db.add(
            Deployment(
                id=deployment_id,
                endpoint_id=endpoint.id,
                model_repo="acme/model",
                runtime="stub",
                compute_size="CPU",
                target_kind="modal",
                status="running",
            )
        )
        db.commit()

    await service.reconcile_startup()

    assert target.inspect_calls == 0
    assert target.stop_calls == 0
    with Session(engine) as db:
        deployment = db.get(Deployment, deployment_id)
        endpoint = db.get(InferenceEndpoint, deployment.endpoint_id)
        assert deployment.status == endpoint.status == "running"


def test_deployment_target_kind_has_database_default_and_constraint() -> None:
    column = Deployment.__table__.c.target_kind

    assert column.default is not None
    assert column.server_default is not None
    assert any(
        constraint.name == "ck_deployments_target_kind"
        for constraint in Deployment.__table__.constraints
    )
