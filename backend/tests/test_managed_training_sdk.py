from __future__ import annotations

import json
import uuid

import httpx
import pytest

from ctrl_pi import (
    ConsoleLogPage,
    CtrlPiClient,
    CtrlPiError,
    ManagedTrainingCheckpoints,
    ManagedTrainingJob,
    ManagedTrainingJobPage,
    ManagedTrainingMetrics,
)
from ctrl_pi.trainer import Client as LegacyClient

JOB_ID = "11111111-1111-4111-8111-111111111111"
RUN_ID = "22222222-2222-4222-8222-222222222222"
KEY = "33333333-3333-4333-8333-333333333333"
NOW = "2026-08-29T00:00:00Z"


def _job() -> dict[str, object]:
    return {
        "id": JOB_ID,
        "training_run_id": RUN_ID,
        "idempotency_key": KEY,
        "request_hash": "a" * 64,
        "status": "running",
        "outcome": "pending",
        "target_kind": "modal",
        "provider_state": "running",
        "compute_size": "Modal: 2xA100",
        "runtime": "lerobot",
        "dataset_repo": "acme/data",
        "requested_dataset_revision": "main",
        "dataset_revision": "b" * 40,
        "base_model": "lerobot/act-base",
        "requested_base_model_revision": None,
        "base_model_revision": "c" * 40,
        "output_model_repo": "acme/output",
        "output_private": True,
        "output_marker_revision": "d" * 40,
        "output_revision": None,
        "max_steps": 100,
        "batch_size": 8,
        "log_every": 1,
        "save_every": 10,
        "seed": 42,
        "num_workers": 4,
        "timeout_seconds": 3600,
        "deadline_at": NOW,
        "provider_app_id": "ap-123",
        "provider_function_call_id": "fc-123",
        "last_event_sequence": 2,
        "event_gap": False,
        "launch_attempted_at": NOW,
        "provider_launch_started_at": NOW,
        "started_at": NOW,
        "execution_finished_at": None,
        "cancel_requested_at": None,
        "teardown_verified": False,
        "teardown_verified_at": None,
        "last_error": None,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _response(request: httpx.Request) -> dict[str, object]:
    path = request.url.path
    if path == "/api/trainer/jobs":
        return {"jobs": [_job()], "next_cursor": None} if request.method == "GET" else _job()
    if path.endswith("/logs"):
        return {
            "logs": [],
            "oldest_sequence": None,
            "latest_sequence": None,
            "next_sequence": 0,
            "truncated": False,
            "has_more": False,
        }
    if path.endswith("/metrics"):
        return {
            "job_id": JOB_ID,
            "training_run_id": RUN_ID,
            "current_step": 10,
            "metrics": {"loss": [{"step": 10, "value": 0.5}]},
        }
    if path.endswith("/checkpoints"):
        return {
            "job_id": JOB_ID,
            "training_run_id": RUN_ID,
            "checkpoints": [
                {"repo_id": "acme/output", "revision": "e" * 40, "step": 10}
            ],
        }
    if path.startswith(f"/api/trainer/jobs/{JOB_ID}"):
        return _job()
    raise AssertionError(f"unexpected request {request.method} {path}")


def _run_with_managed_summary(marker_revision: str) -> dict[str, object]:
    return {
        "id": RUN_ID,
        "name": "managed ACT",
        "status": "running",
        "current_step": 10,
        "dataset_repo": "acme/data",
        "base_model": "acme/base",
        "runtime": "lerobot",
        "framework": "lerobot",
        "output_model_repo": "acme/output",
        "checkpoint_revision": None,
        "config": {"managed": True},
        "metrics": {},
        "checkpoints": [],
        "managed_job": {
            "id": JOB_ID,
            "status": "running",
            "outcome": "pending",
            "target_kind": "modal",
            "compute_size": "Modal: 2xA100",
            "deadline_at": NOW,
            "provider_state": "running",
            "teardown_verified": False,
            "output_model_repo": "acme/output",
            "output_marker_revision": marker_revision,
            "output_revision": None,
            "last_error": None,
            "event_gap": False,
        },
        "created_at": NOW,
        "updated_at": NOW,
    }


def test_universal_sdk_managed_training_method_and_route_matrix() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response(request))

    with CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    ) as client:
        launched = client.launch_managed_training(
            "managed ACT",
            idempotency_key=KEY,
            dataset_repo="acme/data",
            dataset_revision="main",
            base_model="lerobot/act-base",
            output_model_repo="acme/output",
            output_private=True,
            acknowledge_public_model_risk=False,
            compute_size="Modal: 2xA100",
            max_steps=100,
            batch_size=8,
            log_every=1,
            save_every=10,
            acknowledge_compute_cost=True,
        )
        assert isinstance(launched, ManagedTrainingJob)
        assert isinstance(client.list_managed_training_jobs(), ManagedTrainingJobPage)
        assert isinstance(client.get_managed_training_job(JOB_ID), ManagedTrainingJob)
        assert isinstance(client.cancel_managed_training_job(JOB_ID), ManagedTrainingJob)
        assert isinstance(client.list_managed_training_logs(JOB_ID), ConsoleLogPage)
        assert isinstance(
            client.get_managed_training_metrics(JOB_ID), ManagedTrainingMetrics
        )
        assert isinstance(
            client.list_managed_training_checkpoints(JOB_ID),
            ManagedTrainingCheckpoints,
        )

    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/trainer/jobs"),
        ("GET", "/api/trainer/jobs"),
        ("GET", f"/api/trainer/jobs/{JOB_ID}"),
        ("POST", f"/api/trainer/jobs/{JOB_ID}/cancel"),
        ("GET", f"/api/trainer/jobs/{JOB_ID}/logs"),
        ("GET", f"/api/trainer/jobs/{JOB_ID}/metrics"),
        ("GET", f"/api/trainer/jobs/{JOB_ID}/checkpoints"),
    ]
    launch_body = json.loads(requests[0].content)
    assert launch_body["idempotency_key"] == KEY
    assert launch_body["acknowledge_compute_cost"] is True
    assert launch_body["output_private"] is True
    assert "token" not in launch_body


def test_universal_sdk_rejects_unsafe_managed_launch_before_transport() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_job())

    client = CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    )
    common = {
        "idempotency_key": KEY,
        "dataset_repo": "acme/data",
        "base_model": "acme/base",
        "output_model_repo": "acme/output",
        "compute_size": "Modal: A10G",
        "max_steps": 100,
        "log_every": 1,
        "save_every": 10,
        "acknowledge_compute_cost": True,
    }
    with pytest.raises(CtrlPiError, match="invalid"):
        client.launch_managed_training("job", **{**common, "idempotency_key": "../bad"})
    with pytest.raises(CtrlPiError, match="invalid"):
        client.launch_managed_training(
            "job",
            **{
                **common,
                "output_private": False,
                "acknowledge_public_model_risk": False,
            },
        )
    with pytest.raises(CtrlPiError, match="invalid"):
        client.launch_managed_training(
            "job", **{**common, "acknowledge_compute_cost": False}
        )
    assert calls == 0
    client.close()


def test_legacy_trainer_client_adds_managed_methods_without_changing_transport() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response(request))

    with LegacyClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    ) as client:
        job = client.launch_managed_training(
            "managed ACT",
            idempotency_key=KEY,
            dataset_repo="acme/data",
            base_model="acme/base",
            output_model_repo="acme/output",
            compute_size="Modal: A10G",
            max_steps=100,
            log_every=1,
            save_every=10,
            acknowledge_compute_cost=True,
        )
        assert job["id"] == JOB_ID
        assert client.list_managed_training_jobs()["jobs"][0]["id"] == JOB_ID
        assert client.get_managed_training_job(JOB_ID)["id"] == JOB_ID
        assert client.cancel_managed_training_job(JOB_ID)["id"] == JOB_ID
        client.list_managed_training_logs(JOB_ID)
        client.get_managed_training_metrics(JOB_ID)
        client.list_managed_training_checkpoints(JOB_ID)

    assert len(requests) == 7
    assert all(KEY not in str(request.url) for request in requests)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_revision", "not-an-immutable-sha"),
        ("base_model_revision", "B" * 40),
        ("output_marker_revision", "d" * 39),
        ("output_revision", "e" * 41),
        ("compute_size", "Modal: Auto"),
        ("provider_app_id", "a" * 256),
        ("provider_function_call_id", "f" * 256),
        ("last_error", "x" * 241),
    ],
)
def test_sdk_rejects_corrupt_managed_lifecycle_identity(
    field: str, value: object
) -> None:
    payload = _job()
    payload[field] = value

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(CtrlPiError, match="invalid response") as caught:
            client.get_managed_training_job(JOB_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_sdk_training_run_accepts_typed_managed_summary_and_rejects_bad_marker() -> None:
    marker = ["d" * 40]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_run_with_managed_summary(marker[0]))

    with CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    ) as client:
        run = client.get_run(RUN_ID)
        assert run.managed_job is not None
        assert run.managed_job.output_marker_revision == "d" * 40

        marker[0] = "not-an-immutable-sha"
        with pytest.raises(CtrlPiError, match="invalid response"):
            client.get_run(RUN_ID)
