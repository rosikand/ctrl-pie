from __future__ import annotations

import json
import math
from collections.abc import Generator
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ctrl_pi.db import Base, get_db
from ctrl_pi.main import create_app
from ctrl_pi.models import TrainingRun
import ctrl_pi.api.trainer as trainer_api


@pytest.fixture
def trainer_app():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    app = create_app()

    def database() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = database
    return app, engine


def _create(client: TestClient, **overrides):
    payload = {
        "name": "ACT block pickup",
        "dataset_repo": "acme/block-pickup",
        "base_model": "lerobot/act_aloha_sim_transfer_cube_human",
        "runtime": "lerobot",
        "framework": "pytorch",
        "config": {"batch_size": 32, "optimizer": {"lr": 0.0001}},
    }
    payload.update(overrides)
    return client.post(
        "/api/trainer/runs",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_run_crud_filter_and_serialization(trainer_app) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        created = _create(client)
        second = _create(client, name="SmolVLA baseline", status="running")
        fetched = client.get(f"/api/trainer/runs/{created.json()['id']}")
        running = client.get("/api/trainer/runs?status=running")
        updated = client.patch(
            f"/api/trainer/runs/{created.json()['id']}",
            json={
                "status": "running",
                "current_step": 4,
                "runtime": "LeRobot",
                "config": {"batch_size": 64},
            },
        )
        all_runs = client.get("/api/trainer/runs")

    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["status"] == "created"
    assert payload["current_step"] == 0
    assert payload["metrics"] == {}
    assert payload["checkpoints"] == []
    assert payload["output_model_repo"] is None
    assert payload["checkpoint_revision"] is None
    assert payload["created_at"] and payload["updated_at"]
    assert fetched.json() == payload
    assert fetched.headers["cache-control"] == "private, no-store"
    assert running.status_code == 200
    assert [run["id"] for run in running.json()["runs"]] == [second.json()["id"]]
    assert updated.status_code == 200, updated.text
    assert updated.json()["status"] == "running"
    assert updated.json()["current_step"] == 4
    assert updated.json()["runtime"] == "LeRobot"
    assert updated.json()["config"] == {"batch_size": 64}
    assert {run["id"] for run in all_runs.json()["runs"]} == {
        created.json()["id"],
        second.json()["id"],
    }


def test_metrics_replace_same_step_and_current_step_never_regresses(
    trainer_app,
) -> None:
    app, engine = trainer_app
    with TestClient(app) as client:
        run_id = _create(client).json()["id"]
        first = client.post(
            f"/api/trainer/runs/{run_id}/metrics",
            json={"step": 10, "metrics": {"train/loss": 0.5, "lr": 0.001}},
        )
        replacement = client.post(
            f"/api/trainer/runs/{run_id}/metrics",
            json={"step": 10, "metrics": {"train/loss": 0.4}},
        )
        older = client.post(
            f"/api/trainer/runs/{run_id}/metrics",
            json={"step": 5, "metrics": {"eval.loss": 0.6}},
        )

    assert first.status_code == replacement.status_code == older.status_code == 200
    payload = older.json()
    assert payload["current_step"] == 10
    assert payload["metrics"] == {
        "train/loss": [{"step": 10, "value": 0.4}],
        "lr": [{"step": 10, "value": 0.001}],
        "eval.loss": [{"step": 5, "value": 0.6}],
    }
    with Session(engine) as db:
        stored = db.scalar(select(TrainingRun))
        assert stored is not None
        assert stored.current_step == 10
        assert stored.metrics["train/loss"] == [{"step": 10, "value": 0.4}]


def test_checkpoint_registration_is_idempotent_and_advances_run(trainer_app) -> None:
    app, _ = trainer_app
    revision = "a" * 40
    with TestClient(app) as client:
        run_id = _create(client).json()["id"]
        first = client.post(
            f"/api/trainer/runs/{run_id}/checkpoints",
            json={"repo_id": "acme/act-block", "revision": revision, "step": 100},
        )
        duplicate = client.post(
            f"/api/trainer/runs/{run_id}/checkpoints",
            json={"repo_id": "acme/act-block", "revision": revision, "step": 100},
        )

    assert first.status_code == duplicate.status_code == 200
    payload = duplicate.json()
    assert payload["current_step"] == 100
    assert payload["output_model_repo"] == "acme/act-block"
    assert payload["checkpoint_revision"] == revision
    assert payload["checkpoints"] == [
        {"repo_id": "acme/act-block", "revision": revision, "step": 100}
    ]


def test_console_logs_are_sequenced_timestamped_and_advance_run(
    trainer_app,
) -> None:
    app, engine = trainer_app
    with TestClient(app) as client:
        run_id = _create(client, status="running", current_step=4).json()["id"]
        first = client.post(
            f"/api/trainer/runs/{run_id}/logs",
            json={"source": "stdout", "line": "step 8: loss=0.42", "step": 8},
        )
        second = client.post(
            f"/api/trainer/runs/{run_id}/logs",
            json={"source": "stderr", "line": "checkpoint upload pending"},
        )
        logs_response = client.get(f"/api/trainer/runs/{run_id}/logs")
        run_response = client.get(f"/api/trainer/runs/{run_id}")

    assert first.status_code == second.status_code == 201
    assert logs_response.status_code == run_response.status_code == 200
    assert logs_response.headers["cache-control"] == "private, no-store"
    assert run_response.json()["current_step"] == 8
    payload = logs_response.json()
    assert payload["oldest_sequence"] == 1
    assert payload["latest_sequence"] == payload["next_sequence"] == 2
    assert payload["truncated"] is payload["has_more"] is False
    assert [item["sequence"] for item in payload["logs"]] == [1, 2]
    assert payload["logs"][0] == {
        "sequence": 1,
        "source": "stdout",
        "line": "step 8: loss=0.42",
        "step": 8,
        "timestamp": payload["logs"][0]["timestamp"],
    }
    assert payload["logs"][1]["source"] == "stderr"
    assert payload["logs"][1]["step"] is None
    for item in payload["logs"]:
        timestamp = datetime.fromisoformat(item["timestamp"])
        assert timestamp.utcoffset() is not None
    with Session(engine) as db:
        stored = db.scalar(select(TrainingRun))
        assert stored is not None
        assert [item["sequence"] for item in stored.console_logs] == [1, 2]


def test_console_log_retention_is_bounded_and_sequences_do_not_reset(
    trainer_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, engine = trainer_app
    monkeypatch.setattr(trainer_api, "_MAX_CONSOLE_LOGS", 3)
    monkeypatch.setattr(trainer_api, "_MAX_CONSOLE_LOG_BYTES", 430)
    with TestClient(app) as client:
        run_id = _create(client).json()["id"]
        for index in range(1, 7):
            response = client.post(
                f"/api/trainer/runs/{run_id}/logs",
                json={
                    "source": "system",
                    "line": f"event {index}: " + ("x" * 80),
                    "step": index,
                },
            )
            assert response.status_code == 201, response.text
        latest = client.get(
            f"/api/trainer/runs/{run_id}/logs", params={"limit": 2}
        )
        stale_cursor = client.get(
            f"/api/trainer/runs/{run_id}/logs",
            params={"after_sequence": 0, "limit": 1},
        )
        next_cursor = client.get(
            f"/api/trainer/runs/{run_id}/logs",
            params={"after_sequence": stale_cursor.json()["next_sequence"]},
        )

    assert latest.status_code == stale_cursor.status_code == next_cursor.status_code == 200
    logs = latest.json()["logs"]
    assert [item["sequence"] for item in logs] == [5, 6]
    assert latest.json()["truncated"] is True
    assert latest.json()["has_more"] is False
    assert stale_cursor.json()["truncated"] is True
    assert stale_cursor.json()["has_more"] is True
    assert all(
        item["sequence"] > stale_cursor.json()["next_sequence"]
        for item in next_cursor.json()["logs"]
    )
    assert next_cursor.json()["logs"][-1]["sequence"] == 6
    assert logs[-1]["sequence"] == 6
    assert logs == sorted(logs, key=lambda item: item["sequence"])
    with Session(engine) as db:
        stored = db.scalar(select(TrainingRun))
        assert stored is not None
        assert len(stored.console_logs) <= 3
        assert len(
            json.dumps(stored.console_logs, separators=(",", ":")).encode("utf-8")
        ) <= 430
        assert [item["sequence"] for item in stored.console_logs][-2:] == [5, 6]


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "worker", "line": "hello"},
        {"source": "stdout", "line": ""},
        {"source": "stdout", "line": "first\nsecond"},
        {"source": "stdout", "line": "ansi:\x1b[31m"},
        {"source": "stdout", "line": "💥" * 2_000},
        {"source": "stdout", "line": "x", "step": -1},
        {"source": "stdout", "line": "x", "step": True},
        {"source": "stdout", "line": "x", "timestamp": "2026-08-29T00:00:00Z"},
    ],
)
def test_console_log_fields_are_strict(trainer_app, payload: dict) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        run_id = _create(client).json()["id"]
        response = client.post(f"/api/trainer/runs/{run_id}/logs", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "line",
    [
        "HF_TOKEN=hf_must_not_echo_123456",
        "Authorization: Bearer super_secret_value",
        "connecting to postgresql://user:password@db.internal/app",
        "Modal credential wk-super_secret_value",
        "WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567",
        "OPENAI_API_KEY=sk-proj-0123456789abcdef",
        "AWS_SECRET_ACCESS_KEY=0123456789abcdef0123456789abcdef01234567",
        "wandbApiKey=0123456789abcdef0123456789abcdef01234567",
        "env: WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567",
    ],
)
def test_console_logs_reject_secret_like_lines_without_echo(
    trainer_app, line: str
) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        run_id = _create(client).json()["id"]
        response = client.post(
            f"/api/trainer/runs/{run_id}/logs",
            json={"source": "stdout", "line": line},
        )

    assert response.status_code == 422
    assert line not in response.text


@pytest.mark.parametrize(
    "line",
    [
        "num_tokens=4096",
        "tokenizerName=lerobot/tokenizer",
        "api_key_length=32",
        "secretary=alice",
        "monkey=banana",
    ],
)
def test_console_logs_allow_nonsecret_training_assignments(
    trainer_app, line: str
) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        run_id = _create(client).json()["id"]
        response = client.post(
            f"/api/trainer/runs/{run_id}/logs",
            json={"source": "system", "line": line},
        )

    assert response.status_code == 201, response.text
    assert response.json()["line"] == line


def test_step_conflict_and_external_status_resume_are_supported(trainer_app) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        run_id = _create(client, status="running", current_step=20).json()["id"]
        regression = client.patch(
            f"/api/trainer/runs/{run_id}", json={"current_step": 19}
        )
        completed = client.patch(
            f"/api/trainer/runs/{run_id}", json={"status": "completed"}
        )
        late_metric = client.post(
            f"/api/trainer/runs/{run_id}/metrics",
            json={"step": 21, "metrics": {"loss": 0.1}},
        )
        restart = client.patch(
            f"/api/trainer/runs/{run_id}", json={"status": "running"}
        )
        resumed_metadata = client.patch(
            f"/api/trainer/runs/{run_id}", json={"framework": "jax"}
        )

    assert regression.status_code == 409
    assert completed.status_code == 200
    assert late_metric.status_code == restart.status_code == 200
    assert resumed_metadata.status_code == 200
    assert resumed_metadata.json()["framework"] == "jax"


@pytest.mark.parametrize(
    ("config", "secret_value"),
    [
        ({"hf_token": "hf_literal_1"}, "hf_literal_1"),
        ({"HFToken": "hf_literal_2"}, "hf_literal_2"),
        ({"nested": {"accessToken": "hf_literal_3"}}, "hf_literal_3"),
        ({"clientSecret": "client_literal"}, "client_literal"),
        ({"privateKey": "private_literal"}, "private_literal"),
        ({"DATABASE_URL": "postgresql://db_literal"}, "db_literal"),
        ({"hubToken": "hub_literal"}, "hub_literal"),
        ({"githubToken": "github_literal"}, "github_literal"),
        ({"secretValue": "secret_literal"}, "secret_literal"),
        ({"passwordValue": "password_literal"}, "password_literal"),
        ({"credentialValue": "credential_literal"}, "credential_literal"),
        ({"aws_access_key_id": "access_literal"}, "access_literal"),
        ({"postgres_dsn": "dsn_literal"}, "dsn_literal"),
        ({"tokenizerToken": "tokenizer_literal"}, "tokenizer_literal"),
        ({"tokenizerAuthToken": "auth_literal"}, "auth_literal"),
        ({"hf_auth": "hf_auth_literal"}, "hf_auth_literal"),
        ({"bearer": "bearer_literal"}, "bearer_literal"),
        ({"hub_key": "hub_key_literal"}, "hub_key_literal"),
        ({"oauth": "oauth_literal"}, "oauth_literal"),
        ({"session": "session_literal"}, "session_literal"),
        ({"hfAuth": "hf_camel_literal"}, "hf_camel_literal"),
        ({"HFAuth": "hf_acronym_literal"}, "hf_acronym_literal"),
        ({"hubKey": "hub_camel_literal"}, "hub_camel_literal"),
        ({"signingKey": "signing_literal"}, "signing_literal"),
        ({"keyId": "key_id_literal"}, "key_id_literal"),
        ({"sessionCookie": "cookie_literal"}, "cookie_literal"),
        ({"loss": math.inf}, "Infinity"),
    ],
)
def test_config_rejects_secrets_and_nonfinite_values(
    trainer_app, config, secret_value: str
) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        response = _create(client, config=config)

    assert response.status_code == 422
    assert secret_value not in response.text


def test_config_allows_nonsecret_token_training_vocabulary(trainer_app) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        response = _create(
            client,
            config={
                "tokenizer_name": "lerobot/tokenizer",
                "num_tokens": 1024,
                "max_new_tokens": 32,
                "image_key": "observation.images.mock",
                "imageKey": "observation.images.mock",
                "monkey": "benign",
            },
        )

    assert response.status_code == 201, response.text


def test_validation_errors_do_not_echo_forbidden_top_level_secrets(
    trainer_app,
) -> None:
    app, _ = trainer_app
    with TestClient(app) as client:
        response = client.post(
            "/api/trainer/runs",
            json={"name": "unsafe", "hf_token": "hf_must_not_echo"},
        )

    assert response.status_code == 422
    assert "hf_must_not_echo" not in response.text
    assert set(response.json()["detail"][0]) == {"loc", "msg", "type"}


def test_input_validation_missing_runs_and_empty_patch(trainer_app) -> None:
    app, _ = trainer_app
    unknown = "00000000-0000-0000-0000-000000000000"
    with TestClient(app) as client:
        bad_metric = client.post(
            f"/api/trainer/runs/{unknown}/metrics",
            content='{"step":0,"metrics":{"loss":NaN}}',
            headers={"Content-Type": "application/json"},
        )
        bad_repo = _create(client, dataset_repo="https://huggingface.co/acme/data")
        missing = client.get(f"/api/trainer/runs/{unknown}")
        created = _create(client)
        empty_patch = client.patch(
            f"/api/trainer/runs/{created.json()['id']}", json={}
        )
        invalid_status = client.get("/api/trainer/runs?status=unknown")

    assert bad_metric.status_code == bad_repo.status_code == 422
    assert missing.status_code == 404
    assert empty_patch.status_code == 422
    assert invalid_status.status_code == 422


def test_training_run_json_columns_have_callable_and_database_defaults() -> None:
    metrics = TrainingRun.__table__.c.metrics
    checkpoints = TrainingRun.__table__.c.checkpoints
    console_logs = TrainingRun.__table__.c.console_logs

    assert metrics.default is not None and metrics.default.is_callable
    assert checkpoints.default is not None and checkpoints.default.is_callable
    assert console_logs.default is not None and console_logs.default.is_callable
    assert metrics.server_default is not None
    assert checkpoints.server_default is not None
    assert console_logs.server_default is not None
