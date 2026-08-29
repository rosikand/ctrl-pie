from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Literal, NotRequired, TypedDict
import uuid

import httpx

from ctrl_pi._http import SafeHttpClient

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
ConsoleLogSource = Literal["stdout", "stderr", "system"]
ManagedTrainingComputeSize = Literal[
    "Modal: A10G",
    "Modal: A100",
    "Modal: 2xA100",
    "Modal: 4xA100",
    "Modal: 8xA100",
    "Modal: H100",
    "Modal: 2xH100",
    "Modal: 4xH100",
    "Modal: 8xH100",
]


class MetricPoint(TypedDict):
    step: int
    value: float


class Checkpoint(TypedDict):
    repo_id: str
    revision: str
    step: int


class ConsoleLog(TypedDict):
    sequence: int
    source: ConsoleLogSource
    line: str
    step: int | None
    timestamp: str


class ManagedTrainingJobSummary(TypedDict):
    id: str
    status: str
    outcome: str
    target_kind: Literal["stub", "modal"]
    compute_size: ManagedTrainingComputeSize
    deadline_at: str
    provider_state: str
    teardown_verified: bool
    output_model_repo: str
    output_marker_revision: str | None
    output_revision: str | None
    last_error: str | None
    event_gap: bool


class TrainingRun(TypedDict):
    id: str
    name: str
    status: RunStatus
    current_step: int
    dataset_repo: str | None
    base_model: str | None
    runtime: str | None
    framework: str | None
    output_model_repo: str | None
    checkpoint_revision: str | None
    config: dict[str, Any]
    metrics: dict[str, list[MetricPoint]]
    checkpoints: list[Checkpoint]
    managed_job: ManagedTrainingJobSummary | None
    created_at: str
    updated_at: str


class TrainingRunCreate(TypedDict):
    name: str
    status: NotRequired[RunStatus]
    current_step: NotRequired[int]
    dataset_repo: NotRequired[str | None]
    base_model: NotRequired[str | None]
    runtime: NotRequired[str | None]
    framework: NotRequired[str | None]
    output_model_repo: NotRequired[str | None]
    checkpoint_revision: NotRequired[str | None]
    config: NotRequired[dict[str, Any]]


class ManagedTrainingJob(TypedDict):
    id: str
    training_run_id: str
    idempotency_key: str
    request_hash: str
    status: str
    outcome: str
    target_kind: Literal["stub", "modal"]
    provider_state: str
    compute_size: ManagedTrainingComputeSize
    runtime: Literal["lerobot"]
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
    deadline_at: str
    provider_app_id: str | None
    provider_function_call_id: str | None
    last_event_sequence: int
    event_gap: bool
    launch_attempted_at: str | None
    provider_launch_started_at: str | None
    started_at: str | None
    execution_finished_at: str | None
    cancel_requested_at: str | None
    teardown_verified: bool
    teardown_verified_at: str | None
    last_error: str | None
    created_at: str
    updated_at: str


class ManagedTrainingJobPage(TypedDict):
    jobs: list[ManagedTrainingJob]
    next_cursor: str | None


class ManagedTrainingMetrics(TypedDict):
    job_id: str
    training_run_id: str
    current_step: int
    metrics: dict[str, list[MetricPoint]]


class ManagedTrainingCheckpoints(TypedDict):
    job_id: str
    training_run_id: str
    checkpoints: list[Checkpoint]


class TrainerClientError(RuntimeError):
    """A sanitized ctrl-pi API or transport error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_UNSET = object()


class Client:
    """Small synchronous client for training scripts to report progress."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = SafeHttpClient(
            base_url,
            timeout=timeout,
            error_type=TrainerClientError,
            _transport=_transport,
        )

    @property
    def is_closed(self) -> bool:
        return self._http.is_closed

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def create_run(
        self,
        name: str,
        *,
        status: RunStatus = "created",
        current_step: int = 0,
        dataset_repo: str | None = None,
        base_model: str | None = None,
        runtime: str | None = None,
        framework: str | None = None,
        output_model_repo: str | None = None,
        checkpoint_revision: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingRun:
        payload: TrainingRunCreate = {
            "name": name,
            "status": status,
            "current_step": current_step,
            "dataset_repo": dataset_repo,
            "base_model": base_model,
            "runtime": runtime,
            "framework": framework,
            "output_model_repo": output_model_repo,
            "checkpoint_revision": checkpoint_revision,
            "config": {} if config is None else config,
        }
        return self._request("POST", "/api/trainer/runs", json=payload)

    def list_runs(self, *, status: RunStatus | None = None) -> list[TrainingRun]:
        params = {} if status is None else {"status": status}
        payload = self._request("GET", "/api/trainer/runs", params=params)
        runs = payload.get("runs")
        if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
            raise TrainerClientError("ctrl-pi returned an invalid response.")
        return runs

    def get_run(self, run_id: str) -> TrainingRun:
        return self._request("GET", f"/api/trainer/runs/{self._run_id(run_id)}")

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | object = _UNSET,
        current_step: int | object = _UNSET,
        dataset_repo: str | None | object = _UNSET,
        base_model: str | None | object = _UNSET,
        runtime: str | None | object = _UNSET,
        framework: str | None | object = _UNSET,
        output_model_repo: str | None | object = _UNSET,
        checkpoint_revision: str | None | object = _UNSET,
        config: dict[str, Any] | object = _UNSET,
    ) -> TrainingRun:
        values = {
            "status": status,
            "current_step": current_step,
            "dataset_repo": dataset_repo,
            "base_model": base_model,
            "runtime": runtime,
            "framework": framework,
            "output_model_repo": output_model_repo,
            "checkpoint_revision": checkpoint_revision,
            "config": config,
        }
        payload = {key: value for key, value in values.items() if value is not _UNSET}
        return self._request(
            "PATCH", f"/api/trainer/runs/{self._run_id(run_id)}", json=payload
        )

    def log_metrics(
        self,
        run_id: str,
        *,
        step: int,
        metrics: dict[str, float],
    ) -> TrainingRun:
        return self._request(
            "POST",
            f"/api/trainer/runs/{self._run_id(run_id)}/metrics",
            json={"step": step, "metrics": metrics},
        )

    def log_console(
        self,
        run_id: str,
        *,
        line: str,
        source: ConsoleLogSource = "stdout",
        step: int | None = None,
    ) -> ConsoleLog:
        return self._request(
            "POST",
            f"/api/trainer/runs/{self._run_id(run_id)}/logs",
            json={"source": source, "line": line, "step": step},
        )

    def register_checkpoint(
        self,
        run_id: str,
        *,
        repo_id: str,
        revision: str,
        step: int,
    ) -> TrainingRun:
        return self._request(
            "POST",
            f"/api/trainer/runs/{self._run_id(run_id)}/checkpoints",
            json={"repo_id": repo_id, "revision": revision, "step": step},
        )

    def launch_managed_training(
        self,
        name: str,
        *,
        idempotency_key: str,
        dataset_repo: str,
        base_model: str,
        output_model_repo: str,
        compute_size: ManagedTrainingComputeSize,
        max_steps: int,
        acknowledge_compute_cost: bool,
        dataset_revision: str | None = None,
        base_model_revision: str | None = None,
        output_private: bool = True,
        acknowledge_public_model_risk: bool = False,
        batch_size: int = 8,
        log_every: int = 10,
        save_every: int = 1_000,
        seed: int = 42,
        num_workers: int = 4,
        timeout_minutes: int = 60,
    ) -> ManagedTrainingJob:
        payload = {
            "idempotency_key": self._uuid_id(
                idempotency_key, "Managed training idempotency"
            ),
            "name": name,
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "base_model": base_model,
            "base_model_revision": base_model_revision,
            "output_model_repo": output_model_repo,
            "output_private": output_private,
            "acknowledge_public_model_risk": acknowledge_public_model_risk,
            "acknowledge_compute_cost": acknowledge_compute_cost,
            "runtime": "lerobot",
            "compute_size": compute_size,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "log_every": log_every,
            "save_every": save_every,
            "seed": seed,
            "num_workers": num_workers,
            "timeout_minutes": timeout_minutes,
        }
        return self._request("POST", "/api/trainer/jobs", json=payload)

    def list_managed_training_jobs(
        self, *, limit: int = 50, cursor: str | None = None
    ) -> ManagedTrainingJobPage:
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        payload = self._request("GET", "/api/trainer/jobs", params=params)
        jobs = payload.get("jobs")
        if not isinstance(jobs, list) or not all(isinstance(job, dict) for job in jobs):
            raise TrainerClientError("ctrl-pi returned an invalid response.") from None
        return payload

    def get_managed_training_job(self, job_id: str) -> ManagedTrainingJob:
        return self._request(
            "GET", f"/api/trainer/jobs/{self._uuid_id(job_id, 'Managed training job')}"
        )

    def cancel_managed_training_job(self, job_id: str) -> ManagedTrainingJob:
        return self._request(
            "POST",
            f"/api/trainer/jobs/{self._uuid_id(job_id, 'Managed training job')}/cancel",
        )

    def list_managed_training_logs(
        self,
        job_id: str,
        *,
        after_sequence: int | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if after_sequence is not None:
            params["after_sequence"] = after_sequence
        return self._request(
            "GET",
            f"/api/trainer/jobs/{self._uuid_id(job_id, 'Managed training job')}/logs",
            params=params,
        )

    def get_managed_training_metrics(self, job_id: str) -> ManagedTrainingMetrics:
        return self._request(
            "GET",
            f"/api/trainer/jobs/{self._uuid_id(job_id, 'Managed training job')}/metrics",
        )

    def list_managed_training_checkpoints(
        self, job_id: str
    ) -> ManagedTrainingCheckpoints:
        return self._request(
            "GET",
            f"/api/trainer/jobs/{self._uuid_id(job_id, 'Managed training job')}/checkpoints",
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        messages = {
            404: "The requested training run was not found.",
            409: "The training run update conflicts with current state.",
            422: "The trainer request failed validation.",
            503: "The ctrl-pi database or configuration is unavailable.",
        }
        body = self._http.request(
            method,
            path,
            params=kwargs.get("params"),
            json=kwargs.get("json"),
            json_supplied="json" in kwargs,
            status_messages=messages,
        )
        payload: Any = None
        invalid_json = False
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            invalid_json = True
        if invalid_json or not isinstance(payload, dict):
            raise TrainerClientError("ctrl-pi returned an invalid response.") from None
        return payload

    @staticmethod
    def _run_id(value: str) -> str:
        return Client._uuid_id(value, "Training run")

    @staticmethod
    def _uuid_id(value: str, label: str) -> str:
        parsed: uuid.UUID | None = None
        try:
            parsed = uuid.UUID(value)
        except (TypeError, ValueError, AttributeError):
            pass
        if parsed is None:
            raise TrainerClientError(f"{label} ID is invalid.")
        return str(parsed)
