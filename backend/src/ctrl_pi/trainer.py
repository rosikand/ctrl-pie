from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Literal, NotRequired, TypedDict
import uuid

import httpx

from ctrl_pi._http import SafeHttpClient

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
ConsoleLogSource = Literal["stdout", "stderr", "system"]


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
        parsed: uuid.UUID | None = None
        try:
            parsed = uuid.UUID(value)
        except (TypeError, ValueError, AttributeError):
            pass
        if parsed is None:
            raise TrainerClientError("Training run ID is invalid.")
        return str(parsed)
