from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeAlias, runtime_checkable


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
TrainingTargetKind = Literal["stub", "modal"]
TrainingExecutionState = Literal[
    "pending", "running", "succeeded", "failed", "cancelled", "unknown"
]
TrainingResourceLifecycle = Literal[
    "deploying", "running", "stopping", "stopped", "failed", "unknown"
]
TrainingLogSource = Literal["stdout", "stderr", "system"]

_PROVIDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}")
_SHA = re.compile(r"[0-9a-f]{40}")
_METRIC_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.\-/]{0,63})")
_MAX_SEQUENCE = 9_007_199_254_740_991
MANAGED_TRAINING_MARKER_PATH = ".ctrl-pi-managed-training.json"
MANAGED_TRAINING_MARKER_SCHEMA = "ctrl-pi.managed-training-model"
MANAGED_TRAINING_MARKER_VERSION = 1
MANAGED_SMOLVLA_DEPENDENCY_DIR = ".ctrl-pi-smolvlm"
MANAGED_SMOLVLA_DEPENDENCY_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class ManagedTrainingTargetError(RuntimeError):
    """A safe provider-neutral managed-training error."""


class ManagedTrainingConfigurationError(ManagedTrainingTargetError):
    """Required provider configuration is unavailable."""


class ManagedTrainingOwnershipError(ManagedTrainingTargetError):
    """A provider resource failed exact ctrl-π ownership validation."""


class ManagedTrainingProtocolError(ManagedTrainingTargetError):
    """A provider worker returned a deterministic invalid protocol value."""


class ManagedTrainingTransientError(ManagedTrainingTargetError):
    """A bounded provider observation failed and may be retried safely."""


def training_app_name(job_id: uuid.UUID) -> str:
    return f"ctrl-pi-training-{job_id}"


def training_ownership_tag(job_id: uuid.UUID) -> str:
    return f"ctrl-pi-training-job={job_id}"


def training_gpu_spec(compute_size: ManagedTrainingComputeSize) -> tuple[str, int]:
    values: dict[ManagedTrainingComputeSize, tuple[str, int]] = {
        "Modal: A10G": ("A10G", 1),
        "Modal: A100": ("A100", 1),
        "Modal: 2xA100": ("A100", 2),
        "Modal: 4xA100": ("A100", 4),
        "Modal: 8xA100": ("A100", 8),
        "Modal: H100": ("H100", 1),
        "Modal: 2xH100": ("H100", 2),
        "Modal: 4xH100": ("H100", 4),
        "Modal: 8xH100": ("H100", 8),
    }
    try:
        return values[compute_size]
    except KeyError:
        raise ValueError("managed training compute size is invalid") from None


def managed_training_marker(job_id: uuid.UUID, request_hash: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{64}", request_hash) is None:
        raise ValueError("managed training request hash is invalid")
    return (
        json.dumps(
            {
                "job_id": str(job_id),
                "request_hash": request_hash,
                "schema": MANAGED_TRAINING_MARKER_SCHEMA,
                "version": MANAGED_TRAINING_MARKER_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ManagedTrainingSpec:
    job_id: uuid.UUID
    request_hash: str
    app_name: str
    ownership_tag: str
    dataset_repo: str
    dataset_revision: str
    base_model: str
    base_model_revision: str
    output_model_repo: str
    output_marker_revision: str
    output_private: bool
    runtime: Literal["lerobot"]
    max_steps: int
    batch_size: int
    log_every: int
    save_every: int
    seed: int
    num_workers: int
    compute_size: ManagedTrainingComputeSize
    timeout_seconds: int
    deadline_at: datetime

    def __post_init__(self) -> None:
        managed_training_marker(self.job_id, self.request_hash)
        if self.app_name != training_app_name(self.job_id):
            raise ValueError("training app name does not match job ID")
        if self.ownership_tag != training_ownership_tag(self.job_id):
            raise ValueError("training ownership tag does not match job ID")
        for value, label in (
            (self.dataset_repo, "dataset repo"),
            (self.base_model, "base model"),
            (self.output_model_repo, "output model repo"),
        ):
            if not 3 <= len(value) <= 255 or value.count("/") != 1:
                raise ValueError(f"{label} is invalid")
        for value, label in (
            (self.dataset_revision, "dataset revision"),
            (self.base_model_revision, "base model revision"),
            (self.output_marker_revision, "output marker revision"),
        ):
            if _SHA.fullmatch(value) is None:
                raise ValueError(f"{label} must be an immutable Hub SHA")
        if not isinstance(self.output_private, bool):
            raise ValueError("output visibility flag is invalid")
        if self.runtime != "lerobot":
            raise ValueError("managed training runtime is unsupported")
        if not 1 <= self.max_steps <= 2_147_483_647:
            raise ValueError("max_steps is invalid")
        if not 1 <= self.batch_size <= 4_096:
            raise ValueError("batch_size is invalid")
        if not 1 <= self.log_every <= self.max_steps:
            raise ValueError("log_every is invalid")
        if math.ceil(self.max_steps / self.log_every) > 10_000:
            raise ValueError("managed training would exceed the metric event limit")
        if not 1 <= self.save_every <= self.max_steps:
            raise ValueError("save_every is invalid")
        if math.ceil(self.max_steps / self.save_every) > 512:
            raise ValueError("managed training would exceed the checkpoint limit")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("seed is invalid")
        if not 0 <= self.num_workers <= 64:
            raise ValueError("num_workers is invalid")
        training_gpu_spec(self.compute_size)
        # The public request is bounded to whole minutes, but Hub preparation
        # consumes part of that persisted deadline before compute is spawned.
        if not 1 <= self.timeout_seconds <= 86_400:
            raise ValueError("managed training timeout is invalid")
        if self.deadline_at.tzinfo is None or self.deadline_at.utcoffset() is None:
            raise ValueError("managed training deadline must include a timezone")


@dataclass(frozen=True)
class TrainingHandle:
    job_id: uuid.UUID
    provider_app_id: str
    provider_function_call_id: str | None
    app_name: str
    ownership_tag: str
    request_hash: str | None = None

    def __post_init__(self) -> None:
        if self.request_hash is not None:
            managed_training_marker(self.job_id, self.request_hash)
        _validate_identity(
            job_id=self.job_id,
            provider_app_id=self.provider_app_id,
            provider_function_call_id=self.provider_function_call_id,
            app_name=self.app_name,
            ownership_tag=self.ownership_tag,
        )


@dataclass(frozen=True)
class TrainingLogEvent:
    sequence: int
    source: TrainingLogSource
    line: str
    step: int | None = None

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        if (
            not self.line
            or len(self.line.encode("utf-8")) > 4 * 1024
            or "\n" in self.line
            or "\r" in self.line
            or any(ord(character) < 32 and character != "\t" for character in self.line)
        ):
            raise ValueError("managed training log line is invalid")
        _validate_step(self.step)


@dataclass(frozen=True)
class TrainingMetricEvent:
    sequence: int
    step: int
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        _validate_step(self.step, required=True)
        if not 1 <= len(self.metrics) <= 64:
            raise ValueError("managed training metric event is invalid")
        for name, value in self.metrics.items():
            if (
                _METRIC_NAME.fullmatch(name) is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("managed training metric event is invalid")


@dataclass(frozen=True)
class TrainingCheckpointEvent:
    sequence: int
    repo_id: str
    revision: str
    step: int
    final: bool = False

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        _validate_step(self.step, required=True)
        if not 3 <= len(self.repo_id) <= 255 or self.repo_id.count("/") != 1:
            raise ValueError("managed training checkpoint repo is invalid")
        if _SHA.fullmatch(self.revision) is None:
            raise ValueError("managed training checkpoint revision is invalid")
        if not isinstance(self.final, bool):
            raise ValueError("managed training final checkpoint flag is invalid")


TrainingEvent: TypeAlias = (
    TrainingLogEvent | TrainingMetricEvent | TrainingCheckpointEvent
)


@dataclass(frozen=True)
class TrainingResult:
    job_id: uuid.UUID
    request_hash: str
    output_model_repo: str
    revision: str
    step: int

    def __post_init__(self) -> None:
        managed_training_marker(self.job_id, self.request_hash)
        if (
            not 3 <= len(self.output_model_repo) <= 255
            or self.output_model_repo.count("/") != 1
        ):
            raise ValueError("managed training result repo is invalid")
        if _SHA.fullmatch(self.revision) is None:
            raise ValueError("managed training result revision is invalid")
        _validate_step(self.step, required=True)


@dataclass(frozen=True)
class TrainingTargetState:
    job_id: uuid.UUID
    provider_app_id: str
    provider_function_call_id: str | None
    app_name: str
    ownership_tag: str
    exists: bool
    resource_lifecycle: TrainingResourceLifecycle
    execution_state: TrainingExecutionState
    running_tasks: int

    def __post_init__(self) -> None:
        _validate_identity(
            job_id=self.job_id,
            provider_app_id=self.provider_app_id,
            provider_function_call_id=self.provider_function_call_id,
            app_name=self.app_name,
            ownership_tag=self.ownership_tag,
        )
        if not isinstance(self.exists, bool) or self.running_tasks < 0:
            raise ValueError("managed training provider state is invalid")

    @property
    def stopped_verified(self) -> bool:
        return (
            (
                self.resource_lifecycle == "stopped"
                or (not self.exists and self.resource_lifecycle == "unknown")
            )
            and self.running_tasks == 0
            and self.execution_state not in {"pending", "running"}
        )

    def handle(self, *, request_hash: str | None = None) -> TrainingHandle:
        return TrainingHandle(
            job_id=self.job_id,
            provider_app_id=self.provider_app_id,
            provider_function_call_id=self.provider_function_call_id,
            app_name=self.app_name,
            ownership_tag=self.ownership_tag,
            request_hash=request_hash,
        )


@dataclass(frozen=True)
class TrainingPoll:
    state: TrainingTargetState
    events: tuple[TrainingEvent, ...]
    next_sequence: int
    truncated: bool
    has_more: bool
    result: TrainingResult | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.next_sequence <= _MAX_SEQUENCE:
            raise ValueError("managed training event cursor is invalid")
        sequences = [event.sequence for event in self.events]
        if sequences != sorted(set(sequences)):
            raise ValueError("managed training events are out of order")
        if sequences and self.next_sequence < sequences[-1]:
            raise ValueError("managed training event cursor regressed")
        if not isinstance(self.truncated, bool) or not isinstance(self.has_more, bool):
            raise ValueError("managed training event page metadata is invalid")


@runtime_checkable
class ManagedTrainingTarget(Protocol):
    @property
    def kind(self) -> TrainingTargetKind: ...

    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle: ...

    def poll(
        self,
        handle: TrainingHandle,
        *,
        after_sequence: int,
        limit: int,
    ) -> TrainingPoll: ...

    def inspect(self, handle: TrainingHandle) -> TrainingTargetState: ...

    def cancel(self, handle: TrainingHandle) -> None: ...

    def stop(self, handle: TrainingHandle) -> None: ...

    def list_owned(self) -> list[TrainingTargetState]: ...


def validate_training_handle(handle: TrainingHandle) -> None:
    _validate_identity(
        job_id=handle.job_id,
        provider_app_id=handle.provider_app_id,
        provider_function_call_id=handle.provider_function_call_id,
        app_name=handle.app_name,
        ownership_tag=handle.ownership_tag,
    )


def validate_training_state(state: TrainingTargetState) -> None:
    _validate_identity(
        job_id=state.job_id,
        provider_app_id=state.provider_app_id,
        provider_function_call_id=state.provider_function_call_id,
        app_name=state.app_name,
        ownership_tag=state.ownership_tag,
    )


def _validate_identity(
    *,
    job_id: uuid.UUID,
    provider_app_id: str,
    provider_function_call_id: str | None,
    app_name: str,
    ownership_tag: str,
) -> None:
    if app_name != training_app_name(job_id):
        raise ManagedTrainingOwnershipError(
            "Managed training app name does not match its job."
        )
    if ownership_tag != training_ownership_tag(job_id):
        raise ManagedTrainingOwnershipError(
            "Managed training ownership tag does not match its job."
        )
    if _PROVIDER_ID.fullmatch(provider_app_id) is None:
        raise ManagedTrainingOwnershipError(
            "Managed training provider App ID is invalid."
        )
    if (
        provider_function_call_id is not None
        and _PROVIDER_ID.fullmatch(provider_function_call_id) is None
    ):
        raise ManagedTrainingOwnershipError(
            "Managed training provider FunctionCall ID is invalid."
        )


def _validate_sequence(value: int) -> None:
    if isinstance(value, bool) or not 1 <= value <= _MAX_SEQUENCE:
        raise ValueError("managed training event sequence is invalid")


def _validate_step(value: int | None, *, required: bool = False) -> None:
    if value is None:
        if required:
            raise ValueError("managed training event step is required")
        return
    if isinstance(value, bool) or not 0 <= value <= 2_147_483_647:
        raise ValueError("managed training event step is invalid")


__all__ = [
    "MANAGED_TRAINING_MARKER_PATH",
    "MANAGED_TRAINING_MARKER_SCHEMA",
    "MANAGED_TRAINING_MARKER_VERSION",
    "ManagedTrainingComputeSize",
    "ManagedTrainingConfigurationError",
    "ManagedTrainingOwnershipError",
    "ManagedTrainingProtocolError",
    "ManagedTrainingSpec",
    "ManagedTrainingTarget",
    "ManagedTrainingTargetError",
    "ManagedTrainingTransientError",
    "TrainingCheckpointEvent",
    "TrainingEvent",
    "TrainingExecutionState",
    "TrainingHandle",
    "TrainingLogEvent",
    "TrainingMetricEvent",
    "TrainingPoll",
    "TrainingResourceLifecycle",
    "TrainingResult",
    "TrainingTargetKind",
    "TrainingTargetState",
    "managed_training_marker",
    "training_app_name",
    "training_gpu_spec",
    "training_ownership_tag",
    "validate_training_handle",
    "validate_training_state",
]
