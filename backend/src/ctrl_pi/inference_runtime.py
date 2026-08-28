from __future__ import annotations

import base64
import binascii
import math
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

RuntimeKind = Literal["stub", "lerobot", "openpi"]
MOCK_MODEL_REPO = "ctrl-pi/mock-policy"
MOCK_MODEL_REVISION = "0" * 40
RuntimeFeatureKind = Literal["state", "environment", "visual", "language", "action"]

MAX_INPUT_FEATURES = 32
MAX_VECTOR_FEATURES = 32
MAX_VECTOR_DIMENSION = 4096
MAX_VECTOR_SCALARS = 16_384
MAX_IMAGE_FEATURES = 8
MAX_IMAGE_BASE64_BYTES = 4 * 1024 * 1024
MAX_IMAGE_DECODED_BYTES = 3 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
MAX_IMAGE_PIXELS = 4_194_304
MAX_ACTION_DIMENSION = 256
MAX_ACTIONS_PER_CHUNK = 100
MAX_TASK_LENGTH = 512
MAX_SEQUENCE = 2**63 - 1
MAX_ABSOLUTE_SCALAR = 1_000_000.0

_FEATURE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_POLICY_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class InferenceRuntimeError(RuntimeError):
    """A safe provider-neutral inference runtime error."""


class RuntimeConfigurationError(InferenceRuntimeError):
    """The requested model/runtime configuration is invalid."""


class RuntimeNotLoadedError(InferenceRuntimeError):
    """The runtime has not loaded a policy."""


class RuntimeProtocolError(InferenceRuntimeError):
    """A bounded inference request or response violated the wire contract."""


class RuntimeUnavailableError(InferenceRuntimeError):
    """A real runtime adapter is intentionally unavailable in V1."""


def _safe_feature_name(value: str) -> str:
    if (
        not _FEATURE_NAME.fullmatch(value)
        or ".." in value
        or "//" in value
        or "\\" in value
    ):
        raise ValueError("feature name is invalid")
    return value


def _safe_repo_id(value: str) -> str:
    value = value.strip()
    if value.count("/") != 1:
        raise ValueError("model_repo must contain one namespace")
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        validate_repo_id(value)
    except (HFValidationError, ValueError):
        raise ValueError("model_repo is not a valid Hugging Face repo ID") from None
    return value


def _immutable_revision(value: str) -> str:
    if not _REVISION.fullmatch(value):
        raise ValueError("revision must be a lowercase 40-character Hub SHA")
    return value


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class RuntimeLoadSpec:
    model_repo: str
    revision: str
    local_model_path: Path | None
    device: Literal["cpu", "cuda"]
    actions_per_chunk: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.model_repo, str) or not isinstance(self.revision, str):
            raise RuntimeConfigurationError("The runtime model identity is invalid.")
        try:
            repo_id = _safe_repo_id(self.model_repo)
            revision = _immutable_revision(self.revision)
        except ValueError as error:
            raise RuntimeConfigurationError(str(error)) from None
        if (
            isinstance(self.actions_per_chunk, bool)
            or not isinstance(self.actions_per_chunk, int)
            or not 1 <= self.actions_per_chunk <= MAX_ACTIONS_PER_CHUNK
        ):
            raise RuntimeConfigurationError(
                f"actions_per_chunk must be between 1 and {MAX_ACTIONS_PER_CHUNK}"
            )
        if self.device not in {"cpu", "cuda"}:
            raise RuntimeConfigurationError("runtime device must be cpu or cuda")
        object.__setattr__(self, "model_repo", repo_id)
        object.__setattr__(self, "revision", revision)
        if self.local_model_path is not None:
            object.__setattr__(self, "local_model_path", Path(self.local_model_path))


class RuntimeFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    kind: RuntimeFeatureKind
    shape: tuple[StrictInt, ...] = Field(min_length=1, max_length=3)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _safe_feature_name(value)

    @model_validator(mode="after")
    def validate_shape(self) -> RuntimeFeature:
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError("feature dimensions must be positive integers")
        if self.kind == "visual":
            if len(self.shape) != 3:
                raise ValueError("visual feature shape must be [C,H,W]")
            channels, height, width = self.shape
            if channels not in {1, 3, 4}:
                raise ValueError("visual feature channels must be 1, 3, or 4")
            if (
                height > MAX_IMAGE_SIDE
                or width > MAX_IMAGE_SIDE
                or height * width > MAX_IMAGE_PIXELS
            ):
                raise ValueError("visual feature dimensions exceed the safe limit")
        elif len(self.shape) != 1 or self.shape[0] > MAX_VECTOR_DIMENSION:
            raise ValueError("non-visual features must be bounded vectors")
        return self


class RuntimeDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["runtime"] = "runtime"
    runtime: RuntimeKind
    policy_type: str = Field(min_length=1, max_length=64)
    model_repo: str = Field(min_length=3, max_length=255)
    revision: str = Field(min_length=40, max_length=40)
    inputs: list[RuntimeFeature] = Field(min_length=1, max_length=MAX_INPUT_FEATURES)
    action: RuntimeFeature
    actions_per_chunk: StrictInt = Field(ge=1, le=MAX_ACTIONS_PER_CHUNK)

    @field_validator("policy_type")
    @classmethod
    def validate_policy_type(cls, value: str) -> str:
        if not _POLICY_TYPE.fullmatch(value):
            raise ValueError("policy_type is invalid")
        return value

    @field_validator("model_repo")
    @classmethod
    def validate_model_repo(cls, value: str) -> str:
        return _safe_repo_id(value)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _immutable_revision(value)

    @model_validator(mode="after")
    def validate_features(self) -> RuntimeDescriptor:
        names = [feature.name for feature in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError("runtime input feature names must be unique")
        if any(feature.kind == "action" for feature in self.inputs):
            raise ValueError("runtime inputs cannot contain an action feature")
        if self.action.kind != "action" or self.action.shape[0] > MAX_ACTION_DIMENSION:
            raise ValueError("runtime action descriptor is invalid")
        if self.action.name in names:
            raise ValueError("runtime action name conflicts with an input")
        if sum(feature.kind == "visual" for feature in self.inputs) > MAX_IMAGE_FEATURES:
            raise ValueError("runtime has too many visual inputs")
        if (
            sum(
                feature.shape[0]
                for feature in self.inputs
                if feature.kind in {"state", "environment"}
            )
            > MAX_VECTOR_SCALARS
        ):
            raise ValueError("runtime vector inputs exceed the safe scalar limit")
        return self


class HealthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["health"] = "health"
    nonce: str = Field(min_length=1, max_length=128)


class DescribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["describe"] = "describe"


class RuntimeHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["health"] = "health"
    healthy: StrictBool
    echo: str = Field(max_length=128)
    runtime: RuntimeKind
    model_repo: str | None = Field(default=None, max_length=255)
    revision: str | None = Field(default=None, max_length=40)

    @model_validator(mode="after")
    def validate_identity(self) -> RuntimeHealth:
        if self.model_repo is not None:
            _safe_repo_id(self.model_repo)
        if self.revision is not None:
            _immutable_revision(self.revision)
        if self.healthy and (self.model_repo is None or self.revision is None):
            raise ValueError("healthy runtime identity is incomplete")
        return self


class EncodedImage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    encoding: Literal["jpeg"] = "jpeg"
    data_base64: str = Field(min_length=4, max_length=MAX_IMAGE_BASE64_BYTES)

    @field_validator("data_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image data is not valid base64") from None
        if not decoded or len(decoded) > MAX_IMAGE_DECODED_BYTES:
            raise ValueError("decoded image exceeds the safe size limit")
        return value

    def decoded(self) -> bytes:
        try:
            return base64.b64decode(self.data_base64, validate=True)
        except (binascii.Error, ValueError):
            raise RuntimeProtocolError("The encoded image is invalid.") from None


class ObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["observation"] = "observation"
    request_id: uuid.UUID
    sequence: StrictInt = Field(ge=0, le=MAX_SEQUENCE)
    captured_at: AwareDatetime
    vectors: dict[str, list[float]] = Field(
        default_factory=dict,
        max_length=MAX_VECTOR_FEATURES,
    )
    images: dict[str, EncodedImage] = Field(
        default_factory=dict,
        max_length=MAX_IMAGE_FEATURES,
    )
    task: str = Field(default="", max_length=MAX_TASK_LENGTH)

    @field_validator("captured_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @field_validator("vectors")
    @classmethod
    def validate_vectors(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        total = 0
        for name, vector in value.items():
            _safe_feature_name(name)
            if not 1 <= len(vector) <= MAX_VECTOR_DIMENSION:
                raise ValueError("observation vector dimension is outside the safe limit")
            total += len(vector)
            if total > MAX_VECTOR_SCALARS:
                raise ValueError("observation vectors exceed the safe scalar limit")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or abs(float(item)) > MAX_ABSOLUTE_SCALAR
                for item in vector
            ):
                raise ValueError("observation vectors must contain finite numbers")
        return {name: [float(item) for item in vector] for name, vector in value.items()}

    @field_validator("vectors", mode="before")
    @classmethod
    def validate_raw_vectors(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("observation vectors must be an object")
        for name, vector in value.items():
            if not isinstance(name, str) or not isinstance(vector, list):
                raise ValueError("observation vectors are invalid")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or abs(float(item)) > MAX_ABSOLUTE_SCALAR
                for item in vector
            ):
                raise ValueError("observation vectors must contain finite numbers")
        return value

    @field_validator("images")
    @classmethod
    def validate_image_names(cls, value: dict[str, EncodedImage]) -> dict[str, EncodedImage]:
        for name in value:
            _safe_feature_name(name)
        return value

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("task contains an invalid control character")
        return value


class ActionChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["action_chunk"] = "action_chunk"
    request_id: uuid.UUID
    observation_sequence: StrictInt = Field(ge=0, le=MAX_SEQUENCE)
    revision: str = Field(min_length=40, max_length=40)
    actions: list[list[float]] = Field(min_length=1, max_length=MAX_ACTIONS_PER_CHUNK)
    server_received_at: AwareDatetime
    server_completed_at: AwareDatetime

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _immutable_revision(value)

    @field_validator("server_received_at", "server_completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, value: list[list[float]]) -> list[list[float]]:
        dimension: int | None = None
        normalized: list[list[float]] = []
        for row in value:
            if not 1 <= len(row) <= MAX_ACTION_DIMENSION:
                raise ValueError("action dimension is outside the safe limit")
            if dimension is None:
                dimension = len(row)
            elif len(row) != dimension:
                raise ValueError("all action rows must have one dimension")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or abs(float(item)) > MAX_ABSOLUTE_SCALAR
                for item in row
            ):
                raise ValueError("actions must contain finite numbers")
            normalized.append([float(item) for item in row])
        return normalized

    @field_validator("actions", mode="before")
    @classmethod
    def validate_raw_actions(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("actions must be an array")
        for row in value:
            if not isinstance(row, list):
                raise ValueError("each action must be an array")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or abs(float(item)) > MAX_ABSOLUTE_SCALAR
                for item in row
            ):
                raise ValueError("actions must contain finite numbers")
        return value

    @model_validator(mode="after")
    def validate_timing(self) -> ActionChunk:
        if self.server_completed_at < self.server_received_at:
            raise ValueError("server completion precedes receipt")
        return self


RuntimeWireRequest = Annotated[
    HealthRequest | DescribeRequest | ObservationRequest,
    Field(discriminator="type"),
]


def validate_observation(
    request: ObservationRequest,
    descriptor: RuntimeDescriptor,
) -> None:
    expected_vectors = {
        feature.name: feature.shape[0]
        for feature in descriptor.inputs
        if feature.kind in {"state", "environment"}
    }
    expected_images = {
        feature.name
        for feature in descriptor.inputs
        if feature.kind == "visual"
    }
    needs_language = any(feature.kind == "language" for feature in descriptor.inputs)
    if set(request.vectors) != set(expected_vectors):
        raise RuntimeProtocolError("Observation vector features do not match the policy.")
    if set(request.images) != expected_images:
        raise RuntimeProtocolError("Observation image features do not match the policy.")
    if needs_language and not request.task.strip():
        raise RuntimeProtocolError("The policy requires a task instruction.")
    if any(
        len(request.vectors[name]) != dimension
        for name, dimension in expected_vectors.items()
    ):
        raise RuntimeProtocolError("Observation vector shape does not match the policy.")


def validate_action_chunk(
    request: ObservationRequest,
    chunk: ActionChunk,
    descriptor: RuntimeDescriptor,
) -> None:
    if (
        chunk.request_id != request.request_id
        or chunk.observation_sequence != request.sequence
        or chunk.revision != descriptor.revision
    ):
        raise RuntimeProtocolError("The action chunk does not match its observation.")
    if not 1 <= len(chunk.actions) <= descriptor.actions_per_chunk:
        raise RuntimeProtocolError("The action chunk length exceeds the runtime contract.")
    if any(
        len(action) != descriptor.action.shape[0]
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value)) > MAX_ABSOLUTE_SCALAR
            for value in action
        )
        for action in chunk.actions
    ):
        raise RuntimeProtocolError("The action shape does not match the policy.")


@runtime_checkable
class InferenceRuntime(Protocol):
    @property
    def kind(self) -> RuntimeKind: ...

    def load(self, spec: RuntimeLoadSpec) -> RuntimeDescriptor: ...

    def describe(self) -> RuntimeDescriptor: ...

    def reset_session(self) -> None: ...

    def predict(self, request: ObservationRequest) -> ActionChunk: ...

    def close(self) -> None: ...


class StubInferenceRuntime:
    """Deterministic action-chunk runtime used for every mock runtime choice."""

    def __init__(self, runtime: RuntimeKind = "stub") -> None:
        self._kind = runtime
        self._descriptor: RuntimeDescriptor | None = None
        self._spec: RuntimeLoadSpec | None = None
        self._lock = threading.RLock()

    @property
    def kind(self) -> RuntimeKind:
        return self._kind

    def load(self, spec: RuntimeLoadSpec) -> RuntimeDescriptor:
        with self._lock:
            if self._spec is not None and self._spec != spec:
                raise RuntimeConfigurationError("The stub runtime already loaded another policy.")
            descriptor = RuntimeDescriptor(
                runtime=self.kind,
                policy_type="stub" if self.kind == "stub" else f"{self.kind}-stub",
                model_repo=spec.model_repo,
                revision=spec.revision,
                inputs=[
                    RuntimeFeature(
                        name="observation.state",
                        kind="state",
                        shape=(7,),
                    )
                ],
                action=RuntimeFeature(name="action", kind="action", shape=(7,)),
                actions_per_chunk=spec.actions_per_chunk,
            )
            self._spec = spec
            self._descriptor = descriptor
            return descriptor.model_copy(deep=True)

    def describe(self) -> RuntimeDescriptor:
        with self._lock:
            if self._descriptor is None:
                raise RuntimeNotLoadedError("The inference runtime is not loaded.")
            return self._descriptor.model_copy(deep=True)

    def predict(self, request: ObservationRequest) -> ActionChunk:
        received_at = datetime.now(UTC)
        with self._lock:
            descriptor = self.describe()
            validate_observation(request, descriptor)
            state = request.vectors["observation.state"]
            actions: list[list[float]] = []
            for offset in range(descriptor.actions_per_chunk):
                phase = request.sequence + offset
                row = [
                    state[index]
                    + 0.01 * math.sin((phase + index + 1) * 0.25)
                    for index in range(6)
                ]
                row.append(
                    min(
                        1.0,
                        max(0.0, state[6] + 0.01 * math.cos((phase + 1) * 0.25)),
                    )
                )
                actions.append(row)
            chunk = ActionChunk(
                request_id=request.request_id,
                observation_sequence=request.sequence,
                revision=descriptor.revision,
                actions=actions,
                server_received_at=received_at,
                server_completed_at=datetime.now(UTC),
            )
            validate_action_chunk(request, chunk, descriptor)
            return chunk

    def reset_session(self) -> None:
        with self._lock:
            if self._descriptor is None:
                raise RuntimeNotLoadedError("The inference runtime is not loaded.")

    def close(self) -> None:
        with self._lock:
            self._descriptor = None
            self._spec = None


class OpenPIInferenceRuntime:
    """Explicit real-runtime placeholder; mock mode selects StubInferenceRuntime."""

    @property
    def kind(self) -> RuntimeKind:
        return "openpi"

    @staticmethod
    def _unavailable() -> RuntimeUnavailableError:
        return RuntimeUnavailableError("The real OpenPI runtime is not available in V1.")

    def load(self, spec: RuntimeLoadSpec) -> RuntimeDescriptor:
        del spec
        raise self._unavailable()

    def describe(self) -> RuntimeDescriptor:
        raise self._unavailable()

    def predict(self, request: ObservationRequest) -> ActionChunk:
        del request
        raise self._unavailable()

    def reset_session(self) -> None:
        raise self._unavailable()

    def close(self) -> None:
        return None
