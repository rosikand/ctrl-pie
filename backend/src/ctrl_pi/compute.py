from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

TargetKind = Literal["stub", "modal"]
TargetLifecycle = Literal[
    "deploying",
    "running",
    "stopping",
    "stopped",
    "failed",
    "unknown",
]

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}")


class ComputeTargetError(RuntimeError):
    """A safe provider-neutral compute operation error."""


class ComputeConfigurationError(ComputeTargetError):
    """Required target configuration or credentials are unavailable."""


class ComputeOwnershipError(ComputeTargetError):
    """A provider resource failed ctrl-pi ownership validation."""


def deployment_app_name(deployment_id: uuid.UUID) -> str:
    return f"ctrl-pi-{deployment_id}"


def deployment_ownership_tag(deployment_id: uuid.UUID) -> str:
    return f"ctrl-pi/deployment/{deployment_id}"


@dataclass(frozen=True)
class ResourcePolicy:
    compute_size: str
    timeout_seconds: int
    min_containers: int = 0
    buffer_containers: int = 0
    max_containers: int = 1
    scaledown_window_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.compute_size or len(self.compute_size) > 64:
            raise ValueError("compute_size must be 1-64 characters")
        if not 1 <= self.timeout_seconds <= 1800:
            raise ValueError("compute timeout must be between 1 and 1800 seconds")
        if self.min_containers != 0 or self.buffer_containers != 0:
            raise ValueError("V1 compute targets cannot keep warm containers")
        if self.max_containers != 1:
            raise ValueError("V1 deployments must use exactly one container")
        if not 1 <= self.scaledown_window_seconds <= 60:
            raise ValueError("scaledown window must be between 1 and 60 seconds")


@dataclass(frozen=True)
class DeploymentSpec:
    deployment_id: uuid.UUID
    app_name: str
    ownership_tag: str
    model_repo: str
    checkpoint_revision: str | None
    runtime: str
    resources: ResourcePolicy

    def __post_init__(self) -> None:
        if self.app_name != deployment_app_name(self.deployment_id):
            raise ValueError("app_name does not match deployment_id")
        if self.ownership_tag != deployment_ownership_tag(self.deployment_id):
            raise ValueError("ownership_tag does not match deployment_id")
        if not self.model_repo or len(self.model_repo) > 255:
            raise ValueError("model_repo must be 1-255 characters")
        if not self.runtime or len(self.runtime) > 64:
            raise ValueError("runtime must be 1-64 characters")
        if self.checkpoint_revision is not None and not self.checkpoint_revision:
            raise ValueError("checkpoint_revision cannot be blank")


@dataclass(frozen=True)
class DeploymentHandle:
    deployment_id: uuid.UUID
    provider_app_id: str
    app_name: str
    ownership_tag: str
    endpoint_url: str | None

    def __post_init__(self) -> None:
        _validate_owned_identity(
            deployment_id=self.deployment_id,
            provider_app_id=self.provider_app_id,
            app_name=self.app_name,
            ownership_tag=self.ownership_tag,
        )
        if self.endpoint_url is not None:
            _validate_endpoint_url(self.endpoint_url)


@dataclass(frozen=True)
class HealthResult:
    healthy: bool
    echo: str


@dataclass(frozen=True)
class TargetState:
    deployment_id: uuid.UUID
    provider_app_id: str
    app_name: str
    ownership_tag: str
    exists: bool
    lifecycle: TargetLifecycle
    running_tasks: int
    endpoint_url: str | None

    def __post_init__(self) -> None:
        _validate_owned_identity(
            deployment_id=self.deployment_id,
            provider_app_id=self.provider_app_id,
            app_name=self.app_name,
            ownership_tag=self.ownership_tag,
        )
        if self.running_tasks < 0:
            raise ValueError("running_tasks cannot be negative")
        if self.endpoint_url is not None:
            _validate_endpoint_url(self.endpoint_url)

    @property
    def stopped_verified(self) -> bool:
        return (
            (
                self.lifecycle == "stopped"
                or (not self.exists and self.lifecycle == "unknown")
            )
            and self.running_tasks == 0
        )

    @property
    def running_verified(self) -> bool:
        return self.exists and self.lifecycle == "running"

    def handle(self) -> DeploymentHandle:
        return DeploymentHandle(
            deployment_id=self.deployment_id,
            provider_app_id=self.provider_app_id,
            app_name=self.app_name,
            ownership_tag=self.ownership_tag,
            endpoint_url=self.endpoint_url,
        )


@runtime_checkable
class ComputeTarget(Protocol):
    @property
    def kind(self) -> TargetKind: ...

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle: ...

    def health(self, handle: DeploymentHandle, nonce: str) -> HealthResult: ...

    def inspect(self, handle: DeploymentHandle) -> TargetState: ...

    def stop(self, handle: DeploymentHandle) -> None: ...

    def list_owned(self) -> list[TargetState]: ...


def validate_owned_handle(handle: DeploymentHandle) -> None:
    _validate_owned_identity(
        deployment_id=handle.deployment_id,
        provider_app_id=handle.provider_app_id,
        app_name=handle.app_name,
        ownership_tag=handle.ownership_tag,
    )


def _validate_owned_identity(
    *,
    deployment_id: uuid.UUID,
    provider_app_id: str,
    app_name: str,
    ownership_tag: str,
) -> None:
    if app_name != deployment_app_name(deployment_id):
        raise ComputeOwnershipError("Compute app name does not match its deployment.")
    if ownership_tag != deployment_ownership_tag(deployment_id):
        raise ComputeOwnershipError("Compute ownership tag does not match its deployment.")
    if not _SAFE_IDENTIFIER.fullmatch(provider_app_id):
        raise ComputeOwnershipError("Compute provider app ID is invalid.")


def _validate_endpoint_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise ValueError("compute endpoint URL is invalid") from error
    if (
        parsed.scheme not in {"http", "https", "stub"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("compute endpoint URL is invalid")
