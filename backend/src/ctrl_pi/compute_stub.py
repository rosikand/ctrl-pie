from __future__ import annotations

import re
import threading
import uuid
from dataclasses import replace

from ctrl_pi.compute import (
    ComputeOwnershipError,
    ComputeTargetError,
    DeploymentHandle,
    DeploymentSpec,
    HealthResult,
    TargetKind,
    TargetState,
    deployment_app_name,
    deployment_ownership_tag,
    validate_owned_handle,
)


class StubComputeTarget:
    """Deterministic, credential-free lifecycle target used by mock mode."""

    def __init__(self) -> None:
        self._states: dict[uuid.UUID, TargetState] = {}
        self._runtime_identities: dict[uuid.UUID, tuple[str, str, str]] = {}
        self._lock = threading.Lock()

    @property
    def kind(self) -> TargetKind:
        return "stub"

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        runtime_identity: tuple[str, str, str] | None = None
        if spec.runtime == "stub" and spec.resources.compute_size == "CPU":
            pass
        elif spec.runtime in {"lerobot", "openpi"} and spec.resources.compute_size in {
            "Modal: A10G",
            "Modal: A100",
            "Modal: H100",
        }:
            if (
                spec.checkpoint_revision is None
                or re.fullmatch(r"[0-9a-f]{40}", spec.checkpoint_revision) is None
            ):
                raise ComputeTargetError(
                    "The mock runtime requires an immutable model revision."
                )
            runtime_identity = (
                spec.runtime,
                spec.model_repo,
                spec.checkpoint_revision,
            )
        else:
            raise ComputeTargetError(
                "The stub compute target does not support this runtime and compute size."
            )
        provider_app_id = f"stub-{spec.deployment_id.hex}"
        handle = DeploymentHandle(
            deployment_id=spec.deployment_id,
            provider_app_id=provider_app_id,
            app_name=spec.app_name,
            ownership_tag=spec.ownership_tag,
            endpoint_url=f"stub://ctrl-pi/{spec.deployment_id}",
        )
        with self._lock:
            if spec.deployment_id in self._states:
                raise ComputeTargetError(
                    "A compute resource already exists for this deployment."
                )
            self._states[spec.deployment_id] = TargetState(
                deployment_id=handle.deployment_id,
                provider_app_id=handle.provider_app_id,
                app_name=handle.app_name,
                ownership_tag=handle.ownership_tag,
                exists=True,
                lifecycle="running",
                running_tasks=1,
                endpoint_url=handle.endpoint_url,
            )
            if runtime_identity is not None:
                self._runtime_identities[spec.deployment_id] = runtime_identity
        return handle

    def health(self, handle: DeploymentHandle, nonce: str) -> HealthResult:
        if not nonce or len(nonce) > 128:
            raise ComputeTargetError("The health nonce is invalid.")
        if handle.endpoint_url is None:
            raise ComputeTargetError("The stub endpoint URL is unavailable.")
        state = self.inspect(handle)
        if not state.running_verified:
            return HealthResult(healthy=False, echo="")
        with self._lock:
            runtime_identity = self._runtime_identities.get(handle.deployment_id)
        if runtime_identity is None:
            return HealthResult(healthy=True, echo=nonce)
        runtime, model_repo, revision = runtime_identity
        return HealthResult(
            healthy=True,
            echo=nonce,
            runtime=runtime,
            model_repo=model_repo,
            revision=revision,
        )

    def inspect(self, handle: DeploymentHandle) -> TargetState:
        self._validate_handle(handle)
        with self._lock:
            state = self._states.get(handle.deployment_id)
        if state is None:
            return TargetState(
                deployment_id=handle.deployment_id,
                provider_app_id=handle.provider_app_id,
                app_name=handle.app_name,
                ownership_tag=handle.ownership_tag,
                exists=False,
                lifecycle="stopped",
                running_tasks=0,
                endpoint_url=handle.endpoint_url,
            )
        if (
            state.provider_app_id != handle.provider_app_id
            or state.app_name != handle.app_name
            or state.ownership_tag != handle.ownership_tag
        ):
            raise ComputeOwnershipError(
                "The stub resource identity does not match this deployment."
            )
        return state

    def stop(self, handle: DeploymentHandle) -> None:
        state = self.inspect(handle)
        if state.stopped_verified:
            return
        with self._lock:
            self._states[handle.deployment_id] = replace(
                state,
                lifecycle="stopped",
                running_tasks=0,
            )

    def list_owned(self) -> list[TargetState]:
        with self._lock:
            return sorted(
                self._states.values(),
                key=lambda state: str(state.deployment_id),
            )

    @staticmethod
    def _validate_handle(handle: DeploymentHandle) -> None:
        validate_owned_handle(handle)
        expected_provider_id = f"stub-{handle.deployment_id.hex}"
        if (
            handle.app_name != deployment_app_name(handle.deployment_id)
            or handle.ownership_tag
            != deployment_ownership_tag(handle.deployment_id)
            or handle.provider_app_id != expected_provider_id
        ):
            raise ComputeOwnershipError(
                "The stub resource is not owned by this deployment."
            )
