from __future__ import annotations

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
        self._lock = threading.Lock()

    @property
    def kind(self) -> TargetKind:
        return "stub"

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        if spec.resources.compute_size != "CPU":
            raise ComputeTargetError("The stub compute target supports CPU only.")
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
        return handle

    def health(self, handle: DeploymentHandle, nonce: str) -> HealthResult:
        if not nonce or len(nonce) > 128:
            raise ComputeTargetError("The health nonce is invalid.")
        if handle.endpoint_url is None:
            raise ComputeTargetError("The stub endpoint URL is unavailable.")
        state = self.inspect(handle)
        if not state.running_verified:
            return HealthResult(healthy=False, echo="")
        return HealthResult(healthy=True, echo=nonce)

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
