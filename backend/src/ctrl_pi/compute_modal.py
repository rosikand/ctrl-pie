from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from urllib.parse import urlsplit

import httpx
import modal
from modal._object import _get_environment_name
from modal._utils.async_utils import synchronizer
from modal.exception import AuthError, NotFoundError
from modal_proto import api_pb2

from ctrl_pi.compute import (
    ComputeConfigurationError,
    ComputeOwnershipError,
    ComputeTargetError,
    DeploymentHandle,
    DeploymentSpec,
    HealthResult,
    TargetKind,
    TargetLifecycle,
    TargetState,
    deployment_app_name,
    deployment_ownership_tag,
    validate_owned_handle,
)
from ctrl_pi.modal_workload import (
    MODAL_HEALTH_FUNCTION,
    MODAL_OWNERSHIP_TAG_KEY,
    build_modal_workload,
)

if TYPE_CHECKING:
    from ctrl_pi.config import AppConfig


_OWNED_APP_NAME = re.compile(
    r"ctrl-pi-(?P<deployment_id>[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})"
)
_ACTIVE_LIFECYCLES = {"deploying", "running", "stopping", "failed", "unknown"}
_FAILED = object()
_T = TypeVar("_T")


@dataclass(frozen=True)
class _ProviderDeployment:
    app_id: str
    endpoint_url: str | None


@dataclass(frozen=True)
class _ProviderApp:
    app_id: str
    app_name: str
    lifecycle: TargetLifecycle
    running_tasks: int


@dataclass(frozen=True)
class _ResolvedProviderApp:
    app_id: str
    lifecycle: TargetLifecycle


class _GatewayConfigurationError(RuntimeError):
    pass


class _ModalGateway(Protocol):
    def deploy(self, spec: DeploymentSpec) -> _ProviderDeployment: ...

    def list_apps(self) -> list[_ProviderApp]: ...

    def resolve_name(self, app_name: str) -> _ResolvedProviderApp | None: ...

    def get_lifecycle(self, app_id: str) -> TargetLifecycle | None: ...

    def get_tags(self, app_id: str) -> dict[str, str]: ...

    def get_endpoint_url(self, app_name: str) -> str | None: ...

    def stop(self, app_id: str) -> None: ...


def _lifecycle_from_modal(state: int) -> TargetLifecycle:
    if state == api_pb2.APP_STATE_INITIALIZING:
        return "deploying"
    if state in {
        api_pb2.APP_STATE_DEPLOYED,
        api_pb2.APP_STATE_EPHEMERAL,
        api_pb2.APP_STATE_DETACHED,
        api_pb2.APP_STATE_DETACHED_DISCONNECTED,
    }:
        return "running"
    if state == api_pb2.APP_STATE_STOPPING:
        return "stopping"
    if state == api_pb2.APP_STATE_STOPPED:
        return "stopped"
    return "unknown"


# Modal 1.5.4 does not expose task counts or stop-by-ID through its stable App
# methods. These tiny wrappers mirror the installed official CLI and keep the
# pinned protobuf dependency out of the rest of ctrl-pi.
@synchronizer.create_blocking
async def _rpc_list_apps(client: object, environment_name: str) -> list[tuple[str, str, int, int]]:
    response = await client.stub.AppList(  # type: ignore[attr-defined]
        api_pb2.AppListRequest(environment_name=environment_name)
    )
    return [
        (
            item.app_id,
            item.description or item.name,
            item.state,
            item.n_running_tasks,
        )
        for item in response.apps
    ]


@synchronizer.create_blocking
async def _rpc_resolve_name(
    client: object, environment_name: str, app_name: str
) -> tuple[str, int] | None:
    response = await client.stub.AppGetByDeploymentName(  # type: ignore[attr-defined]
        api_pb2.AppGetByDeploymentNameRequest(
            name=app_name,
            environment_name=environment_name,
        )
    )
    app_id = response.app_id or response.previous_app_id
    if not app_id:
        return None
    return app_id, response.lifecycle.app_state


@synchronizer.create_blocking
async def _rpc_get_lifecycle(client: object, app_id: str) -> int:
    response = await client.stub.AppGetLifecycle(  # type: ignore[attr-defined]
        api_pb2.AppGetLifecycleRequest(app_id=app_id)
    )
    return response.lifecycle.app_state


@synchronizer.create_blocking
async def _rpc_get_tags(client: object, app_id: str) -> dict[str, str]:
    response = await client.stub.AppGetTags(  # type: ignore[attr-defined]
        api_pb2.AppGetTagsRequest(app_id=app_id)
    )
    return dict(response.tags)


@synchronizer.create_blocking
async def _rpc_stop_app(client: object, app_id: str) -> None:
    await client.stub.AppStop(  # type: ignore[attr-defined]
        api_pb2.AppStopRequest(
            app_id=app_id,
            source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT,
        )
    )


class _OfficialModalGateway:
    def __init__(
        self,
        *,
        token_id: str | None,
        token_secret: str | None,
        environment_name: str | None,
    ) -> None:
        self._token_id = token_id
        self._token_secret = token_secret
        self._environment_name = _get_environment_name(environment_name)
        self._modal_client: object | None = None

    def _client(self) -> object:
        if self._modal_client is not None:
            return self._modal_client
        configuration_failed = False
        try:
            if (self._token_id is None) != (self._token_secret is None):
                configuration_failed = True
                client = None
            elif self._token_id is not None and self._token_secret is not None:
                client = modal.Client.from_credentials(
                    self._token_id,
                    self._token_secret,
                )
            else:
                client = modal.Client.from_env()
        except AuthError:
            configuration_failed = True
            client = None
        if configuration_failed or client is None:
            raise _GatewayConfigurationError
        self._modal_client = client
        return client

    def deploy(self, spec: DeploymentSpec) -> _ProviderDeployment:
        client = self._client()
        workload = build_modal_workload(
            app_name=spec.app_name,
            deployment_id=spec.deployment_id,
            resources=spec.resources,
        )
        deploy_failed = False
        deployed_app: object | None = None
        try:
            deployed_app = workload.app.deploy(
                name=spec.app_name,
                environment_name=self._environment_name,
                client=client,
                strategy="recreate",
            )
        except Exception:
            deploy_failed = True
        app_id = getattr(deployed_app, "app_id", None)
        if deploy_failed or not app_id:
            recovered_ids: set[str] = set()
            recovered_tags: dict[str, str] = {}
            recovery_failed = False
            try:
                resolved = self.resolve_name(spec.app_name)
                if resolved is not None:
                    recovered_ids.add(resolved.app_id)
                recovered_ids.update(
                    app.app_id
                    for app in self.list_apps()
                    if app.app_name == spec.app_name
                    and app.lifecycle != "stopped"
                )
                if len(recovered_ids) == 1:
                    recovered_tags = self.get_tags(next(iter(recovered_ids)))
            except Exception:
                recovery_failed = True
            if (
                not recovery_failed
                and len(recovered_ids) == 1
                and recovered_tags.get(MODAL_OWNERSHIP_TAG_KEY)
                == str(spec.deployment_id)
            ):
                return _ProviderDeployment(
                    app_id=next(iter(recovered_ids)),
                    endpoint_url=None,
                )
            raise RuntimeError("Modal deployment did not complete")
        endpoint_url: str | None = None
        if not deploy_failed:
            try:
                endpoint_url = workload.health_function.get_web_url()
                if endpoint_url is None:
                    endpoint_url = self.get_endpoint_url(spec.app_name)
            except Exception:
                endpoint_url = None
        return _ProviderDeployment(
            app_id=cast(str, app_id),
            endpoint_url=endpoint_url,
        )

    def list_apps(self) -> list[_ProviderApp]:
        return [
            _ProviderApp(
                app_id=app_id,
                app_name=app_name,
                lifecycle=_lifecycle_from_modal(state),
                running_tasks=running_tasks,
            )
            for app_id, app_name, state, running_tasks in _rpc_list_apps(
                self._client(), self._environment_name
            )
        ]

    def resolve_name(self, app_name: str) -> _ResolvedProviderApp | None:
        resolved = _rpc_resolve_name(
            self._client(), self._environment_name, app_name
        )
        if resolved is None:
            return None
        app_id, state = resolved
        return _ResolvedProviderApp(
            app_id=app_id,
            lifecycle=_lifecycle_from_modal(state),
        )

    def get_lifecycle(self, app_id: str) -> TargetLifecycle | None:
        not_found = False
        try:
            state = _rpc_get_lifecycle(self._client(), app_id)
        except NotFoundError:
            not_found = True
            state = None
        if not_found or state is None:
            return None
        return _lifecycle_from_modal(state)

    def get_tags(self, app_id: str) -> dict[str, str]:
        return _rpc_get_tags(self._client(), app_id)

    def get_endpoint_url(self, app_name: str) -> str | None:
        return modal.Function.from_name(
            app_name,
            MODAL_HEALTH_FUNCTION,
            environment_name=self._environment_name,
            client=self._client(),
        ).get_web_url()

    def stop(self, app_id: str) -> None:
        _rpc_stop_app(self._client(), app_id)


class ModalComputeTarget:
    """Modal lifecycle adapter for ctrl-pi-owned, CPU-only M9 workloads."""

    def __init__(
        self,
        *,
        token_id: str | None = None,
        token_secret: str | None = None,
        environment_name: str | None = None,
        gateway: _ModalGateway | None = None,
        http_transport: httpx.BaseTransport | None = None,
        health_timeout_seconds: float = 60.0,
        stop_timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if health_timeout_seconds <= 0:
            raise ValueError("health_timeout_seconds must be positive")
        if stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._gateway = (
            gateway
            if gateway is not None
            else _OfficialModalGateway(
                token_id=token_id,
                token_secret=token_secret,
                environment_name=environment_name,
            )
        )
        self._http_transport = http_transport
        self._health_timeout_seconds = health_timeout_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._sleeper = sleeper

    @classmethod
    def from_config(cls, config: AppConfig) -> "ModalComputeTarget":
        token_id = (
            config.modal_token_id.get_secret_value()
            if config.modal_token_id is not None
            else None
        )
        token_secret = (
            config.modal_token_secret.get_secret_value()
            if config.modal_token_secret is not None
            else None
        )
        return cls(token_id=token_id, token_secret=token_secret)

    @property
    def kind(self) -> TargetKind:
        return "modal"

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        self._validate_spec(spec)
        self._reject_name_collision(spec)
        deployment = self._provider_call(
            lambda: self._gateway.deploy(spec),
            "The Modal workload could not be deployed.",
        )
        identity_verified = False
        try:
            self._verify_remote_identity(
                deployment_id=spec.deployment_id,
                provider_app_id=deployment.app_id,
                app_name=spec.app_name,
            )
            identity_verified = True
            endpoint_url = deployment.endpoint_url
            if endpoint_url is not None:
                try:
                    self._validate_modal_endpoint(endpoint_url)
                except ComputeOwnershipError:
                    endpoint_url = None
            return DeploymentHandle(
                deployment_id=spec.deployment_id,
                provider_app_id=deployment.app_id,
                app_name=spec.app_name,
                ownership_tag=spec.ownership_tag,
                endpoint_url=endpoint_url,
            )
        except (ComputeTargetError, ValueError):
            if identity_verified:
                self._best_effort_stop_owned(
                    deployment_id=spec.deployment_id,
                    provider_app_id=deployment.app_id,
                    app_name=spec.app_name,
                    ownership_tag=spec.ownership_tag,
                )
            raise

    def health(self, handle: DeploymentHandle, nonce: str) -> HealthResult:
        validate_owned_handle(handle)
        if handle.endpoint_url is None:
            raise ComputeTargetError("The Modal health endpoint is unavailable.")
        self._validate_modal_endpoint(handle.endpoint_url)
        if not nonce or len(nonce) > 128:
            raise ComputeTargetError("The health nonce is invalid.")
        request_failed = False
        response: httpx.Response | None = None
        try:
            with httpx.Client(
                transport=self._http_transport,
                timeout=self._health_timeout_seconds,
                follow_redirects=False,
            ) as client:
                response = client.post(handle.endpoint_url, json={"nonce": nonce})
        except Exception:
            request_failed = True
        if request_failed or response is None:
            raise ComputeTargetError("The Modal health check could not be reached.")
        if response.status_code != 200:
            return HealthResult(healthy=False, echo="")
        invalid_response = False
        try:
            payload = response.json()
        except Exception:
            invalid_response = True
            payload = None
        if invalid_response or not isinstance(payload, dict):
            return HealthResult(healthy=False, echo="")
        echo = payload.get("echo")
        if not isinstance(echo, str) or len(echo) > 128:
            echo = ""
        matches = echo == nonce
        return HealthResult(
            healthy=payload.get("healthy") is True and matches,
            echo=echo if matches else "",
        )

    def inspect(self, handle: DeploymentHandle) -> TargetState:
        validate_owned_handle(handle)
        return self._inspect_identity(
            deployment_id=handle.deployment_id,
            provider_app_id=handle.provider_app_id,
            app_name=handle.app_name,
            ownership_tag=handle.ownership_tag,
            endpoint_url=handle.endpoint_url,
        )

    def stop(self, handle: DeploymentHandle) -> None:
        validate_owned_handle(handle)
        self._stop_identity(
            deployment_id=handle.deployment_id,
            provider_app_id=handle.provider_app_id,
            app_name=handle.app_name,
            ownership_tag=handle.ownership_tag,
            endpoint_url=handle.endpoint_url,
        )

    def stop_owned(self, state: TargetState) -> None:
        """Operator-only stop path for an enumerated app with no endpoint URL."""

        self._validate_state_identity(state)
        apps = self._provider_call(
            self._gateway.list_apps,
            "The Modal App could not be inspected before cleanup.",
        )
        listed = next(
            (app for app in apps if app.app_id == state.provider_app_id),
            None,
        )
        if listed is None:
            current = self._inspect_identity(
                deployment_id=state.deployment_id,
                provider_app_id=state.provider_app_id,
                app_name=state.app_name,
                ownership_tag=state.ownership_tag,
                endpoint_url=state.endpoint_url,
            )
            if current.stopped_verified:
                return
            raise ComputeOwnershipError(
                "The enumerated Modal App is no longer addressable."
            )
        if listed.app_name != state.app_name:
            raise ComputeOwnershipError(
                "The enumerated Modal App name does not match its ID."
            )
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(state.app_name),
            "The Modal App name could not be verified before cleanup.",
        )
        if resolved is not None and resolved.app_id != state.provider_app_id:
            raise ComputeOwnershipError(
                "The enumerated Modal App ID does not match its name."
            )
        tags = self._provider_call(
            lambda: self._gateway.get_tags(state.provider_app_id),
            "The Modal App ownership could not be verified before cleanup.",
        )
        if not self._tags_match(tags, state.deployment_id):
            raise ComputeOwnershipError(
                "The enumerated Modal App ownership tag is invalid."
            )
        self._provider_call(
            lambda: self._gateway.stop(state.provider_app_id),
            "The Modal App could not be stopped.",
        )
        self._wait_for_stop(
            deployment_id=state.deployment_id,
            provider_app_id=state.provider_app_id,
            app_name=state.app_name,
            ownership_tag=state.ownership_tag,
            endpoint_url=state.endpoint_url,
        )

    def list_owned(self) -> list[TargetState]:
        owned, unverifiable = self.list_owned_for_panic()
        if unverifiable:
            raise ComputeTargetError(
                "Some ctrl-pi Modal Apps could not be verified."
            )
        return owned

    def list_owned_for_panic(self) -> tuple[list[TargetState], list[str]]:
        """Enumerate each exact-name candidate without one bad App hiding others."""

        apps = self._provider_call(
            self._gateway.list_apps,
            "Modal Apps could not be listed.",
        )
        owned: list[TargetState] = []
        unverifiable: list[str] = []
        for app in apps:
            deployment_id = self._deployment_id_from_name(app.app_name)
            if deployment_id is None:
                continue
            try:
                tags = self._provider_call(
                    lambda app_id=app.app_id: self._gateway.get_tags(app_id),
                    "Modal App ownership could not be verified.",
                )
            except ComputeTargetError:
                unverifiable.append(app.app_name)
                continue
            marker = tags.get(MODAL_OWNERSHIP_TAG_KEY)
            if marker is None:
                if app.lifecycle == "stopped" and app.running_tasks == 0:
                    continue
                unverifiable.append(app.app_name)
                continue
            if marker != str(deployment_id):
                continue
            try:
                lifecycle = self._provider_call(
                    lambda app_id=app.app_id: self._gateway.get_lifecycle(app_id),
                    "Modal App lifecycle could not be inspected.",
                )
            except ComputeTargetError:
                lifecycle = app.lifecycle
                unverifiable.append(app.app_name)
            effective_lifecycle = lifecycle or app.lifecycle
            if effective_lifecycle == "stopped" and app.running_tasks == 0:
                continue
            endpoint_url = self._recover_endpoint_url(app.app_name)
            try:
                state = TargetState(
                    deployment_id=deployment_id,
                    provider_app_id=app.app_id,
                    app_name=app.app_name,
                    ownership_tag=deployment_ownership_tag(deployment_id),
                    exists=True,
                    lifecycle=effective_lifecycle,
                    running_tasks=app.running_tasks,
                    endpoint_url=endpoint_url,
                )
            except (ComputeTargetError, ValueError):
                unverifiable.append(app.app_name)
                continue
            owned.append(state)
        return (
            sorted(owned, key=lambda state: str(state.deployment_id)),
            sorted(set(unverifiable)),
        )

    def _reject_name_collision(self, spec: DeploymentSpec) -> None:
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(spec.app_name),
            "Modal App ownership could not be checked before deployment.",
        )
        apps = self._provider_call(
            self._gateway.list_apps,
            "Modal App ownership could not be checked before deployment.",
        )
        candidate_ids = {
            app.app_id for app in apps if app.app_name == spec.app_name
        }
        if resolved is not None:
            candidate_ids.add(resolved.app_id)
        if not candidate_ids:
            return
        for app_id in sorted(candidate_ids):
            tags = self._provider_call(
                lambda app_id=app_id: self._gateway.get_tags(app_id),
                "Modal App ownership could not be checked before deployment.",
            )
            if not self._tags_match(tags, spec.deployment_id):
                raise ComputeOwnershipError(
                    "A Modal App with this name is not owned by the deployment."
                )
        raise ComputeTargetError(
            "A Modal App already exists for this deployment."
        )

    def _verify_remote_identity(
        self,
        *,
        deployment_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
    ) -> None:
        apps = self._provider_call(
            self._gateway.list_apps,
            "The deployed Modal App identity could not be verified.",
        )
        listed = next(
            (app for app in apps if app.app_id == provider_app_id),
            None,
        )
        if listed is not None and listed.app_name != app_name:
            raise ComputeOwnershipError(
                "The deployed Modal App name does not match its ID."
            )
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(app_name),
            "The deployed Modal App identity could not be verified.",
        )
        if resolved is not None and resolved.app_id != provider_app_id:
            raise ComputeOwnershipError(
                "The deployed Modal App ID does not match its name."
            )
        if resolved is None and listed is None:
            raise ComputeOwnershipError(
                "The deployed Modal App is not addressable by name or listing."
            )
        tags = self._provider_call(
            lambda: self._gateway.get_tags(provider_app_id),
            "The deployed Modal App ownership could not be verified.",
        )
        if not self._tags_match(tags, deployment_id):
            raise ComputeOwnershipError(
                "The deployed Modal App ownership tag is invalid."
            )

    def _inspect_identity(
        self,
        *,
        deployment_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
        ownership_tag: str,
        endpoint_url: str | None,
        ownership_preverified: bool = False,
    ) -> TargetState:
        apps = self._provider_call(
            self._gateway.list_apps,
            "The Modal App could not be inspected.",
        )
        listed = next((app for app in apps if app.app_id == provider_app_id), None)
        lifecycle = self._provider_call(
            lambda: self._gateway.get_lifecycle(provider_app_id),
            "The Modal App lifecycle could not be inspected.",
        )
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(app_name),
            "The Modal App name could not be verified.",
        )
        exists = listed is not None or lifecycle is not None or (
            resolved is not None and resolved.app_id == provider_app_id
        )
        if not exists:
            return TargetState(
                deployment_id=deployment_id,
                provider_app_id=provider_app_id,
                app_name=app_name,
                ownership_tag=ownership_tag,
                exists=False,
                lifecycle="stopped",
                running_tasks=0,
                endpoint_url=endpoint_url,
            )
        if listed is not None and listed.app_name != app_name:
            raise ComputeOwnershipError(
                "The Modal App name does not match its provider ID."
            )
        if lifecycle is not None:
            effective_lifecycle = lifecycle
        elif listed is not None:
            effective_lifecycle = listed.lifecycle
        elif resolved is not None:
            effective_lifecycle = resolved.lifecycle
        else:
            effective_lifecycle = "unknown"
        if effective_lifecycle in _ACTIVE_LIFECYCLES:
            if resolved is not None and resolved.app_id != provider_app_id:
                raise ComputeOwnershipError(
                    "The active Modal App ID does not match its name."
                )
            if listed is None and resolved is None:
                raise ComputeOwnershipError(
                    "The active Modal App is not addressable by name or listing."
                )
        running_tasks = listed.running_tasks if listed is not None else 0
        if (
            lifecycle == "stopped"
            and (
                listed is None
                or (listed.app_name == app_name and running_tasks == 0)
            )
        ):
            return TargetState(
                deployment_id=deployment_id,
                provider_app_id=provider_app_id,
                app_name=app_name,
                ownership_tag=ownership_tag,
                exists=True,
                lifecycle="stopped",
                running_tasks=0,
                endpoint_url=endpoint_url,
            )
        tags: dict[str, str] | None = None
        try:
            tags = self._provider_call(
                lambda: self._gateway.get_tags(provider_app_id),
                "The Modal App ownership could not be inspected.",
            )
        except ComputeTargetError:
            if not ownership_preverified:
                raise
        if tags is not None and not self._tags_match(tags, deployment_id):
            marker = tags.get(MODAL_OWNERSHIP_TAG_KEY)
            if not (ownership_preverified and marker is None):
                raise ComputeOwnershipError(
                    "The Modal App ownership tag does not match its deployment."
                )
        return TargetState(
            deployment_id=deployment_id,
            provider_app_id=provider_app_id,
            app_name=app_name,
            ownership_tag=ownership_tag,
            exists=True,
            lifecycle=effective_lifecycle,
            running_tasks=running_tasks,
            endpoint_url=endpoint_url,
        )

    def _stop_identity(
        self,
        *,
        deployment_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
        ownership_tag: str,
        endpoint_url: str | None,
    ) -> None:
        state = self._inspect_identity(
            deployment_id=deployment_id,
            provider_app_id=provider_app_id,
            app_name=app_name,
            ownership_tag=ownership_tag,
            endpoint_url=endpoint_url,
        )
        if state.stopped_verified:
            return
        self._provider_call(
            lambda: self._gateway.stop(provider_app_id),
            "The Modal App could not be stopped.",
        )
        self._wait_for_stop(
            deployment_id=deployment_id,
            provider_app_id=provider_app_id,
            app_name=app_name,
            ownership_tag=ownership_tag,
            endpoint_url=endpoint_url,
        )

    def _wait_for_stop(
        self,
        *,
        deployment_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
        ownership_tag: str,
        endpoint_url: str | None,
    ) -> None:
        deadline = self._clock() + self._stop_timeout_seconds
        while True:
            state = self._inspect_identity(
                deployment_id=deployment_id,
                provider_app_id=provider_app_id,
                app_name=app_name,
                ownership_tag=ownership_tag,
                endpoint_url=endpoint_url,
                ownership_preverified=True,
            )
            if state.stopped_verified:
                return
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise ComputeTargetError(
                    "Modal teardown could not be verified before the deadline."
                )
            self._sleeper(min(self._poll_interval_seconds, remaining))

    def _best_effort_stop_owned(
        self,
        *,
        deployment_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
        ownership_tag: str,
    ) -> None:
        try:
            self._stop_identity(
                deployment_id=deployment_id,
                provider_app_id=provider_app_id,
                app_name=app_name,
                ownership_tag=ownership_tag,
                endpoint_url=None,
            )
        except Exception:
            return

    def _recover_endpoint_url(self, app_name: str) -> str | None:
        try:
            endpoint_url = self._gateway.get_endpoint_url(app_name)
        except Exception:
            return None
        if endpoint_url is None:
            return None
        try:
            self._validate_modal_endpoint(endpoint_url)
        except ComputeOwnershipError:
            return None
        return endpoint_url

    @staticmethod
    def _validate_spec(spec: DeploymentSpec) -> None:
        resources = spec.resources
        if spec.runtime != "stub" or resources.compute_size != "CPU":
            raise ComputeConfigurationError(
                "The M9 Modal probe supports only the stub CPU workload."
            )
        if (
            resources.min_containers != 0
            or resources.buffer_containers != 0
            or resources.max_containers != 1
            or not 2 <= resources.scaledown_window_seconds <= 60
            or not 1 <= resources.timeout_seconds <= 1800
        ):
            raise ComputeConfigurationError(
                "The Modal resource policy violates the V1 cost guardrails."
            )

    @staticmethod
    def _validate_state_identity(state: TargetState) -> None:
        if (
            state.app_name != deployment_app_name(state.deployment_id)
            or state.ownership_tag
            != deployment_ownership_tag(state.deployment_id)
        ):
            raise ComputeOwnershipError(
                "The Modal App is not owned by this deployment."
            )

    @staticmethod
    def _deployment_id_from_name(app_name: str) -> uuid.UUID | None:
        match = _OWNED_APP_NAME.fullmatch(app_name)
        if match is None:
            return None
        try:
            deployment_id = uuid.UUID(match.group("deployment_id"))
        except ValueError:
            return None
        if app_name != deployment_app_name(deployment_id):
            return None
        return deployment_id

    @staticmethod
    def _tags_match(tags: Mapping[str, str], deployment_id: uuid.UUID) -> bool:
        return tags.get(MODAL_OWNERSHIP_TAG_KEY) == str(deployment_id)

    @staticmethod
    def _validate_modal_endpoint(endpoint_url: str) -> None:
        try:
            parsed = urlsplit(endpoint_url)
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.scheme != "https"
            or parsed.hostname is None
            or not parsed.hostname.endswith(".modal.run")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ComputeOwnershipError("The Modal endpoint URL is invalid.")

    @staticmethod
    def _provider_call(callback: Callable[[], _T], message: str) -> _T:
        configuration_failed = False
        result: _T | object = _FAILED
        try:
            result = callback()
        except _GatewayConfigurationError:
            configuration_failed = True
        except Exception:
            pass
        if configuration_failed:
            raise ComputeConfigurationError(
                "Modal credentials are not configured."
            )
        if result is _FAILED:
            raise ComputeTargetError(message)
        return cast(_T, result)
