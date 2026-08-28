from __future__ import annotations

import asyncio
import copy
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.compute import (
    ComputeConfigurationError,
    ComputeOwnershipError,
    ComputeTarget,
    ComputeTargetError,
    DeploymentHandle,
    DeploymentSpec,
    HealthResult,
    ResourcePolicy,
    TargetKind,
    TargetState,
    deployment_app_name,
    deployment_ownership_tag,
    validate_owned_handle,
)
from ctrl_pi.models import Deployment, InferenceEndpoint, Robot

DeploymentStatus = Literal[
    "created", "deploying", "running", "stopping", "stopped", "failed"
]
_Result = TypeVar("_Result")


class DeploymentServiceError(RuntimeError):
    """A safe lifecycle-service error."""


class DeploymentNotFoundError(DeploymentServiceError):
    pass


class DeploymentConflictError(DeploymentServiceError):
    pass


class DeploymentConfigurationError(DeploymentServiceError):
    pass


class DeploymentProviderError(DeploymentServiceError):
    pass


class DeploymentStorageError(DeploymentServiceError):
    pass


@dataclass(frozen=True)
class ResolvedModelRevision:
    model_repo: str
    revision: str


class ModelRevisionResolver(Protocol):
    def resolve(
        self,
        *,
        model_repo: str,
        revision: str | None,
    ) -> ResolvedModelRevision: ...


class HFModelRevisionResolver:
    """Resolve one requested model ref to its exact Hub repository and SHA."""

    def __init__(
        self,
        token: str | None,
        namespace: str | None,
        *,
        hub_api_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._token = token
        self._namespace = namespace
        self._hub_api_factory = hub_api_factory

    def resolve(
        self,
        *,
        model_repo: str,
        revision: str | None,
    ) -> ResolvedModelRevision:
        namespace = self._namespace
        if (
            namespace is None
            or not namespace
            or namespace.strip() != namespace
            or re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?",
                namespace,
            )
            is None
            or ".." in namespace
            or "--" in namespace
        ):
            raise DeploymentConfigurationError(
                "HF_NAMESPACE is required to resolve model revisions."
            )
        if model_repo.count("/") != 1 or model_repo.split("/", 1)[0] != namespace:
            raise DeploymentConfigurationError(
                "The model repository is outside the configured HF_NAMESPACE."
            )
        token = self._token
        if token is None or not token.strip():
            raise DeploymentConfigurationError(
                "HF_TOKEN is required to resolve model revisions."
            )
        resolution_failed = False
        info: Any = None
        try:
            if self._hub_api_factory is None:
                from huggingface_hub import HfApi

                api = HfApi(token=token)
            else:
                api = self._hub_api_factory(token)
            info = api.model_info(
                repo_id=model_repo,
                revision=revision,
                token=token,
            )
        except Exception:
            resolution_failed = True
        if resolution_failed:
            raise DeploymentProviderError(
                "The Hugging Face model revision could not be resolved."
            )

        returned_repo = getattr(info, "id", None) or getattr(info, "repo_id", None)
        returned_revision = getattr(info, "sha", None)
        if (
            not isinstance(returned_repo, str)
            or returned_repo != model_repo
            or not isinstance(returned_revision, str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", returned_revision) is None
            or (
                revision is not None
                and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None
                and returned_revision.casefold() != revision.casefold()
            )
        ):
            raise DeploymentProviderError(
                "Hugging Face returned an invalid model revision identity."
            )
        return ResolvedModelRevision(
            model_repo=returned_repo,
            revision=returned_revision.casefold(),
        )


@dataclass(frozen=True)
class DeploymentRecord:
    id: uuid.UUID
    endpoint_id: uuid.UUID
    name: str
    target_kind: TargetKind
    status: DeploymentStatus
    model_repo: str
    checkpoint_revision: str | None
    runtime: str
    compute_size: str
    endpoint_url: str | None
    provider_app_id: str | None
    arm_id: str | None
    record_session: bool
    recording_id: uuid.UUID | None
    started_at: datetime | None
    stopped_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DeploymentService:
    """Coordinates durable deployment state with one provider target."""

    def __init__(
        self,
        target: ComputeTarget,
        *,
        model_revision_resolver: ModelRevisionResolver | None = None,
        session_factory: Callable[[], Session] | None = None,
        nonce_factory: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        stop_verify_timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self.target = target
        self._model_revision_resolver = model_revision_resolver
        self._session_factory = session_factory
        self._nonce_factory = nonce_factory or (lambda: secrets.token_urlsafe(24))
        self._monotonic = monotonic
        self._stop_verify_timeout_seconds = max(stop_verify_timeout_seconds, 0.0)
        self._poll_interval_seconds = max(poll_interval_seconds, 0.0)
        self._active_stops: set[uuid.UUID] = set()
        self._active_stops_lock = threading.Lock()

    @property
    def session_factory(self) -> Callable[[], Session] | None:
        return self._session_factory

    async def deploy(
        self,
        db: Session,
        *,
        name: str,
        model_repo: str,
        checkpoint_revision: str | None,
        runtime: str,
        compute_size: str,
        timeout_seconds: int,
    ) -> DeploymentRecord:
        task = asyncio.create_task(
            self._deploy(
                db,
                name=name,
                model_repo=model_repo,
                checkpoint_revision=checkpoint_revision,
                runtime=runtime,
                compute_size=compute_size,
                timeout_seconds=timeout_seconds,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            completed, record = await self._settle_cancelled_task(task)
            if completed and record is not None:
                cleanup = asyncio.create_task(self._stop(db, record.id))
                await self._settle_cancelled_task(cleanup)
            raise

    async def _deploy(
        self,
        db: Session,
        *,
        name: str,
        model_repo: str,
        checkpoint_revision: str | None,
        runtime: str,
        compute_size: str,
        timeout_seconds: int,
    ) -> DeploymentRecord:
        model_repo, checkpoint_revision, resources = await self._prepare_deployment(
            model_repo=model_repo,
            checkpoint_revision=checkpoint_revision,
            runtime=runtime,
            compute_size=compute_size,
            timeout_seconds=timeout_seconds,
        )
        endpoint = InferenceEndpoint(
            name=name,
            runtime=runtime,
            status="created",
        )
        deployment = Deployment(
            endpoint_id=None,
            model_repo=model_repo,
            checkpoint_revision=checkpoint_revision,
            runtime=runtime,
            compute_size=compute_size,
            target_kind=self.target.kind,
            status="created",
            record_session=False,
        )
        try:
            db.add(endpoint)
            db.flush()
            deployment.endpoint_id = endpoint.id
            db.add(deployment)
            self._commit(db)
        except SQLAlchemyError:
            db.rollback()
            raise DeploymentStorageError(
                "PostgreSQL could not create deployment lifecycle state."
            ) from None
        deployment_id = deployment.id

        spec = DeploymentSpec(
            deployment_id=deployment_id,
            app_name=deployment_app_name(deployment_id),
            ownership_tag=deployment_ownership_tag(deployment_id),
            model_repo=model_repo,
            checkpoint_revision=checkpoint_revision,
            runtime=runtime,
            resources=resources,
        )
        self._set_status(
            db,
            deployment_id,
            allowed={"created"},
            status="deploying",
        )

        handle: DeploymentHandle | None = None
        try:
            handle = await asyncio.to_thread(self.target.deploy, spec)
            self._validate_handle(handle, deployment_id)
            self._persist_handle(db, deployment_id, handle)
            if handle.endpoint_url is None:
                raise DeploymentProviderError(
                    "The compute target endpoint URL is unavailable."
                )

            nonce = self._nonce_factory()
            if not nonce or len(nonce) > 128:
                raise DeploymentProviderError(
                    "The compute target health check could not be verified."
                )
            health = await asyncio.to_thread(self.target.health, handle, nonce)
            self._validate_health(
                health,
                nonce,
                runtime=runtime,
                model_repo=model_repo,
                checkpoint_revision=checkpoint_revision,
            )
            state = await asyncio.to_thread(self.target.inspect, handle)
            self._validate_state(state, handle)
            if not state.running_verified:
                raise DeploymentProviderError(
                    "The compute target did not reach a running state."
                )
            self._set_running(db, deployment_id, handle)
            return self.get(db, deployment_id)
        except DeploymentStorageError:
            if handle is None:
                handle = await self._recover_created_handle(deployment_id)
            if handle is not None:
                await self._compensating_stop(handle)
                self._mark_failed(db, deployment_id, handle=handle)
            else:
                self._mark_failed(db, deployment_id)
            raise
        except ComputeConfigurationError:
            if handle is None:
                handle = await self._recover_created_handle(deployment_id)
            if handle is not None:
                await self._compensating_stop(handle)
            self._mark_failed(db, deployment_id, handle=handle)
            raise DeploymentConfigurationError(
                "The compute target is not configured."
            ) from None
        except (ComputeTargetError, DeploymentProviderError, ValueError):
            if handle is None:
                handle = await self._recover_created_handle(deployment_id)
            if handle is not None:
                await self._compensating_stop(handle)
            self._mark_failed(db, deployment_id, handle=handle)
            raise DeploymentProviderError(
                "The compute target deployment or health check failed."
            ) from None
        except Exception:
            if handle is None:
                handle = await self._recover_created_handle(deployment_id)
            if handle is not None:
                await self._compensating_stop(handle)
            self._mark_failed(db, deployment_id, handle=handle)
            raise DeploymentProviderError(
                "The compute target deployment or health check failed."
            ) from None

    async def _prepare_deployment(
        self,
        *,
        model_repo: str,
        checkpoint_revision: str | None,
        runtime: str,
        compute_size: str,
        timeout_seconds: int,
    ) -> tuple[str, str | None, ResourcePolicy]:
        supported_gpu_sizes = {"Modal: A10G", "Modal: A100", "Modal: H100"}
        if runtime == "stub":
            if compute_size != "CPU":
                raise DeploymentConfigurationError(
                    "The stub runtime requires CPU compute."
                )
        elif runtime in {"lerobot", "openpi"}:
            if compute_size not in supported_gpu_sizes:
                raise DeploymentConfigurationError(
                    "The inference runtime requires a supported Modal GPU."
                )
            if runtime == "openpi" and self.target.kind == "modal":
                raise DeploymentConfigurationError(
                    "The real OpenPI Modal runtime is not available in V1."
                )
        else:
            raise DeploymentConfigurationError(
                "The inference runtime is not supported."
            )

        try:
            resources = ResourcePolicy(
                compute_size=compute_size,
                timeout_seconds=timeout_seconds,
            )
        except ValueError:
            raise DeploymentConfigurationError(
                "The deployment resource policy is invalid."
            ) from None
        # Mock mode is a deliberately credential/network-free acceptance path.
        # It may serve or emulate any runtime, but it must be given an immutable
        # identity up front rather than consulting mutable Hub state.
        from ctrl_pi.compute_stub import StubComputeTarget

        if isinstance(self.target, StubComputeTarget):
            if (
                not isinstance(checkpoint_revision, str)
                or re.fullmatch(r"[0-9a-fA-F]{40}", checkpoint_revision) is None
            ):
                raise DeploymentConfigurationError(
                    "Mock inference requires an exact 40-character model SHA."
                )
            return model_repo, checkpoint_revision.casefold(), resources
        if runtime == "stub":
            return model_repo, checkpoint_revision, resources

        resolver = self._model_revision_resolver
        if resolver is None:
            raise DeploymentConfigurationError(
                "Model revision resolution is not configured."
            )
        configuration_failed = False
        resolution_failed = False
        resolved: ResolvedModelRevision | None = None
        try:
            resolved = await asyncio.to_thread(
                resolver.resolve,
                model_repo=model_repo,
                revision=checkpoint_revision,
            )
        except DeploymentConfigurationError:
            configuration_failed = True
        except DeploymentProviderError:
            resolution_failed = True
        except Exception:
            resolution_failed = True
        if configuration_failed:
            raise DeploymentConfigurationError(
                "Model revision resolution is not configured."
            )
        if resolution_failed:
            raise DeploymentProviderError(
                "The Hugging Face model revision could not be resolved."
            )
        if not isinstance(resolved, ResolvedModelRevision) or (
            not isinstance(resolved.model_repo, str)
            or resolved.model_repo != model_repo
            or not isinstance(resolved.revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", resolved.revision) is None
        ):
            raise DeploymentProviderError(
                "Hugging Face returned an invalid model revision identity."
            )
        return resolved.model_repo, resolved.revision, resources

    def get(self, db: Session, deployment_id: uuid.UUID) -> DeploymentRecord:
        deployment, endpoint = self._load_pair(db, deployment_id, lock=False)
        return self._record(deployment, endpoint, arm_id=self._arm_id(db, deployment))

    def list(self, db: Session) -> list[DeploymentRecord]:
        try:
            rows = db.execute(
                select(Deployment, InferenceEndpoint, Robot.driver_id)
                .join(
                    InferenceEndpoint,
                    InferenceEndpoint.id == Deployment.endpoint_id,
                )
                .outerjoin(Robot, Robot.id == Deployment.robot_id)
                .order_by(Deployment.created_at.desc(), Deployment.id.desc())
            ).all()
        except SQLAlchemyError:
            raise DeploymentStorageError(
                "PostgreSQL could not list deployment lifecycle state."
            ) from None
        return [
            self._record(deployment, endpoint, arm_id=arm_id)
            for deployment, endpoint, arm_id in rows
        ]

    async def stop(
        self, db: Session, deployment_id: uuid.UUID
    ) -> DeploymentRecord:
        with self._active_stops_lock:
            if deployment_id in self._active_stops:
                raise DeploymentConflictError(
                    "A stop operation is already active for this deployment."
                )
            self._active_stops.add(deployment_id)
        try:
            return await self._run_to_terminal(self._stop(db, deployment_id))
        finally:
            with self._active_stops_lock:
                self._active_stops.discard(deployment_id)

    async def _stop(
        self, db: Session, deployment_id: uuid.UUID
    ) -> DeploymentRecord:
        deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
        if deployment.target_kind != self.target.kind:
            raise DeploymentConfigurationError(
                "The deployment belongs to a different compute target."
            )
        if deployment.status == "stopped":
            return self._record(
                deployment,
                endpoint,
                arm_id=self._arm_id(db, deployment),
            )
        if deployment.status not in {"running", "failed", "stopping"}:
            raise DeploymentConflictError(
                "The deployment cannot be stopped from its current state."
            )
        if deployment.status != "stopping":
            self._assign_status(deployment, endpoint, "stopping")
            self._commit(db)

        try:
            handle = await self._resolve_handle(deployment_id, endpoint)
            if handle is None:
                confirmed, state = await self._confirm_owned_absence(deployment_id)
                if not confirmed:
                    raise DeploymentProviderError(
                        "The compute target could not verify complete teardown."
                    )
                if state is None:
                    self._set_stopped(db, deployment_id)
                    return self.get(db, deployment_id)
                self._validate_state_identity(state, deployment_id)
                handle = state.handle()
            await self._stop_and_verify(handle)
            self._set_stopped(db, deployment_id, handle=handle)
            return self.get(db, deployment_id)
        except ComputeConfigurationError:
            self._mark_failed(db, deployment_id)
            raise DeploymentConfigurationError(
                "The compute target is not configured."
            ) from None
        except (ComputeTargetError, DeploymentProviderError, ValueError):
            self._mark_failed(db, deployment_id)
            raise DeploymentProviderError(
                "The compute target could not verify complete teardown."
            ) from None
        except DeploymentStorageError:
            raise
        except Exception:
            self._mark_failed(db, deployment_id)
            raise DeploymentProviderError(
                "The compute target could not verify complete teardown."
            ) from None

    async def reconcile_startup(self) -> None:
        if self._session_factory is None:
            return
        try:
            owned = await asyncio.to_thread(self.target.list_owned)
            owned_by_id = self._owned_state_map(owned)
        except Exception:
            return

        try:
            with self._session_factory() as db:
                deployment_ids = list(
                    db.scalars(
                        select(Deployment.id).where(
                            Deployment.target_kind == self.target.kind,
                            Deployment.status.in_(
                                ("deploying", "stopping", "running", "failed")
                            )
                        )
                    )
                )
        except SQLAlchemyError:
            return

        for deployment_id in deployment_ids:
            try:
                await self._reconcile_one(deployment_id, owned_by_id.get(deployment_id))
            except Exception:
                # Startup remains available; unresolved rows stay retryable.
                continue

    async def _reconcile_one(
        self,
        deployment_id: uuid.UUID,
        state: TargetState | None,
    ) -> None:
        if self._session_factory is None:
            return
        status: str
        runtime: str
        model_repo: str
        checkpoint_revision: str | None
        persisted_handle: DeploymentHandle | None = None
        with self._session_factory() as db:
            deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
            status = deployment.status
            runtime = deployment.runtime
            model_repo = deployment.model_repo
            checkpoint_revision = deployment.checkpoint_revision
            if status not in {"deploying", "stopping", "running", "failed"}:
                return
            if endpoint.provider_app_id is not None:
                persisted_handle = DeploymentHandle(
                    deployment_id=deployment_id,
                    provider_app_id=endpoint.provider_app_id,
                    app_name=deployment_app_name(deployment_id),
                    ownership_tag=deployment_ownership_tag(deployment_id),
                    endpoint_url=endpoint.endpoint_url,
                )
            if state is not None:
                self._validate_state_identity(state, deployment_id)
                if (
                    endpoint.provider_app_id is not None
                    and endpoint.provider_app_id != state.provider_app_id
                ):
                    self._assign_status(deployment, endpoint, "failed")
                    self._commit(db)
                    return
                endpoint.provider_app_id = state.provider_app_id
                if state.endpoint_url is not None:
                    endpoint.endpoint_url = state.endpoint_url
                self._commit(db)

        if persisted_handle is not None:
            try:
                inspected = await asyncio.to_thread(
                    self.target.inspect, persisted_handle
                )
                self._validate_state(inspected, persisted_handle)
                state = inspected
            except Exception:
                return
        if state is None:
            confirmed, state = await self._confirm_owned_absence(deployment_id)
            if not confirmed:
                return
            if state is None:
                if status == "failed":
                    return
                with self._session_factory() as db:
                    if status == "stopping":
                        self._set_stopped(db, deployment_id)
                    else:
                        self._mark_failed(db, deployment_id)
                return
        if state.stopped_verified:
            with self._session_factory() as db:
                if status == "stopping":
                    self._set_stopped(db, deployment_id)
                else:
                    self._mark_failed(db, deployment_id)
            return
        handle = state.handle()
        if status == "failed":
            await self._compensating_stop(handle)
            with self._session_factory() as db:
                self._mark_failed(db, deployment_id, handle=handle)
            return
        if status == "stopping":
            if not state.stopped_verified:
                try:
                    await self._stop_and_verify(handle)
                except Exception:
                    with self._session_factory() as db:
                        self._mark_failed(db, deployment_id, handle=handle)
                    return
            with self._session_factory() as db:
                self._set_stopped(db, deployment_id, handle=handle)
            return

        if handle.endpoint_url is None:
            await self._compensating_stop(handle)
            with self._session_factory() as db:
                self._mark_failed(db, deployment_id, handle=handle)
            return
        if not state.running_verified:
            await self._compensating_stop(handle)
            with self._session_factory() as db:
                self._mark_failed(db, deployment_id, handle=handle)
            return
        nonce = self._nonce_factory()
        try:
            health = await asyncio.to_thread(self.target.health, handle, nonce)
            self._validate_health(
                health,
                nonce,
                runtime=runtime,
                model_repo=model_repo,
                checkpoint_revision=checkpoint_revision,
            )
        except Exception:
            await self._compensating_stop(handle)
            with self._session_factory() as db:
                self._mark_failed(db, deployment_id, handle=handle)
            return
        with self._session_factory() as db:
            self._set_running(db, deployment_id, handle)

    async def _confirm_owned_absence(
        self, deployment_id: uuid.UUID
    ) -> tuple[bool, TargetState | None]:
        await asyncio.sleep(self._poll_interval_seconds)
        try:
            states = await asyncio.to_thread(self.target.list_owned)
            owned = self._owned_state_map(states)
        except Exception:
            return False, None
        return True, owned.get(deployment_id)

    async def _recover_created_handle(
        self, deployment_id: uuid.UUID
    ) -> DeploymentHandle | None:
        try:
            states = await asyncio.to_thread(self.target.list_owned)
            matches = [state for state in states if state.deployment_id == deployment_id]
            if len(matches) != 1:
                return None
            state = matches[0]
            self._validate_state_identity(state, deployment_id)
            return state.handle()
        except Exception:
            return None

    @staticmethod
    async def _run_to_terminal(
        operation: Coroutine[Any, Any, _Result],
    ) -> _Result:
        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                try:
                    task.result()
                except Exception:
                    pass
            raise

    @staticmethod
    async def _settle_cancelled_task(
        task: asyncio.Task[_Result],
    ) -> tuple[bool, _Result | None]:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.cancelled():
            return False, None
        try:
            return True, task.result()
        except Exception:
            return False, None

    async def _resolve_handle(
        self,
        deployment_id: uuid.UUID,
        endpoint: InferenceEndpoint,
    ) -> DeploymentHandle | None:
        if endpoint.provider_app_id is not None:
            handle = DeploymentHandle(
                deployment_id=deployment_id,
                provider_app_id=endpoint.provider_app_id,
                app_name=deployment_app_name(deployment_id),
                ownership_tag=deployment_ownership_tag(deployment_id),
                endpoint_url=endpoint.endpoint_url,
            )
            self._validate_handle(handle, deployment_id)
            return handle

        states = await asyncio.to_thread(self.target.list_owned)
        matches = [state for state in states if state.deployment_id == deployment_id]
        if len(matches) > 1:
            raise ComputeOwnershipError(
                "Multiple compute resources claim this deployment."
            )
        if not matches:
            return None
        state = matches[0]
        self._validate_state_identity(state, deployment_id)
        if (
            endpoint.provider_app_id is not None
            and endpoint.provider_app_id != state.provider_app_id
        ):
            raise ComputeOwnershipError(
                "Compute provider identity does not match persisted state."
            )
        if (
            endpoint.endpoint_url is not None
            and state.endpoint_url is not None
            and endpoint.endpoint_url != state.endpoint_url
        ):
            raise ComputeOwnershipError(
                "Compute endpoint URL does not match persisted state."
            )
        return state.handle()

    async def _stop_and_verify(self, handle: DeploymentHandle) -> TargetState:
        self._validate_handle(handle, handle.deployment_id)
        await asyncio.to_thread(self.target.stop, handle)
        deadline = self._monotonic() + self._stop_verify_timeout_seconds
        while True:
            state = await asyncio.to_thread(self.target.inspect, handle)
            self._validate_state(state, handle)
            if state.stopped_verified:
                return state
            if self._monotonic() >= deadline:
                raise DeploymentProviderError(
                    "The compute target did not verify complete teardown."
                )
            await asyncio.sleep(self._poll_interval_seconds)

    async def _compensating_stop(self, handle: DeploymentHandle) -> bool:
        try:
            await self._stop_and_verify(handle)
        except Exception:
            return False
        return True

    def _persist_handle(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        handle: DeploymentHandle,
    ) -> None:
        deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
        if deployment.status != "deploying":
            raise DeploymentConflictError(
                "Deployment state changed while compute was starting."
            )
        endpoint.provider_app_id = handle.provider_app_id
        endpoint.endpoint_url = handle.endpoint_url
        self._commit(db)

    def _set_running(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        handle: DeploymentHandle,
    ) -> None:
        deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
        if deployment.status not in {"deploying", "running"}:
            raise DeploymentConflictError(
                "Deployment state changed before health verification completed."
            )
        endpoint.provider_app_id = handle.provider_app_id
        endpoint.endpoint_url = handle.endpoint_url
        now = datetime.now(UTC)
        deployment.started_at = deployment.started_at or now
        deployment.stopped_at = None
        endpoint.last_heartbeat_at = now
        self._assign_status(deployment, endpoint, "running")
        self._commit(db)

    def _set_stopped(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        *,
        handle: DeploymentHandle | None = None,
    ) -> None:
        deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
        if handle is not None:
            endpoint.provider_app_id = handle.provider_app_id
            endpoint.endpoint_url = handle.endpoint_url
        self._assign_status(deployment, endpoint, "stopped")
        deployment.stopped_at = datetime.now(UTC)
        self._commit(db)

    def _mark_failed(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        *,
        handle: DeploymentHandle | None = None,
    ) -> None:
        try:
            deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
            if deployment.status == "stopped":
                return
            if handle is not None:
                endpoint.provider_app_id = handle.provider_app_id
                endpoint.endpoint_url = handle.endpoint_url
            self._assign_status(deployment, endpoint, "failed")
            self._commit(db)
        except DeploymentStorageError:
            raise

    def _set_status(
        self,
        db: Session,
        deployment_id: uuid.UUID,
        *,
        allowed: set[str],
        status: DeploymentStatus,
    ) -> None:
        deployment, endpoint = self._load_pair(db, deployment_id, lock=True)
        if deployment.status not in allowed:
            raise DeploymentConflictError(
                "The deployment cannot transition from its current state."
            )
        self._assign_status(deployment, endpoint, status)
        self._commit(db)

    @staticmethod
    def _assign_status(
        deployment: Deployment,
        endpoint: InferenceEndpoint,
        status: DeploymentStatus,
    ) -> None:
        deployment.status = status
        endpoint.status = status

    @staticmethod
    def _commit(db: Session) -> None:
        failed = False
        try:
            db.commit()
        except SQLAlchemyError:
            failed = True
            db.rollback()
        if failed:
            raise DeploymentStorageError(
                "PostgreSQL could not persist deployment lifecycle state."
            )

    @staticmethod
    def _load_pair(
        db: Session,
        deployment_id: uuid.UUID,
        *,
        lock: bool,
    ) -> tuple[Deployment, InferenceEndpoint]:
        statement = select(Deployment).where(Deployment.id == deployment_id)
        if lock:
            statement = statement.with_for_update()
        try:
            deployment = db.scalar(statement)
        except SQLAlchemyError:
            raise DeploymentStorageError(
                "PostgreSQL could not read deployment lifecycle state."
            ) from None
        if deployment is None:
            raise DeploymentNotFoundError("Deployment was not found.")
        if deployment.endpoint_id is None:
            raise DeploymentStorageError(
                "Deployment endpoint metadata is unavailable."
            )
        endpoint_statement = select(InferenceEndpoint).where(
            InferenceEndpoint.id == deployment.endpoint_id
        )
        if lock:
            endpoint_statement = endpoint_statement.with_for_update()
        try:
            endpoint = db.scalar(endpoint_statement)
        except SQLAlchemyError:
            raise DeploymentStorageError(
                "PostgreSQL could not read deployment endpoint state."
            ) from None
        if endpoint is None:
            raise DeploymentStorageError(
                "Deployment endpoint metadata is unavailable."
            )
        return deployment, endpoint

    def _record(
        self,
        deployment: Deployment,
        endpoint: InferenceEndpoint,
        *,
        arm_id: str | None,
    ) -> DeploymentRecord:
        if deployment.status != endpoint.status:
            raise DeploymentStorageError(
                "Deployment and endpoint lifecycle states are inconsistent."
            )
        return DeploymentRecord(
            id=deployment.id,
            endpoint_id=endpoint.id,
            name=endpoint.name,
            target_kind=deployment.target_kind,
            status=deployment.status,
            model_repo=deployment.model_repo,
            checkpoint_revision=deployment.checkpoint_revision,
            runtime=deployment.runtime,
            compute_size=deployment.compute_size,
            endpoint_url=endpoint.endpoint_url,
            provider_app_id=endpoint.provider_app_id,
            arm_id=arm_id,
            record_session=deployment.record_session,
            recording_id=deployment.recording_id,
            started_at=self._as_utc(deployment.started_at),
            stopped_at=self._as_utc(deployment.stopped_at),
            created_at=self._as_utc(deployment.created_at),
            updated_at=self._as_utc(deployment.updated_at),
        )

    @staticmethod
    def _arm_id(db: Session, deployment: Deployment) -> str | None:
        if deployment.robot_id is None:
            return None
        try:
            return db.scalar(
                select(Robot.driver_id).where(Robot.id == deployment.robot_id)
            )
        except SQLAlchemyError:
            raise DeploymentStorageError(
                "PostgreSQL could not read the deployment robot identity."
            ) from None

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _validate_handle(
        handle: DeploymentHandle,
        deployment_id: uuid.UUID,
    ) -> None:
        validate_owned_handle(handle)
        if handle.deployment_id != deployment_id:
            raise ComputeOwnershipError(
                "Compute handle belongs to a different deployment."
            )

    @staticmethod
    def _validate_health(
        health: HealthResult,
        nonce: str,
        *,
        runtime: str,
        model_repo: str,
        checkpoint_revision: str | None,
    ) -> None:
        if not health.healthy or not secrets.compare_digest(health.echo, nonce):
            raise DeploymentProviderError(
                "The compute target health check could not be verified."
            )
        identity = (health.runtime, health.model_repo, health.revision)
        if runtime == "stub":
            if any(value is not None for value in identity):
                raise DeploymentProviderError(
                    "The compute target health identity could not be verified."
                )
            return
        if (
            checkpoint_revision is None
            or health.runtime != runtime
            or health.model_repo != model_repo
            or health.revision != checkpoint_revision
        ):
            raise DeploymentProviderError(
                "The compute target health identity could not be verified."
            )

    @classmethod
    def _validate_state(
        cls,
        state: TargetState,
        handle: DeploymentHandle,
    ) -> None:
        cls._validate_state_identity(state, handle.deployment_id)
        if (
            state.provider_app_id != handle.provider_app_id
            or state.app_name != handle.app_name
            or state.ownership_tag != handle.ownership_tag
        ):
            raise ComputeOwnershipError(
                "Compute inspection returned a different resource."
            )

    @staticmethod
    def _validate_state_identity(
        state: TargetState,
        deployment_id: uuid.UUID,
    ) -> None:
        if (
            state.deployment_id != deployment_id
            or state.app_name != deployment_app_name(deployment_id)
            or state.ownership_tag != deployment_ownership_tag(deployment_id)
        ):
            raise ComputeOwnershipError(
                "Compute state belongs to a different deployment."
            )

    @classmethod
    def _owned_state_map(
        cls, states: list[TargetState]
    ) -> dict[uuid.UUID, TargetState]:
        result: dict[uuid.UUID, TargetState] = {}
        for state in states:
            cls._validate_state_identity(state, state.deployment_id)
            if state.deployment_id in result:
                raise ComputeOwnershipError(
                    "Multiple compute resources claim one deployment."
                )
            result[state.deployment_id] = copy.deepcopy(state)
        return result
