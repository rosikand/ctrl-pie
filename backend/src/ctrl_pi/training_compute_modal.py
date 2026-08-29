from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import modal
from modal._utils.async_utils import synchronizer
from modal.exception import TimeoutError as ModalTimeoutError
from modal_proto import api_pb2

from ctrl_pi.compute_modal import (
    _GatewayConfigurationError,
    _OfficialModalGateway,
    _ProviderApp,
    _ResolvedProviderApp,
)
from ctrl_pi.modal_training_workload import (
    MAX_WORKER_EVENT_LINE_BYTES,
    MODAL_TRAINING_EVENT_PREFIX,
    MODAL_TRAINING_EVENT_SCHEMA,
    MODAL_TRAINING_FUNCTION,
    MODAL_TRAINING_OWNERSHIP_TAG_KEY,
    MODAL_TRAINING_PROTOCOL_VERSION,
    MODAL_TRAINING_RESULT_SCHEMA,
    _strict_json,
    build_modal_training_workload,
    managed_training_spec_payload,
)
from ctrl_pi.training_compute import (
    ManagedTrainingConfigurationError,
    ManagedTrainingOwnershipError,
    ManagedTrainingProtocolError,
    ManagedTrainingSpec,
    ManagedTrainingTargetError,
    ManagedTrainingTransientError,
    TrainingCheckpointEvent,
    TrainingEvent,
    TrainingExecutionState,
    TrainingHandle,
    TrainingLogEvent,
    TrainingMetricEvent,
    TrainingPoll,
    TrainingResult,
    TrainingTargetKind,
    TrainingTargetState,
    managed_training_marker,
    training_app_name,
    training_ownership_tag,
    validate_training_handle,
    validate_training_state,
)

if TYPE_CHECKING:
    from ctrl_pi.config import AppConfig


_OWNED_TRAINING_APP_NAME = re.compile(
    r"ctrl-pi-training-(?P<job_id>[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})"
)
_REQUEST_HASH = re.compile(r"[0-9a-f]{64}")
_ACTIVE_LIFECYCLES = {"deploying", "running", "stopping", "failed", "unknown"}
MAX_MODAL_LOG_TAIL_ENTRIES = 1_000
MAX_MODAL_LOG_TAIL_BYTES = 1024 * 1024
MAX_MODAL_POLL_EVENTS = 200
DEFAULT_MODAL_LOG_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class _ProviderTrainingDeployment:
    app_id: str


@dataclass(frozen=True)
class _ProviderCallInfo:
    app_id: str
    execution_state: TrainingExecutionState


@dataclass(frozen=True)
class _ProviderCallPoll:
    info: _ProviderCallInfo
    result: object | None


@dataclass(frozen=True)
class _ProviderLogTail:
    lines: tuple[str, ...]
    truncated: bool


class _ModalTrainingGateway(Protocol):
    def deploy_training(self, spec: ManagedTrainingSpec) -> _ProviderTrainingDeployment: ...

    def spawn_training(self, app_name: str, payload: dict[str, object]) -> str: ...

    def inspect_call(self, function_call_id: str) -> _ProviderCallInfo: ...

    def poll_call(self, function_call_id: str) -> _ProviderCallPoll: ...

    def tail_call_logs(
        self,
        function_call_id: str,
        *,
        entries: int,
        max_bytes: int,
        timeout_seconds: float,
    ) -> _ProviderLogTail: ...

    def cancel_call(self, function_call_id: str) -> None: ...

    def list_apps(self) -> list[_ProviderApp]: ...

    def resolve_name(self, app_name: str) -> _ResolvedProviderApp | None: ...

    def get_lifecycle(self, app_id: str) -> str | None: ...

    def get_tags(self, app_id: str) -> dict[str, str]: ...

    def stop(self, app_id: str) -> None: ...


@synchronizer.create_blocking
async def _rpc_function_call_info(
    client: object,
    function_call_id: str,
) -> tuple[str, str, int, int, int, int, int, int]:
    from_id = await client.stub.FunctionCallFromId(  # type: ignore[attr-defined]
        api_pb2.FunctionCallFromIdRequest(function_call_id=function_call_id)
    )
    app_id = from_id.metadata.app_id
    function_id = from_id.metadata.function_id
    if not app_id or not function_id:
        raise RuntimeError("FunctionCall identity is unavailable")
    response = await client.stub.FunctionCallGetInfo(  # type: ignore[attr-defined]
        api_pb2.FunctionCallGetInfoRequest(
            function_id=function_id,
            function_call_id=function_call_id,
        )
    )
    info = response.info
    return (
        app_id,
        function_id,
        info.total_inputs,
        info.pending_inputs.total,
        info.succeeded_inputs.total,
        info.failed_inputs.total,
        info.timeout_inputs.total,
        info.cancelled_inputs.total,
    )


@synchronizer.create_blocking
async def _rpc_tail_protocol_logs(
    function_call: object,
    *,
    entries: int,
    max_bytes: int,
    timeout_seconds: float,
) -> tuple[list[str], bool]:
    async def collect() -> tuple[list[str], bool]:
        lines: list[str] = []
        total_bytes = 0
        scanned_characters = 0
        truncated = False
        async for entry in function_call.logs.tail(entries=entries):  # type: ignore[attr-defined]
            message = getattr(entry, "message", None)
            if not isinstance(message, str):
                continue
            scan_limit = max_bytes * 2
            remaining_characters = scan_limit - scanned_characters
            if remaining_characters <= 0:
                truncated = True
                break
            bounded_message = message[:remaining_characters]
            scanned_characters += len(bounded_message)
            if len(bounded_message) != len(message):
                truncated = True
            for line in bounded_message.splitlines():
                if not line.startswith(MODAL_TRAINING_EVENT_PREFIX):
                    continue
                size = len(line.encode("utf-8"))
                if size > MAX_WORKER_EVENT_LINE_BYTES or size > max_bytes - total_bytes:
                    truncated = True
                    continue
                lines.append(line)
                total_bytes += size
            if truncated and scanned_characters >= scan_limit:
                break
        return lines, truncated

    return await asyncio.wait_for(collect(), timeout=timeout_seconds)


def _execution_state_from_counts(
    *,
    total: int,
    pending: int,
    succeeded: int,
    failed: int,
    timed_out: int,
    cancelled: int,
) -> TrainingExecutionState:
    if cancelled > 0:
        return "cancelled"
    if failed > 0 or timed_out > 0:
        return "failed"
    if total > 0 and succeeded == total:
        return "succeeded"
    if pending > 0 or total == 0:
        return "pending"
    return "running"


class _OfficialModalTrainingGateway(_OfficialModalGateway):
    def deploy_training(self, spec: ManagedTrainingSpec) -> _ProviderTrainingDeployment:
        workload = build_modal_training_workload(
            spec=spec,
            hf_token=self._hf_token or "",
        )
        deployed_app: object | None = None
        deploy_failed = False
        try:
            deployed_app = workload.app.deploy(
                name=spec.app_name,
                environment_name=self._environment_name,
                client=self._client(),
                strategy="recreate",
            )
        except Exception:
            deploy_failed = True
        app_id = getattr(deployed_app, "app_id", None)
        if deploy_failed or not isinstance(app_id, str) or not app_id:
            recovered_ids: set[str] = set()
            recovery_failed = False
            try:
                resolved = self.resolve_name(spec.app_name)
                if resolved is not None and resolved.lifecycle != "stopped":
                    recovered_ids.add(resolved.app_id)
                recovered_ids.update(
                    app.app_id
                    for app in self.list_apps()
                    if app.app_name == spec.app_name and app.lifecycle != "stopped"
                )
                if len(recovered_ids) != 1:
                    recovery_failed = True
                else:
                    tags = self.get_tags(next(iter(recovered_ids)))
                    if tags.get(MODAL_TRAINING_OWNERSHIP_TAG_KEY) != str(spec.job_id):
                        recovery_failed = True
            except Exception:
                recovery_failed = True
            if recovery_failed or len(recovered_ids) != 1:
                raise RuntimeError("Modal training deployment did not complete")
            app_id = next(iter(recovered_ids))
        return _ProviderTrainingDeployment(app_id=app_id)

    def spawn_training(self, app_name: str, payload: dict[str, object]) -> str:
        function = modal.Function.from_name(
            app_name,
            MODAL_TRAINING_FUNCTION,
            environment_name=self._environment_name,
            client=self._client(),
        )
        call = function.spawn(payload)
        call_id = getattr(call, "object_id", None)
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError("Modal training FunctionCall ID is unavailable")
        return call_id

    def inspect_call(self, function_call_id: str) -> _ProviderCallInfo:
        (
            app_id,
            _function_id,
            total,
            pending,
            succeeded,
            failed,
            timed_out,
            cancelled,
        ) = _rpc_function_call_info(self._client(), function_call_id)
        return _ProviderCallInfo(
            app_id=app_id,
            execution_state=_execution_state_from_counts(
                total=total,
                pending=pending,
                succeeded=succeeded,
                failed=failed,
                timed_out=timed_out,
                cancelled=cancelled,
            ),
        )

    def poll_call(self, function_call_id: str) -> _ProviderCallPoll:
        info = self.inspect_call(function_call_id)
        result: object | None = None
        if info.execution_state == "succeeded":
            call = modal.FunctionCall.from_id(
                function_call_id,
                client=self._client(),
            )
            try:
                result = call.get(timeout=0)
            except ModalTimeoutError:
                return _ProviderCallPoll(
                    info=_ProviderCallInfo(info.app_id, "running"),
                    result=None,
                )
            except Exception:
                return _ProviderCallPoll(
                    # The provider already reported success; a transport or
                    # result-fetch failure is retryable and must not rewrite it
                    # as a failed training execution.
                    info=_ProviderCallInfo(info.app_id, "unknown"),
                    result=None,
                )
        return _ProviderCallPoll(info=info, result=result)

    def tail_call_logs(
        self,
        function_call_id: str,
        *,
        entries: int,
        max_bytes: int,
        timeout_seconds: float,
    ) -> _ProviderLogTail:
        call = modal.FunctionCall.from_id(
            function_call_id,
            client=self._client(),
        )
        lines, truncated = _rpc_tail_protocol_logs(
            call,
            entries=entries,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )
        return _ProviderLogTail(tuple(lines), truncated)

    def cancel_call(self, function_call_id: str) -> None:
        modal.FunctionCall.from_id(
            function_call_id,
            client=self._client(),
        ).cancel(terminate_containers=True)


class ModalTrainingTarget:
    """Exact-ownership Modal adapter for one long-running training call per App."""

    def __init__(
        self,
        *,
        token_id: str | None = None,
        token_secret: str | None = None,
        hf_token: str | None = None,
        environment_name: str | None = None,
        gateway: _ModalTrainingGateway | None = None,
        stop_timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.25,
        log_timeout_seconds: float = DEFAULT_MODAL_LOG_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if stop_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("Modal training teardown bounds must be positive")
        if log_timeout_seconds <= 0 or log_timeout_seconds > 30:
            raise ValueError("Modal training log timeout is invalid")
        self._gateway = gateway or _OfficialModalTrainingGateway(
            token_id=token_id,
            token_secret=token_secret,
            environment_name=environment_name,
            hf_token=hf_token,
        )
        self._hf_token = hf_token
        self._stop_timeout_seconds = stop_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._log_timeout_seconds = log_timeout_seconds
        self._clock = clock
        self._sleeper = sleeper

    @classmethod
    def from_config(cls, config: AppConfig) -> ModalTrainingTarget:
        return cls(
            token_id=(
                config.modal_token_id.get_secret_value()
                if config.modal_token_id is not None
                else None
            ),
            token_secret=(
                config.modal_token_secret.get_secret_value()
                if config.modal_token_secret is not None
                else None
            ),
            hf_token=(
                config.hf_token.get_secret_value()
                if config.hf_token is not None
                else None
            ),
        )

    @property
    def kind(self) -> TrainingTargetKind:
        return "modal"

    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle:
        # Re-run marker/spec construction to fail before touching Modal.
        managed_training_marker(spec.job_id, spec.request_hash)
        if self._hf_token is None or not self._hf_token.strip():
            raise ManagedTrainingConfigurationError(
                "Hugging Face credentials are required for managed training."
            )
        self._reject_name_collision(spec)
        deployment = self._provider_call(
            lambda: self._gateway.deploy_training(spec),
            "The Modal training App could not be deployed.",
        )
        verified = False
        call_id: str | None = None
        try:
            self._verify_remote_identity(
                job_id=spec.job_id,
                provider_app_id=deployment.app_id,
                app_name=spec.app_name,
            )
            verified = True
            call_id = self._provider_call(
                lambda: self._gateway.spawn_training(
                    spec.app_name,
                    managed_training_spec_payload(spec),
                ),
                "The Modal training call could not be started.",
            )
            call_info = self._provider_call(
                lambda: self._gateway.inspect_call(cast(str, call_id)),
                "The Modal training call identity could not be verified.",
                transient=True,
            )
            if call_info.app_id != deployment.app_id:
                raise ManagedTrainingOwnershipError(
                    "The Modal training call does not belong to its App."
                )
            handle = TrainingHandle(
                job_id=spec.job_id,
                provider_app_id=deployment.app_id,
                provider_function_call_id=call_id,
                app_name=spec.app_name,
                ownership_tag=spec.ownership_tag,
                request_hash=spec.request_hash,
            )
            return handle
        except (ManagedTrainingTargetError, ValueError):
            if verified:
                if call_id is not None:
                    self._best_effort_cancel_exact_call(
                        call_id=call_id,
                        provider_app_id=deployment.app_id,
                    )
                self._best_effort_stop_owned(
                    job_id=spec.job_id,
                    provider_app_id=deployment.app_id,
                    app_name=spec.app_name,
                    ownership_tag=spec.ownership_tag,
                )
            raise

    def poll(
        self,
        handle: TrainingHandle,
        *,
        after_sequence: int,
        limit: int,
    ) -> TrainingPoll:
        validate_training_handle(handle)
        if handle.request_hash is None:
            raise ManagedTrainingTargetError(
                "The managed training request identity is unavailable."
            )
        if isinstance(after_sequence, bool) or not 0 <= after_sequence <= 9_007_199_254_740_991:
            raise ManagedTrainingTargetError("The managed training event cursor is invalid.")
        if isinstance(limit, bool) or not 1 <= limit <= MAX_MODAL_POLL_EVENTS:
            raise ManagedTrainingTargetError("The managed training event page size is invalid.")
        state = self.inspect(handle)
        if handle.provider_function_call_id is None:
            return TrainingPoll(
                state=state,
                events=(),
                next_sequence=after_sequence,
                truncated=False,
                has_more=False,
                result=None,
            )
        tail = self._provider_call(
            lambda: self._gateway.tail_call_logs(
                cast(str, handle.provider_function_call_id),
                entries=MAX_MODAL_LOG_TAIL_ENTRIES,
                max_bytes=MAX_MODAL_LOG_TAIL_BYTES,
                timeout_seconds=self._log_timeout_seconds,
            ),
            "Managed training events could not be fetched.",
            transient=True,
        )
        expected_hash = handle.request_hash
        events: list[TrainingEvent] = []
        highest_observed_sequence = after_sequence
        previous_tail_sequence = 0
        observed_hash: str | None = expected_hash
        for line in tail.lines:
            try:
                event, event_hash = self._parse_event(line, handle.job_id)
                if observed_hash is None:
                    observed_hash = event_hash
                elif event_hash != observed_hash:
                    raise ValueError("managed training request hash changed")
                if event.sequence <= previous_tail_sequence:
                    raise ValueError("managed training event sequence is not monotonic")
                previous_tail_sequence = event.sequence
                highest_observed_sequence = max(highest_observed_sequence, event.sequence)
                if event.sequence > after_sequence:
                    events.append(event)
            except (TypeError, ValueError):
                raise ManagedTrainingProtocolError(
                    "The Modal training event protocol is invalid."
                ) from None
        deduplicated = events
        gap = bool(deduplicated and deduplicated[0].sequence > after_sequence + 1)
        gap = gap or any(
            right.sequence != left.sequence + 1
            for left, right in zip(deduplicated, deduplicated[1:], strict=False)
        )
        has_more = len(deduplicated) > limit
        page = tuple(deduplicated[:limit])
        next_sequence = page[-1].sequence if page else after_sequence
        result: TrainingResult | None = None
        call_poll = self._provider_call(
            lambda: self._gateway.poll_call(cast(str, handle.provider_function_call_id)),
            "The Modal training call could not be inspected.",
            transient=True,
        )
        if call_poll.info.app_id != handle.provider_app_id:
            raise ManagedTrainingOwnershipError(
                "The Modal training call does not belong to its App."
            )
        if call_poll.info.execution_state != state.execution_state:
            state = TrainingTargetState(
                **{
                    **state.__dict__,
                    "execution_state": call_poll.info.execution_state,
                }
            )
        result_truncated = False
        if call_poll.result is not None:
            result, result_truncated, result_last_sequence = self._parse_result(
                call_poll.result,
                job_id=handle.job_id,
                expected_request_hash=observed_hash,
            )
            if result_last_sequence < highest_observed_sequence:
                raise ManagedTrainingProtocolError(
                    "The Modal training result cursor regressed."
                )
            result_truncated = result_truncated or (
                result_last_sequence > highest_observed_sequence
            )
        return TrainingPoll(
            state=state,
            events=page,
            next_sequence=next_sequence,
            truncated=tail.truncated or gap or result_truncated,
            has_more=has_more,
            result=result,
        )

    def inspect(self, handle: TrainingHandle) -> TrainingTargetState:
        validate_training_handle(handle)
        state = self._inspect_identity(
            job_id=handle.job_id,
            provider_app_id=handle.provider_app_id,
            provider_function_call_id=handle.provider_function_call_id,
            app_name=handle.app_name,
            ownership_tag=handle.ownership_tag,
        )
        return state

    def cancel(self, handle: TrainingHandle) -> None:
        validate_training_handle(handle)
        # App ownership is verified independently. If FunctionCall metadata has
        # expired, exact AppStop still provides the cancellation boundary.
        self._verify_handle_app(handle)
        if handle.provider_function_call_id is not None:
            try:
                info = self._provider_call(
                    lambda: self._gateway.inspect_call(
                        cast(str, handle.provider_function_call_id)
                    ),
                    "The Modal training call identity could not be verified.",
                    transient=True,
                )
                if info.app_id != handle.provider_app_id:
                    raise ManagedTrainingOwnershipError(
                        "The Modal training call does not belong to its App."
                    )
                self._provider_call(
                    lambda: self._gateway.cancel_call(
                        cast(str, handle.provider_function_call_id)
                    ),
                    "The Modal training call could not be cancelled.",
                )
            except ManagedTrainingOwnershipError:
                self._stop_app_identity_without_call(handle)
                raise
            except ManagedTrainingTargetError:
                # Do not mutate an unverified call ID; exact owned App cleanup
                # below still terminates its tasks and is verified.
                pass
        self._stop_identity(handle)

    def _stop_app_identity_without_call(self, handle: TrainingHandle) -> None:
        app_only = TrainingHandle(
            job_id=handle.job_id,
            provider_app_id=handle.provider_app_id,
            provider_function_call_id=None,
            app_name=handle.app_name,
            ownership_tag=handle.ownership_tag,
            request_hash=handle.request_hash,
        )
        self._stop_identity(app_only)

    def stop(self, handle: TrainingHandle) -> None:
        validate_training_handle(handle)
        self._stop_identity(handle)

    def list_owned(self) -> list[TrainingTargetState]:
        owned, unverifiable = self.list_owned_for_panic()
        if unverifiable:
            raise ManagedTrainingTargetError(
                "Some ctrl-pi Modal training Apps could not be verified."
            )
        return owned

    def list_owned_for_panic(self) -> tuple[list[TrainingTargetState], list[str]]:
        apps = self._provider_call(
            self._gateway.list_apps,
            "Modal training Apps could not be listed.",
            transient=True,
        )
        owned: list[TrainingTargetState] = []
        unverifiable: list[str] = []
        for app in apps:
            job_id = self._job_id_from_name(app.app_name)
            if job_id is None:
                continue
            try:
                tags = self._provider_call(
                    lambda app_id=app.app_id: self._gateway.get_tags(app_id),
                    "Modal training App ownership could not be verified.",
                    transient=True,
                )
            except ManagedTrainingTargetError:
                unverifiable.append(app.app_name)
                continue
            marker = tags.get(MODAL_TRAINING_OWNERSHIP_TAG_KEY)
            if marker is None:
                if app.lifecycle == "stopped" and app.running_tasks == 0:
                    continue
                unverifiable.append(app.app_name)
                continue
            if marker != str(job_id):
                continue
            lifecycle = cast(Any, app.lifecycle)
            try:
                current = self._provider_call(
                    lambda app_id=app.app_id: self._gateway.get_lifecycle(app_id),
                    "Modal training App lifecycle could not be inspected.",
                    transient=True,
                )
                if current is not None:
                    lifecycle = current
            except ManagedTrainingTargetError:
                unverifiable.append(app.app_name)
            if lifecycle == "stopped" and app.running_tasks == 0:
                continue
            try:
                owned.append(
                    TrainingTargetState(
                        job_id=job_id,
                        provider_app_id=app.app_id,
                        provider_function_call_id=None,
                        app_name=app.app_name,
                        ownership_tag=training_ownership_tag(job_id),
                        exists=True,
                        resource_lifecycle=lifecycle,
                        execution_state="unknown",
                        running_tasks=app.running_tasks,
                    )
                )
            except (ManagedTrainingTargetError, ValueError):
                unverifiable.append(app.app_name)
        return (
            sorted(owned, key=lambda state: str(state.job_id)),
            sorted(set(unverifiable)),
        )

    def stop_owned(self, state: TrainingTargetState) -> None:
        validate_training_state(state)
        handle = state.handle()
        self._verify_handle_app(handle)
        self._stop_identity(handle)

    def _inspect_identity(
        self,
        *,
        job_id: uuid.UUID,
        provider_app_id: str,
        provider_function_call_id: str | None,
        app_name: str,
        ownership_tag: str,
        ownership_preverified: bool = False,
    ) -> TrainingTargetState:
        apps = self._provider_call(
            self._gateway.list_apps,
            "The Modal training App could not be inspected.",
            transient=True,
        )
        listed = next((app for app in apps if app.app_id == provider_app_id), None)
        lifecycle = self._provider_call(
            lambda: self._gateway.get_lifecycle(provider_app_id),
            "The Modal training App lifecycle could not be inspected.",
            transient=True,
        )
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(app_name),
            "The Modal training App name could not be verified.",
            transient=True,
        )
        exists = listed is not None or lifecycle is not None or (
            resolved is not None and resolved.app_id == provider_app_id
        )
        if not exists:
            execution = self._call_execution_state(
                provider_function_call_id,
                provider_app_id,
            )
            return TrainingTargetState(
                job_id=job_id,
                provider_app_id=provider_app_id,
                provider_function_call_id=provider_function_call_id,
                app_name=app_name,
                ownership_tag=ownership_tag,
                exists=False,
                resource_lifecycle=(
                    "unknown" if execution in {"pending", "running"} else "stopped"
                ),
                execution_state=execution,
                # A live FunctionCall contradicts an AppList miss. Preserve a
                # conservative nonzero task count until Modal proves teardown.
                running_tasks=1 if execution in {"pending", "running"} else 0,
            )
        if listed is not None and listed.app_name != app_name:
            raise ManagedTrainingOwnershipError(
                "The Modal training App name does not match its ID."
            )
        effective_lifecycle = lifecycle or (
            listed.lifecycle if listed is not None else resolved.lifecycle if resolved else "unknown"
        )
        execution = self._call_execution_state(
            provider_function_call_id,
            provider_app_id,
        )
        running_tasks = listed.running_tasks if listed is not None else 0
        if (
            effective_lifecycle == "stopped"
            and running_tasks == 0
            and execution not in {"pending", "running"}
            and (listed is None or listed.app_name == app_name)
        ):
            # Modal may remove tags/name resolution from stopped history. A
            # lifecycle-by-ID response plus zero listed tasks is the pinned
            # SDK's provider proof for idempotent teardown.
            return TrainingTargetState(
                job_id=job_id,
                provider_app_id=provider_app_id,
                provider_function_call_id=provider_function_call_id,
                app_name=app_name,
                ownership_tag=ownership_tag,
                exists=True,
                resource_lifecycle="stopped",
                execution_state=execution,
                running_tasks=0,
            )
        if effective_lifecycle in _ACTIVE_LIFECYCLES:
            if resolved is not None and resolved.app_id != provider_app_id:
                raise ManagedTrainingOwnershipError(
                    "The active Modal training App ID does not match its name."
                )
            if listed is None and resolved is None:
                raise ManagedTrainingOwnershipError(
                    "The active Modal training App is not addressable."
                )
        tags: dict[str, str] | None = None
        try:
            tags = self._provider_call(
                lambda: self._gateway.get_tags(provider_app_id),
                "The Modal training App ownership could not be inspected.",
                transient=True,
            )
        except ManagedTrainingTargetError:
            if not ownership_preverified:
                raise
        if tags is not None and not self._tags_match(tags, job_id):
            marker = tags.get(MODAL_TRAINING_OWNERSHIP_TAG_KEY)
            if not (ownership_preverified and marker is None):
                raise ManagedTrainingOwnershipError(
                    "The Modal training App ownership tag is invalid."
                )
        return TrainingTargetState(
            job_id=job_id,
            provider_app_id=provider_app_id,
            provider_function_call_id=provider_function_call_id,
            app_name=app_name,
            ownership_tag=ownership_tag,
            exists=True,
            resource_lifecycle=cast(Any, effective_lifecycle),
            execution_state=execution,
            running_tasks=running_tasks,
        )

    def _call_execution_state(
        self,
        call_id: str | None,
        provider_app_id: str,
    ) -> TrainingExecutionState:
        if call_id is None:
            return "unknown"
        try:
            info = self._provider_call(
                lambda: self._gateway.inspect_call(call_id),
                "The Modal training call could not be inspected.",
                transient=True,
            )
        except ManagedTrainingTargetError:
            return "unknown"
        if info.app_id != provider_app_id:
            raise ManagedTrainingOwnershipError(
                "The Modal training call does not belong to its App."
            )
        return info.execution_state

    def _verify_handle_app(self, handle: TrainingHandle) -> None:
        self._verify_remote_identity(
            job_id=handle.job_id,
            provider_app_id=handle.provider_app_id,
            app_name=handle.app_name,
        )

    def _stop_identity(self, handle: TrainingHandle) -> None:
        state = self.inspect(handle)
        if state.stopped_verified:
            return
        self._verify_handle_app(handle)
        self._provider_call(
            lambda: self._gateway.stop(handle.provider_app_id),
            "The Modal training App could not be stopped.",
        )
        deadline = self._clock() + self._stop_timeout_seconds
        while True:
            state = self._inspect_identity(
                job_id=handle.job_id,
                provider_app_id=handle.provider_app_id,
                provider_function_call_id=handle.provider_function_call_id,
                app_name=handle.app_name,
                ownership_tag=handle.ownership_tag,
                ownership_preverified=True,
            )
            if state.stopped_verified:
                return
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise ManagedTrainingTargetError(
                    "Modal training teardown could not be verified before the deadline."
                )
            self._sleeper(min(self._poll_interval_seconds, remaining))

    def _reject_name_collision(self, spec: ManagedTrainingSpec) -> None:
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(spec.app_name),
            "Modal training App ownership could not be checked before launch.",
            transient=True,
        )
        apps = self._provider_call(
            self._gateway.list_apps,
            "Modal training App ownership could not be checked before launch.",
            transient=True,
        )
        candidate_ids = {app.app_id for app in apps if app.app_name == spec.app_name}
        if resolved is not None:
            candidate_ids.add(resolved.app_id)
        if not candidate_ids:
            return
        for app_id in sorted(candidate_ids):
            tags = self._provider_call(
                lambda app_id=app_id: self._gateway.get_tags(app_id),
                "Modal training App ownership could not be checked before launch.",
                transient=True,
            )
            if not self._tags_match(tags, spec.job_id):
                raise ManagedTrainingOwnershipError(
                    "A Modal App with this training name is not owned by the job."
                )
        raise ManagedTrainingTargetError(
            "A Modal training App already exists for this job."
        )

    def _verify_remote_identity(
        self,
        *,
        job_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
    ) -> None:
        apps = self._provider_call(
            self._gateway.list_apps,
            "The Modal training App identity could not be verified.",
            transient=True,
        )
        listed = next((app for app in apps if app.app_id == provider_app_id), None)
        if listed is not None and listed.app_name != app_name:
            raise ManagedTrainingOwnershipError(
                "The Modal training App name does not match its ID."
            )
        resolved = self._provider_call(
            lambda: self._gateway.resolve_name(app_name),
            "The Modal training App identity could not be verified.",
            transient=True,
        )
        if resolved is not None and resolved.app_id != provider_app_id:
            raise ManagedTrainingOwnershipError(
                "The Modal training App ID does not match its name."
            )
        if resolved is None and listed is None:
            raise ManagedTrainingOwnershipError(
                "The Modal training App is not addressable."
            )
        tags = self._provider_call(
            lambda: self._gateway.get_tags(provider_app_id),
            "The Modal training App ownership could not be verified.",
            transient=True,
        )
        if not self._tags_match(tags, job_id):
            raise ManagedTrainingOwnershipError(
                "The Modal training App ownership tag is invalid."
            )

    def _best_effort_cancel_exact_call(self, *, call_id: str, provider_app_id: str) -> None:
        try:
            info = self._gateway.inspect_call(call_id)
            if info.app_id == provider_app_id:
                self._gateway.cancel_call(call_id)
        except BaseException:
            return

    def _best_effort_stop_owned(
        self,
        *,
        job_id: uuid.UUID,
        provider_app_id: str,
        app_name: str,
        ownership_tag: str,
    ) -> None:
        try:
            self._stop_identity(
                TrainingHandle(
                    job_id=job_id,
                    provider_app_id=provider_app_id,
                    provider_function_call_id=None,
                    app_name=app_name,
                    ownership_tag=ownership_tag,
                )
            )
        except BaseException:
            return

    @staticmethod
    def _parse_event(line: str, job_id: uuid.UUID) -> tuple[TrainingEvent, str]:
        if not line.startswith(MODAL_TRAINING_EVENT_PREFIX):
            raise ValueError("managed training event prefix is invalid")
        encoded = line[len(MODAL_TRAINING_EVENT_PREFIX) :]
        if len(encoded.encode("utf-8")) > MAX_WORKER_EVENT_LINE_BYTES:
            raise ValueError("managed training event is too large")
        payload = _strict_json(encoded)
        common = {"schema", "version", "job_id", "request_hash", "sequence", "type"}
        if not isinstance(payload, dict):
            raise ValueError("managed training event is invalid")
        if (
            payload.get("schema") != MODAL_TRAINING_EVENT_SCHEMA
            or type(payload.get("version")) is not int
            or payload.get("version") != MODAL_TRAINING_PROTOCOL_VERSION
            or payload.get("job_id") != str(job_id)
        ):
            raise ValueError("managed training event identity is invalid")
        request_hash = payload.get("request_hash")
        if not isinstance(request_hash, str) or _REQUEST_HASH.fullmatch(request_hash) is None:
            raise ValueError("managed training event request hash is invalid")
        sequence = payload.get("sequence")
        kind = payload.get("type")
        payload_fields = set(payload)
        if kind == "log" and payload_fields in (
            common | {"source", "line"},
            common | {"source", "line", "step"},
        ):
            if payload.get("source") not in {"stdout", "stderr", "system"}:
                raise ValueError("managed training log source is invalid")
            event = TrainingLogEvent(
                sequence=cast(int, sequence),
                source=cast(Any, payload["source"]),
                line=cast(str, payload["line"]),
                step=cast(int | None, payload.get("step")),
            )
        elif kind == "metric" and payload_fields == common | {"step", "metrics"}:
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict) or any(
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                for name, value in metrics.items()
            ):
                raise ValueError("managed training metric values are invalid")
            event = TrainingMetricEvent(
                sequence=cast(int, sequence),
                step=cast(int, payload["step"]),
                metrics={str(name): float(value) for name, value in metrics.items()},
            )
        elif kind == "checkpoint" and payload_fields == common | {
            "repo_id",
            "revision",
            "step",
            "final",
        }:
            event = TrainingCheckpointEvent(
                sequence=cast(int, sequence),
                repo_id=cast(str, payload["repo_id"]),
                revision=cast(str, payload["revision"]),
                step=cast(int, payload["step"]),
                final=cast(bool, payload["final"]),
            )
        else:
            raise ValueError("managed training event schema is invalid")
        return event, request_hash

    @staticmethod
    def _parse_result(
        payload: object,
        *,
        job_id: uuid.UUID,
        expected_request_hash: str | None,
    ) -> tuple[TrainingResult, bool, int]:
        fields = {
            "schema",
            "version",
            "job_id",
            "request_hash",
            "output_model_repo",
            "revision",
            "step",
            "last_sequence",
            "events_truncated",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ManagedTrainingProtocolError(
                "The Modal training result protocol is invalid."
            )
        request_hash = payload.get("request_hash")
        if (
            payload.get("schema") != MODAL_TRAINING_RESULT_SCHEMA
            or type(payload.get("version")) is not int
            or payload.get("version") != MODAL_TRAINING_PROTOCOL_VERSION
            or payload.get("job_id") != str(job_id)
            or not isinstance(request_hash, str)
            or _REQUEST_HASH.fullmatch(request_hash) is None
            or (expected_request_hash is not None and request_hash != expected_request_hash)
            or not isinstance(payload.get("events_truncated"), bool)
        ):
            raise ManagedTrainingProtocolError(
                "The Modal training result identity is invalid."
            )
        last_sequence = payload.get("last_sequence")
        if isinstance(last_sequence, bool) or not isinstance(last_sequence, int) or not 0 <= last_sequence <= 9_007_199_254_740_991:
            raise ManagedTrainingProtocolError(
                "The Modal training result cursor is invalid."
            )
        try:
            result = TrainingResult(
                job_id=job_id,
                request_hash=request_hash,
                output_model_repo=cast(str, payload["output_model_repo"]),
                revision=cast(str, payload["revision"]),
                step=cast(int, payload["step"]),
            )
        except (TypeError, ValueError):
            raise ManagedTrainingProtocolError(
                "The Modal training result artifact is invalid."
            ) from None
        return (
            result,
            cast(bool, payload["events_truncated"]),
            cast(int, last_sequence),
        )

    @staticmethod
    def _tags_match(tags: dict[str, str], job_id: uuid.UUID) -> bool:
        return tags.get(MODAL_TRAINING_OWNERSHIP_TAG_KEY) == str(job_id)

    @staticmethod
    def _job_id_from_name(app_name: str) -> uuid.UUID | None:
        match = _OWNED_TRAINING_APP_NAME.fullmatch(app_name)
        if match is None:
            return None
        try:
            job_id = uuid.UUID(match.group("job_id"))
        except ValueError:
            return None
        return job_id if app_name == training_app_name(job_id) else None

    @staticmethod
    def _provider_call(
        function: Callable[[], Any],
        message: str,
        *,
        transient: bool = False,
    ) -> Any:
        try:
            return function()
        except _GatewayConfigurationError:
            raise ManagedTrainingConfigurationError(
                "Modal credentials are not configured."
            ) from None
        except ManagedTrainingTargetError:
            raise
        except Exception:
            error_type = (
                ManagedTrainingTransientError
                if transient
                else ManagedTrainingTargetError
            )
            raise error_type(message) from None


__all__ = [
    "DEFAULT_MODAL_LOG_TIMEOUT_SECONDS",
    "MAX_MODAL_LOG_TAIL_BYTES",
    "MAX_MODAL_LOG_TAIL_ENTRIES",
    "MAX_MODAL_POLL_EVENTS",
    "ModalTrainingTarget",
]
