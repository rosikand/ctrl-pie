from __future__ import annotations

import io
import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ctrl_pi.compute import TargetState, deployment_app_name, deployment_ownership_tag
from ctrl_pi.compute_modal import _ProviderApp, _ResolvedProviderApp
from ctrl_pi.modal_panic import CombinedModalPanicTarget, run_modal_panic
from ctrl_pi.modal_training_workload import (
    MODAL_TRAINING_EVENT_PREFIX,
    MODAL_TRAINING_EVENT_SCHEMA,
    MODAL_TRAINING_OWNERSHIP_TAG_KEY,
    MODAL_TRAINING_PROTOCOL_VERSION,
    MODAL_TRAINING_RESULT_SCHEMA,
)
from ctrl_pi.training_compute import (
    ManagedTrainingOwnershipError,
    ManagedTrainingProtocolError,
    ManagedTrainingSpec,
    ManagedTrainingTargetError,
    ManagedTrainingTransientError,
    TrainingHandle,
    TrainingTargetState,
    training_app_name,
    training_ownership_tag,
)
from ctrl_pi.training_compute_modal import (
    MAX_MODAL_LOG_TAIL_BYTES,
    MAX_MODAL_LOG_TAIL_ENTRIES,
    ModalTrainingTarget,
    _OfficialModalTrainingGateway,
    _ProviderCallInfo,
    _ProviderCallPoll,
    _ProviderLogTail,
    _ProviderTrainingDeployment,
)


JOB_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
OTHER_JOB_ID = uuid.UUID("44444444-4444-4444-8444-444444444444")
APP_ID = "ap-training111111"
CALL_ID = "fc-training111111"
REQUEST_HASH = "a" * 64
REVISION = "b" * 40
RESULT_REVISION = "c" * 40
SECRET = "modal-secret-must-not-leak"


def _spec() -> ManagedTrainingSpec:
    return ManagedTrainingSpec(
        job_id=JOB_ID,
        request_hash=REQUEST_HASH,
        app_name=training_app_name(JOB_ID),
        ownership_tag=training_ownership_tag(JOB_ID),
        dataset_repo="owner/dataset",
        dataset_revision=REVISION,
        base_model="owner/model",
        base_model_revision=REVISION,
        output_model_repo="owner/output",
        output_marker_revision=REVISION,
        output_private=True,
        runtime="lerobot",
        max_steps=100,
        batch_size=8,
        log_every=10,
        save_every=50,
        seed=1,
        num_workers=2,
        compute_size="Modal: A10G",
        timeout_seconds=3_600,
        deadline_at=datetime.now(UTC) + timedelta(hours=1),
    )


def _handle(*, request_hash: str | None = REQUEST_HASH) -> TrainingHandle:
    return TrainingHandle(
        job_id=JOB_ID,
        provider_app_id=APP_ID,
        provider_function_call_id=CALL_ID,
        app_name=training_app_name(JOB_ID),
        ownership_tag=training_ownership_tag(JOB_ID),
        request_hash=request_hash,
    )


def _event(sequence: int, kind: str = "log", **values: object) -> str:
    payload: dict[str, object] = {
        "schema": MODAL_TRAINING_EVENT_SCHEMA,
        "version": MODAL_TRAINING_PROTOCOL_VERSION,
        "job_id": str(JOB_ID),
        "request_hash": REQUEST_HASH,
        "sequence": sequence,
        "type": kind,
    }
    if kind == "log":
        payload.update({"source": "stdout", "line": f"line {sequence}"})
    elif kind == "metric":
        payload.update({"step": sequence, "metrics": {"loss": 0.5}})
    elif kind == "checkpoint":
        payload.update(
            {
                "repo_id": "owner/output",
                "revision": RESULT_REVISION,
                "step": sequence,
                "final": False,
            }
        )
    payload.update(values)
    return MODAL_TRAINING_EVENT_PREFIX + json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _result(*, last_sequence: int = 3, events_truncated: bool = False) -> dict[str, object]:
    return {
        "schema": MODAL_TRAINING_RESULT_SCHEMA,
        "version": MODAL_TRAINING_PROTOCOL_VERSION,
        "job_id": str(JOB_ID),
        "request_hash": REQUEST_HASH,
        "output_model_repo": "owner/output",
        "revision": RESULT_REVISION,
        "step": 100,
        "last_sequence": last_sequence,
        "events_truncated": events_truncated,
    }


class FakeGateway:
    def __init__(self) -> None:
        self.apps: list[_ProviderApp] = []
        self.tags: dict[str, dict[str, str]] = {}
        self.lifecycles: dict[str, str | None] = {}
        self.resolved: dict[str, _ResolvedProviderApp] = {}
        self.call_info = _ProviderCallInfo(APP_ID, "running")
        self.call_result: object | None = None
        self.log_tail = _ProviderLogTail((), False)
        self.payload: dict[str, object] | None = None
        self.deploy_calls = 0
        self.spawn_calls = 0
        self.cancel_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.spawn_error: Exception | None = None
        self.call_error: Exception | None = None
        self.tail_error: Exception | None = None

    def add_app(
        self,
        *,
        job_id: uuid.UUID = JOB_ID,
        app_id: str = APP_ID,
        app_name: str | None = None,
        tag_value: str | None = None,
        lifecycle: str = "running",
        running_tasks: int = 1,
    ) -> None:
        name = app_name or training_app_name(job_id)
        self.apps.append(
            _ProviderApp(
                app_id=app_id,
                app_name=name,
                lifecycle=lifecycle,  # type: ignore[arg-type]
                running_tasks=running_tasks,
            )
        )
        self.tags[app_id] = {
            MODAL_TRAINING_OWNERSHIP_TAG_KEY: tag_value or str(job_id)
        }
        self.lifecycles[app_id] = lifecycle
        self.resolved[name] = _ResolvedProviderApp(
            app_id=app_id,
            lifecycle=lifecycle,  # type: ignore[arg-type]
        )

    def deploy_training(self, spec: ManagedTrainingSpec) -> _ProviderTrainingDeployment:
        self.deploy_calls += 1
        self.add_app(job_id=spec.job_id)
        return _ProviderTrainingDeployment(APP_ID)

    def spawn_training(self, app_name: str, payload: dict[str, object]) -> str:
        self.spawn_calls += 1
        self.payload = payload
        if self.spawn_error is not None:
            raise self.spawn_error
        return CALL_ID

    def inspect_call(self, function_call_id: str) -> _ProviderCallInfo:
        assert function_call_id == CALL_ID
        if self.call_error is not None:
            raise self.call_error
        return self.call_info

    def poll_call(self, function_call_id: str) -> _ProviderCallPoll:
        return _ProviderCallPoll(self.inspect_call(function_call_id), self.call_result)

    def tail_call_logs(self, function_call_id: str, **kwargs: Any) -> _ProviderLogTail:
        assert function_call_id == CALL_ID
        assert kwargs == {
            "entries": MAX_MODAL_LOG_TAIL_ENTRIES,
            "max_bytes": MAX_MODAL_LOG_TAIL_BYTES,
            "timeout_seconds": 5.0,
        }
        if self.tail_error is not None:
            raise self.tail_error
        return self.log_tail

    def cancel_call(self, function_call_id: str) -> None:
        self.cancel_calls.append(function_call_id)
        self.call_info = replace(self.call_info, execution_state="cancelled")

    def list_apps(self) -> list[_ProviderApp]:
        return list(self.apps)

    def resolve_name(self, app_name: str) -> _ResolvedProviderApp | None:
        return self.resolved.get(app_name)

    def get_lifecycle(self, app_id: str) -> str | None:
        return self.lifecycles.get(app_id)

    def get_tags(self, app_id: str) -> dict[str, str]:
        return self.tags.get(app_id, {})

    def stop(self, app_id: str) -> None:
        self.stop_calls.append(app_id)
        self.lifecycles[app_id] = "stopped"
        self.apps = [
            replace(app, lifecycle="stopped", running_tasks=0)
            if app.app_id == app_id
            else app
            for app in self.apps
        ]
        for name, resolved in list(self.resolved.items()):
            if resolved.app_id == app_id:
                self.resolved[name] = replace(resolved, lifecycle="stopped")


def test_launch_uses_exact_owned_app_and_credential_free_payload() -> None:
    gateway = FakeGateway()
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    handle = target.launch(_spec())

    assert handle == _handle()
    assert gateway.deploy_calls == 1
    assert gateway.spawn_calls == 1
    assert gateway.payload is not None
    assert gateway.payload["job_id"] == str(JOB_ID)
    assert gateway.payload["request_hash"] == REQUEST_HASH
    assert "hf-safe" not in repr(gateway.payload)


def test_foreign_name_collision_fails_before_deploy_or_spawn() -> None:
    gateway = FakeGateway()
    gateway.add_app(tag_value=str(OTHER_JOB_ID))
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingOwnershipError, match="not owned"):
        target.launch(_spec())

    assert gateway.deploy_calls == 0
    assert gateway.spawn_calls == 0
    assert gateway.stop_calls == []


def test_spawn_failure_exact_stops_deployed_app_without_leaking_provider_error() -> None:
    gateway = FakeGateway()
    gateway.spawn_error = RuntimeError(f"spawn failed {SECRET}")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingTargetError) as caught:
        target.launch(_spec())

    assert SECRET not in str(caught.value)
    assert gateway.stop_calls == [APP_ID]
    assert gateway.apps[0].lifecycle == "stopped"
    assert gateway.apps[0].running_tasks == 0


def test_poll_returns_strict_events_and_surfaces_tail_and_result_sequence_gaps() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.call_info = _ProviderCallInfo(APP_ID, "succeeded")
    gateway.log_tail = _ProviderLogTail((_event(3), _event(4, "metric")), False)
    gateway.call_result = _result(last_sequence=5)
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    poll = target.poll(_handle(), after_sequence=0, limit=10)

    assert [event.sequence for event in poll.events] == [3, 4]
    assert poll.next_sequence == 4
    assert poll.truncated is True
    assert poll.has_more is False
    assert poll.result is not None
    assert poll.result.job_id == JOB_ID
    assert poll.result.request_hash == REQUEST_HASH
    assert poll.result.revision == RESULT_REVISION


def test_poll_rejects_missing_durable_request_identity() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingTargetError, match="identity"):
        target.poll(_handle(request_hash=None), after_sequence=0, limit=10)


def test_log_transport_failure_is_explicitly_retryable_and_secret_safe() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.tail_error = RuntimeError(f"provider transport {SECRET}")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingTransientError) as caught:
        target.poll(_handle(), after_sequence=0, limit=10)

    assert SECRET not in str(caught.value)


def test_poll_fails_closed_on_duplicate_nonfinite_or_wrong_hash_protocol_lines() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    wrong_hash = _event(1).replace(REQUEST_HASH, "f" * 64)
    duplicate_json = (
        MODAL_TRAINING_EVENT_PREFIX
        + '{"schema":"ctrl-pi.modal-training-event","schema":"duplicate"}'
    )
    nonfinite = _event(2, "metric").replace("0.5", "NaN")
    gateway.log_tail = _ProviderLogTail((wrong_hash, duplicate_json, nonfinite), False)
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingProtocolError, match="event protocol"):
        target.poll(_handle(), after_sequence=0, limit=10)


@pytest.mark.parametrize(
    "lines",
    [
        (_event(1).replace('"version":1', '"version":true'),),
        (_event(1, "metric").replace("0.5", "true"),),
    ],
)
def test_poll_fails_closed_on_loosely_typed_events(
    lines: tuple[str, ...],
) -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.log_tail = _ProviderLogTail(lines, False)
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingProtocolError, match="event protocol"):
        target.poll(_handle(), after_sequence=0, limit=10)


@pytest.mark.parametrize(
    ("lines", "expected_sequences", "expected_gap"),
    [
        # Modal log delivery is not exactly-once: replayed or reordered lines
        # are deduplicated and sorted without failing a paid job, while true
        # sequence gaps stay visible.
        ((_event(1), _event(1)), [1], False),
        ((_event(2), _event(1)), [1, 2], False),
        ((_event(1), _event(2), _event(2), _event(3)), [1, 2, 3], False),
        ((_event(1), _event(3)), [1, 3], True),
    ],
)
def test_poll_deduplicates_and_orders_provider_lines(
    lines: tuple[str, ...],
    expected_sequences: list[int],
    expected_gap: bool,
) -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.call_info = _ProviderCallInfo(APP_ID, "running")
    gateway.log_tail = _ProviderLogTail(lines, False)
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    poll = target.poll(_handle(), after_sequence=0, limit=10)

    assert [event.sequence for event in poll.events] == expected_sequences
    assert poll.truncated is expected_gap


def test_poll_rejects_conflicting_payloads_for_one_sequence() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.log_tail = _ProviderLogTail(
        (_event(1), _event(1, line="conflicting line")),
        False,
    )
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingProtocolError, match="event protocol"):
        target.poll(_handle(), after_sequence=0, limit=10)


def test_absent_app_with_live_function_call_is_never_stopped_verified() -> None:
    gateway = FakeGateway()
    gateway.call_info = _ProviderCallInfo(APP_ID, "running")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    state = target.inspect(_handle())

    assert state.exists is False
    assert state.resource_lifecycle == "unknown"
    assert state.running_tasks == 1
    assert state.execution_state == "running"
    assert state.stopped_verified is False


@pytest.mark.parametrize("execution", ["succeeded", "failed", "cancelled", "unknown"])
def test_stopped_history_remains_idempotent_after_tags_and_name_resolution_disappear(
    execution: str,
) -> None:
    gateway = FakeGateway()
    gateway.lifecycles[APP_ID] = "stopped"
    gateway.call_info = _ProviderCallInfo(APP_ID, execution)  # type: ignore[arg-type]
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    state = target.inspect(_handle())
    target.stop(_handle())

    assert state.stopped_verified is True
    assert gateway.stop_calls == []


def test_stopped_applist_history_without_lifecycle_or_tags_is_not_verified() -> None:
    gateway = FakeGateway()
    gateway.add_app(lifecycle="stopped", running_tasks=0)
    gateway.lifecycles[APP_ID] = None
    gateway.tags[APP_ID] = {}
    gateway.resolved.clear()
    gateway.call_info = _ProviderCallInfo(APP_ID, "succeeded")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingOwnershipError, match="ownership tag"):
        target.inspect(_handle())

    assert gateway.stop_calls == []


def test_stopped_applist_history_requires_the_exact_ownership_tag() -> None:
    gateway = FakeGateway()
    gateway.add_app(lifecycle="stopped", running_tasks=0)
    gateway.lifecycles[APP_ID] = None
    gateway.resolved.clear()
    gateway.call_info = _ProviderCallInfo(APP_ID, "succeeded")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    state = target.inspect(_handle())

    assert state.stopped_verified is True
    assert gateway.stop_calls == []


def test_cancel_checks_call_app_then_cancels_and_verifies_zero_tasks() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    target.cancel(_handle())

    assert gateway.cancel_calls == [CALL_ID]
    assert gateway.stop_calls == [APP_ID]
    assert target.inspect(_handle()).stopped_verified is True


def test_mismatched_call_is_never_cancelled_but_exact_owned_app_is_still_stopped() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.call_info = _ProviderCallInfo("ap-foreign222222", "running")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingOwnershipError, match="does not belong"):
        target.cancel(_handle())

    assert gateway.cancel_calls == []
    assert gateway.stop_calls == [APP_ID]


def test_unavailable_call_identity_falls_back_to_exact_app_stop() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.call_error = RuntimeError(f"metadata unavailable {SECRET}")
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    target.cancel(_handle())

    assert gateway.cancel_calls == []
    assert gateway.stop_calls == [APP_ID]


def test_owned_listing_includes_duplicate_exact_apps_but_ignores_foreign_and_near_prefix() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.add_app(app_id="ap-duplicate22222")
    gateway.add_app(
        app_id="ap-foreign333333",
        tag_value=str(OTHER_JOB_ID),
    )
    gateway.add_app(
        app_id="ap-near44444444",
        app_name=training_app_name(OTHER_JOB_ID) + "-worker",
        tag_value=str(OTHER_JOB_ID),
    )
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    states, unverifiable = target.list_owned_for_panic()

    assert {state.provider_app_id for state in states} == {
        APP_ID,
        "ap-duplicate22222",
    }
    assert unverifiable == []


def test_result_cursor_regression_fails_safely() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.call_info = _ProviderCallInfo(APP_ID, "succeeded")
    gateway.log_tail = _ProviderLogTail((_event(4),), False)
    gateway.call_result = _result(last_sequence=3)
    target = ModalTrainingTarget(gateway=gateway, hf_token="hf-safe")

    with pytest.raises(ManagedTrainingTargetError, match="cursor regressed"):
        target.poll(_handle(), after_sequence=0, limit=10)


def test_success_result_fetch_transport_error_is_retryable_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _OfficialModalTrainingGateway(
        token_id=None,
        token_secret=None,
        environment_name=None,
        hf_token="hf-safe",
    )
    gateway._modal_client = object()
    gateway.inspect_call = lambda call_id: _ProviderCallInfo(APP_ID, "succeeded")  # type: ignore[method-assign]

    class FailedCall:
        def get(self, *, timeout: float) -> object:
            raise RuntimeError(f"transport failed {SECRET}")

    monkeypatch.setattr(
        "ctrl_pi.training_compute_modal.modal.FunctionCall.from_id",
        lambda *args, **kwargs: FailedCall(),
    )

    poll = gateway.poll_call(CALL_ID)

    assert poll.info.execution_state == "unknown"
    assert poll.result is None


class FakeOwnedTarget:
    def __init__(self, states: list[Any], *, enumeration_error: bool = False) -> None:
        self.states = {state.provider_app_id: state for state in states}
        self.enumeration_error = enumeration_error
        self.stop_calls: list[str] = []
        self.stop_failures: set[str] = set()

    def list_owned_for_panic(self) -> tuple[list[Any], list[str]]:
        if self.enumeration_error:
            raise RuntimeError(f"enumeration failed {SECRET}")
        return list(self.states.values()), []

    def stop_owned(self, state: Any) -> None:
        self.stop_calls.append(state.provider_app_id)
        if state.provider_app_id in self.stop_failures:
            raise RuntimeError(f"stop failed {SECRET}")
        self.states.pop(state.provider_app_id, None)


def _inference_state() -> TargetState:
    deployment_id = uuid.UUID("55555555-5555-4555-8555-555555555555")
    return TargetState(
        deployment_id=deployment_id,
        provider_app_id="ap-inference55555",
        app_name=deployment_app_name(deployment_id),
        ownership_tag=deployment_ownership_tag(deployment_id),
        exists=True,
        lifecycle="running",
        running_tasks=1,
        endpoint_url=None,
    )


def _training_state() -> TrainingTargetState:
    return TrainingTargetState(
        job_id=JOB_ID,
        provider_app_id=APP_ID,
        provider_function_call_id=None,
        app_name=training_app_name(JOB_ID),
        ownership_tag=training_ownership_tag(JOB_ID),
        exists=True,
        resource_lifecycle="running",
        execution_state="unknown",
        running_tasks=1,
    )


def test_modal_panic_aggregates_and_verifies_inference_and_training_cleanup() -> None:
    inference = FakeOwnedTarget([_inference_state()])
    training = FakeOwnedTarget([_training_state()])
    target = CombinedModalPanicTarget(inference, training)  # type: ignore[arg-type]
    output = io.StringIO()

    status = run_modal_panic(target, output=output)

    assert status == 0
    assert inference.stop_calls == ["ap-inference55555"]
    assert training.stop_calls == [APP_ID]
    assert "verified zero active" in output.getvalue()


def test_modal_panic_continues_training_cleanup_when_inference_enumeration_fails() -> None:
    inference = FakeOwnedTarget([], enumeration_error=True)
    training = FakeOwnedTarget([_training_state()])
    target = CombinedModalPanicTarget(inference, training)  # type: ignore[arg-type]
    output = io.StringIO()

    status = run_modal_panic(target, output=output)

    assert status == 1
    assert training.stop_calls == [APP_ID]
    assert "inference-enumeration" in output.getvalue()
    assert SECRET not in output.getvalue()


def test_modal_panic_continues_past_training_stop_failure() -> None:
    inference_state = _inference_state()
    training_state = _training_state()
    inference = FakeOwnedTarget([inference_state])
    training = FakeOwnedTarget([training_state])
    training.stop_failures.add(APP_ID)
    target = CombinedModalPanicTarget(inference, training)  # type: ignore[arg-type]
    output = io.StringIO()

    status = run_modal_panic(target, output=output)

    assert status == 1
    assert inference.stop_calls == [inference_state.provider_app_id]
    assert training.stop_calls == [APP_ID]
    assert training_state.app_name in output.getvalue()
    assert SECRET not in output.getvalue()
