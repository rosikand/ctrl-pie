from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import replace

from ctrl_pi.training_compute import (
    ManagedTrainingSpec,
    ManagedTrainingTargetError,
    TrainingCheckpointEvent,
    TrainingHandle,
    TrainingLogEvent,
    TrainingMetricEvent,
    TrainingPoll,
    TrainingResult,
    TrainingTargetKind,
    TrainingTargetState,
    training_app_name,
    training_ownership_tag,
    validate_training_handle,
)


class StubManagedTrainingTarget:
    """Deterministic credential-free managed training target for mock mode."""

    def __init__(self) -> None:
        self._states: dict[uuid.UUID, TrainingTargetState] = {}
        self._events: dict[uuid.UUID, tuple[object, ...]] = {}
        self._results: dict[uuid.UUID, TrainingResult] = {}
        self._polled: set[uuid.UUID] = set()
        self._lock = threading.Lock()

    @property
    def kind(self) -> TrainingTargetKind:
        return "stub"

    def launch(self, spec: ManagedTrainingSpec) -> TrainingHandle:
        handle = TrainingHandle(
            job_id=spec.job_id,
            provider_app_id=f"stub-{spec.job_id.hex}",
            provider_function_call_id=f"call-{spec.job_id.hex}",
            app_name=spec.app_name,
            ownership_tag=spec.ownership_tag,
            request_hash=spec.request_hash,
        )
        revision = hashlib.sha1(
            f"ctrl-pi-stub-training:{spec.job_id}:{spec.request_hash}".encode()
        ).hexdigest()
        checkpoint_step = min(spec.save_every, spec.max_steps)
        metric_step = min(spec.log_every, spec.max_steps)
        events = (
            TrainingLogEvent(
                sequence=1,
                source="system",
                line="Mock managed training started.",
                step=0,
            ),
            TrainingMetricEvent(
                sequence=2,
                step=metric_step,
                metrics={"loss": 0.25},
            ),
            TrainingCheckpointEvent(
                sequence=3,
                repo_id=spec.output_model_repo,
                revision=revision,
                step=checkpoint_step,
                final=checkpoint_step == spec.max_steps,
            ),
            TrainingLogEvent(
                sequence=4,
                source="system",
                line="Mock managed training completed.",
                step=spec.max_steps,
            ),
        )
        result = TrainingResult(
            job_id=spec.job_id,
            request_hash=spec.request_hash,
            output_model_repo=spec.output_model_repo,
            revision=revision,
            step=spec.max_steps,
        )
        with self._lock:
            if spec.job_id in self._states:
                raise ManagedTrainingTargetError(
                    "A managed training resource already exists for this job."
                )
            self._states[spec.job_id] = TrainingTargetState(
                job_id=spec.job_id,
                provider_app_id=handle.provider_app_id,
                provider_function_call_id=handle.provider_function_call_id,
                app_name=handle.app_name,
                ownership_tag=handle.ownership_tag,
                exists=True,
                resource_lifecycle="running",
                execution_state="running",
                running_tasks=1,
            )
            self._events[spec.job_id] = events
            self._results[spec.job_id] = result
        return handle

    def poll(
        self,
        handle: TrainingHandle,
        *,
        after_sequence: int,
        limit: int,
    ) -> TrainingPoll:
        self._validate_handle(handle)
        if handle.request_hash is None:
            raise ManagedTrainingTargetError(
                "The managed training request identity is unavailable."
            )
        if not 0 <= after_sequence <= 9_007_199_254_740_991 or not 1 <= limit <= 256:
            raise ManagedTrainingTargetError(
                "The managed training event request is invalid."
            )
        with self._lock:
            state = self._state(handle)
            if state.execution_state == "running":
                state = replace(
                    state,
                    execution_state="succeeded",
                    running_tasks=0,
                )
                self._states[handle.job_id] = state
                self._polled.add(handle.job_id)
            available = tuple(
                event
                for event in self._events.get(handle.job_id, ())
                if event.sequence > after_sequence
            )
            selected = available[:limit]
            next_sequence = selected[-1].sequence if selected else after_sequence
            return TrainingPoll(
                state=state,
                events=selected,
                next_sequence=next_sequence,
                truncated=False,
                has_more=len(available) > len(selected),
                result=(
                    self._results.get(handle.job_id)
                    if state.execution_state == "succeeded"
                    else None
                ),
            )

    def inspect(self, handle: TrainingHandle) -> TrainingTargetState:
        self._validate_handle(handle)
        with self._lock:
            return self._state(handle)

    def cancel(self, handle: TrainingHandle) -> None:
        self._validate_handle(handle)
        with self._lock:
            state = self._state(handle)
            if state.execution_state in {"running", "pending"}:
                self._states[handle.job_id] = replace(
                    state,
                    execution_state="cancelled",
                    resource_lifecycle="stopping",
                )

    def stop(self, handle: TrainingHandle) -> None:
        self._validate_handle(handle)
        with self._lock:
            state = self._state(handle)
            self._states[handle.job_id] = replace(
                state,
                resource_lifecycle="stopped",
                running_tasks=0,
            )

    def list_owned(self) -> list[TrainingTargetState]:
        with self._lock:
            return sorted(self._states.values(), key=lambda state: str(state.job_id))

    def _state(self, handle: TrainingHandle) -> TrainingTargetState:
        state = self._states.get(handle.job_id)
        if state is None:
            return TrainingTargetState(
                job_id=handle.job_id,
                provider_app_id=handle.provider_app_id,
                provider_function_call_id=handle.provider_function_call_id,
                app_name=handle.app_name,
                ownership_tag=handle.ownership_tag,
                exists=False,
                resource_lifecycle="unknown",
                execution_state="unknown",
                running_tasks=0,
            )
        if (
            state.provider_app_id != handle.provider_app_id
            or state.provider_function_call_id != handle.provider_function_call_id
            or state.app_name != handle.app_name
            or state.ownership_tag != handle.ownership_tag
        ):
            raise ManagedTrainingTargetError(
                "The managed training resource identity does not match this job."
            )
        return state

    @staticmethod
    def _validate_handle(handle: TrainingHandle) -> None:
        validate_training_handle(handle)
        if (
            handle.app_name != training_app_name(handle.job_id)
            or handle.ownership_tag != training_ownership_tag(handle.job_id)
            or handle.provider_app_id != f"stub-{handle.job_id.hex}"
            or handle.provider_function_call_id != f"call-{handle.job_id.hex}"
        ):
            raise ManagedTrainingTargetError(
                "The managed training resource is not owned by this job."
            )


__all__ = ["StubManagedTrainingTarget"]
