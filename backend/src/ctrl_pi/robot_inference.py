from __future__ import annotations

import asyncio
import base64
import math
import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from ctrl_pi.camera import MockCamera
from ctrl_pi.drivers.mock_yam import JOINT_LIMIT_RADIANS, MockYAMDriver
from ctrl_pi.drivers.yam import JOINT_NAMES, ArmAction, ArmTelemetry, YAMDriver
from ctrl_pi.inference_runtime import (
    ActionChunk,
    EncodedImage,
    ObservationRequest,
    RuntimeDescriptor,
    RuntimeKind,
    RuntimeLoadSpec,
    RuntimeProtocolError,
    validate_action_chunk,
)
from ctrl_pi.inference_transport import (
    InferenceTransport,
    InferenceTransportError,
)
from ctrl_pi.rig import (
    RigLease,
    RigLeaseConflictError,
    RigLeaseOwnershipError,
    RigLeaseToken,
)


class RobotInferenceError(RuntimeError):
    """A safe robot-side inference loop error."""


class RobotInferenceConflictError(RobotInferenceError):
    pass


class RobotInferenceConfigurationError(RobotInferenceError):
    pass


class RobotInferencePreparationError(RobotInferenceError):
    pass


@dataclass(frozen=True)
class RobotInferenceSnapshot:
    session_id: uuid.UUID
    arm_id: str
    running: bool
    steps_executed: int
    requests_completed: int
    dropped_chunks: int
    queue_depth: int
    last_latency_ms: float | None
    average_latency_ms: float | None
    frequency_hz: float
    last_error: str | None
    started_at: datetime | None
    stopped_at: datetime | None


@dataclass(frozen=True)
class _PendingInference:
    request: ObservationRequest
    started_monotonic: float


@dataclass(frozen=True)
class _QueuedAction:
    values: list[float]
    observation_sequence: int
    captured_monotonic: float


class RobotInferenceLoop:
    """Single-owner, bounded action-chunk loop for one YAM follower arm."""

    def __init__(
        self,
        *,
        driver: YAMDriver,
        camera: MockCamera,
        rig_lease: RigLease,
        transport: InferenceTransport,
        arm_id: str,
        task: str,
        expected_runtime: RuntimeKind,
        expected_model_repo: str,
        expected_revision: str,
        session_id: uuid.UUID | None = None,
        control_frequency_hz: float = 50.0,
        max_queue_actions: int = 200,
        max_action_age_seconds: float = 2.0,
        max_joint_step_radians: float = 0.35,
        max_gripper_step: float = 0.20,
        allow_opaque_mock_actions: bool = False,
        max_steps: int | None = None,
        action_observer: Callable[
            [ArmTelemetry, ArmAction, ArmTelemetry], Awaitable[None]
        ]
        | None = None,
        start_hook: Callable[[], Awaitable[None]] | None = None,
        nonce_factory: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(arm_id, str):
            raise RobotInferenceConfigurationError("arm_id must be 1-200 characters")
        arm_id = arm_id.strip()
        if not arm_id or len(arm_id) > 200:
            raise RobotInferenceConfigurationError("arm_id must be 1-200 characters")
        if not isinstance(task, str) or len(task) > 512 or "\x00" in task:
            raise RobotInferenceConfigurationError("task is invalid")
        if session_id is not None and not isinstance(session_id, uuid.UUID):
            raise RobotInferenceConfigurationError("session_id must be a UUID")
        if not isinstance(expected_runtime, str) or expected_runtime not in {
            "stub",
            "lerobot",
            "openpi",
        }:
            raise RobotInferenceConfigurationError("expected runtime is invalid")
        try:
            expected_identity = RuntimeLoadSpec(
                model_repo=expected_model_repo,
                revision=expected_revision,
                local_model_path=None,
                device="cpu",
                actions_per_chunk=1,
            )
        except Exception:
            raise RobotInferenceConfigurationError(
                "expected runtime identity is invalid"
            ) from None
        if (
            isinstance(control_frequency_hz, bool)
            or not isinstance(control_frequency_hz, (int, float))
            or not 1 <= float(control_frequency_hz) <= 200
        ):
            raise RobotInferenceConfigurationError(
                "control frequency must be between 1 and 200 Hz"
            )
        if (
            isinstance(max_queue_actions, bool)
            or not isinstance(max_queue_actions, int)
            or not 1 <= max_queue_actions <= 1000
        ):
            raise RobotInferenceConfigurationError(
                "action queue limit must be between 1 and 1000"
            )
        if (
            isinstance(max_action_age_seconds, bool)
            or not isinstance(max_action_age_seconds, (int, float))
            or not 0.05 <= float(max_action_age_seconds) <= 30
        ):
            raise RobotInferenceConfigurationError(
                "action age limit must be between 0.05 and 30 seconds"
            )
        if (
            isinstance(max_joint_step_radians, bool)
            or not isinstance(max_joint_step_radians, (int, float))
            or not math.isfinite(float(max_joint_step_radians))
            or not 0 < float(max_joint_step_radians) <= 0.5
        ):
            raise RobotInferenceConfigurationError("joint action step limit is invalid")
        if (
            isinstance(max_gripper_step, bool)
            or not isinstance(max_gripper_step, (int, float))
            or not math.isfinite(float(max_gripper_step))
            or not 0 < float(max_gripper_step) <= 0.5
        ):
            raise RobotInferenceConfigurationError("gripper action step limit is invalid")
        if not isinstance(allow_opaque_mock_actions, bool):
            raise RobotInferenceConfigurationError(
                "allow_opaque_mock_actions must be a boolean"
            )
        if allow_opaque_mock_actions and not isinstance(driver, MockYAMDriver):
            raise RobotInferenceConfigurationError(
                "opaque action projection is available only with MockYAMDriver"
            )
        if (
            max_steps is not None
            and (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or not 1 <= max_steps <= 1_000_000
            )
        ):
            raise RobotInferenceConfigurationError(
                "max_steps must be between 1 and 1000000"
            )
        if action_observer is not None and not callable(action_observer):
            raise RobotInferenceConfigurationError("action_observer must be callable")
        if start_hook is not None and not callable(start_hook):
            raise RobotInferenceConfigurationError("start_hook must be callable")

        self.driver = driver
        self.camera = camera
        self.rig_lease = rig_lease
        self.transport = transport
        self.arm_id = arm_id
        self.task = task
        self.expected_runtime = expected_runtime
        self.expected_model_repo = expected_identity.model_repo
        self.expected_revision = expected_identity.revision
        self.session_id = session_id or uuid.uuid4()
        self.control_frequency_hz = float(control_frequency_hz)
        self.max_queue_actions = max_queue_actions
        self.max_action_age_seconds = float(max_action_age_seconds)
        self.max_joint_step_radians = float(max_joint_step_radians)
        self.max_gripper_step = float(max_gripper_step)
        self.allow_opaque_mock_actions = allow_opaque_mock_actions
        self.max_steps = max_steps
        self.action_observer = action_observer
        self.start_hook = start_hook
        self._nonce_factory = nonce_factory or (lambda: secrets.token_urlsafe(24))
        self._monotonic = monotonic
        self._now = now or (lambda: datetime.now(UTC))

        self._lifecycle_lock = asyncio.Lock()
        self._state_lock = threading.RLock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._lease_token: RigLeaseToken | None = None
        self._descriptor: RuntimeDescriptor | None = None
        self._started_once = False
        self._running = False
        self._steps_executed = 0
        self._requests_completed = 0
        self._dropped_chunks = 0
        self._queue_depth = 0
        self._last_latency_ms: float | None = None
        self._latency_total_ms = 0.0
        self._frequency_hz = 0.0
        self._last_error: str | None = None
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None

    async def start(self) -> RobotInferenceSnapshot:
        async with self._lifecycle_lock:
            if self._task is not None and not self._task.done():
                raise RobotInferenceConflictError("Inference is already running.")
            if self._started_once:
                raise RobotInferenceConflictError(
                    "This inference loop is single-use; create a new session."
                )
            try:
                arm = self.driver.get_arm(self.arm_id)
            except Exception:
                raise RobotInferenceConfigurationError(
                    "The selected inference arm is unavailable."
                ) from None
            if not arm.connected or arm.role != "follower":
                raise RobotInferenceConfigurationError(
                    "Inference requires a connected follower arm."
                )
            try:
                lease_token = self.rig_lease.acquire(
                    "inference",
                    str(self.session_id),
                )
            except RigLeaseConflictError:
                raise RobotInferenceConflictError(
                    "The arm rig is already controlled by another operation."
                ) from None
            except Exception:
                raise RobotInferenceConfigurationError(
                    "The inference rig lease could not be acquired."
                ) from None

            initialized = False
            descriptor: RuntimeDescriptor | None = None
            try:
                descriptor = await self._to_thread_terminal(self.transport.describe)
                if (
                    descriptor.runtime != self.expected_runtime
                    or descriptor.model_repo != self.expected_model_repo
                    or descriptor.revision != self.expected_revision
                ):
                    raise RobotInferenceConfigurationError(
                        "The runtime descriptor does not match the deployment."
                    )
                if self.max_queue_actions < descriptor.actions_per_chunk:
                    raise RobotInferenceConfigurationError(
                        "The action queue is smaller than one policy chunk."
                    )
                nonce = self._nonce_factory()
                if not isinstance(nonce, str) or not 1 <= len(nonce) <= 128:
                    raise RobotInferenceConfigurationError(
                        "The runtime health nonce is invalid."
                    )
                health = await self._to_thread_terminal(self.transport.health, nonce)
                if (
                    not health.healthy
                    or not secrets.compare_digest(health.echo, nonce)
                    or health.runtime != descriptor.runtime
                    or health.model_repo != descriptor.model_repo
                    or health.revision != descriptor.revision
                    or health.runtime != self.expected_runtime
                    or health.model_repo != self.expected_model_repo
                    or health.revision != self.expected_revision
                ):
                    raise RobotInferenceConfigurationError(
                        "The loaded runtime identity could not be verified."
                    )
                initialized = True
            except asyncio.CancelledError:
                raise
            except RobotInferenceError:
                raise
            except Exception:
                raise RobotInferenceConfigurationError(
                    "The remote inference runtime could not be initialized."
                ) from None
            finally:
                if not initialized:
                    try:
                        await self._safe_close_transport()
                    finally:
                        self._safe_release_lease(lease_token)

            assert descriptor is not None
            if self.start_hook is not None:
                try:
                    await self.start_hook()
                except asyncio.CancelledError:
                    try:
                        await self._safe_close_transport()
                    finally:
                        self._safe_release_lease(lease_token)
                    raise
                except Exception:
                    try:
                        await self._safe_close_transport()
                    finally:
                        self._safe_release_lease(lease_token)
                    raise RobotInferencePreparationError(
                        "The inference session could not be prepared safely."
                    ) from None
            with self._state_lock:
                self._descriptor = descriptor
                self._lease_token = lease_token
                self._stop_event = asyncio.Event()
                self._started_once = True
                self._running = True
                self._started_at = self._as_utc(self._now())
                self._stopped_at = None
                self._last_error = None
            self._task = asyncio.create_task(
                self._run(),
                name=f"ctrl-pi-inference-{self.session_id}",
            )
            return self.snapshot()

    async def stop(self) -> RobotInferenceSnapshot:
        async with self._lifecycle_lock:
            task = self._task
            stop_event = self._stop_event
            if task is None:
                if not self._started_once:
                    raise RobotInferenceConflictError("Inference has not started.")
                return self.snapshot()
            if stop_event is not None:
                stop_event.set()
        await self._await_terminal(task)
        return self.snapshot()

    async def shutdown(self) -> None:
        async with self._lifecycle_lock:
            task = self._task
            if self._stop_event is not None:
                self._stop_event.set()
        if task is not None:
            await self._await_terminal(task)

    async def wait(self) -> RobotInferenceSnapshot:
        """Wait for this single-use loop to reach its terminal state."""

        async with self._lifecycle_lock:
            task = self._task
            if task is None:
                if not self._started_once:
                    raise RobotInferenceConflictError("Inference has not started.")
                return self.snapshot()
        await self._await_terminal(task)
        return self.snapshot()

    async def wait_for_steps(
        self,
        steps: int,
        *,
        timeout_seconds: float = 10.0,
    ) -> RobotInferenceSnapshot:
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 1_000_000:
            raise ValueError("steps must be between 0 and 1000000")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        deadline = self._monotonic() + timeout_seconds
        while True:
            snapshot = self.snapshot()
            if snapshot.steps_executed >= steps or not snapshot.running:
                return snapshot
            if self._monotonic() >= deadline:
                raise TimeoutError("inference step wait timed out")
            await asyncio.sleep(0.002)

    def snapshot(self) -> RobotInferenceSnapshot:
        with self._state_lock:
            average = (
                None
                if self._requests_completed == 0
                else self._latency_total_ms / self._requests_completed
            )
            return RobotInferenceSnapshot(
                session_id=self.session_id,
                arm_id=self.arm_id,
                running=self._running,
                steps_executed=self._steps_executed,
                requests_completed=self._requests_completed,
                dropped_chunks=self._dropped_chunks,
                queue_depth=self._queue_depth,
                last_latency_ms=self._last_latency_ms,
                average_latency_ms=average,
                frequency_hz=self._frequency_hz,
                last_error=self._last_error,
                started_at=self._started_at,
                stopped_at=self._stopped_at,
            )

    async def active_resource_counts(self) -> tuple[int, int]:
        with self._state_lock:
            task_count = int(self._task is not None and not self._task.done())
            return task_count, self._queue_depth

    async def _run(self) -> None:
        descriptor = self._descriptor
        stop_event = self._stop_event
        lease_token = self._lease_token
        assert descriptor is not None and stop_event is not None and lease_token is not None
        interval = 1.0 / self.control_frequency_hz
        deadline = self._monotonic()
        queue: deque[_QueuedAction] = deque()
        in_flight: asyncio.Task[ActionChunk] | None = None
        pending: _PendingInference | None = None
        next_sequence = 0
        last_chunk_sequence = -1
        low_watermark = min(
            descriptor.actions_per_chunk // 2,
            self.max_queue_actions - descriptor.actions_per_chunk,
        )
        started_monotonic = self._monotonic()
        safe_error: str | None = None
        try:
            while not stop_event.is_set():
                cycle_started = self._monotonic()
                if in_flight is not None and in_flight.done():
                    assert pending is not None
                    chunk = in_flight.result()
                    validate_action_chunk(pending.request, chunk, descriptor)
                    latency_ms = (
                        self._monotonic() - pending.started_monotonic
                    ) * 1000.0
                    if latency_ms / 1000.0 > self.max_action_age_seconds:
                        with self._state_lock:
                            self._dropped_chunks += 1
                    else:
                        if chunk.observation_sequence <= last_chunk_sequence:
                            raise RuntimeProtocolError(
                                "The runtime returned an out-of-order action chunk."
                            )
                        if len(queue) + len(chunk.actions) > self.max_queue_actions:
                            raise RuntimeProtocolError(
                                "The runtime action queue would exceed its bound."
                            )
                        queue.extend(
                            _QueuedAction(
                                values=list(action),
                                observation_sequence=chunk.observation_sequence,
                                captured_monotonic=pending.started_monotonic,
                            )
                            for action in chunk.actions
                        )
                        last_chunk_sequence = chunk.observation_sequence
                    with self._state_lock:
                        self._requests_completed += 1
                        self._last_latency_ms = latency_ms
                        self._latency_total_ms += latency_ms
                    in_flight = None
                    pending = None

                if stop_event.is_set():
                    break
                if queue:
                    queued = queue.popleft()
                    if (
                        self._monotonic() - queued.captured_monotonic
                        > self.max_action_age_seconds
                    ):
                        while (
                            queue
                            and queue[0].observation_sequence
                            == queued.observation_sequence
                        ):
                            queue.popleft()
                        with self._state_lock:
                            self._dropped_chunks += 1
                    else:
                        if stop_event.is_set():
                            break
                        current = self.driver.get_arm(self.arm_id)
                        action = self._map_action(queued.values, current, descriptor)
                        if stop_event.is_set():
                            break
                        if (
                            self._monotonic() - queued.captured_monotonic
                            > self.max_action_age_seconds
                        ):
                            while (
                                queue
                                and queue[0].observation_sequence
                                == queued.observation_sequence
                            ):
                                queue.popleft()
                            with self._state_lock:
                                self._dropped_chunks += 1
                        else:
                            applied = self.driver.apply_action(self.arm_id, action)
                            with self._state_lock:
                                self._steps_executed += 1
                                elapsed = max(
                                    self._monotonic() - started_monotonic,
                                    1e-9,
                                )
                                self._frequency_hz = self._steps_executed / elapsed
                                reached_step_limit = (
                                    self.max_steps is not None
                                    and self._steps_executed >= self.max_steps
                                )
                            if self.action_observer is not None:
                                await self.action_observer(current, action, applied)
                            if reached_step_limit:
                                stop_event.set()

                if (
                    not stop_event.is_set()
                    and in_flight is None
                    and len(queue) <= low_watermark
                ):
                    request = self._capture_observation(descriptor, next_sequence)
                    pending = _PendingInference(
                        request=request,
                        started_monotonic=self._monotonic(),
                    )
                    in_flight = asyncio.create_task(
                        asyncio.to_thread(self.transport.infer, request),
                        name=f"ctrl-pi-infer-request-{self.session_id}-{next_sequence}",
                    )
                    next_sequence += 1

                with self._state_lock:
                    self._queue_depth = len(queue)
                deadline += interval
                if deadline <= cycle_started:
                    missed = math.floor((cycle_started - deadline) / interval) + 1
                    deadline += missed * interval
                await self._wait_until(stop_event, max(0.0, deadline - self._monotonic()))
        except asyncio.CancelledError:
            stop_event.set()
            raise
        except (InferenceTransportError, RuntimeProtocolError):
            safe_error = "The inference transport returned an invalid action stream."
        except Exception:
            safe_error = "The robot inference loop stopped safely."
        finally:
            stop_event.set()
            if in_flight is not None:
                await self._settle_in_flight(in_flight)
            queue.clear()
            close_ok = False
            try:
                close_ok = await self._safe_close_transport()
            finally:
                release_ok = self._safe_release_lease(lease_token)
                if safe_error is None and (not close_ok or not release_ok):
                    safe_error = (
                        "The inference resources could not be released safely."
                    )
                with self._state_lock:
                    self._queue_depth = 0
                    self._running = False
                    self._stopped_at = self._as_utc(self._now())
                    self._last_error = safe_error

    def _capture_observation(
        self,
        descriptor: RuntimeDescriptor,
        sequence: int,
    ) -> ObservationRequest:
        telemetry = self.driver.get_arm(self.arm_id)
        raw_state = self._yam_state(telemetry)
        vectors: dict[str, list[float]] = {}
        visual_names: list[str] = []
        for feature in descriptor.inputs:
            if feature.kind in {"state", "environment"}:
                vectors[feature.name] = [
                    raw_state[index % len(raw_state)]
                    for index in range(feature.shape[0])
                ]
            elif feature.kind == "visual":
                visual_names.append(feature.name)
        images: dict[str, EncodedImage] = {}
        if visual_names:
            frame = self.camera.capture()
            encoded = EncodedImage(
                data_base64=base64.b64encode(self.camera.jpeg(frame)).decode("ascii")
            )
            images = dict.fromkeys(visual_names, encoded)
        return ObservationRequest(
            request_id=uuid.uuid4(),
            sequence=sequence,
            captured_at=telemetry.timestamp,
            vectors=vectors,
            images=images,
            task=self.task,
        )

    def _map_action(
        self,
        values: list[float],
        current: ArmTelemetry,
        descriptor: RuntimeDescriptor,
    ) -> ArmAction:
        if len(values) != descriptor.action.shape[0] or not all(
            math.isfinite(value) for value in values
        ):
            raise RuntimeProtocolError("The runtime action vector is invalid.")
        current_joints = {
            joint.name: joint.position_radians for joint in current.joints
        }
        if set(current_joints) != set(JOINT_NAMES):
            raise RuntimeProtocolError("The YAM joint state is incomplete.")
        if len(values) == 7:
            targets = dict(zip(JOINT_NAMES, values[:6], strict=True))
            gripper = values[6]
            if any(
                not JOINT_LIMIT_RADIANS[0] <= target <= JOINT_LIMIT_RADIANS[1]
                or abs(target - current_joints[name]) > self.max_joint_step_radians
                for name, target in targets.items()
            ):
                raise RuntimeProtocolError("The runtime joint action exceeds its safe bound.")
            if (
                not 0.0 <= gripper <= 1.0
                or abs(gripper - current.gripper.position) > self.max_gripper_step
            ):
                raise RuntimeProtocolError("The runtime gripper action exceeds its safe bound.")
        else:
            if not self.allow_opaque_mock_actions:
                raise RuntimeProtocolError(
                    "The policy action shape is incompatible with the YAM arm."
                )
            targets = {
                name: max(
                    JOINT_LIMIT_RADIANS[0],
                    min(
                        JOINT_LIMIT_RADIANS[1],
                        current_joints[name]
                        + 0.02 * math.tanh(values[index % len(values)]),
                    ),
                )
                for index, name in enumerate(JOINT_NAMES)
            }
            gripper = max(
                0.0,
                min(
                    1.0,
                    current.gripper.position
                    + 0.02 * math.tanh(values[6 % len(values)]),
                ),
            )
        return ArmAction(
            timestamp=self._as_utc(self._now()),
            joint_positions_radians=targets,
            gripper_position=gripper,
        )

    @staticmethod
    def _yam_state(telemetry: ArmTelemetry) -> list[float]:
        positions = {
            joint.name: joint.position_radians for joint in telemetry.joints
        }
        if set(positions) != set(JOINT_NAMES):
            raise RuntimeProtocolError("The YAM joint state is incomplete.")
        state = [positions[name] for name in JOINT_NAMES]
        state.append(telemetry.gripper.position)
        if not all(math.isfinite(value) for value in state):
            raise RuntimeProtocolError("The YAM joint state is invalid.")
        return state

    @staticmethod
    async def _wait_until(stop_event: asyncio.Event, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except TimeoutError:
            return

    @staticmethod
    async def _to_thread_terminal(callback: Callable[..., object], *args: object):
        task = asyncio.create_task(asyncio.to_thread(callback, *args))
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
            raise

    @staticmethod
    async def _await_terminal(task: asyncio.Task[None]) -> None:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            raise

    @staticmethod
    async def _settle_in_flight(task: asyncio.Task[ActionChunk]) -> None:
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

    async def _safe_close_transport(self) -> bool:
        try:
            await self._to_thread_terminal(self.transport.close)
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
        return True

    def _safe_release_lease(self, token: RigLeaseToken) -> bool:
        try:
            self.rig_lease.release(token)
        except RigLeaseOwnershipError:
            return False
        return True

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
