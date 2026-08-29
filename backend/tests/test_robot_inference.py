from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from ctrl_pi.camera import MockCamera
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import ArmAction, ArmTelemetry
from ctrl_pi.inference_runtime import (
    ActionChunk,
    ObservationRequest,
    RuntimeDescriptor,
    RuntimeFeature,
    RuntimeHealth,
    RuntimeLoadSpec,
    StubInferenceRuntime,
    validate_observation,
)
from ctrl_pi.inference_transport import InProcessInferenceTransport
from ctrl_pi.rig import RigLease
from ctrl_pi.rig import RigLeaseOwnershipError
from ctrl_pi.robot_inference import (
    RobotInferenceConfigurationError,
    RobotInferenceConflictError,
    RobotInferenceLoop,
)

REPO_ID = "acme/mock-policy"
REVISION = "d" * 40


def _descriptor(
    *,
    action_dimension: int = 7,
    actions_per_chunk: int = 4,
    inputs: list[RuntimeFeature] | None = None,
) -> RuntimeDescriptor:
    return RuntimeDescriptor(
        runtime="lerobot",
        policy_type="act",
        model_repo=REPO_ID,
        revision=REVISION,
        inputs=inputs
        or [RuntimeFeature(name="observation.state", kind="state", shape=(7,))],
        action=RuntimeFeature(
            name="action",
            kind="action",
            shape=(action_dimension,),
        ),
        actions_per_chunk=actions_per_chunk,
    )


def _safe_chunk(
    request: ObservationRequest,
    descriptor: RuntimeDescriptor,
    *,
    action_factory: Callable[[int], list[float]] | None = None,
) -> ActionChunk:
    now = datetime.now(UTC)
    state = request.vectors.get("observation.state", [0.0] * 7)
    if action_factory is None:
        if descriptor.action.shape[0] == 7:
            base = (state + [0.0] * 7)[:7]
            action_factory = lambda _index: base.copy()
        else:
            action_factory = lambda index: [
                math.sin(request.sequence + index + dimension)
                for dimension in range(descriptor.action.shape[0])
            ]
    return ActionChunk(
        request_id=request.request_id,
        observation_sequence=request.sequence,
        revision=descriptor.revision,
        actions=[
            action_factory(index) for index in range(descriptor.actions_per_chunk)
        ],
        server_received_at=now,
        server_completed_at=now,
    )


class FakeTransport:
    def __init__(
        self,
        descriptor: RuntimeDescriptor | None = None,
        *,
        infer_callback: Callable[[ObservationRequest], ActionChunk] | None = None,
        health_callback: Callable[[str], RuntimeHealth] | None = None,
    ) -> None:
        self.descriptor = descriptor or _descriptor()
        self.infer_callback = infer_callback
        self.health_callback = health_callback
        self.requests: list[ObservationRequest] = []
        self.closed = False
        self.active_infers = 0
        self.maximum_active_infers = 0
        self._lock = threading.Lock()

    def describe(self) -> RuntimeDescriptor:
        return self.descriptor.model_copy(deep=True)

    def health(self, nonce: str) -> RuntimeHealth:
        if self.health_callback is not None:
            return self.health_callback(nonce)
        return RuntimeHealth(
            healthy=True,
            echo=nonce,
            runtime=self.descriptor.runtime,
            model_repo=self.descriptor.model_repo,
            revision=self.descriptor.revision,
        )

    def infer(self, request: ObservationRequest) -> ActionChunk:
        with self._lock:
            self.active_infers += 1
            self.maximum_active_infers = max(
                self.maximum_active_infers,
                self.active_infers,
            )
            self.requests.append(request.model_copy(deep=True))
        try:
            if self.infer_callback is not None:
                return self.infer_callback(request)
            return _safe_chunk(request, self.descriptor)
        finally:
            with self._lock:
                self.active_infers -= 1

    def close(self) -> None:
        self.closed = True


class _UnsafeIdleDriver(MockYAMDriver):
    def __init__(self, *, latch_fails: bool = False) -> None:
        super().__init__()
        self.latch_fails = latch_fails

    def safe_idle(self, arm_id: str):
        del arm_id
        raise RuntimeError("injected safe-idle failure")

    def latch_fault(self, arm_ids: list[str], detail: str) -> None:
        if self.latch_fails:
            raise RuntimeError("injected fault-latch failure")
        super().latch_fault(arm_ids, detail)


def _loop(
    transport: FakeTransport | InProcessInferenceTransport,
    *,
    driver: MockYAMDriver | None = None,
    lease: RigLease | None = None,
    **kwargs: object,
) -> RobotInferenceLoop:
    return RobotInferenceLoop(
        driver=driver or MockYAMDriver(),
        camera=MockCamera(width=32, height=24),
        rig_lease=lease or RigLease(),
        transport=transport,
        arm_id="yam-follower",
        task="move the mock arm",
        expected_runtime="lerobot",
        expected_model_repo=REPO_ID,
        expected_revision=REVISION,
        control_frequency_hz=200,
        max_queue_actions=40,
        **kwargs,
    )


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        await asyncio.sleep(0.002)


@pytest.mark.asyncio
async def test_mock_loop_executes_100_steps_single_flight_and_releases_everything() -> None:
    driver = MockYAMDriver()
    before = driver.get_arm("yam-follower")
    runtime = StubInferenceRuntime(runtime="lerobot")
    runtime.load(
        RuntimeLoadSpec(
            model_repo=REPO_ID,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
            actions_per_chunk=20,
        )
    )
    transport = InProcessInferenceTransport(runtime)
    lease = RigLease()
    loop = _loop(transport, driver=driver, lease=lease)

    started = await loop.start()
    reached = await loop.wait_for_steps(100, timeout_seconds=3)
    stopped = await loop.stop()

    after = driver.get_arm("yam-follower")
    assert started.running
    assert reached.steps_executed >= 100
    assert 100 <= stopped.steps_executed <= 102
    assert stopped.requests_completed >= 5
    assert stopped.last_latency_ms is not None
    assert stopped.average_latency_ms is not None
    assert stopped.frequency_hz > 0
    assert not stopped.running and stopped.queue_depth == 0
    assert stopped.last_error is None
    assert stopped.started_at is not None and stopped.stopped_at is not None
    assert lease.current() is None
    assert await loop.active_resource_counts() == (0, 0)
    assert after.control_state == "gravity_comp"
    assert after.energized is True
    assert after.holding is True
    assert [joint.position_radians for joint in before.joints] != [
        joint.position_radians for joint in after.joints
    ]


@pytest.mark.asyncio
async def test_inference_fault_latches_before_release_when_safe_idle_fails() -> None:
    driver = _UnsafeIdleDriver()
    lease = RigLease()
    loop = _loop(FakeTransport(), driver=driver, lease=lease, max_steps=1)

    await loop.start()
    stopped = await loop.wait()

    assert stopped.running is False
    assert "fault-latched" in (stopped.last_error or "")
    assert lease.current("yam-follower") is None
    follower = driver.get_arm("yam-follower")
    assert follower.control_state == "error"
    assert follower.connected is False


@pytest.mark.asyncio
async def test_inference_retains_lease_when_safe_idle_and_latch_are_uncertain() -> None:
    driver = _UnsafeIdleDriver(latch_fails=True)
    lease = RigLease()
    loop = _loop(FakeTransport(), driver=driver, lease=lease, max_steps=1)

    await loop.start()
    stopped = await loop.wait()

    assert "ownership remains blocked" in (stopped.last_error or "")
    token = lease.current("yam-follower")
    assert token is not None and token.owner == "inference"


@pytest.mark.asyncio
async def test_rig_lease_conflict_is_non_mutating_and_loop_can_start_after_release() -> None:
    lease = RigLease()
    owner = lease.acquire("teleop", "recording-id")
    transport = FakeTransport()
    loop = _loop(transport, lease=lease)

    with pytest.raises(RobotInferenceConflictError, match="already controlled"):
        await loop.start()
    assert lease.current() == owner
    assert not transport.closed

    lease.release(owner)
    await loop.start()
    await _wait_until(lambda: loop.snapshot().steps_executed >= 1)
    await loop.stop()
    assert transport.closed
    assert lease.current() is None


@pytest.mark.asyncio
async def test_start_requires_exact_runtime_health_identity_and_releases_lease() -> None:
    descriptor = _descriptor()
    transport = FakeTransport(
        descriptor,
        health_callback=lambda nonce: RuntimeHealth(
            healthy=True,
            echo=nonce,
            runtime=descriptor.runtime,
            model_repo="acme/wrong-policy",
            revision=descriptor.revision,
        ),
    )
    lease = RigLease()
    loop = _loop(transport, lease=lease)

    with pytest.raises(RobotInferenceConfigurationError, match="identity"):
        await loop.start()

    assert transport.closed
    assert lease.current() is None
    assert await loop.active_resource_counts() == (0, 0)


@pytest.mark.asyncio
async def test_self_consistent_wrong_descriptor_cannot_drive_the_selected_deployment() -> None:
    wrong = RuntimeDescriptor(
        runtime="lerobot",
        policy_type="act",
        model_repo="acme/wrong-policy",
        revision="e" * 40,
        inputs=[RuntimeFeature(name="observation.state", kind="state", shape=(7,))],
        action=RuntimeFeature(name="action", kind="action", shape=(7,)),
        actions_per_chunk=2,
    )
    transport = FakeTransport(wrong)
    lease = RigLease()
    loop = _loop(transport, lease=lease)

    with pytest.raises(RobotInferenceConfigurationError, match="deployment"):
        await loop.start()

    assert transport.closed
    assert not transport.requests
    assert lease.current() is None


@pytest.mark.asyncio
async def test_blocked_inference_is_single_flight_and_stop_waits_for_worker() -> None:
    entered = threading.Event()
    release = threading.Event()
    descriptor = _descriptor(actions_per_chunk=4)

    def infer(request: ObservationRequest) -> ActionChunk:
        entered.set()
        assert release.wait(timeout=2)
        return _safe_chunk(request, descriptor)

    transport = FakeTransport(descriptor, infer_callback=infer)
    lease = RigLease()
    loop = _loop(transport, lease=lease)
    await loop.start()
    assert await asyncio.to_thread(entered.wait, 1)

    stop_task = asyncio.create_task(loop.stop())
    await asyncio.sleep(0.03)
    assert not stop_task.done()
    assert transport.active_infers == transport.maximum_active_infers == 1
    assert lease.current() is not None

    release.set()
    stopped = await asyncio.wait_for(stop_task, timeout=1)
    assert not stopped.running
    assert transport.active_infers == 0
    assert transport.closed
    assert lease.current() is None


@pytest.mark.asyncio
async def test_stale_chunks_are_dropped_without_reaching_the_arm() -> None:
    descriptor = _descriptor(actions_per_chunk=2)

    def infer(request: ObservationRequest) -> ActionChunk:
        time.sleep(0.07)
        return _safe_chunk(request, descriptor)

    transport = FakeTransport(descriptor, infer_callback=infer)
    loop = _loop(
        transport,
        max_action_age_seconds=0.05,
    )
    await loop.start()
    await _wait_until(lambda: loop.snapshot().dropped_chunks >= 1)
    stopped = await loop.stop()

    assert stopped.dropped_chunks >= 1
    assert stopped.requests_completed >= 1
    assert stopped.steps_executed == 0
    assert stopped.last_error is None


@pytest.mark.asyncio
async def test_actions_expire_while_waiting_in_the_bounded_queue() -> None:
    descriptor = _descriptor(actions_per_chunk=5)
    transport = FakeTransport(descriptor)
    loop = RobotInferenceLoop(
        driver=MockYAMDriver(),
        camera=MockCamera(width=32, height=24),
        rig_lease=RigLease(),
        transport=transport,
        arm_id="yam-follower",
        task="move the mock arm",
        expected_runtime="lerobot",
        expected_model_repo=REPO_ID,
        expected_revision=REVISION,
        control_frequency_hz=50,
        max_queue_actions=10,
        max_action_age_seconds=0.05,
    )

    await loop.start()
    await _wait_until(lambda: loop.snapshot().dropped_chunks >= 1)
    stopped = await loop.stop()

    assert stopped.requests_completed >= 1
    assert stopped.dropped_chunks >= 1
    assert 1 <= stopped.steps_executed < descriptor.actions_per_chunk
    assert stopped.queue_depth == 0


@pytest.mark.asyncio
async def test_nonfinite_chunk_bypassing_model_validation_fails_closed() -> None:
    descriptor = _descriptor(actions_per_chunk=1)

    def infer(request: ObservationRequest) -> ActionChunk:
        now = datetime.now(UTC)
        return ActionChunk.model_construct(
            request_id=request.request_id,
            observation_sequence=request.sequence,
            revision=descriptor.revision,
            actions=[[float("nan")] * 7],
            server_received_at=now,
            server_completed_at=now,
        )

    transport = FakeTransport(descriptor, infer_callback=infer)
    lease = RigLease()
    loop = _loop(transport, lease=lease)
    await loop.start()
    await _wait_until(lambda: not loop.snapshot().running)
    stopped = loop.snapshot()

    assert stopped.steps_executed == 0
    assert stopped.last_error == (
        "The inference transport returned an invalid action stream."
    )
    assert transport.closed
    assert lease.current() is None


@pytest.mark.asyncio
async def test_observation_projection_has_exact_features_and_bounded_jpeg() -> None:
    descriptor = _descriptor(
        actions_per_chunk=1,
        inputs=[
            RuntimeFeature(name="observation.state", kind="state", shape=(10,)),
            RuntimeFeature(name="observation.environment", kind="environment", shape=(3,)),
            RuntimeFeature(name="observation.images.top", kind="visual", shape=(3, 8, 8)),
            RuntimeFeature(name="task", kind="language", shape=(1,)),
        ],
    )
    observed = threading.Event()

    def infer(request: ObservationRequest) -> ActionChunk:
        validate_observation(request, descriptor)
        observed.set()
        state = request.vectors["observation.state"]
        return _safe_chunk(
            request,
            descriptor,
            action_factory=lambda _index: state[:7],
        )

    transport = FakeTransport(descriptor, infer_callback=infer)
    loop = _loop(transport)
    await loop.start()
    assert await asyncio.to_thread(observed.wait, 1)
    await loop.stop()

    request = transport.requests[0]
    assert set(request.vectors) == {
        "observation.state",
        "observation.environment",
    }
    assert len(request.vectors["observation.state"]) == 10
    assert len(request.vectors["observation.environment"]) == 3
    assert set(request.images) == {"observation.images.top"}
    assert request.images["observation.images.top"].decoded().startswith(b"\xff\xd8")
    assert request.task == "move the mock arm"


@pytest.mark.asyncio
async def test_opaque_action_projection_is_explicit_mock_only_and_bounded() -> None:
    descriptor = _descriptor(action_dimension=2, actions_per_chunk=2)
    transport = FakeTransport(descriptor)
    driver = MockYAMDriver()
    before = driver.get_arm("yam-follower")
    loop = _loop(
        transport,
        driver=driver,
        allow_opaque_mock_actions=True,
    )
    await loop.start()
    await loop.wait_for_steps(4, timeout_seconds=1)
    stopped = await loop.stop()
    after = driver.get_arm("yam-follower")

    deltas = [
        abs(left.position_radians - right.position_radians)
        for left, right in zip(before.joints, after.joints, strict=True)
    ]
    assert stopped.steps_executed >= 4 and stopped.last_error is None
    assert all(delta <= 0.02 * stopped.steps_executed + 1e-9 for delta in deltas)
    assert abs(after.gripper.position - before.gripper.position) <= (
        0.02 * stopped.steps_executed + 1e-9
    )

    rejected = _loop(FakeTransport(descriptor))
    await rejected.start()
    await _wait_until(lambda: not rejected.snapshot().running)
    assert rejected.snapshot().steps_executed == 0
    assert rejected.snapshot().last_error is not None


class FailingDriver(MockYAMDriver):
    def __init__(self, *, fail_after: int) -> None:
        super().__init__()
        self.fail_after = fail_after
        self.apply_calls = 0

    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry:
        self.apply_calls += 1
        if self.apply_calls > self.fail_after:
            raise RuntimeError("raw hardware secret")
        return super().apply_action(arm_id, action)


class SlowReadDriver(MockYAMDriver):
    def __init__(self) -> None:
        super().__init__()
        self.slow_reads = False
        self.apply_calls = 0

    def get_arm(self, arm_id: str) -> ArmTelemetry:
        if self.slow_reads:
            time.sleep(0.06)
        return super().get_arm(arm_id)

    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry:
        self.apply_calls += 1
        return super().apply_action(arm_id, action)


@pytest.mark.asyncio
async def test_driver_failure_stops_safely_without_leaking_raw_error() -> None:
    transport = FakeTransport(_descriptor(actions_per_chunk=2))
    driver = FailingDriver(fail_after=1)
    lease = RigLease()
    loop = _loop(transport, driver=driver, lease=lease)
    await loop.start()
    await _wait_until(lambda: not loop.snapshot().running)
    stopped = loop.snapshot()

    assert stopped.steps_executed == 1
    assert stopped.last_error == "The robot inference loop stopped safely."
    assert "secret" not in stopped.last_error
    assert transport.closed
    assert lease.current() is None


@pytest.mark.asyncio
async def test_action_age_is_rechecked_after_slow_telemetry_before_driver_write() -> None:
    driver = SlowReadDriver()
    loop = _loop(
        FakeTransport(_descriptor(actions_per_chunk=1)),
        driver=driver,
        max_action_age_seconds=0.05,
    )
    await loop.start()
    driver.slow_reads = True
    await _wait_until(lambda: loop.snapshot().dropped_chunks >= 1)
    stopped = await loop.stop()

    assert stopped.dropped_chunks >= 1
    assert stopped.steps_executed == 0
    assert driver.apply_calls == 0


class BlockingCloseTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__(_descriptor(actions_per_chunk=1))
        self.close_entered = threading.Event()
        self.close_release = threading.Event()

    def close(self) -> None:
        self.close_entered.set()
        assert self.close_release.wait(timeout=2)
        super().close()


@pytest.mark.asyncio
async def test_repeated_cancellation_during_close_still_releases_the_rig() -> None:
    transport = BlockingCloseTransport()
    lease = RigLease()
    loop = _loop(transport, lease=lease)
    await loop.start()
    await loop.wait_for_steps(1, timeout_seconds=1)

    stop_task = asyncio.create_task(loop.stop())
    assert await asyncio.to_thread(transport.close_entered.wait, 1)
    run_task = loop._task
    assert run_task is not None
    run_task.cancel()
    run_task.cancel()
    transport.close_release.set()

    with pytest.raises(asyncio.CancelledError):
        await stop_task
    snapshot = loop.snapshot()
    assert not snapshot.running and snapshot.queue_depth == 0
    assert transport.closed
    assert lease.current() is None
    assert await loop.active_resource_counts() == (0, 0)


class ReportingReleaseFailureLease(RigLease):
    def release(self, token) -> None:  # type: ignore[no-untyped-def]
        super().release(token)
        raise RigLeaseOwnershipError("raw lease detail")


@pytest.mark.asyncio
async def test_cleanup_failure_is_preserved_in_terminal_snapshot() -> None:
    lease = ReportingReleaseFailureLease()
    loop = _loop(FakeTransport(_descriptor(actions_per_chunk=1)), lease=lease)
    await loop.start()
    await loop.wait_for_steps(1, timeout_seconds=1)
    stopped = await loop.stop()

    assert stopped.last_error == (
        "The inference resources could not be released safely; command ownership remains blocked."
    )
    assert "raw" not in stopped.last_error
    assert lease.current() is None


def test_loop_configuration_and_wait_bounds_are_strict() -> None:
    transport = FakeTransport()
    with pytest.raises(RobotInferenceConfigurationError):
        _loop(transport, max_joint_step_radians="0.1")
    with pytest.raises(RobotInferenceConfigurationError):
        _loop(transport, max_gripper_step=float("nan"))
    with pytest.raises(RobotInferenceConfigurationError):
        _loop(transport, allow_opaque_mock_actions=1)
    with pytest.raises(RobotInferenceConfigurationError):
        RobotInferenceLoop(
            driver=MockYAMDriver(),
            camera=MockCamera(width=32, height=24),
            rig_lease=RigLease(),
            transport=transport,
            arm_id="yam-follower",
            task="move",
            expected_runtime="lerobot",
            expected_model_repo="invalid",
            expected_revision=REVISION,
        )


@pytest.mark.asyncio
async def test_wait_validation_and_single_use_lifecycle() -> None:
    loop = _loop(FakeTransport())
    with pytest.raises(ValueError):
        await loop.wait_for_steps(True)
    with pytest.raises(ValueError):
        await loop.wait_for_steps(1, timeout_seconds=float("nan"))
    with pytest.raises(RobotInferenceConflictError, match="not started"):
        await loop.stop()

    await loop.start()
    await loop.stop()
    second = await loop.stop()
    assert not second.running
    with pytest.raises(RobotInferenceConflictError, match="single-use"):
        await loop.start()
