"""Short-lived, handle-only SocketCAN diagnostic boundary.

The child opens one SocketCAN bus only after the separately acknowledged API
operation.  It sends only the ioheart passive encoder report request
(``0x50E`` with ``[0xFF, 0x02]``), accepts only ``0x50F`` reports, and always
closes the bus.  It never imports i2rt, constructs a robot/motor object, writes
encoder configuration, changes link state, or exposes a general CAN surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import math
import multiprocessing
from multiprocessing.context import BaseContext
import os
import queue
import struct
import signal
import sys
import time
from typing import Any, Callable, Protocol


HANDLE_REQUEST_ID = 0x50E
HANDLE_REPORT_ID = 0x50F
HANDLE_REPORT_REQUEST = bytes((0xFF, 0x02))
HANDLE_RANGE_RADIANS = 0.7


class YAMHandleDiagnosticError(RuntimeError):
    """Sanitized failure of the isolated handle reader."""


class YAMHandleDiagnosticTeardownUncertainError(YAMHandleDiagnosticError):
    """The diagnostic child may still own its SocketCAN file descriptor."""


@dataclass(frozen=True, slots=True)
class HandleDiagnosticConfig:
    logical_id: str
    stable_identity: str
    runtime_interface: str
    duration_seconds: float
    sample_frequency_hz: float = 20.0
    response_timeout_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not self.logical_id or not self.stable_identity:
            raise ValueError("handle diagnostic identity may not be empty")
        if not self.runtime_interface or len(self.runtime_interface.encode("utf-8")) > 15:
            raise ValueError("handle diagnostic interface is invalid")
        if not math.isfinite(self.duration_seconds) or not 0.1 <= self.duration_seconds <= 30.0:
            raise ValueError("handle diagnostic duration must be between 0.1 and 30 seconds")
        if not math.isfinite(self.sample_frequency_hz) or not 1.0 <= self.sample_frequency_hz <= 50.0:
            raise ValueError("handle diagnostic frequency must be between 1 and 50 Hz")
        if not math.isfinite(self.response_timeout_seconds) or not 0.005 <= self.response_timeout_seconds <= 0.5:
            raise ValueError("handle diagnostic response timeout is invalid")


@dataclass(frozen=True, slots=True)
class HandleDiagnosticSample:
    reachable: bool
    observed_minimum: float | None
    observed_maximum: float | None
    latest_buttons: tuple[bool, ...]
    sample_count: int


class HandleRangeReader(Protocol):
    def sample(self, config: HandleDiagnosticConfig) -> HandleDiagnosticSample: ...


class _CANBus(Protocol):
    def set_filters(self, filters: list[dict[str, int]]) -> None: ...

    def send(self, message: Any) -> None: ...

    def recv(self, timeout: float) -> Any | None: ...

    def shutdown(self) -> None: ...


BusFactory = Callable[[str], tuple[_CANBus, Callable[[int, bytes], Any]]]


def _default_bus_factory(
    runtime_interface: str,
) -> tuple[_CANBus, Callable[[int, bytes], Any]]:
    # Imported only inside the explicit diagnostic child.  Constructing a
    # python-can SocketCAN bus opens the CAN socket but no YAM motor/device.
    import can

    bus = can.interface.Bus(interface="socketcan", channel=runtime_interface)

    def message(arbitration_id: int, payload: bytes) -> Any:
        return can.Message(
            arbitration_id=arbitration_id,
            data=payload,
            is_extended_id=False,
        )

    return bus, message


def sample_handle_requests_only(
    config: HandleDiagnosticConfig,
    *,
    bus_factory: BusFactory = _default_bus_factory,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> HandleDiagnosticSample:
    """Execute the bounded wire protocol; injectable for hardware-free tests."""

    bus, message_factory = bus_factory(config.runtime_interface)
    minimum: float | None = None
    maximum: float | None = None
    buttons: tuple[bool, ...] = ()
    count = 0
    deadline = monotonic() + config.duration_seconds
    period = 1.0 / config.sample_frequency_hz
    try:
        bus.set_filters([{"can_id": HANDLE_REPORT_ID, "can_mask": 0x7FF}])
        while True:
            bus.send(message_factory(HANDLE_REQUEST_ID, HANDLE_REPORT_REQUEST))
            response_deadline = monotonic() + config.response_timeout_seconds
            response = None
            while monotonic() < response_deadline:
                response = bus.recv(
                    timeout=max(0.0, response_deadline - monotonic())
                )
                if response is None:
                    break
                if (
                    getattr(response, "arbitration_id", None) == HANDLE_REPORT_ID
                    and not bool(getattr(response, "is_extended_id", False))
                ):
                    break
                response = None
            if response is not None:
                payload = bytes(getattr(response, "data", b""))
                if len(payload) == struct.calcsize("!BhhB"):
                    _device, position_counts, _velocity_counts, inputs = struct.unpack(
                        "!BhhB", payload
                    )
                    position_radians = position_counts * 2.0 * math.pi / 4096.0
                    trigger = min(1.0, abs(position_radians) / HANDLE_RANGE_RADIANS)
                    minimum = trigger if minimum is None else min(minimum, trigger)
                    maximum = trigger if maximum is None else max(maximum, trigger)
                    buttons = (bool(inputs & 0x01), bool(inputs & 0x02))
                    count += 1
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                break
            sleep(min(period, remaining))
    finally:
        bus.shutdown()
    return HandleDiagnosticSample(
        reachable=count > 0,
        observed_minimum=minimum,
        observed_maximum=maximum,
        latest_buttons=buttons,
        sample_count=count,
    )


def _handle_child(
    config: HandleDiagnosticConfig,
    result_queue: Any,
    expected_parent_pid: int,
) -> None:
    try:
        _set_parent_death_signal(expected_parent_pid)
        result = sample_handle_requests_only(config)
    except BaseException:
        result = None
    try:
        result_queue.put_nowait(result)
    except queue.Full:
        return


def _set_parent_death_signal(expected_parent_pid: int) -> None:
    """Ensure a Linux diagnostic child cannot outlive a dying supervisor.

    The parent PID is checked again after ``prctl`` to close the fork/prctl
    race; if the parent changed in that window, the child exits before opening
    SocketCAN.
    """

    if not sys.platform.startswith("linux"):
        return
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("handle diagnostic supervisor exited before child start")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("handle diagnostic supervisor exited during child start")


class SupervisedHandleRangeReader:
    """Run the raw handle-only reader in one bounded, always-reaped child."""

    def __init__(
        self,
        *,
        context: BaseContext | None = None,
        child_target: Callable[..., None] = _handle_child,
        teardown_grace_seconds: float = 1.0,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        self._child_target = child_target
        self._teardown_grace = teardown_grace_seconds

    def sample(self, config: HandleDiagnosticConfig) -> HandleDiagnosticSample:
        result_queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=self._child_target,
            args=(config, result_queue, os.getpid()),
            name=f"ctrl-pi-yam-handle-{config.logical_id}",
            daemon=False,
        )
        started = False
        process_error: BaseException | None = None
        try:
            process.start()
            started = True
            process.join(timeout=config.duration_seconds + self._teardown_grace)
        except BaseException as error:
            process_error = error

        reaped = not started or self._terminate_and_reap(process)
        if started and not reaped:
            try:
                raise YAMHandleDiagnosticTeardownUncertainError(
                    "The handle-only diagnostic child could not be reaped."
                ) from process_error
            finally:
                self._close_queue(result_queue)
        if process_error is not None:
            try:
                if isinstance(process_error, YAMHandleDiagnosticError):
                    raise process_error
                raise YAMHandleDiagnosticError(
                    "The handle-only diagnostic could not be started safely."
                ) from process_error
            finally:
                self._close_queue(result_queue)
        try:
            try:
                result = result_queue.get(timeout=0.2)
            except queue.Empty as error:
                raise YAMHandleDiagnosticError(
                    "The handle-only diagnostic did not return a readable sample."
                ) from error
            if not isinstance(result, HandleDiagnosticSample):
                raise YAMHandleDiagnosticError(
                    "The handle-only diagnostic could not read the encoder safely."
                )
            return result
        except YAMHandleDiagnosticError:
            raise
        except BaseException as error:
            raise YAMHandleDiagnosticError(
                "The handle-only diagnostic result could not be read safely."
            ) from error
        finally:
            self._close_queue(result_queue)

    def _terminate_and_reap(self, process: Any) -> bool:
        if self._is_reaped(process):
            return True
        try:
            process.terminate()
        except BaseException:
            pass
        try:
            process.join(timeout=self._teardown_grace)
        except BaseException:
            pass
        if self._is_reaped(process):
            return True
        try:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
        except BaseException:
            pass
        try:
            process.join(timeout=self._teardown_grace)
        except BaseException:
            pass
        return self._is_reaped(process)

    @staticmethod
    def _is_reaped(process: Any) -> bool:
        try:
            return not process.is_alive() and process.exitcode is not None
        except BaseException:
            return False

    @staticmethod
    def _close_queue(result_queue: Any) -> None:
        try:
            result_queue.close()
            result_queue.cancel_join_thread()
        except BaseException:
            pass
