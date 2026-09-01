"""Supervised per-arm process boundary for the field-tested i2rt YAM stack.

This module deliberately does not import :mod:`i2rt` at module import time.  A
motion-capable vendor object is constructed only in the child process started
by an explicit connect operation.  The FastAPI process owns identity,
supervision, command freshness, and teardown; the child owns the CAN object and
all threads created by i2rt.

The protocol is local-only multiprocessing IPC.  It does not start Portal, an
RPC server, a shell launcher, or a network listener.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import importlib
import io
import json
import logging
import math
import multiprocessing
from multiprocessing.context import BaseContext
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from types import TracebackType
from typing import Any, Callable, Literal, Protocol, TypeAlias
from uuid import uuid4


PROTOCOL_VERSION = 1
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CAN_INTERFACE_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,14})")
_MAX_LOG_CHARS = 512
_MAX_DIAGNOSTIC_CHARS = 240
_MAX_GIT_STATUS_BYTES = 64 * 1024
_PR_SET_PDEATHSIG = 1
logger = logging.getLogger(__name__)


class I2RTWorkerError(RuntimeError):
    """Base class for sanitized worker-boundary failures."""


class I2RTCheckoutError(I2RTWorkerError):
    """The operator checkout cannot be proven to be the configured source."""


class I2RTWorkerStartupError(I2RTWorkerError):
    """The arm worker did not reach a verified ready state."""


class I2RTWorkerProtocolError(I2RTWorkerError):
    """A child sent an invalid or mismatched IPC message."""


class I2RTWorkerUnavailableError(I2RTWorkerError):
    """The worker is not healthy enough to read or command safely."""


@dataclass(frozen=True, slots=True)
class I2RTCheckoutIdentity:
    root: str
    commit: str
    read_only: bool


@dataclass(frozen=True, slots=True)
class I2RTArmWorkerConfig:
    """Motion-critical configuration copied into exactly one arm worker."""

    logical_id: str
    role: Literal["leader", "follower"]
    stable_identity: str
    runtime_interface: str
    end_effector_kind: Literal[
        "yam_teaching_handle", "linear_4310", "crank_4310"
    ]
    checkout_root: str
    expected_commit: str
    require_read_only_checkout: bool = True
    command_max_age_seconds: float = 0.250
    heartbeat_interval_seconds: float = 0.100
    telemetry_interval_seconds: float = 0.020

    def __post_init__(self) -> None:
        root = Path(self.checkout_root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("i2rt checkout_root must be an absolute bounded path")
        if _COMMIT_RE.fullmatch(self.expected_commit) is None:
            raise ValueError("expected_commit must be an exact lowercase 40-hex commit")
        if _CAN_INTERFACE_RE.fullmatch(self.runtime_interface) is None:
            raise ValueError("runtime_interface must be a Linux-safe interface name")
        if not self.logical_id or len(self.logical_id.encode("utf-8")) > 120:
            raise ValueError("logical_id must be 1 through 120 bytes")
        if not self.stable_identity or len(self.stable_identity.encode("utf-8")) > 256:
            raise ValueError("stable_identity must be 1 through 256 bytes")
        if any(
            ord(character) < 32 or ord(character) == 127
            for value in (self.logical_id, self.stable_identity)
            for character in value
        ):
            raise ValueError("worker identity may not contain control characters")
        if self.role == "leader" and self.end_effector_kind != "yam_teaching_handle":
            raise ValueError("CAN leaders require a yam_teaching_handle")
        if self.role == "follower" and self.end_effector_kind not in {
            "linear_4310",
            "crank_4310",
        }:
            raise ValueError("CAN followers require a supported actuated gripper")
        for name, value, lower, upper in (
            ("command_max_age_seconds", self.command_max_age_seconds, 0.020, 2.0),
            ("heartbeat_interval_seconds", self.heartbeat_interval_seconds, 0.020, 1.0),
            ("telemetry_interval_seconds", self.telemetry_interval_seconds, 0.005, 1.0),
        ):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")

    @property
    def fingerprint(self) -> str:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "logical_id": self.logical_id,
            "role": self.role,
            "stable_identity": self.stable_identity,
            "runtime_interface": self.runtime_interface,
            "end_effector_kind": self.end_effector_kind,
            "checkout_root": str(Path(self.checkout_root)),
            "expected_commit": self.expected_commit,
            "require_read_only_checkout": self.require_read_only_checkout,
            "command_max_age_seconds": self.command_max_age_seconds,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    sequence: int
    issued_monotonic: float
    deadline_monotonic: float
    positions: tuple[float, float, float, float, float, float, float]


ControlOperation = Literal["safe_idle", "resume_commands", "shutdown"]


@dataclass(frozen=True, slots=True)
class WorkerControlRequest:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    request_id: str
    operation: ControlOperation
    issued_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerReady:
    protocol_version: int
    logical_id: str
    stable_identity: str
    runtime_interface: str
    config_fingerprint: str
    checkout_commit: str
    pid: int
    process_group_id: int | None
    parent_death_signal: bool
    inner_liveness: dict[str, bool]
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    inner_liveness: dict[str, bool]
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerControlReply:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    request_id: str
    operation: ControlOperation
    ok: bool
    detail: str
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerCommandDropped:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    sequence: int
    reason: Literal["stale", "not_enabled", "invalid"]
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerFault:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    detail: str
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerStopped:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    safe_idle_confirmed: bool
    device_close_confirmed: bool
    detail: str
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerLog:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    level: Literal["debug", "info", "warning", "error"]
    message: str
    emitted_monotonic: float


@dataclass(frozen=True, slots=True)
class WorkerTelemetry:
    protocol_version: int
    logical_id: str
    config_fingerprint: str
    sample_monotonic: float
    joint_positions: tuple[float, float, float, float, float, float]
    joint_velocities: tuple[float, float, float, float, float, float]
    joint_efforts: tuple[float, float, float, float, float, float]
    gripper_position: float
    gripper_velocity: float
    gripper_effort: float | None
    handle_trigger_position: float | None
    handle_buttons: tuple[bool, ...]
    control_loop_frequency_hz: float
    inner_liveness: dict[str, bool]
    command_sequence_applied: int | None


WorkerStatusMessage: TypeAlias = (
    WorkerReady
    | WorkerHeartbeat
    | WorkerControlReply
    | WorkerCommandDropped
    | WorkerFault
    | WorkerStopped
)


@dataclass(frozen=True, slots=True)
class I2RTWorkerSnapshot:
    phase: Literal["new", "starting", "ready", "stopping", "stopped", "error"]
    pid: int | None
    process_group_id: int | None
    accepting_commands: bool
    heartbeat_age_seconds: float | None
    telemetry_age_seconds: float | None
    inner_liveness: dict[str, bool]
    dropped_commands: int
    fault: str | None
    shutdown_uncertain: bool
    reaped: bool
    logs: tuple[WorkerLog, ...]
    exitcode: int | None = None


@dataclass(frozen=True, slots=True)
class I2RTWorkerShutdownResult:
    clean: bool
    safe_idle_confirmed: bool
    device_close_confirmed: bool
    forced: bool
    reaped: bool
    uncertain: bool
    detail: str


class _RobotLike(Protocol):
    motor_chain: Any

    def get_observations(self) -> dict[str, Any]: ...

    def command_joint_pos(self, joint_pos: Any) -> None: ...

    def enter_gravity_comp_idle(self) -> None: ...

    def close(self) -> None: ...


RobotLoader = Callable[[I2RTArmWorkerConfig], _RobotLike]
ChildTarget = Callable[..., None]


class _I2RTProcessSpawner:
    """Create i2rt children from one thread that outlives every child.

    Linux ties ``PR_SET_PDEATHSIG`` to the thread that created the child, not
    merely to the containing process. FastAPI synchronous routes run in AnyIO
    worker threads that may retire while the application remains healthy, so
    they must never call ``Process.start()`` directly.

    Leases keep the single executor thread alive until every process it
    created has been reaped. Releasing the last lease joins the thread; a
    later hardware connection lazily creates a fresh one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._leases = 0

    def acquire(self) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="ctrl-pi-i2rt-spawner",
                )
            self._leases += 1

    def start_process(self, process: Any) -> None:
        with self._lock:
            executor = self._executor
            if executor is None or self._leases <= 0:
                raise RuntimeError("the i2rt process spawner has no active lease")
            future = executor.submit(process.start)
        # Process.start() exceptions retain their original type and cause the
        # normal fail-closed startup cleanup in I2RTArmWorker.
        future.result()

    def release(self) -> None:
        executor: ThreadPoolExecutor | None = None
        with self._lock:
            if self._leases <= 0:
                raise RuntimeError("the i2rt process spawner lease is unbalanced")
            self._leases -= 1
            if self._leases == 0:
                executor = self._executor
                self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


_I2RT_PROCESS_SPAWNER = _I2RTProcessSpawner()


class _LatestCommandMailbox:
    """One-slot shared-memory mailbox with atomic overwrite semantics.

    ``multiprocessing.Queue(maxsize=1)`` has a feeder-thread window in which a
    full item cannot yet be drained with ``get_nowait``.  That can accidentally
    preserve the old command.  This fixed-size mailbox stores only numeric
    command fields and reconstructs the typed envelope on read, so every write
    atomically replaces the previous pending action.
    """

    def __init__(self, context: BaseContext, config: I2RTArmWorkerConfig) -> None:
        self._config = config
        self._lock = context.Lock()
        self._pending = context.RawValue("b", False)
        self._sequence = context.RawValue("Q", 0)
        self._issued = context.RawValue("d", 0.0)
        self._deadline = context.RawValue("d", 0.0)
        self._positions = context.RawArray("d", 7)

    def put_nowait(self, item: object) -> None:
        if not isinstance(item, WorkerCommand):
            raise TypeError("the command mailbox accepts only WorkerCommand")
        with self._lock:
            self._sequence.value = item.sequence
            self._issued.value = item.issued_monotonic
            self._deadline.value = item.deadline_monotonic
            for index, value in enumerate(item.positions):
                self._positions[index] = value
            self._pending.value = True

    def get_nowait(self) -> WorkerCommand:
        with self._lock:
            if not self._pending.value:
                raise queue.Empty
            command = WorkerCommand(
                protocol_version=PROTOCOL_VERSION,
                logical_id=self._config.logical_id,
                config_fingerprint=self._config.fingerprint,
                sequence=int(self._sequence.value),
                issued_monotonic=float(self._issued.value),
                deadline_monotonic=float(self._deadline.value),
                positions=tuple(  # type: ignore[arg-type]
                    float(value) for value in self._positions
                ),
            )
            self._pending.value = False
            return command

    def close(self) -> None:
        return None

    def cancel_join_thread(self) -> None:
        return None


def _clean_text(value: object, *, maximum: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    text = " ".join(str(value).split())
    text = "".join(character for character in text if 32 <= ord(character) != 127)
    if not text:
        return "unspecified worker failure"
    return text[:maximum]


def _checkout_is_read_only(root: Path) -> bool:
    """Return whether the checkout filesystem is mounted read-only.

    Only the kernel's ``ST_RDONLY`` mount flag is a positive production proof.
    Directory permissions and ``os.access`` are insufficient because tracked
    files below a non-writable directory may remain writable. No probe file is
    created because the verification path is itself passive.
    """

    try:
        read_only_flag = getattr(os, "ST_RDONLY", 1)
        return bool(os.statvfs(root).f_flag & read_only_flag)
    except OSError:
        return False


def verify_i2rt_checkout(
    checkout_root: str,
    expected_commit: str,
    *,
    require_read_only: bool = True,
    timeout_seconds: float = 2.0,
) -> I2RTCheckoutIdentity:
    """Verify a local checkout without fetching or consulting a remote.

    Bounded, read-only Git queries prove the root, exact commit, clean tracked
    worktree/index, and absence of untracked shadow files. Environment values
    that could redirect Git to another worktree are removed. The function
    never invokes a shell, package installer, network client, checkout fetch,
    or checkout mutation.
    """

    root = Path(checkout_root)
    if not root.is_absolute() or ".." in root.parts:
        raise I2RTCheckoutError("The i2rt checkout path must be absolute.")
    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise I2RTCheckoutError("The expected i2rt commit must be exact 40-hex.")
    # The field-tested checkout uses ``i2rt`` as a PEP 420 namespace package,
    # so an ``i2rt/__init__.py`` marker would reject the exact supported tree.
    # These two ordinary files identify the checkout without importing i2rt or
    # touching a motion-capable vendor boundary.
    if (
        not root.is_dir()
        or not (root / "pyproject.toml").is_file()
        or not (root / "i2rt" / "robots" / "get_robot.py").is_file()
    ):
        raise I2RTCheckoutError("The configured i2rt checkout is not present.")

    read_only = _checkout_is_read_only(root)
    if require_read_only and not read_only:
        raise I2RTCheckoutError(
            "The configured i2rt checkout is writable; mount or expose it read-only."
        )

    environment = os.environ.copy()
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(key, None)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    resolved_root = root.resolve()
    git_prefix = [
        "git",
        "-c",
        f"safe.directory={resolved_root}",
        "-C",
        str(root),
    ]

    def git_read(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                [*git_prefix, *arguments],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise I2RTCheckoutError(
                "The configured i2rt Git identity could not be verified."
            ) from error
        if len(completed.stdout.encode("utf-8")) > _MAX_GIT_STATUS_BYTES:
            raise I2RTCheckoutError(
                "The configured i2rt Git verification output exceeded its bound."
            )
        return completed.stdout.strip()

    def git_clean_check(*arguments: str) -> None:
        try:
            completed = subprocess.run(
                [*git_prefix, *arguments],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise I2RTCheckoutError(
                "The configured i2rt Git identity could not be verified."
            ) from error
        if completed.returncode != 0:
            raise I2RTCheckoutError(
                "The local i2rt checkout has tracked or staged changes."
            )

    top_level = Path(git_read("rev-parse", "--show-toplevel")).resolve()
    if top_level != resolved_root:
        raise I2RTCheckoutError("The configured path is not the i2rt checkout root.")
    head = git_read("rev-parse", "--verify", "HEAD^{commit}").lower()
    if _COMMIT_RE.fullmatch(head) is None or head != expected_commit:
        raise I2RTCheckoutError("The local i2rt checkout is not at the expected commit.")
    # ``diff-index`` produces no content in quiet mode and detects both staged
    # and unstaged tracked changes relative to the claimed HEAD.
    git_clean_check("diff-index", "--quiet", "HEAD", "--")
    untracked_or_ignored = git_read(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored",
    )
    if untracked_or_ignored:
        raise I2RTCheckoutError(
            "The local i2rt checkout contains untracked or ignored files."
        )
    return I2RTCheckoutIdentity(root=str(resolved_root), commit=head, read_only=read_only)


def _configure_linux_parent_death_signal(parent_pid: int) -> bool:
    """Ask Linux to SIGTERM this child if its spawning parent disappears.

    The caller captures ``parent_pid`` before ``prctl`` and checks it again
    immediately after, closing the documented race where the parent exits
    before the kernel setting takes effect. Non-Linux platforms return False;
    production YAM hardware remains Linux-only.
    """

    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (AttributeError, OSError) as error:
        raise RuntimeError("Linux parent-death supervision is unavailable") from error
    if result != 0:
        error_number = ctypes.get_errno()
        raise RuntimeError(
            f"Linux parent-death supervision failed with errno {error_number}"
        )
    return os.getppid() == parent_pid


def load_pinned_i2rt_robot(config: I2RTArmWorkerConfig) -> _RobotLike:
    """Import and construct the pinned i2rt robot inside the active child only."""

    if any(name == "i2rt" or name.startswith("i2rt.") for name in sys.modules):
        raise RuntimeError("i2rt was imported before checkout identity verification")
    checkout_root = Path(config.checkout_root).resolve()
    sys.path.insert(0, str(checkout_root))
    get_robot_module = importlib.import_module("i2rt.robots.get_robot")
    utils_module = importlib.import_module("i2rt.robots.utils")
    for module in (get_robot_module, utils_module):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError("the loaded i2rt module has no source identity")
        try:
            Path(module_file).resolve().relative_to(checkout_root)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "i2rt resolved outside the verified operator checkout"
            ) from error
    gripper_type = utils_module.GripperType.from_string_name(config.end_effector_kind)
    return get_robot_module.get_yam_robot(
        channel=config.runtime_interface,
        arm_type=utils_module.ArmType.YAM,
        gripper_type=gripper_type,
        zero_gravity_mode=True,
        enable_auto_recovery=False,
    )


def _put_latest(target_queue: Any, item: object) -> None:
    """Non-blocking, bounded latest-wins queue write."""

    for _ in range(3):
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
    # Never block a hardware/control loop on observability IPC.


def _drain_latest(source_queue: Any) -> object | None:
    latest: object | None = None
    while True:
        try:
            latest = source_queue.get_nowait()
        except queue.Empty:
            return latest


class _IPCLogHandler(logging.Handler):
    def __init__(self, config: I2RTArmWorkerConfig, log_queue: Any) -> None:
        super().__init__()
        self._config = config
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: Literal["debug", "info", "warning", "error"]
            if record.levelno >= logging.ERROR:
                level = "error"
            elif record.levelno >= logging.WARNING:
                level = "warning"
            elif record.levelno >= logging.INFO:
                level = "info"
            else:
                level = "debug"
            _put_latest(
                self._queue,
                WorkerLog(
                    protocol_version=PROTOCOL_VERSION,
                    logical_id=self._config.logical_id,
                    config_fingerprint=self._config.fingerprint,
                    level=level,
                    message=_clean_text(self.format(record), maximum=_MAX_LOG_CHARS),
                    emitted_monotonic=time.monotonic(),
                ),
            )
        except Exception:
            return


class _IPCLogStream(io.TextIOBase):
    """Line-buffer vendor ``print`` calls into the same bounded log channel."""

    def __init__(self, handler: _IPCLogHandler, level: int) -> None:
        self._handler = handler
        self._level = level
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self._buffer = (self._buffer + value)[-_MAX_LOG_CHARS * 2 :]
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                record = logging.LogRecord(
                    "i2rt.vendor",
                    self._level,
                    "",
                    0,
                    _clean_text(line, maximum=_MAX_LOG_CHARS),
                    (),
                    None,
                )
                self._handler.emit(record)
        return len(value)

    def flush(self) -> None:
        if self._buffer.strip():
            self.write("\n")


def _configure_child_logging(config: I2RTArmWorkerConfig, log_queue: Any) -> None:
    handler = _IPCLogHandler(config, log_queue)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    sys.stdout = _IPCLogStream(handler, logging.INFO)
    sys.stderr = _IPCLogStream(handler, logging.ERROR)


def _inner_liveness(robot: _RobotLike) -> dict[str, bool]:
    motor_chain = getattr(robot, "motor_chain", None)
    robot_thread = getattr(robot, "_server_thread", None)
    return {
        "motor_chain_running": bool(getattr(motor_chain, "running", False)),
        "robot_server_alive": bool(
            robot_thread is not None
            and callable(getattr(robot_thread, "is_alive", None))
            and robot_thread.is_alive()
        ),
    }


def _finite_vector(value: Any, *, length: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {label}") from error
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"invalid {label}")
    return result


def _sample_robot(
    config: I2RTArmWorkerConfig,
    robot: _RobotLike,
    applied_sequence: int | None,
) -> WorkerTelemetry:
    observations = robot.get_observations()
    joint_positions = _finite_vector(
        observations.get("joint_pos"), length=6, label="joint positions"
    )
    joint_velocities = _finite_vector(
        observations.get("joint_vel"), length=6, label="joint velocities"
    )
    joint_efforts = _finite_vector(
        observations.get("joint_eff"), length=6, label="joint efforts"
    )

    handle_trigger: float | None = None
    handle_buttons: tuple[bool, ...] = ()
    if config.role == "leader":
        motor_chain = robot.motor_chain
        reader = getattr(motor_chain, "get_same_bus_device_states", None)
        states = reader() if callable(reader) else None
        if not states:
            raise RuntimeError("teaching handle telemetry is unavailable")
        encoder = states[0]
        handle_trigger = float(encoder.position)
        if not math.isfinite(handle_trigger) or not 0.0 <= handle_trigger <= 1.0:
            raise RuntimeError("teaching handle telemetry is invalid")
        handle_buttons = tuple(bool(value) for value in encoder.io_inputs)
        if len(handle_buttons) > 8:
            raise RuntimeError("teaching handle button telemetry is invalid")
        gripper_position = 1.0 - handle_trigger
        gripper_velocity = 0.0
        gripper_effort = None
    else:
        gripper_position = _finite_vector(
            observations.get("gripper_pos"), length=1, label="gripper position"
        )[0]
        gripper_velocity = _finite_vector(
            observations.get("gripper_vel"), length=1, label="gripper velocity"
        )[0]
        gripper_effort_values = observations.get("gripper_eff")
        gripper_effort = (
            _finite_vector(gripper_effort_values, length=1, label="gripper effort")[0]
            if gripper_effort_values is not None
            else None
        )
    if not 0.0 <= gripper_position <= 1.0:
        raise RuntimeError("normalized gripper telemetry is outside [0, 1]")

    loop_frequency = float(getattr(robot.motor_chain, "comm_freq", 0.0))
    if not math.isfinite(loop_frequency) or loop_frequency < 0.0:
        raise RuntimeError("control-loop frequency is invalid")
    liveness = _inner_liveness(robot)
    if not all(liveness.values()):
        raise RuntimeError("an i2rt inner control loop stopped")
    return WorkerTelemetry(
        protocol_version=PROTOCOL_VERSION,
        logical_id=config.logical_id,
        config_fingerprint=config.fingerprint,
        sample_monotonic=time.monotonic(),
        joint_positions=joint_positions,  # type: ignore[arg-type]
        joint_velocities=joint_velocities,  # type: ignore[arg-type]
        joint_efforts=joint_efforts,  # type: ignore[arg-type]
        gripper_position=gripper_position,
        gripper_velocity=gripper_velocity,
        gripper_effort=gripper_effort,
        handle_trigger_position=handle_trigger,
        handle_buttons=handle_buttons,
        control_loop_frequency_hz=loop_frequency,
        inner_liveness=liveness,
        command_sequence_applied=applied_sequence,
    )


def _validate_command(
    config: I2RTArmWorkerConfig, command: object, now: float
) -> Literal["fresh", "stale", "invalid"]:
    if not isinstance(command, WorkerCommand):
        return "invalid"
    if (
        command.protocol_version != PROTOCOL_VERSION
        or command.logical_id != config.logical_id
        or command.config_fingerprint != config.fingerprint
        or command.sequence < 1
        or command.deadline_monotonic < command.issued_monotonic
        or command.deadline_monotonic - command.issued_monotonic
        > config.command_max_age_seconds + 1e-6
        or command.issued_monotonic > now + 1.0
        or not all(math.isfinite(value) for value in command.positions)
        or not 0.0 <= command.positions[6] <= 1.0
    ):
        return "invalid"
    if now > command.deadline_monotonic:
        return "stale"
    return "fresh"


def _safe_idle(robot: _RobotLike) -> bool:
    method = getattr(robot, "enter_gravity_comp_idle", None)
    if not callable(method):
        return False
    method()
    return True


def i2rt_arm_worker_main(
    config: I2RTArmWorkerConfig,
    control_queue: Any,
    command_queue: Any,
    status_queue: Any,
    telemetry_queue: Any,
    log_queue: Any,
    expected_parent_pid: int,
    robot_loader: RobotLoader = load_pinned_i2rt_robot,
) -> None:
    """Child entry point.  Tests may inject a fake ``robot_loader``."""

    termination_requested = threading.Event()
    # The spawning process captures this PID before Process.start(). If that
    # parent died before this child was scheduled, getppid() may already be a
    # subreaper/init PID; never install supervision against that replacement.
    if expected_parent_pid <= 1 or os.getppid() != expected_parent_pid:
        return

    def request_termination(_signum: int, _frame: Any) -> None:
        termination_requested.set()

    try:
        signal.signal(signal.SIGTERM, request_termination)
        signal.signal(signal.SIGINT, request_termination)
    except (ValueError, OSError):
        pass

    parent_death_signal = _configure_linux_parent_death_signal(
        expected_parent_pid
    )
    if sys.platform.startswith("linux") and not parent_death_signal:
        # The parent exited in the narrow window before prctl completed. Do
        # not construct a motion-capable object as an already orphaned child.
        termination_requested.set()

    process_group_id: int | None = None
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        if hasattr(os, "getpgrp"):
            process_group_id = os.getpgrp()
    except OSError:
        process_group_id = None

    _configure_child_logging(config, log_queue)
    robot: _RobotLike | None = None
    safe_idle_confirmed = False
    close_confirmed = False
    fault_detail: str | None = None
    applied_sequence: int | None = None
    last_accepted_command_monotonic: float | None = None
    accepting_commands = False
    try:
        checkout = verify_i2rt_checkout(
            config.checkout_root,
            config.expected_commit,
            require_read_only=config.require_read_only_checkout,
        )
        if termination_requested.is_set():
            raise RuntimeError("the worker parent exited before i2rt construction")
        # This is the first point at which a motion-capable import/factory is
        # allowed.  Checkout identity has already been verified locally.
        robot = robot_loader(config)
        telemetry = _sample_robot(config, robot, applied_sequence)
        _put_latest(telemetry_queue, telemetry)
        _put_latest(
            status_queue,
            WorkerReady(
                protocol_version=PROTOCOL_VERSION,
                logical_id=config.logical_id,
                stable_identity=config.stable_identity,
                runtime_interface=config.runtime_interface,
                config_fingerprint=config.fingerprint,
                checkout_commit=checkout.commit,
                pid=os.getpid(),
                process_group_id=process_group_id,
                parent_death_signal=parent_death_signal,
                inner_liveness=telemetry.inner_liveness,
                emitted_monotonic=time.monotonic(),
            ),
        )
        next_heartbeat = time.monotonic()
        next_telemetry = time.monotonic() + config.telemetry_interval_seconds
        stop_requested = False
        while not stop_requested and not termination_requested.is_set():
            while True:
                try:
                    request = control_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(request, WorkerControlRequest) or (
                    request.protocol_version != PROTOCOL_VERSION
                    or request.logical_id != config.logical_id
                    or request.config_fingerprint != config.fingerprint
                ):
                    raise RuntimeError("invalid worker control request")
                ok = True
                detail = "acknowledged"
                if request.operation == "safe_idle":
                    accepting_commands = False
                    last_accepted_command_monotonic = None
                    _drain_latest(command_queue)
                    safe_idle_confirmed = _safe_idle(robot)
                    ok = safe_idle_confirmed
                    detail = (
                        "gravity-compensation idle confirmed"
                        if ok
                        else "safe-idle operation is unavailable"
                    )
                elif request.operation == "resume_commands":
                    accepting_commands = True
                    last_accepted_command_monotonic = None
                    detail = "fresh commands enabled"
                elif request.operation == "shutdown":
                    accepting_commands = False
                    last_accepted_command_monotonic = None
                    _drain_latest(command_queue)
                    safe_idle_confirmed = _safe_idle(robot)
                    ok = safe_idle_confirmed
                    detail = (
                        "shutdown accepted after safe idle"
                        if ok
                        else "shutdown accepted without safe-idle confirmation"
                    )
                    stop_requested = True
                else:  # pragma: no cover - Literal plus strict dataclass boundary
                    raise RuntimeError("unsupported worker control request")
                _put_latest(
                    status_queue,
                    WorkerControlReply(
                        protocol_version=PROTOCOL_VERSION,
                        logical_id=config.logical_id,
                        config_fingerprint=config.fingerprint,
                        request_id=request.request_id,
                        operation=request.operation,
                        ok=ok,
                        detail=detail,
                        emitted_monotonic=time.monotonic(),
                    ),
                )
            if stop_requested:
                break

            command = _drain_latest(command_queue)
            if command is not None:
                now = time.monotonic()
                validity = _validate_command(config, command, now)
                if validity != "fresh" or not accepting_commands:
                    reason: Literal["stale", "not_enabled", "invalid"] = (
                        "not_enabled" if validity == "fresh" else validity
                    )
                    sequence = command.sequence if isinstance(command, WorkerCommand) else 0
                    _put_latest(
                        status_queue,
                        WorkerCommandDropped(
                            protocol_version=PROTOCOL_VERSION,
                            logical_id=config.logical_id,
                            config_fingerprint=config.fingerprint,
                            sequence=sequence,
                            reason=reason,
                            emitted_monotonic=now,
                        ),
                    )
                elif config.role != "follower":
                    _put_latest(
                        status_queue,
                        WorkerCommandDropped(
                            protocol_version=PROTOCOL_VERSION,
                            logical_id=config.logical_id,
                            config_fingerprint=config.fingerprint,
                            sequence=command.sequence,
                            reason="invalid",
                            emitted_monotonic=now,
                        ),
                    )
                else:
                    # NumPy is imported only in the active child; passing a
                    # mutable ndarray matches i2rt's clipping implementation.
                    import numpy as np

                    robot.command_joint_pos(np.asarray(command.positions, dtype=float))
                    applied_sequence = command.sequence
                    last_accepted_command_monotonic = now

            now = time.monotonic()
            if (
                accepting_commands
                and last_accepted_command_monotonic is not None
                and now - last_accepted_command_monotonic
                > config.command_max_age_seconds
            ):
                accepting_commands = False
                last_accepted_command_monotonic = None
                _drain_latest(command_queue)
                if not _safe_idle(robot):
                    raise RuntimeError(
                        "the follower command stream timed out and safe idle was unavailable"
                    )
                safe_idle_confirmed = True
                raise RuntimeError(
                    "the follower command stream timed out; gravity-compensation idle was entered"
                )
            if now >= next_telemetry:
                _put_latest(
                    telemetry_queue, _sample_robot(config, robot, applied_sequence)
                )
                next_telemetry = now + config.telemetry_interval_seconds
            if now >= next_heartbeat:
                liveness = _inner_liveness(robot)
                if not all(liveness.values()):
                    raise RuntimeError("an i2rt inner control loop stopped")
                _put_latest(
                    status_queue,
                    WorkerHeartbeat(
                        protocol_version=PROTOCOL_VERSION,
                        logical_id=config.logical_id,
                        config_fingerprint=config.fingerprint,
                        inner_liveness=liveness,
                        emitted_monotonic=now,
                    ),
                )
                next_heartbeat = now + config.heartbeat_interval_seconds
            time.sleep(0.002)
    except BaseException as error:
        fault_detail = _clean_text(f"{type(error).__name__}: {error}")
        _put_latest(
            status_queue,
            WorkerFault(
                protocol_version=PROTOCOL_VERSION,
                logical_id=config.logical_id,
                config_fingerprint=config.fingerprint,
                detail=fault_detail,
                emitted_monotonic=time.monotonic(),
            ),
        )
    finally:
        if robot is not None:
            try:
                safe_idle_confirmed = _safe_idle(robot) or safe_idle_confirmed
            except BaseException:
                safe_idle_confirmed = False
            try:
                robot.close()
                close_confirmed = True
            except BaseException:
                close_confirmed = False
        detail = (
            "worker stopped after a runtime fault"
            if fault_detail is not None
            else "worker closed its i2rt device"
        )
        _put_latest(
            status_queue,
            WorkerStopped(
                protocol_version=PROTOCOL_VERSION,
                logical_id=config.logical_id,
                config_fingerprint=config.fingerprint,
                safe_idle_confirmed=safe_idle_confirmed,
                device_close_confirmed=close_confirmed,
                detail=detail,
                emitted_monotonic=time.monotonic(),
            ),
        )


class I2RTArmWorker:
    """Parent-side supervisor for exactly one connected SocketCAN YAM arm."""

    def __init__(
        self,
        config: I2RTArmWorkerConfig,
        *,
        startup_timeout_seconds: float = 20.0,
        heartbeat_timeout_seconds: float = 0.750,
        telemetry_timeout_seconds: float = 0.750,
        operation_timeout_seconds: float = 1.0,
        shutdown_timeout_seconds: float = 2.0,
        terminate_timeout_seconds: float = 0.500,
        context: BaseContext | None = None,
        child_target: ChildTarget = i2rt_arm_worker_main,
        robot_loader: RobotLoader | None = None,
        process_spawner: _I2RTProcessSpawner = _I2RT_PROCESS_SPAWNER,
    ) -> None:
        for name, value, lower, upper in (
            ("startup_timeout_seconds", startup_timeout_seconds, 0.05, 120.0),
            ("heartbeat_timeout_seconds", heartbeat_timeout_seconds, 0.05, 10.0),
            ("telemetry_timeout_seconds", telemetry_timeout_seconds, 0.05, 10.0),
            ("operation_timeout_seconds", operation_timeout_seconds, 0.05, 10.0),
            ("shutdown_timeout_seconds", shutdown_timeout_seconds, 0.05, 30.0),
            ("terminate_timeout_seconds", terminate_timeout_seconds, 0.05, 10.0),
        ):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        self.config = config
        self._startup_timeout = startup_timeout_seconds
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._telemetry_timeout = telemetry_timeout_seconds
        self._operation_timeout = operation_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._terminate_timeout = terminate_timeout_seconds
        self._context = context or multiprocessing.get_context("spawn")
        self._child_target = child_target
        self._robot_loader = robot_loader
        self._process_spawner = process_spawner

        self._condition = threading.Condition(threading.RLock())
        self._process_lock = threading.Lock()
        self._phase: Literal[
            "new", "starting", "ready", "stopping", "stopped", "error"
        ] = "new"
        self._process: Any | None = None
        self._process_group_id: int | None = None
        self._control_queue: Any | None = None
        self._command_queue: Any | None = None
        self._status_queue: Any | None = None
        self._telemetry_queue: Any | None = None
        self._log_queue: Any | None = None
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._ready = False
        self._accepting_commands = False
        self._last_heartbeat: float | None = None
        self._last_telemetry: WorkerTelemetry | None = None
        self._inner_liveness: dict[str, bool] = {}
        self._fault: str | None = None
        self._fault_latched_at: float | None = None
        self._shutdown_uncertain = False
        self._stopped_event: WorkerStopped | None = None
        self._replies: dict[str, WorkerControlReply] = {}
        self._logs: deque[WorkerLog] = deque(maxlen=64)
        self._sequence = 0
        self._dropped_commands = 0
        self._reaped = False
        self._spawner_acquired = False

    def __enter__(self) -> I2RTArmWorker:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()

    @property
    def process_pid(self) -> int | None:
        process = self._process
        return int(process.pid) if process is not None and process.pid is not None else None

    def start(self) -> None:
        with self._condition:
            if self._phase != "new":
                raise I2RTWorkerStartupError(
                    "An arm worker instance may be started only once; faults never auto-respawn."
                )
            self._phase = "starting"
            self._control_queue = self._context.Queue(maxsize=8)
            self._command_queue = _LatestCommandMailbox(self._context, self.config)
            self._status_queue = self._context.Queue(maxsize=16)
            self._telemetry_queue = self._context.Queue(maxsize=1)
            self._log_queue = self._context.Queue(maxsize=32)
            expected_parent_pid = os.getpid()
            arguments: tuple[object, ...] = (
                self.config,
                self._control_queue,
                self._command_queue,
                self._status_queue,
                self._telemetry_queue,
                self._log_queue,
                expected_parent_pid,
            )
            if self._robot_loader is not None:
                arguments += (self._robot_loader,)
            self._process = self._context.Process(
                target=self._child_target,
                args=arguments,
                name=f"ctrl-pi-i2rt-{self.config.logical_id}",
                daemon=False,
            )
            try:
                self._process_spawner.acquire()
                self._spawner_acquired = True
                self._process_spawner.start_process(self._process)
            except BaseException as error:
                self._phase = "error"
                self._fault = "The i2rt arm worker process could not be started."
                self._shutdown_uncertain = True
                process = self._process
                if process is not None and process.pid is not None:
                    self._force_stop_process()
                    self._release_spawner_if_reaped()
                else:
                    self._release_spawner()
                self._close_queues()
                raise I2RTWorkerStartupError(self._fault) from error
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name=f"ctrl-pi-i2rt-monitor-{self.config.logical_id}",
                daemon=True,
            )
            self._monitor_thread.start()

            deadline = time.monotonic() + self._startup_timeout
            while (
                not self._ready
                or self._last_telemetry is None
            ) and self._fault is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            if self._ready and self._last_telemetry is not None and self._fault is None:
                self._phase = "ready"
                return

        reason = self._fault or "The i2rt arm worker did not become ready in time."
        self._failed_start_cleanup(reason)
        raise I2RTWorkerStartupError(reason)

    def _failed_start_cleanup(self, reason: str) -> None:
        with self._condition:
            self._fault = reason
            self._phase = "error"
            self._accepting_commands = False
            self._shutdown_uncertain = True
        self._send_control_without_wait("shutdown")
        process = self._process
        if process is not None:
            process.join(timeout=min(self._terminate_timeout, 0.2))
            if process.is_alive():
                self._force_stop_process()
            self._mark_reaped()
            self._release_spawner_if_reaped()
        self._stop_monitor()
        self._close_queues()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.is_set():
            self._drain_incoming()
            request_shutdown = False
            force_cleanup = False
            process_exited = False
            now = time.monotonic()
            with self._condition:
                process = self._process
                process_alive = process is not None and process.is_alive()
                if self._phase == "ready" or (self._ready and self._phase == "starting"):
                    if (
                        self._last_heartbeat is not None
                        and now - self._last_heartbeat > self._heartbeat_timeout
                    ):
                        request_shutdown = self._latch_fault_locked(
                            "The i2rt arm worker heartbeat is stale."
                        )
                    elif (
                        self._last_telemetry is not None
                        and now - self._last_telemetry.sample_monotonic
                        > self._telemetry_timeout
                    ):
                        request_shutdown = self._latch_fault_locked(
                            "The i2rt arm telemetry is stale."
                        )
                    elif self._inner_liveness and not all(self._inner_liveness.values()):
                        request_shutdown = self._latch_fault_locked(
                            "An i2rt inner control loop is not alive."
                        )
                if (
                    process is not None
                    and not process_alive
                    and not self._reaped
                    and self._phase not in {"stopping", "stopped"}
                ):
                    exitcode = process.exitcode
                    request_shutdown = self._latch_fault_locked(
                        "The i2rt arm worker exited unexpectedly "
                        f"(exitcode={exitcode})."
                    ) or request_shutdown
                    process_exited = True
                if (
                    self._fault_latched_at is not None
                    and process_alive
                    and now - self._fault_latched_at > self._shutdown_timeout
                ):
                    force_cleanup = True
                self._condition.notify_all()
            if request_shutdown:
                self._send_control_without_wait("shutdown")
            if force_cleanup:
                self._force_stop_process()
            if process_exited:
                logger.error(
                    "i2rt arm worker %s exited unexpectedly (pid=%s, exitcode=%s)",
                    self.config.logical_id,
                    process.pid if process is not None else None,
                    process.exitcode if process is not None else None,
                )
                self._mark_reaped()
                self._release_spawner_if_reaped()
            self._monitor_stop.wait(0.010)
        self._drain_incoming()

    def _drain_incoming(self) -> None:
        status_queue = self._status_queue
        telemetry_queue = self._telemetry_queue
        log_queue = self._log_queue
        if status_queue is not None:
            while True:
                try:
                    message = status_queue.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    break
                self._consume_status(message)
        if telemetry_queue is not None:
            latest = _drain_latest(telemetry_queue)
            if latest is not None:
                self._consume_telemetry(latest)
        if log_queue is not None:
            while True:
                try:
                    message = log_queue.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    break
                self._consume_log(message)

    def _valid_envelope(self, message: object) -> bool:
        return (
            getattr(message, "protocol_version", None) == PROTOCOL_VERSION
            and getattr(message, "logical_id", None) == self.config.logical_id
            and getattr(message, "config_fingerprint", None) == self.config.fingerprint
        )

    def _consume_status(self, message: object) -> None:
        request_shutdown = False
        with self._condition:
            if not isinstance(
                message,
                (
                    WorkerReady,
                    WorkerHeartbeat,
                    WorkerControlReply,
                    WorkerCommandDropped,
                    WorkerFault,
                    WorkerStopped,
                ),
            ) or not self._valid_envelope(message):
                request_shutdown = self._latch_fault_locked(
                    "The i2rt arm worker sent a mismatched protocol message."
                )
            elif isinstance(message, WorkerReady):
                process = self._process
                expected_pid = process.pid if process is not None else None
                valid = (
                    message.stable_identity == self.config.stable_identity
                    and message.runtime_interface == self.config.runtime_interface
                    and message.checkout_commit == self.config.expected_commit
                    and message.pid == expected_pid
                    and message.parent_death_signal
                    == sys.platform.startswith("linux")
                    and bool(message.inner_liveness)
                    and all(message.inner_liveness.values())
                )
                if not valid:
                    request_shutdown = self._latch_fault_locked(
                        "The i2rt arm worker ready handshake did not match its configuration."
                    )
                else:
                    self._ready = True
                    self._process_group_id = message.process_group_id
                    self._last_heartbeat = message.emitted_monotonic
                    self._inner_liveness = dict(message.inner_liveness)
            elif isinstance(message, WorkerHeartbeat):
                self._last_heartbeat = message.emitted_monotonic
                self._inner_liveness = dict(message.inner_liveness)
                if not message.inner_liveness or not all(message.inner_liveness.values()):
                    request_shutdown = self._latch_fault_locked(
                        "An i2rt inner control loop stopped."
                    )
            elif isinstance(message, WorkerControlReply):
                self._replies[message.request_id] = message
            elif isinstance(message, WorkerCommandDropped):
                self._dropped_commands += 1
            elif isinstance(message, WorkerFault):
                request_shutdown = self._latch_fault_locked(
                    f"The i2rt arm worker faulted: {_clean_text(message.detail)}"
                )
            elif isinstance(message, WorkerStopped):
                self._stopped_event = message
                if not message.safe_idle_confirmed or not message.device_close_confirmed:
                    self._shutdown_uncertain = True
            self._condition.notify_all()
        if request_shutdown:
            self._send_control_without_wait("shutdown")

    def _consume_telemetry(self, message: object) -> None:
        request_shutdown = False
        with self._condition:
            if not isinstance(message, WorkerTelemetry) or not self._valid_envelope(message):
                request_shutdown = self._latch_fault_locked(
                    "The i2rt arm worker sent mismatched telemetry."
                )
            elif (
                len(message.joint_positions) != 6
                or len(message.joint_velocities) != 6
                or len(message.joint_efforts) != 6
                or not all(
                    math.isfinite(value)
                    for values in (
                        message.joint_positions,
                        message.joint_velocities,
                        message.joint_efforts,
                    )
                    for value in values
                )
                or not 0.0 <= message.gripper_position <= 1.0
                or not math.isfinite(message.control_loop_frequency_hz)
                or message.control_loop_frequency_hz < 0.0
            ):
                request_shutdown = self._latch_fault_locked(
                    "The i2rt arm worker sent invalid telemetry."
                )
            else:
                self._last_telemetry = message
                self._inner_liveness = dict(message.inner_liveness)
            self._condition.notify_all()
        if request_shutdown:
            self._send_control_without_wait("shutdown")

    def _consume_log(self, message: object) -> None:
        if not isinstance(message, WorkerLog) or not self._valid_envelope(message):
            return
        bounded = WorkerLog(
            protocol_version=message.protocol_version,
            logical_id=message.logical_id,
            config_fingerprint=message.config_fingerprint,
            level=message.level,
            message=_clean_text(message.message, maximum=_MAX_LOG_CHARS),
            emitted_monotonic=message.emitted_monotonic,
        )
        with self._condition:
            self._logs.append(bounded)

    def _latch_fault_locked(self, detail: str) -> bool:
        if self._fault is not None:
            return False
        self._fault = _clean_text(detail)
        self._fault_latched_at = time.monotonic()
        self._accepting_commands = False
        self._shutdown_uncertain = True
        if self._phase not in {"stopping", "stopped"}:
            self._phase = "error"
        return True

    def _request_control(
        self, operation: ControlOperation, *, timeout: float | None = None
    ) -> WorkerControlReply:
        request_id = uuid4().hex
        request = WorkerControlRequest(
            protocol_version=PROTOCOL_VERSION,
            logical_id=self.config.logical_id,
            config_fingerprint=self.config.fingerprint,
            request_id=request_id,
            operation=operation,
            issued_monotonic=time.monotonic(),
        )
        control_queue = self._control_queue
        if control_queue is None:
            raise I2RTWorkerUnavailableError("The i2rt arm worker is not running.")
        try:
            control_queue.put_nowait(request)
        except (queue.Full, OSError, ValueError) as error:
            raise I2RTWorkerUnavailableError(
                "The i2rt arm worker control channel is unavailable."
            ) from error
        deadline = time.monotonic() + (timeout or self._operation_timeout)
        with self._condition:
            while request_id not in self._replies:
                process = self._process
                if process is None or not process.is_alive():
                    raise I2RTWorkerUnavailableError(
                        "The i2rt arm worker exited before acknowledging the operation."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise I2RTWorkerUnavailableError(
                        "The i2rt arm worker did not acknowledge the operation in time."
                    )
                self._condition.wait(timeout=remaining)
            return self._replies.pop(request_id)

    def _send_control_without_wait(self, operation: ControlOperation) -> None:
        control_queue = self._control_queue
        if control_queue is None:
            return
        request = WorkerControlRequest(
            protocol_version=PROTOCOL_VERSION,
            logical_id=self.config.logical_id,
            config_fingerprint=self.config.fingerprint,
            request_id=uuid4().hex,
            operation=operation,
            issued_monotonic=time.monotonic(),
        )
        try:
            control_queue.put_nowait(request)
        except (queue.Full, OSError, ValueError):
            return

    def enable_commands(self) -> None:
        self.ensure_healthy()
        reply = self._request_control("resume_commands")
        if not reply.ok:
            raise I2RTWorkerUnavailableError(reply.detail)
        with self._condition:
            self._accepting_commands = True

    def safe_idle(self) -> None:
        with self._condition:
            self._accepting_commands = False
        if self._command_queue is not None:
            _drain_latest(self._command_queue)
        reply = self._request_control("safe_idle")
        if not reply.ok:
            with self._condition:
                self._shutdown_uncertain = True
            raise I2RTWorkerUnavailableError(reply.detail)

    def send_command(
        self,
        positions: tuple[float, float, float, float, float, float, float]
        | list[float],
        *,
        issued_monotonic: float | None = None,
    ) -> int:
        self.ensure_healthy()
        values = tuple(float(value) for value in positions)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError("i2rt arm commands must contain seven finite positions")
        if not 0.0 <= values[6] <= 1.0:
            raise ValueError("the normalized gripper command must be inside [0, 1]")
        if self.config.role != "follower":
            raise I2RTWorkerUnavailableError("Position commands are follower-only.")
        with self._condition:
            if not self._accepting_commands:
                raise I2RTWorkerUnavailableError(
                    "Commands are disabled until an explicit control boundary enables them."
                )
            self._sequence += 1
            sequence = self._sequence
        issued = time.monotonic() if issued_monotonic is None else issued_monotonic
        command = WorkerCommand(
            protocol_version=PROTOCOL_VERSION,
            logical_id=self.config.logical_id,
            config_fingerprint=self.config.fingerprint,
            sequence=sequence,
            issued_monotonic=issued,
            deadline_monotonic=issued + self.config.command_max_age_seconds,
            positions=values,  # type: ignore[arg-type]
        )
        if self._command_queue is None:
            raise I2RTWorkerUnavailableError("The i2rt arm worker is not running.")
        _put_latest(self._command_queue, command)
        return sequence

    # Cell-driver vocabulary: an action is the normalized six-joint + gripper
    # command accepted by this worker.  Keep ``send_command`` as the explicit
    # protocol name for lower-level tests and diagnostics.
    send_action = send_command

    def telemetry(self) -> WorkerTelemetry:
        self.ensure_healthy()
        with self._condition:
            telemetry = self._last_telemetry
        if telemetry is None:  # guarded by ensure_healthy; kept for type narrowing
            raise I2RTWorkerUnavailableError("The i2rt arm telemetry is unavailable.")
        return telemetry

    def ensure_healthy(self) -> None:
        self._drain_incoming()
        now = time.monotonic()
        with self._condition:
            if self._fault is not None:
                raise I2RTWorkerUnavailableError(self._fault)
            process = self._process
            if self._phase != "ready" or process is None or not process.is_alive():
                raise I2RTWorkerUnavailableError("The i2rt arm worker is not ready.")
            if self._last_heartbeat is None or now - self._last_heartbeat > self._heartbeat_timeout:
                self._latch_fault_locked("The i2rt arm worker heartbeat is stale.")
                raise I2RTWorkerUnavailableError(self._fault or "Worker heartbeat is stale.")
            telemetry = self._last_telemetry
            if telemetry is None or now - telemetry.sample_monotonic > self._telemetry_timeout:
                self._latch_fault_locked("The i2rt arm telemetry is stale.")
                raise I2RTWorkerUnavailableError(self._fault or "Worker telemetry is stale.")
            if not self._inner_liveness or not all(self._inner_liveness.values()):
                self._latch_fault_locked("An i2rt inner control loop is not alive.")
                raise I2RTWorkerUnavailableError(self._fault or "Worker loop stopped.")

    def snapshot(self) -> I2RTWorkerSnapshot:
        self._drain_incoming()
        now = time.monotonic()
        with self._condition:
            process = self._process
            reaped = self._reaped or (
                process is not None and not process.is_alive() and process.exitcode is not None
            )
            return I2RTWorkerSnapshot(
                phase=self._phase,
                pid=self.process_pid,
                process_group_id=self._process_group_id,
                accepting_commands=self._accepting_commands,
                heartbeat_age_seconds=(
                    None
                    if self._last_heartbeat is None
                    else max(0.0, now - self._last_heartbeat)
                ),
                telemetry_age_seconds=(
                    None
                    if self._last_telemetry is None
                    else max(0.0, now - self._last_telemetry.sample_monotonic)
                ),
                inner_liveness=dict(self._inner_liveness),
                dropped_commands=self._dropped_commands,
                fault=self._fault,
                shutdown_uncertain=self._shutdown_uncertain,
                reaped=reaped,
                logs=tuple(self._logs),
                exitcode=(None if process is None else process.exitcode),
            )

    def shutdown(self) -> I2RTWorkerShutdownResult:
        process = self._process
        if process is None:
            clean = self._phase in {"new", "stopped"}
            return I2RTWorkerShutdownResult(
                clean=clean,
                safe_idle_confirmed=clean,
                device_close_confirmed=clean,
                forced=False,
                reaped=clean,
                uncertain=not clean,
                detail="No arm worker process is running.",
            )

        with self._condition:
            self._phase = "stopping"
            self._accepting_commands = False
        if self._command_queue is not None:
            _drain_latest(self._command_queue)

        safe_idle_confirmed = False
        try:
            reply = self._request_control("safe_idle", timeout=self._operation_timeout)
            safe_idle_confirmed = reply.ok
        except I2RTWorkerError:
            pass
        try:
            self._request_control("shutdown", timeout=self._operation_timeout)
        except I2RTWorkerError:
            pass

        process.join(timeout=self._shutdown_timeout)
        forced = False
        if process.is_alive():
            forced = True
            self._force_stop_process()
        self._drain_incoming()
        self._mark_reaped()
        self._stop_monitor()
        self._drain_incoming()

        stopped = self._stopped_event
        if stopped is not None:
            safe_idle_confirmed = safe_idle_confirmed or stopped.safe_idle_confirmed
        close_confirmed = bool(stopped and stopped.device_close_confirmed)
        reaped = not process.is_alive() and process.exitcode is not None
        clean = bool(
            not forced
            and safe_idle_confirmed
            and close_confirmed
            and reaped
            and process.exitcode == 0
        )
        uncertain = not clean
        with self._condition:
            self._reaped = reaped
            # A runtime fault blocks commands and auto-respawn, but a later
            # explicit shutdown can resolve that earlier uncertainty with
            # positive safe-idle, close, exit, and reap evidence.
            self._shutdown_uncertain = uncertain
            self._phase = "stopped" if clean else "error"
            if uncertain and self._fault is None:
                self._fault = "The i2rt arm shutdown could not be fully confirmed."
        self._release_spawner_if_reaped()
        self._close_queues()
        return I2RTWorkerShutdownResult(
            clean=clean,
            safe_idle_confirmed=safe_idle_confirmed,
            device_close_confirmed=close_confirmed,
            forced=forced,
            reaped=reaped,
            uncertain=uncertain,
            detail=(
                "The i2rt arm worker entered safe idle, closed, and was reaped."
                if clean
                else "The i2rt arm worker was reaped, but safe device shutdown is uncertain."
            ),
        )

    close = shutdown

    def _force_stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        with self._process_lock:
            if not process.is_alive():
                self._mark_reaped()
                return
            with self._condition:
                self._shutdown_uncertain = True
                process_group_id = self._process_group_id
                process_pid = process.pid
            sent_group_signal = False
            if (
                process_group_id is not None
                and process_pid is not None
                and process_group_id == process_pid
                and hasattr(os, "killpg")
            ):
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                    sent_group_signal = True
                except (OSError, ProcessLookupError):
                    pass
            if not sent_group_signal:
                try:
                    process.terminate()
                except (OSError, ProcessLookupError, ValueError):
                    pass
            process.join(timeout=self._terminate_timeout)
            if process.is_alive():
                if sent_group_signal and process_group_id is not None:
                    try:
                        os.killpg(process_group_id, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                else:
                    try:
                        process.kill()
                    except (OSError, ProcessLookupError, ValueError):
                        pass
                process.join(timeout=self._terminate_timeout)
            self._mark_reaped()

        self._release_spawner_if_reaped()

    def _mark_reaped(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            process.join(timeout=0)
        except (AssertionError, ValueError):
            return
        with self._condition:
            self._reaped = not process.is_alive() and process.exitcode is not None
            self._condition.notify_all()

    def _release_spawner_if_reaped(self) -> None:
        process = self._process
        if process is None:
            self._release_spawner()
            return
        try:
            reaped = not process.is_alive() and process.exitcode is not None
        except (AssertionError, ValueError):
            return
        if reaped:
            self._release_spawner()

    def _release_spawner(self) -> None:
        with self._condition:
            if not self._spawner_acquired:
                return
            self._spawner_acquired = False
        self._process_spawner.release()

    def _stop_monitor(self) -> None:
        self._monitor_stop.set()
        thread = self._monitor_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _close_queues(self) -> None:
        for ipc_queue in (
            self._control_queue,
            self._command_queue,
            self._status_queue,
            self._telemetry_queue,
            self._log_queue,
        ):
            if ipc_queue is None:
                continue
            try:
                ipc_queue.close()
                ipc_queue.cancel_join_thread()
            except (OSError, ValueError):
                pass
        self._control_queue = None
        self._command_queue = None
        self._status_queue = None
        self._telemetry_queue = None
        self._log_queue = None


# Concise integration names retained alongside the i2rt-explicit names used by
# diagnostics and tests.
WorkerLaunchConfig = I2RTArmWorkerConfig
SupervisedArmWorker = I2RTArmWorker


__all__ = [
    "I2RTArmWorker",
    "I2RTArmWorkerConfig",
    "I2RTCheckoutError",
    "I2RTCheckoutIdentity",
    "I2RTWorkerError",
    "I2RTWorkerProtocolError",
    "I2RTWorkerShutdownResult",
    "I2RTWorkerSnapshot",
    "I2RTWorkerStartupError",
    "I2RTWorkerUnavailableError",
    "PROTOCOL_VERSION",
    "SupervisedArmWorker",
    "WorkerCommand",
    "WorkerCommandDropped",
    "WorkerControlReply",
    "WorkerControlRequest",
    "WorkerFault",
    "WorkerHeartbeat",
    "WorkerLog",
    "WorkerLaunchConfig",
    "WorkerReady",
    "WorkerStopped",
    "WorkerTelemetry",
    "i2rt_arm_worker_main",
    "load_pinned_i2rt_robot",
    "verify_i2rt_checkout",
]
