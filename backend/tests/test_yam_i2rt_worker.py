from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import pytest

from ctrl_pi.drivers.yam_i2rt_worker import (
    I2RTArmWorker,
    I2RTArmWorkerConfig,
    I2RTCheckoutError,
    I2RTWorkerStartupError,
    I2RTWorkerUnavailableError,
    PROTOCOL_VERSION,
    WorkerReady,
    WorkerTelemetry,
    _configure_linux_parent_death_signal,
    i2rt_arm_worker_main,
    verify_i2rt_checkout,
)


def _git(*arguments: str, cwd: Path) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "ctrl-pi test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "ctrl-pi test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
    )
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    ).stdout.strip()


@pytest.fixture
def fake_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "operator-i2rt"
    package = root / "i2rt" / "robots"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'fake'\n",
        encoding="utf-8",
    )
    (package / "get_robot.py").write_text(
        "# namespace-package fixture; never imported\n", encoding="utf-8"
    )
    (root / ".gitignore").write_text(
        "i2rt/robots/ignored-shadow.py\n", encoding="utf-8"
    )
    _git("init", "--quiet", cwd=root)
    _git(
        "add",
        ".gitignore",
        "pyproject.toml",
        "i2rt/robots/get_robot.py",
        cwd=root,
    )
    _git("commit", "--quiet", "-m", "fixture", cwd=root)
    return root, _git("rev-parse", "HEAD", cwd=root)


def _config(root: Path, commit: str, **changes: Any) -> I2RTArmWorkerConfig:
    values: dict[str, Any] = {
        "logical_id": "follower-right",
        "role": "follower",
        "stable_identity": "adapter-serial-right",
        "runtime_interface": "can7",
        "end_effector_kind": "linear_4310",
        "checkout_root": str(root),
        "expected_commit": commit,
        "require_read_only_checkout": False,
        "heartbeat_interval_seconds": 0.03,
        "telemetry_interval_seconds": 0.01,
        "command_max_age_seconds": 0.1,
    }
    values.update(changes)
    return I2RTArmWorkerConfig(**values)


def _fork_context() -> multiprocessing.context.BaseContext:
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


class _AliveThread:
    def is_alive(self) -> bool:
        return True


class _BrokenControlQueue:
    def put_nowait(self, _item: object) -> None:
        raise ValueError("control queue feeder is closed")

    def close(self) -> None:
        return None

    def cancel_join_thread(self) -> None:
        return None


class _FakeMotorChain:
    def __init__(self) -> None:
        self.running = True
        self.comm_freq = 268.5


class _FakeRobot:
    def __init__(self) -> None:
        self.motor_chain = _FakeMotorChain()
        self._server_thread = _AliveThread()
        self._positions = [0.0] * 6
        self._gripper = 1.0

    def get_observations(self) -> dict[str, list[float]]:
        return {
            "joint_pos": list(self._positions),
            "joint_vel": [0.0] * 6,
            "joint_eff": [0.1] * 6,
            "gripper_pos": [self._gripper],
            "gripper_vel": [0.0],
            "gripper_eff": [0.2],
        }

    def command_joint_pos(self, joint_pos: Any) -> None:
        values = [float(value) for value in joint_pos]
        self._positions = values[:6]
        self._gripper = values[6]

    def enter_gravity_comp_idle(self) -> None:
        return None

    def close(self) -> None:
        self.motor_chain.running = False


def _load_fake_robot(_config: I2RTArmWorkerConfig) -> _FakeRobot:
    # Both ordinary logging and vendor print output stay bounded inside IPC.
    logging.getLogger("fake-i2rt").info("L" * 1_000)
    print("P" * 1_000)
    return _FakeRobot()


def _telemetry(config: I2RTArmWorkerConfig) -> WorkerTelemetry:
    return WorkerTelemetry(
        protocol_version=PROTOCOL_VERSION,
        logical_id=config.logical_id,
        config_fingerprint=config.fingerprint,
        sample_monotonic=time.monotonic(),
        joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        joint_velocities=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        joint_efforts=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_position=1.0,
        gripper_velocity=0.0,
        gripper_effort=0.0,
        handle_trigger_position=None,
        handle_buttons=(),
        control_loop_frequency_hz=250.0,
        inner_liveness={"motor_chain_running": True, "robot_server_alive": True},
        command_sequence_applied=None,
    )


def _ready(config: I2RTArmWorkerConfig) -> WorkerReady:
    return WorkerReady(
        protocol_version=PROTOCOL_VERSION,
        logical_id=config.logical_id,
        stable_identity=config.stable_identity,
        runtime_interface=config.runtime_interface,
        config_fingerprint=config.fingerprint,
        checkout_commit=config.expected_commit,
        pid=os.getpid(),
        process_group_id=None,
        parent_death_signal=sys.platform.startswith("linux"),
        inner_liveness={"motor_chain_running": True, "robot_server_alive": True},
        emitted_monotonic=time.monotonic(),
    )


def _hang_before_ready(
    _config: I2RTArmWorkerConfig,
    _control: Any,
    _commands: Any,
    _status: Any,
    _telemetry_queue: Any,
    _logs: Any,
    expected_parent_pid: int,
) -> None:
    assert expected_parent_pid == os.getppid()
    while True:
        time.sleep(1.0)


def _ready_then_crash(
    config: I2RTArmWorkerConfig,
    _control: Any,
    _commands: Any,
    status: Any,
    telemetry_queue: Any,
    _logs: Any,
    expected_parent_pid: int,
) -> None:
    assert expected_parent_pid == os.getppid()
    telemetry_queue.put(_telemetry(config))
    status.put(_ready(config))
    time.sleep(0.1)
    os._exit(17)


def _ready_then_stall(
    config: I2RTArmWorkerConfig,
    _control: Any,
    _commands: Any,
    status: Any,
    telemetry_queue: Any,
    _logs: Any,
    expected_parent_pid: int,
) -> None:
    assert expected_parent_pid == os.getppid()
    telemetry_queue.put(_telemetry(config))
    status.put(_ready(config))
    while True:
        time.sleep(1.0)


def _mismatched_ready(
    config: I2RTArmWorkerConfig,
    _control: Any,
    _commands: Any,
    status: Any,
    telemetry_queue: Any,
    _logs: Any,
    expected_parent_pid: int,
) -> None:
    assert expected_parent_pid == os.getppid()
    telemetry_queue.put(_telemetry(config))
    ready = _ready(config)
    status.put(
        WorkerReady(
            protocol_version=ready.protocol_version,
            logical_id=ready.logical_id,
            stable_identity="wrong-adapter",
            runtime_interface=ready.runtime_interface,
            config_fingerprint=ready.config_fingerprint,
            checkout_commit=ready.checkout_commit,
            pid=ready.pid,
            process_group_id=ready.process_group_id,
            parent_death_signal=ready.parent_death_signal,
            inner_liveness=ready.inner_liveness,
            emitted_monotonic=ready.emitted_monotonic,
        )
    )
    while True:
        time.sleep(1.0)


def _wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.01)


def test_config_requires_exact_identity_and_absolute_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        _config(tmp_path, "0" * 40, checkout_root="relative/i2rt")
    with pytest.raises(ValueError, match="40-hex"):
        _config(tmp_path, "main")
    with pytest.raises(ValueError, match="teaching_handle"):
        _config(tmp_path, "0" * 40, role="leader")


def test_checkout_verification_is_local_exact_and_read_only(
    fake_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, commit = fake_checkout
    identity = verify_i2rt_checkout(str(root), commit, require_read_only=False)
    assert identity.commit == commit
    assert identity.root == str(root.resolve())
    with pytest.raises(I2RTCheckoutError, match="expected commit"):
        verify_i2rt_checkout(str(root), "0" * 40, require_read_only=False)
    with pytest.raises(I2RTCheckoutError, match="writable"):
        verify_i2rt_checkout(str(root), commit, require_read_only=True)

    # Read-only bind-mount flags and effective access differ by CI uid; inject
    # the already-unit-tested passive permission result for the accepted path.
    monkeypatch.setattr(
        "ctrl_pi.drivers.yam_i2rt_worker._checkout_is_read_only", lambda _root: True
    )
    identity = verify_i2rt_checkout(str(root), commit, require_read_only=True)
    assert identity.read_only is True


def test_checkout_verification_accepts_field_namespace_package_layout(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout

    assert not (root / "i2rt" / "__init__.py").exists()
    identity = verify_i2rt_checkout(str(root), commit, require_read_only=False)

    assert identity.commit == commit
    assert identity.root == str(root.resolve())


def test_checkout_chmod_is_not_a_read_only_mount_proof(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    motion_source = root / "i2rt" / "robots" / "get_robot.py"
    assert motion_source.stat().st_mode & 0o200
    root.chmod(0o555)
    try:
        with pytest.raises(I2RTCheckoutError, match="writable"):
            verify_i2rt_checkout(str(root), commit, require_read_only=True)
    finally:
        root.chmod(0o755)


def test_checkout_verification_scopes_safe_directory_to_exact_root(
    fake_checkout: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, commit = fake_checkout
    real_run = subprocess.run
    git_commands: list[list[str]] = []

    def recording_run(arguments: list[str], *args: Any, **kwargs: Any):
        if arguments and arguments[0] == "git":
            git_commands.append(arguments)
        return real_run(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)

    verify_i2rt_checkout(str(root), commit, require_read_only=False)

    expected = f"safe.directory={root.resolve()}"
    assert git_commands
    assert all(command[1:3] == ["-c", expected] for command in git_commands)
    assert all("safe.directory=*" not in command for command in git_commands)


@pytest.mark.parametrize(
    "dirty_kind", ["tracked", "staged", "untracked", "ignored"]
)
def test_checkout_verification_rejects_every_executable_worktree_difference(
    fake_checkout: tuple[Path, str], dirty_kind: str
) -> None:
    root, commit = fake_checkout
    get_robot = root / "i2rt" / "robots" / "get_robot.py"
    if dirty_kind in {"untracked", "ignored"}:
        filename = "shadow.py" if dirty_kind == "untracked" else "ignored-shadow.py"
        (root / "i2rt" / "robots" / filename).write_text(
            "# could shadow committed motion code\n", encoding="utf-8"
        )
    else:
        get_robot.write_text("# modified motion code\n", encoding="utf-8")
        if dirty_kind == "staged":
            _git("add", "i2rt/robots/get_robot.py", cwd=root)

    with pytest.raises(I2RTCheckoutError, match="tracked|staged|untracked"):
        verify_i2rt_checkout(str(root), commit, require_read_only=False)


def test_real_child_loop_uses_fake_factory_latest_fresh_commands_and_clean_reap(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    assert not any(name == "i2rt" or name.startswith("i2rt.") for name in sys.modules)
    worker = I2RTArmWorker(
        _config(root, commit),
        context=multiprocessing.get_context("spawn"),
        robot_loader=_load_fake_robot,
        startup_timeout_seconds=3.0,
        heartbeat_timeout_seconds=0.5,
        telemetry_timeout_seconds=0.5,
        operation_timeout_seconds=0.5,
        shutdown_timeout_seconds=1.0,
        terminate_timeout_seconds=0.2,
    )
    worker.start()
    assert worker.snapshot().phase == "ready"
    assert worker.telemetry().control_loop_frequency_hz == pytest.approx(268.5)
    with pytest.raises(I2RTWorkerUnavailableError, match="explicit control boundary"):
        worker.send_command([0.0] * 6 + [1.0])

    worker.enable_commands()
    stale_sequence = worker.send_command(
        [0.1] * 6 + [0.8], issued_monotonic=time.monotonic() - 1.0
    )
    _wait_until(lambda: worker.snapshot().dropped_commands >= 1)
    # A burst may be sampled midway, but the bounded mailbox always retains
    # the final action rather than replaying an older backlog.
    for index in range(20):
        value = index / 100.0
        sequence = worker.send_command(
            [value, 0.1, 0.0, -0.1, -0.2, 0.3, 0.4]
        )
    assert sequence > stale_sequence
    _wait_until(lambda: worker.telemetry().command_sequence_applied == sequence)
    assert worker.telemetry().joint_positions == pytest.approx(
        (0.19, 0.1, 0.0, -0.1, -0.2, 0.3)
    )
    assert worker.telemetry().gripper_position == pytest.approx(0.4)

    worker.safe_idle()
    assert worker.snapshot().accepting_commands is False
    with pytest.raises(I2RTWorkerUnavailableError, match="explicit control boundary"):
        worker.send_command([0.0] * 6 + [1.0])
    assert all(len(item.message) <= 512 for item in worker.snapshot().logs)

    result = worker.shutdown()
    assert result.clean is True
    assert result.safe_idle_confirmed is True
    assert result.device_close_confirmed is True
    assert result.reaped is True
    assert result.forced is False
    assert worker.snapshot().phase == "stopped"
    assert not any(name == "i2rt" or name.startswith("i2rt.") for name in sys.modules)


def test_child_command_stream_watchdog_enters_safe_idle_and_faults_closed(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    worker = I2RTArmWorker(
        _config(root, commit, command_max_age_seconds=0.05),
        context=_fork_context(),
        robot_loader=_load_fake_robot,
        startup_timeout_seconds=2.0,
        heartbeat_timeout_seconds=0.5,
        telemetry_timeout_seconds=0.5,
        operation_timeout_seconds=0.5,
        shutdown_timeout_seconds=0.5,
        terminate_timeout_seconds=0.2,
    )
    worker.start()
    worker.enable_commands()
    sequence = worker.send_command([0.1] * 6 + [0.6])
    _wait_until(lambda: worker.telemetry().command_sequence_applied == sequence)

    _wait_until(
        lambda: "command stream timed out" in (worker.snapshot().fault or ""),
        timeout=2.0,
    )
    snapshot = worker.snapshot()
    assert snapshot.accepting_commands is False
    result = worker.shutdown()
    assert result.clean is True
    assert result.safe_idle_confirmed is True
    assert result.device_close_confirmed is True
    assert result.reaped is True
    assert result.uncertain is False
    assert worker.snapshot().phase == "stopped"
    assert worker.snapshot().shutdown_uncertain is False


def test_linux_parent_death_signal_is_configurable_in_an_isolated_child() -> None:
    if not sys.platform.startswith("linux"):
        assert _configure_linux_parent_death_signal(os.getppid()) is False
        return
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "from ctrl_pi.drivers.yam_i2rt_worker import "
                "_configure_linux_parent_death_signal as configure; "
                "print(configure(os.getppid()))"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5.0,
    )
    assert completed.stdout.strip() == "True"


def test_linux_parent_death_signal_is_installed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLibc:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int, int]] = []

        def prctl(
            self, option: int, value: int, arg3: int, arg4: int, arg5: int
        ) -> int:
            self.calls.append((option, value, arg3, arg4, arg5))
            return 0

    libc = FakeLibc()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getppid", lambda: 42_424)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: libc)

    assert _configure_linux_parent_death_signal(42_424) is True
    assert libc.calls == [(1, signal.SIGTERM, 0, 0, 0)]


def test_already_orphaned_child_main_exits_before_prctl_checkout_or_vendor(
    fake_checkout: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, commit = fake_checkout
    calls: list[str] = []
    monkeypatch.setattr(os, "getppid", lambda: 9_999)
    monkeypatch.setattr(
        "ctrl_pi.drivers.yam_i2rt_worker._configure_linux_parent_death_signal",
        lambda _pid: calls.append("prctl") or True,
    )
    monkeypatch.setattr(
        "ctrl_pi.drivers.yam_i2rt_worker.verify_i2rt_checkout",
        lambda *_args, **_kwargs: calls.append("checkout"),
    )

    def load_robot(_config: I2RTArmWorkerConfig) -> _FakeRobot:
        calls.append("vendor")
        return _FakeRobot()

    i2rt_arm_worker_main(
        _config(root, commit),
        None,
        None,
        None,
        None,
        None,
        1_234,
        load_robot,
    )

    assert calls == []


def test_start_timeout_forces_partial_startup_cleanup_and_reaps(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    worker = I2RTArmWorker(
        _config(root, commit),
        context=_fork_context(),
        child_target=_hang_before_ready,
        startup_timeout_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        terminate_timeout_seconds=0.1,
    )
    with pytest.raises(I2RTWorkerStartupError, match="did not become ready"):
        worker.start()
    assert worker.process_pid is not None
    snapshot = worker.snapshot()
    assert snapshot.phase == "error"
    assert snapshot.reaped is True
    assert snapshot.shutdown_uncertain is True


def test_broken_control_queue_still_forces_reap_and_no_wait_cleanup(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    worker = I2RTArmWorker(
        _config(root, commit),
        context=_fork_context(),
        child_target=_ready_then_stall,
        startup_timeout_seconds=1.0,
        heartbeat_timeout_seconds=10.0,
        telemetry_timeout_seconds=10.0,
        operation_timeout_seconds=0.05,
        shutdown_timeout_seconds=0.05,
        terminate_timeout_seconds=0.1,
    )
    worker.start()
    original_control_queue = worker._control_queue
    worker._control_queue = _BrokenControlQueue()

    # Monitor/startup cleanup uses this fire-and-forget path. A failed queue
    # feeder must not escape and abort the remaining terminate/kill/reap work.
    worker._send_control_without_wait("shutdown")
    result = worker.shutdown()

    assert result.forced is True
    assert result.reaped is True
    assert result.uncertain is True
    assert worker.snapshot().reaped is True
    assert worker.snapshot().shutdown_uncertain is True
    if original_control_queue is not None:
        original_control_queue.close()
        original_control_queue.cancel_join_thread()


def test_mismatched_ready_handshake_fails_closed_and_reaps(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    worker = I2RTArmWorker(
        _config(root, commit),
        context=_fork_context(),
        child_target=_mismatched_ready,
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.1,
        terminate_timeout_seconds=0.1,
    )
    with pytest.raises(I2RTWorkerStartupError, match="handshake"):
        worker.start()
    assert worker.snapshot().reaped is True


def test_post_ready_crash_is_latched_without_respawn_and_reaped(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    worker = I2RTArmWorker(
        _config(root, commit),
        context=_fork_context(),
        child_target=_ready_then_crash,
        startup_timeout_seconds=1.0,
        heartbeat_timeout_seconds=0.5,
        telemetry_timeout_seconds=0.5,
        shutdown_timeout_seconds=0.1,
        terminate_timeout_seconds=0.1,
    )
    worker.start()
    _wait_until(lambda: worker.snapshot().fault is not None)
    with pytest.raises(I2RTWorkerUnavailableError, match="exited unexpectedly"):
        worker.ensure_healthy()
    result = worker.shutdown()
    assert result.reaped is True
    assert result.uncertain is True
    with pytest.raises(I2RTWorkerStartupError, match="only once"):
        worker.start()


def test_stale_heartbeat_and_telemetry_trigger_fail_closed_teardown(
    fake_checkout: tuple[Path, str],
) -> None:
    root, commit = fake_checkout
    worker = I2RTArmWorker(
        _config(root, commit),
        context=_fork_context(),
        child_target=_ready_then_stall,
        startup_timeout_seconds=1.0,
        heartbeat_timeout_seconds=0.08,
        telemetry_timeout_seconds=0.08,
        shutdown_timeout_seconds=0.1,
        terminate_timeout_seconds=0.1,
    )
    worker.start()
    _wait_until(lambda: worker.snapshot().fault is not None)
    _wait_until(lambda: worker.snapshot().reaped)
    snapshot = worker.snapshot()
    assert "stale" in (snapshot.fault or "")
    assert snapshot.shutdown_uncertain is True
    result = worker.shutdown()
    assert result.reaped is True
    assert result.uncertain is True
