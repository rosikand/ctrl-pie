from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ctrl_pi.config import AppConfig
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.real_yam import (
    CTRL_JOINT_LIMITS_RADIANS,
    DefaultVendorFactory,
    FOLLOWER_ID,
    LEADER_ID,
    RealYAMConfig,
    RealYAMDriver,
    VendorDevices,
    create_yam_driver,
)
from ctrl_pi.drivers.yam import (
    JOINT_NAMES,
    ActionLimitError,
    ArmAction,
    ControlLoopTelemetry,
    EndEffectorPose,
    GripperTelemetry,
    JointTelemetry,
    JogCommand,
    JogLimitError,
    YAMDriverDiagnostic,
    YAMDriverUnavailableError,
    YAMSetupConfig,
)
from ctrl_pi.yam_probe import run_probe


CONFIGURED = YAMDriverDiagnostic(
    status="configured",
    detail="YAM test preflight passed.",
)


class FakeLeader:
    def __init__(
        self,
        events: list[str],
        *,
        fail_connect: bool = False,
        fail_after_reads: int | None = None,
        blocking_read: threading.Event | None = None,
        read_entered: threading.Event | None = None,
    ) -> None:
        self.events = events
        self.fail_connect = fail_connect
        self.fail_after_reads = fail_after_reads
        self.blocking_read = blocking_read
        self.read_entered = read_entered
        self.reads = 0

    def connect(self, calibrate: bool = True) -> None:
        self.events.append(f"leader.connect:{calibrate}")
        if self.fail_connect:
            raise RuntimeError("leader connect failed")

    def get_action(self) -> dict[str, float]:
        self.reads += 1
        self.events.append("leader.read")
        if self.fail_after_reads is not None and self.reads > self.fail_after_reads:
            raise RuntimeError("/dev/private-leader RAW_PAYLOAD")
        if self.blocking_read is not None and self.reads > 1:
            if self.read_entered is not None:
                self.read_entered.set()
            self.blocking_read.wait(timeout=3.0)
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -100.0,
            "elbow_flex.pos": 100.0,
            "wrist_flex.pos": -50.0,
            "wrist_roll.pos": 50.0,
            "wrist_yaw.pos": 0.0,
            "gripper.pos": 25.0,
        }

    def disconnect(self) -> None:
        self.events.append("leader.disconnect")


class FakeFollower:
    def __init__(
        self,
        events: list[str],
        *,
        fail_connect: bool = False,
        fail_command: bool = False,
        unhealthy_after_checks: int | None = None,
        blocking_command: threading.Event | None = None,
        command_entered: threading.Event | None = None,
    ) -> None:
        self.events = events
        self.fail_connect = fail_connect
        self.fail_command = fail_command
        self.unhealthy_after_checks = unhealthy_after_checks
        self.blocking_command = blocking_command
        self.command_entered = command_entered
        self.health_checks = 0
        self.commands: list[np.ndarray] = []
        self.position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)
        self.gripper = 0.2

    def connect(self, calibrate: bool = True) -> None:
        self.events.append(f"follower.connect:{calibrate}")
        if self.fail_connect:
            raise RuntimeError("follower connect failed")

    def get_observations(self) -> dict[str, np.ndarray]:
        self.events.append("follower.read")
        return {
            "joint_pos": self.position.copy(),
            "gripper_pos": np.array([self.gripper]),
            "joint_vel": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7]),
            "joint_eff": np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
        }

    def worker_healthy(self) -> bool:
        self.health_checks += 1
        return (
            self.unhealthy_after_checks is None
            or self.health_checks <= self.unhealthy_after_checks
        )

    def set_position_control_mode(self) -> None:
        self.events.append("follower.position-control")

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.events.append("follower.command")
        if self.command_entered is not None:
            self.command_entered.set()
        if self.blocking_command is not None:
            self.blocking_command.wait(timeout=3.0)
        if self.fail_command:
            raise RuntimeError("can0 RAW_COMMAND secret-path")
        copied = np.asarray(joint_pos, dtype=float).copy()
        self.commands.append(copied)
        self.position = copied[:6]
        self.gripper = float(copied[6])

    def zero_torque_mode(self) -> None:
        self.events.append("follower.zero")

    def close(self) -> None:
        self.events.append("follower.close")


class FakePoseSolver:
    def pose(self, joint_positions_radians: list[float]) -> tuple[float, ...]:
        return tuple(joint_positions_radians)


class FakeFactory:
    def __init__(self, leader: FakeLeader, follower: FakeFollower) -> None:
        self.leader = leader
        self.follower = follower
        self.calls = 0

    def create(self, _config: RealYAMConfig) -> VendorDevices:
        self.calls += 1
        self.leader.events.append("factory.create")
        return VendorDevices(self.leader, self.follower, FakePoseSolver())


def _config(**updates: Any) -> RealYAMConfig:
    config = RealYAMConfig(
        can_interface="lo",
        leader_port="/tmp/fake-yam-leader",
        mujoco_xml_path="/tmp/fake-yam.xml",
        gripper_type="crank_4310",
        leader_calibration_id="yam-leader",
        leader_calibration_dir="/tmp/fake-yam-calibration",
    )
    return replace(config, **updates)


def _driver(
    leader: FakeLeader,
    follower: FakeFollower,
) -> tuple[RealYAMDriver, FakeFactory]:
    factory = FakeFactory(leader, follower)
    driver = RealYAMDriver(_config(), vendor_factory=factory)
    driver.preflight = lambda: CONFIGURED  # type: ignore[method-assign]
    return driver, factory


def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.005)


def test_factory_selects_mock_or_real_without_hardware_fallback() -> None:
    mock = create_yam_driver(AppConfig(_env_file=None, ctrl_pi_mock_mode=True))
    hardware = create_yam_driver(
        AppConfig(_env_file=None, ctrl_pi_mock_mode=False)
    )

    assert isinstance(mock, MockYAMDriver)
    assert isinstance(hardware, RealYAMDriver)
    assert not isinstance(hardware, MockYAMDriver)
    assert [(arm.id, arm.connected) for arm in hardware.list_arms()] == [
        (LEADER_ID, False),
        (FOLLOWER_ID, False),
    ]


def test_app_config_wires_every_explicit_hardware_value() -> None:
    app_config = AppConfig(
        _env_file=None,
        ctrl_pi_mock_mode=False,
        yam_can_interface="can7",
        yam_leader_port="/dev/serial/by-id/yam-test",
        yam_mujoco_xml_path="/opt/yam/model.xml",
        yam_gripper_type="crank_4310",
        yam_leader_calibration_id="stable-leader",
        yam_leader_calibration_dir="/var/lib/ctrl-pi/calibration",
    )

    config = RealYAMConfig.from_app_config(app_config)

    assert config == RealYAMConfig(
        can_interface="can7",
        leader_port="/dev/serial/by-id/yam-test",
        mujoco_xml_path="/opt/yam/model.xml",
        gripper_type="crank_4310",
        leader_calibration_id="stable-leader",
        leader_calibration_dir="/var/lib/ctrl-pi/calibration",
    )
    assert AppConfig(_env_file=None).yam_leader_calibration_id == "yam-leader"


def test_default_vendor_factory_lazily_builds_pinned_configs_with_stable_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "leader-cal.json").write_text("{}")
    config = _config(
        leader_calibration_id="leader-cal",
        leader_calibration_dir=str(calibration_dir),
    )
    imports: list[str] = []
    captured: dict[str, dict[str, Any]] = {}

    class LeaderConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["leader"] = kwargs

    class CreatedLeader:
        calibration = {"loaded": object()}

        def __init__(self, _config: Any) -> None:
            pass

    class FollowerConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["follower"] = kwargs

    class CreatedFollower:
        def __init__(self, _config: Any) -> None:
            pass

    modules = {
        "lerobot_teleoperator_yam_gello.yam_leader": SimpleNamespace(
            YAMLeader=CreatedLeader
        ),
        "lerobot_teleoperator_yam_gello.config_yam_leader": SimpleNamespace(
            YAMLeaderTeleopConfig=LeaderConfig
        ),
        "lerobot_robot_yam.yam_follower": SimpleNamespace(
            YAMFollower=CreatedFollower
        ),
        "lerobot_robot_yam.config_yam_follower": SimpleNamespace(
            YAMFollowerRobotConfig=FollowerConfig
        ),
    }

    def fake_import(name: str) -> Any:
        imports.append(name)
        return modules[name]

    monkeypatch.setattr("ctrl_pi.drivers.real_yam.importlib.import_module", fake_import)
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam._MujocoPoseSolver",
        lambda _path: FakePoseSolver(),
    )

    devices = DefaultVendorFactory().create(config)

    assert imports == list(modules)
    assert captured["leader"] == {
        "port": config.leader_port,
        "id": "leader-cal",
        "calibration_dir": calibration_dir,
    }
    assert captured["follower"] == {
        "id": FOLLOWER_ID,
        "port": config.can_interface,
        "mujoco_xml_path": config.mujoco_xml_path,
        "gripper_type": "crank_4310",
        "zero_gravity_mode": True,
    }
    assert isinstance(devices.leader, CreatedLeader)
    assert devices.follower._follower.__class__ is CreatedFollower


def test_startup_maps_vendor_order_units_and_nullable_telemetry() -> None:
    events: list[str] = []
    driver, factory = _driver(FakeLeader(events), FakeFollower(events))
    try:
        driver.startup()
        leader, follower = driver.list_arms()

        assert factory.calls == 1
        assert events[:4] == [
            "factory.create",
            "leader.connect:False",
            "follower.connect:False",
            "leader.read",
        ]
        assert [joint.name for joint in follower.joints] == list(JOINT_NAMES)
        assert [joint.position_radians for joint in follower.joints] == pytest.approx(
            [0.1, 0.2, 0.3, 0.5, 0.4, 0.6]
        )
        assert [joint.velocity_radians_per_second for joint in follower.joints] == pytest.approx(
            [1.0, 2.0, 3.0, 5.0, 4.0, 6.0]
        )
        assert [joint.effort_newton_meters for joint in follower.joints] == pytest.approx(
            [11.0, 12.0, 13.0, 15.0, 14.0, 16.0]
        )
        assert all(joint.temperature_celsius is None for joint in follower.joints)
        assert all(joint.effort_newton_meters is None for joint in leader.joints)
        assert follower.gripper.position == pytest.approx(0.8)
        assert follower.gripper.velocity == pytest.approx(-0.7)
        assert follower.gripper.is_closed is False
        assert follower.gripper.force_newtons is None
        assert follower.can.tx_error_count is None
        assert follower.pose.model_dump() == pytest.approx(
            {
                "x_m": 0.1,
                "y_m": 0.2,
                "z_m": 0.3,
                "roll_radians": 0.4,
                "pitch_radians": 0.5,
                "yaw_radians": 0.6,
            }
        )
        assert leader.joints[0].position_radians == pytest.approx(
            sum(CTRL_JOINT_LIMITS_RADIANS["shoulder_yaw"]) / 2
        )
        assert leader.joints[3].position_radians == pytest.approx(0.86)
        assert leader.joints[4].position_radians == pytest.approx(-0.86)
        assert leader.gripper.position == pytest.approx(0.75)
        assert driver.diagnostic().status == "connected"
    finally:
        driver.shutdown()


def test_telemetry_thread_start_failure_disconnects_devices_and_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    driver, factory = _driver(FakeLeader(events), FakeFollower(events))

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("/dev/operator-secret RAW_THREAD_FAILURE")

    monkeypatch.setattr("ctrl_pi.drivers.real_yam.threading.Thread.start", fail_start)
    caplog.set_level(logging.ERROR, logger="ctrl_pi.drivers.real_yam")

    driver.startup()
    driver.shutdown()
    driver.shutdown()

    assert factory.calls == 1
    assert driver.diagnostic().status == "error"
    assert all(not arm.connected for arm in driver.list_arms())
    assert events.count("follower.zero") == 1
    assert events.count("follower.close") == 1
    assert events.count("leader.disconnect") == 1
    assert "operator-secret" not in caplog.text
    assert "RAW_THREAD_FAILURE" not in caplog.text


def test_lifecycle_lock_prevents_reconfiguration_during_device_startup() -> None:
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class BlockingFactory(FakeFactory):
        def create(self, config: RealYAMConfig) -> VendorDevices:
            entered.set()
            assert release.wait(timeout=2.0)
            return super().create(config)

    factory = BlockingFactory(FakeLeader(events), FakeFollower(events))
    driver = RealYAMDriver(_config(), vendor_factory=factory)
    driver.preflight = lambda: CONFIGURED  # type: ignore[method-assign]
    replacement = YAMSetupConfig(
        can_interface="can1",
        leader_port="/dev/serial/by-id/replacement",
        mujoco_xml_path="/opt/ctrl-pi/replacement.xml",
        leader_calibration_id="replacement",
        leader_calibration_dir="/var/lib/ctrl-pi/calibration",
    )

    startup = threading.Thread(target=driver.startup)
    configure = threading.Thread(target=lambda: driver.apply_setup(replacement))
    startup.start()
    assert entered.wait(timeout=1.0)
    configure.start()
    time.sleep(0.03)
    assert configure.is_alive()
    release.set()
    startup.join(timeout=2.0)
    configure.join(timeout=2.0)

    assert not startup.is_alive()
    assert not configure.is_alive()
    assert driver.setup_config() == replacement
    assert factory.calls == 1
    assert "follower.zero" in events
    assert "follower.close" in events
    assert "leader.disconnect" in events
    driver.shutdown()


def test_apply_action_reorders_wrist_axes_and_enables_position_control_once() -> None:
    events: list[str] = []
    follower = FakeFollower(events)
    driver, _ = _driver(FakeLeader(events), follower)
    try:
        driver.startup()
        current = driver.get_arm(FOLLOWER_ID)
        current_positions = {
            joint.name: joint.position_radians for joint in current.joints
        }
        first = ArmAction(
            timestamp=datetime.now(UTC),
            joint_positions_radians={
                name: current_positions[name] + 0.1 for name in JOINT_NAMES
            },
            gripper_position=0.7,
        )
        driver.apply_action(FOLLOWER_ID, first)
        driver.apply_action(
            FOLLOWER_ID,
            ArmAction(
                timestamp=datetime.now(UTC),
                joint_positions_radians={
                    name: first.joint_positions_radians[name] + 0.01
                    for name in JOINT_NAMES
                },
                gripper_position=0.71,
            ),
        )

        assert follower.commands[0] == pytest.approx(
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.3]
        )
        assert events.count("follower.position-control") == 1
    finally:
        driver.shutdown()


def test_crank_gripper_boundary_preserves_public_zero_closed_one_open() -> None:
    events: list[str] = []
    follower = FakeFollower(events)
    follower.gripper = 0.9
    driver, _ = _driver(FakeLeader(events), follower)
    try:
        driver.startup()
        snapshot = driver.get_arm(FOLLOWER_ID)

        assert snapshot.gripper.position == pytest.approx(0.1)
        assert snapshot.gripper.velocity == pytest.approx(-0.7)
        assert snapshot.gripper.is_closed is True

        driver.jog(
            FOLLOWER_ID,
            JogCommand(kind="gripper", axis="position", delta=0.1),
        )
        assert follower.commands[-1][6] == pytest.approx(0.8)
    finally:
        driver.shutdown()


def test_real_driver_rejects_leader_cartesian_limits_and_large_steps() -> None:
    events: list[str] = []
    follower = FakeFollower(events)
    driver, _ = _driver(FakeLeader(events), follower)
    try:
        driver.startup()
        with pytest.raises(JogLimitError, match="Only the connected"):
            driver.jog(
                LEADER_ID,
                JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
            )
        with pytest.raises(JogLimitError, match="Cartesian jog"):
            driver.jog(
                FOLLOWER_ID,
                JogCommand(kind="cartesian", axis="x", delta=0.01),
            )
        current = driver.get_arm(FOLLOWER_ID)
        positions = {joint.name: joint.position_radians for joint in current.joints}
        positions["shoulder_yaw"] = CTRL_JOINT_LIMITS_RADIANS["shoulder_yaw"][1] + 0.01
        with pytest.raises(ActionLimitError, match="safe range"):
            driver.apply_action(
                FOLLOWER_ID,
                ArmAction(
                    timestamp=datetime.now(UTC),
                    joint_positions_radians=positions,
                    gripper_position=current.gripper.position,
                ),
            )
        positions["shoulder_yaw"] = current.joints[0].position_radians + 0.351
        with pytest.raises(ActionLimitError, match="safety bound"):
            driver.apply_action(
                FOLLOWER_ID,
                ArmAction(
                    timestamp=datetime.now(UTC),
                    joint_positions_radians=positions,
                    gripper_position=current.gripper.position,
                ),
            )
        assert follower.commands == []
    finally:
        driver.shutdown()


def test_command_failure_zero_torques_disconnects_and_sanitizes_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    follower = FakeFollower(events, fail_command=True)
    driver, _ = _driver(FakeLeader(events), follower)
    caplog.set_level(logging.ERROR, logger="ctrl_pi.drivers.real_yam")
    try:
        driver.startup()
        with pytest.raises(YAMDriverUnavailableError, match="motion was disabled"):
            driver.jog(
                FOLLOWER_ID,
                JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
            )

        assert "follower.zero" in events
        assert not driver.get_arm(FOLLOWER_ID).connected
        assert driver.diagnostic().status == "error"
        assert "RAW_COMMAND" not in caplog.text
        assert "secret-path" not in caplog.text
    finally:
        driver.shutdown()


def test_dead_follower_worker_rejects_write_before_reporting_success() -> None:
    events: list[str] = []
    release = threading.Event()
    entered = threading.Event()
    follower = FakeFollower(events)
    driver, _ = _driver(
        FakeLeader(events, blocking_read=release, read_entered=entered),
        follower,
    )
    try:
        driver.startup()
        assert entered.wait(timeout=1.0)
        follower.unhealthy_after_checks = follower.health_checks

        with pytest.raises(YAMDriverUnavailableError, match="motion was disabled"):
            driver.jog(
                FOLLOWER_ID,
                JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
            )

        assert follower.commands == []
        assert "follower.zero" in events
        assert driver.diagnostic().status == "error"
        assert not driver.get_arm(FOLLOWER_ID).connected
    finally:
        release.set()
        driver.shutdown()


def test_partial_connect_rolls_back_in_safe_order_and_shutdown_is_idempotent() -> None:
    events: list[str] = []
    follower = FakeFollower(events, fail_connect=True)
    driver, _ = _driver(FakeLeader(events), follower)

    driver.startup()
    first_cleanup = events.copy()
    driver.shutdown()
    driver.shutdown()

    assert first_cleanup[:3] == [
        "factory.create",
        "leader.connect:False",
        "follower.connect:False",
    ]
    assert first_cleanup[-3:] == [
        "follower.zero",
        "follower.close",
        "leader.disconnect",
    ]
    assert events == first_cleanup
    assert driver.diagnostic().status == "error"


@pytest.mark.parametrize(
    "config",
    [
        _config(can_interface=None),
        _config(gripper_type="linear_3507"),
        _config(gripper_type="linear_4310"),
        _config(gripper_type="unknown"),
    ],
)
def test_invalid_or_unsupported_config_never_reaches_vendor(config: RealYAMConfig) -> None:
    events: list[str] = []
    factory = FakeFactory(FakeLeader(events), FakeFollower(events))
    driver = RealYAMDriver(config, vendor_factory=factory)

    driver.startup()

    assert factory.calls == 0
    assert all(not arm.connected for arm in driver.list_arms())
    assert driver.diagnostic().status in {"missing", "error"}


def test_worker_failure_latches_disconnected_without_leaking_vendor_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    driver, _ = _driver(
        FakeLeader(events),
        FakeFollower(events, unhealthy_after_checks=2),
    )
    caplog.set_level(logging.ERROR, logger="ctrl_pi.drivers.real_yam")
    driver.startup()
    _wait_for(lambda: driver.diagnostic().status == "error")
    driver.shutdown()

    assert all(not arm.connected for arm in driver.list_arms())
    assert "RAW_PAYLOAD" not in caplog.text


def test_cached_reads_do_not_wait_for_blocked_serial_io() -> None:
    events: list[str] = []
    release = threading.Event()
    entered = threading.Event()
    driver, _ = _driver(
        FakeLeader(events, blocking_read=release, read_entered=entered),
        FakeFollower(events),
    )
    driver.startup()
    assert entered.wait(timeout=1.0)

    started = time.monotonic()
    arms = driver.list_arms()
    elapsed = time.monotonic() - started

    assert len(arms) == 2
    assert elapsed < 0.05
    release.set()
    driver.shutdown()


def test_follower_command_does_not_wait_for_blocked_leader_serial_io() -> None:
    events: list[str] = []
    release = threading.Event()
    entered = threading.Event()
    follower = FakeFollower(events)
    driver, _ = _driver(
        FakeLeader(events, blocking_read=release, read_entered=entered),
        follower,
    )
    driver.startup()
    assert entered.wait(timeout=1.0)

    started = time.monotonic()
    driver.jog(
        FOLLOWER_ID,
        JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    assert len(follower.commands) == 1
    release.set()
    driver.shutdown()


def test_stale_snapshot_rejects_command_and_zero_torques_without_target_write() -> None:
    events: list[str] = []
    release = threading.Event()
    entered = threading.Event()
    follower = FakeFollower(events)
    driver, _ = _driver(
        FakeLeader(events, blocking_read=release, read_entered=entered),
        follower,
    )
    driver.startup()
    assert entered.wait(timeout=1.0)
    with driver._state_lock:
        driver._last_sample_monotonic = time.monotonic() - 1.0

    with pytest.raises(YAMDriverUnavailableError, match="telemetry is stale"):
        driver.jog(
            FOLLOWER_ID,
            JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
        )

    assert follower.commands == []
    assert "follower.zero" in events
    assert driver.diagnostic().status == "error"
    assert not driver.get_arm(FOLLOWER_ID).connected
    release.set()
    driver.shutdown()


def test_cached_reads_do_not_wait_for_blocked_command() -> None:
    events: list[str] = []
    release = threading.Event()
    entered = threading.Event()
    follower = FakeFollower(
        events,
        blocking_command=release,
        command_entered=entered,
    )
    driver, _ = _driver(FakeLeader(events), follower)
    driver.startup()
    errors: list[BaseException] = []

    def command() -> None:
        try:
            driver.jog(
                FOLLOWER_ID,
                JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=command)
    thread.start()
    assert entered.wait(timeout=1.0)
    started = time.monotonic()
    driver.get_arm(FOLLOWER_ID)
    elapsed = time.monotonic() - started
    release.set()
    thread.join(timeout=1.0)
    driver.shutdown()

    assert elapsed < 0.05
    assert errors == []


@pytest.mark.parametrize(
    ("model", "payload", "field"),
    [
        (
            JointTelemetry,
            {
                "name": "shoulder_yaw",
                "position_radians": 0.0,
                "velocity_radians_per_second": 0.0,
                "effort_newton_meters": 0.0,
                "temperature_celsius": 0.0,
            },
            "position_radians",
        ),
        (
            JointTelemetry,
            {
                "name": "shoulder_yaw",
                "position_radians": 0.0,
                "velocity_radians_per_second": 0.0,
                "effort_newton_meters": 0.0,
                "temperature_celsius": 0.0,
            },
            "effort_newton_meters",
        ),
        (
            EndEffectorPose,
            {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "roll_radians": 0.0,
                "pitch_radians": 0.0,
                "yaw_radians": 0.0,
            },
            "yaw_radians",
        ),
        (
            GripperTelemetry,
            {
                "position": 0.5,
                "velocity": 0.0,
                "force_newtons": None,
                "is_closed": False,
            },
            "velocity",
        ),
        (
            ControlLoopTelemetry,
            {
                "target_frequency_hz": 50.0,
                "frequency_hz": 50.0,
                "cycle_time_ms": 1.0,
                "jitter_ms": 0.0,
                "dropped_cycles": 0,
            },
            "frequency_hz",
        ),
    ],
)
def test_telemetry_models_reject_non_finite_values(
    model: Any,
    payload: dict[str, Any],
    field: str,
) -> None:
    payload[field] = float("nan")

    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_startup_exception_and_public_diagnostic_are_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingFactory:
        calls = 0

        def create(self, _config: RealYAMConfig) -> VendorDevices:
            self.calls += 1
            raise RuntimeError("/dev/secret-device TOKEN RAW_PAYLOAD")

    factory = FailingFactory()
    driver = RealYAMDriver(_config(), vendor_factory=factory)
    driver.preflight = lambda: CONFIGURED  # type: ignore[method-assign]
    caplog.set_level(logging.ERROR, logger="ctrl_pi.drivers.real_yam")

    driver.startup()

    assert factory.calls == 1
    public = driver.diagnostic().model_dump_json()
    assert driver.diagnostic().status == "error"
    for sentinel in ("secret-device", "TOKEN", "RAW_PAYLOAD"):
        assert sentinel not in public
        assert sentinel not in caplog.text


def _preflight_config(tmp_path: Path) -> RealYAMConfig:
    model = tmp_path / "yam.xml"
    model.write_text("<mujoco/>")
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    calibration = {
        name: {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": -100,
            "range_max": 100,
        }
        for index, name in enumerate(
            (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "wrist_yaw",
                "gripper",
            ),
            start=1,
        )
    }
    (calibration_dir / "yam-leader.json").write_text(json.dumps(calibration))
    serial = tmp_path / "ttyYAM"
    serial.write_text("")
    return _config(
        leader_port=str(serial),
        mujoco_xml_path=str(model),
        leader_calibration_dir=str(calibration_dir),
    )


def test_preflight_checks_exact_versions_files_serial_and_can_without_vendor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.metadata.version",
        lambda _distribution: "0.1.1",
    )
    monkeypatch.setattr(
        RealYAMDriver,
        "_serial_device_is_character",
        staticmethod(lambda _path: True),
    )
    monkeypatch.setattr(
        RealYAMDriver,
        "_network_interface_kind",
        staticmethod(lambda _name: "can"),
    )
    events: list[str] = []
    factory = FakeFactory(FakeLeader(events), FakeFollower(events))
    driver = RealYAMDriver(_preflight_config(tmp_path), vendor_factory=factory)

    result = driver.preflight()

    assert result.status == "configured"
    assert factory.calls == 0
    assert events == []


def test_discovery_is_passive_typed_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sys_net = tmp_path / "sys-class-net"
    sys_net.mkdir()
    interface_names = [f"can{index}" for index in range(40)] + ["ethernet-secret"]
    for name in interface_names:
        directory = sys_net / name
        directory.mkdir()
        (directory / "type").write_text("280\n" if name.startswith("can") else "1\n")
    serial_root = tmp_path / "by-id"
    serial_root.mkdir()
    for index in range(40):
        (serial_root / f"operator-secret-{index}").symlink_to("/dev/null")
    (serial_root / "regular-file").write_text("not a device")
    (serial_root / "broken-link").symlink_to(tmp_path / "missing-device")
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.SYS_CLASS_NET_ROOT", sys_net
    )
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.STABLE_SERIAL_ROOT", serial_root
    )
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.CONVENTIONAL_LEADER_DEVICE",
        tmp_path / "missing-conventional",
    )
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.socket.if_nameindex",
        lambda: list(enumerate(interface_names, start=1)),
    )
    events: list[str] = []
    factory = FakeFactory(FakeLeader(events), FakeFollower(events))
    driver = RealYAMDriver(_config(), vendor_factory=factory)

    result = driver.discover_setup()

    assert len(result.can_interfaces) == 32
    assert len(result.leader_ports) == 32
    assert all(item.id.startswith("can") for item in result.can_interfaces)
    assert "ethernet-secret" not in result.model_dump_json()
    assert all("operator-secret" not in item.label for item in result.leader_ports)
    assert all(Path(item.id).is_symlink() for item in result.leader_ports)
    assert factory.calls == 0
    assert events == []


@pytest.mark.parametrize(
    "payload",
    [
        b"{not-json",
        b"{}",
        b"x" * (64 * 1024 + 1),
    ],
)
def test_preflight_rejects_malformed_or_oversized_calibration_without_vendor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: bytes,
) -> None:
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.metadata.version",
        lambda _distribution: "0.1.1",
    )
    config = _preflight_config(tmp_path)
    assert config.leader_calibration_dir is not None
    (Path(config.leader_calibration_dir) / "yam-leader.json").write_bytes(payload)
    events: list[str] = []
    factory = FakeFactory(FakeLeader(events), FakeFollower(events))
    driver = RealYAMDriver(config, vendor_factory=factory)

    result = driver.preflight()

    assert result.status == "missing"
    assert "calibration file" in result.detail
    assert factory.calls == 0
    assert events == []


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ({"gripper_type": "linear_3507"}, "error"),
        ({"can_interface": "ctrl-pi-no-such-interface"}, "missing"),
        ({"leader_port": "/ctrl-pi/no-device"}, "missing"),
        ({"mujoco_xml_path": "/ctrl-pi/no-model"}, "missing"),
    ],
)
def test_preflight_fails_closed_without_disclosing_selected_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: dict[str, str],
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.metadata.version",
        lambda _distribution: "0.1.1",
    )
    monkeypatch.setattr(
        RealYAMDriver,
        "_serial_device_is_character",
        staticmethod(lambda _path: True),
    )
    config = replace(_preflight_config(tmp_path), **mutation)
    driver = RealYAMDriver(config)

    result = driver.preflight()

    assert result.status == expected_status
    assert all(value not in result.detail for value in mutation.values())


def test_preflight_rejects_package_version_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.metadata.version",
        lambda distribution: "0.1.0" if distribution == "yam-common" else "0.1.1",
    )
    driver = RealYAMDriver(_preflight_config(tmp_path))

    result = driver.preflight()

    assert result.status == "error"
    assert "versions" in result.detail


@pytest.mark.parametrize(
    "mutation",
    [
        {"mujoco_xml_path": "x" * 10_000},
        {"leader_calibration_dir": "y" * 10_000},
        {"mujoco_xml_path": "~ctrl_pi_user_does_not_exist/model.xml"},
        {"leader_calibration_dir": "~ctrl_pi_user_does_not_exist/calibration"},
    ],
)
def test_hostile_hardware_paths_become_sanitized_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: dict[str, str],
) -> None:
    monkeypatch.setattr(
        "ctrl_pi.drivers.real_yam.metadata.version",
        lambda _distribution: "0.1.1",
    )

    driver = RealYAMDriver(replace(_preflight_config(tmp_path), **mutation))
    result = driver.preflight()

    assert result.status == "error"
    assert all(value not in result.detail for value in mutation.values())
    assert all(not arm.connected for arm in driver.list_arms())


def test_probe_uses_real_preflight_even_when_application_is_in_mock_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = AppConfig(_env_file=None, ctrl_pi_mock_mode=True)

    exit_code = run_probe(config, connect=False)
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert [arm["driver"] for arm in report["arms"]] == ["yam", "yam"]
    assert report["connect_requested"] is False
    assert all(not arm["connected"] for arm in report["arms"])


def test_probe_default_does_not_construct_vendor_and_connect_always_closes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    driver, factory = _driver(FakeLeader(events), FakeFollower(events))
    app_config = AppConfig(_env_file=None)

    assert run_probe(app_config, connect=False, driver=driver) == 0
    preflight_report = json.loads(capsys.readouterr().out)
    assert preflight_report["status"] == "configured"
    assert factory.calls == 0

    assert run_probe(app_config, connect=True, driver=driver) == 0
    connected_report = json.loads(capsys.readouterr().out)
    assert connected_report["status"] == "connected"
    assert factory.calls == 1
    assert "follower.zero" in events
    assert "follower.close" in events
    assert "leader.disconnect" in events
    serialized = json.dumps(connected_report)
    assert "/tmp/fake" not in serialized


def test_probe_connect_failure_is_actionable_sanitized_and_always_closes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    driver, factory = _driver(
        FakeLeader(events),
        FakeFollower(events, fail_connect=True),
    )

    assert run_probe(AppConfig(_env_file=None), connect=True, driver=driver) == 1
    report = json.loads(capsys.readouterr().out)

    assert factory.calls == 1
    assert report["status"] == "error"
    assert "startup failed" in report["detail"]
    assert "follower.zero" in events
    assert "follower.close" in events
    assert "leader.disconnect" in events
    serialized = json.dumps(report)
    assert "/tmp/fake" not in serialized
    assert "RAW_PAYLOAD" not in serialized
