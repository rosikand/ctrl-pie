from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path
import queue
import struct
import time
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ctrl_pi.db import Base
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import (
    JOINT_NAMES,
    ArmAction,
    LegacyYAMSetupConfig,
    YAMCellArmConfig,
    YAMCellConfig,
    YAMDiscoveryCandidate,
    YAMDiscoveryResult,
    YAMDriverUnavailableError,
)
from ctrl_pi.drivers.yam_cell import YAMCellDriver
from ctrl_pi.drivers.yam_cell_discovery import SocketCANAdapter, SocketCANInventory
from ctrl_pi.drivers.yam_handle_diagnostic import (
    HANDLE_REPORT_ID,
    HANDLE_REPORT_REQUEST,
    HANDLE_REQUEST_ID,
    HandleDiagnosticConfig,
    HandleDiagnosticSample,
    SupervisedHandleRangeReader,
    YAMHandleDiagnosticError,
    YAMHandleDiagnosticTeardownUncertainError,
    sample_handle_requests_only,
)
import ctrl_pi.drivers.yam_handle_diagnostic as handle_diagnostic
from ctrl_pi.drivers.yam_i2rt_worker import (
    I2RTArmWorkerConfig,
    I2RTCheckoutIdentity,
    I2RTWorkerShutdownResult,
    I2RTWorkerSnapshot,
    WorkerTelemetry,
)
from ctrl_pi.rig import RigLease
from ctrl_pi.yam_setup import YAMSetupManager


NOW = datetime(2026, 8, 29, tzinfo=UTC)


class FakeLease:
    def __init__(self, key: tuple[str, str]) -> None:
        self.key = key
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeLocks:
    def __init__(self) -> None:
        self.leases: list[FakeLease] = []

    def acquire(self, *, stable_identity: str, runtime_interface: str) -> FakeLease:
        lease = FakeLease((stable_identity, runtime_interface))
        self.leases.append(lease)
        return lease


class FakeWorker:
    def __init__(
        self,
        config: I2RTArmWorkerConfig,
        *,
        fail_start: bool = False,
        shutdown_clean: bool = True,
    ) -> None:
        self.config = config
        self.fail_start = fail_start
        self.shutdown_clean = shutdown_clean
        self.started = False
        self.accepting = False
        self.faulted = False
        self.safe_idle_fails = False
        self.safe_idle_calls = 0
        self.shutdown_calls = 0
        self.commands: list[tuple[float, ...]] = []
        self.last_command_at: float | None = None
        base = 0.1 if config.role == "follower" else 0.2
        self.sample = WorkerTelemetry(
            protocol_version=1,
            logical_id=config.logical_id,
            config_fingerprint=config.fingerprint,
            sample_monotonic=1.0,
            joint_positions=(base, base, base, base, base, base),
            joint_velocities=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_efforts=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            gripper_position=0.8 if config.role == "leader" else 0.4,
            gripper_velocity=0.0,
            gripper_effort=2.0,
            handle_trigger_position=0.2 if config.role == "leader" else None,
            handle_buttons=(False, True) if config.role == "leader" else (),
            control_loop_frequency_hz=421.0 if config.role == "leader" else 268.0,
            inner_liveness={"motor_chain_running": True},
            command_sequence_applied=None,
        )

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("secret vendor path must not escape")
        self.started = True

    def snapshot(self) -> I2RTWorkerSnapshot:
        if (
            self.accepting
            and self.last_command_at is not None
            and time.monotonic() - self.last_command_at
            > self.config.command_max_age_seconds
        ):
            self.faulted = True
        return I2RTWorkerSnapshot(
            phase="error" if self.faulted else ("ready" if self.started else "new"),
            pid=123 if self.started else None,
            process_group_id=123 if self.started else None,
            accepting_commands=self.accepting,
            heartbeat_age_seconds=0.01,
            telemetry_age_seconds=0.01,
            inner_liveness={"motor_chain_running": not self.faulted},
            dropped_commands=0,
            fault="private vendor failure" if self.faulted else None,
            shutdown_uncertain=False,
            reaped=not self.started,
            logs=(),
        )

    def telemetry(self) -> WorkerTelemetry:
        if self.faulted:
            raise RuntimeError("private vendor failure")
        return self.sample

    def enable_commands(self) -> None:
        if self.faulted:
            raise RuntimeError("private vendor failure")
        self.accepting = True

    def send_action(self, positions: Any) -> int:
        if self.faulted:
            raise RuntimeError("private vendor failure")
        self.commands.append(tuple(float(value) for value in positions))
        self.last_command_at = time.monotonic()
        return len(self.commands)

    def safe_idle(self) -> None:
        self.safe_idle_calls += 1
        self.accepting = False
        if self.safe_idle_fails:
            raise RuntimeError("private safe-idle failure")

    def shutdown(self) -> I2RTWorkerShutdownResult:
        self.shutdown_calls += 1
        self.accepting = False
        if self.shutdown_clean:
            self.started = False
        return I2RTWorkerShutdownResult(
            clean=self.shutdown_clean,
            safe_idle_confirmed=self.shutdown_clean,
            device_close_confirmed=self.shutdown_clean,
            forced=False,
            reaped=self.shutdown_clean,
            uncertain=not self.shutdown_clean,
            detail="clean" if self.shutdown_clean else "uncertain private result",
        )


class WorkerFactory:
    def __init__(self) -> None:
        self.created: list[FakeWorker] = []
        self.fail_ids: set[str] = set()
        self.uncertain_ids: set[str] = set()
        self.raise_ids: set[str] = set()

    def __call__(self, config: I2RTArmWorkerConfig) -> FakeWorker:
        if config.logical_id in self.raise_ids:
            raise RuntimeError("private factory failure")
        worker = FakeWorker(
            config,
            fail_start=config.logical_id in self.fail_ids,
            shutdown_clean=config.logical_id not in self.uncertain_ids,
        )
        self.created.append(worker)
        return worker

    def by_id(self, arm_id: str) -> FakeWorker:
        return next(worker for worker in reversed(self.created) if worker.config.logical_id == arm_id)


class FakeHandleReader:
    def __init__(self, sample: HandleDiagnosticSample) -> None:
        self.result = sample
        self.configs: list[HandleDiagnosticConfig] = []

    def sample(self, config: HandleDiagnosticConfig) -> HandleDiagnosticSample:
        self.configs.append(config)
        return self.result


def _arm(
    logical_id: str,
    role: str,
    pair_id: str,
    serial: str,
    *,
    side: str,
    frame_map_path: str | None = None,
    soft_limits_path: str | None = None,
) -> YAMCellArmConfig:
    return YAMCellArmConfig(
        logical_id=logical_id,
        name=logical_id.replace("-", " ").title(),
        role=role,  # type: ignore[arg-type]
        pair_id=pair_id,
        group_id="bimanual",
        side=side,
        transport_kind="socketcan",
        stable_identity=serial,
        end_effector_kind=(
            "yam_teaching_handle" if role == "leader" else "linear_4310"
        ),
        frame_map_path=frame_map_path,
        soft_limits_path=soft_limits_path,
    )


def _cell(
    *,
    frame_map_path: str | None = None,
    soft_limits_path: str | None = None,
) -> YAMCellConfig:
    return YAMCellConfig(
        name="operator cell",
        i2rt_root="/operator/read-only-i2rt",
        i2rt_commit="a" * 40,
        arms=[
            _arm("right-leader", "leader", "right", "SER-RL", side="right"),
            _arm(
                "right-follower",
                "follower",
                "right",
                "SER-RF",
                side="right",
                frame_map_path=frame_map_path,
                soft_limits_path=soft_limits_path,
            ),
            _arm("left-leader", "leader", "left", "SER-LL", side="left"),
            _arm("left-follower", "follower", "left", "SER-LF", side="left"),
        ],
        pair_ports={"right": 11_333, "left": 11_334},
    )


def _inventory(mapping: dict[str, str]) -> SocketCANInventory:
    return SocketCANInventory(
        adapters=tuple(
            SocketCANAdapter(
                interface=interface,
                stable_identity=serial,
                product="fake adapter",
                manufacturer="fake",
                operstate="up",
                link_up=True,
            )
            for serial, interface in mapping.items()
        )
    )


def _driver(
    inventory: SocketCANInventory,
    workers: WorkerFactory | None = None,
    locks: FakeLocks | None = None,
    handle_reader: FakeHandleReader | None = None,
) -> tuple[YAMCellDriver, WorkerFactory, FakeLocks, list[tuple[str, str]]]:
    factory = workers or WorkerFactory()
    lock_manager = locks or FakeLocks()
    checkout_calls: list[tuple[str, str]] = []

    def verify(root: str, commit: str) -> I2RTCheckoutIdentity:
        checkout_calls.append((root, commit))
        return I2RTCheckoutIdentity(root=root, commit=commit, read_only=True)

    driver = YAMCellDriver(
        MockYAMDriver(),
        inventory_provider=lambda: inventory,
        checkout_verifier=verify,
        worker_factory=factory,
        bus_locks=lock_manager,
        wall_clock=lambda: NOW,
        dependency_commit_marker="a" * 40,
        handle_range_reader=handle_reader,
    )
    return driver, factory, lock_manager, checkout_calls


FULL_INVENTORY = _inventory(
    {
        "SER-RL": "bus-blue",
        "SER-RF": "can7",
        "SER-LL": "can1",
        "SER-LF": "fieldcan",
    }
)


def test_preflight_is_passive_exact_and_reports_stable_runtime_mapping() -> None:
    driver, workers, _, checkout_calls = _driver(FULL_INVENTORY)
    config = _cell()

    result = driver.preflight_setup(config)
    driver.apply_setup(config)
    discovery = driver.discover_setup()

    assert result.ready
    assert result.i2rt_ready is True
    assert checkout_calls == [
        ("/operator/read-only-i2rt", "a" * 40),
        ("/operator/read-only-i2rt", "a" * 40),
    ]
    assert workers.created == []
    assert {arm.arm_id: arm.runtime_interface for arm in result.arms} == {
        "right-leader": "bus-blue",
        "right-follower": "can7",
        "left-leader": "can1",
        "left-follower": "fieldcan",
    }
    assert {item.arm_id: item.stable_identity for item in discovery.resolutions} == {
        arm.logical_id: arm.stable_identity for arm in config.arms
    }
    followers = [arm for arm in result.arms if "follower" in arm.arm_id]
    assert all(arm.frame_map_status == "identity" for arm in followers)
    assert all(arm.soft_limits_status == "missing" for arm in followers)
    assert result.warnings == ["NO SASH GUARD"]


def test_preflight_rejects_mismatched_dependency_commit_marker() -> None:
    driver = YAMCellDriver(
        MockYAMDriver(),
        inventory_provider=lambda: FULL_INVENTORY,
        checkout_verifier=lambda root, commit: I2RTCheckoutIdentity(
            root=root, commit=commit, read_only=True
        ),
        worker_factory=WorkerFactory(),
        bus_locks=FakeLocks(),
        dependency_commit_marker="b" * 40,
    )

    result = driver.preflight_setup(_cell())

    assert not result.ready
    assert result.i2rt_ready is False
    assert "marker does not match" in result.diagnostic.detail


def test_empty_topology_is_a_saveable_checkout_verified_draft_not_connectable() -> None:
    driver, workers, _, _ = _driver(SocketCANInventory(adapters=()))
    config = _cell().model_copy(update={"arms": [], "pair_ports": {}})

    result = driver.preflight_setup(config)

    assert result.ready is False
    assert result.calibration_ready is False
    assert result.i2rt_ready is True
    assert result.arms == []
    assert result.diagnostic.status == "configured"
    assert driver.apply_setup(config).status == "configured"
    assert driver.list_arms() == []
    with pytest.raises(YAMDriverUnavailableError, match="at least one"):
        driver.connect_arms()
    assert workers.created == []


def test_missing_or_down_adapter_is_retryable_but_ambiguity_is_latched_error() -> None:
    missing = _inventory({
        "SER-RL": "bus-blue",
        "SER-RF": "can7",
        "SER-LL": "can1",
    })
    driver, _, _, _ = _driver(missing)

    result = driver.preflight_setup(_cell())
    driver.apply_setup(_cell())

    assert result.ready is False
    assert result.diagnostic.status == "missing"
    assert driver.diagnostic().status == "missing"

    down_inventory = SocketCANInventory(
        adapters=tuple(
            SocketCANAdapter(
                interface=adapter.interface,
                stable_identity=adapter.stable_identity,
                product=adapter.product,
                manufacturer=adapter.manufacturer,
                operstate=(
                    "down" if adapter.stable_identity == "SER-LF" else adapter.operstate
                ),
                link_up=(
                    False if adapter.stable_identity == "SER-LF" else adapter.link_up
                ),
            )
            for adapter in FULL_INVENTORY.adapters
        )
    )
    down_driver, _, _, _ = _driver(down_inventory)
    assert down_driver.preflight_setup(_cell()).diagnostic.status == "missing"

    ambiguous = SocketCANInventory(
        adapters=(
            *FULL_INVENTORY.adapters,
            SocketCANAdapter(
                interface="can9",
                stable_identity="SER-RL",
                product="duplicate fake adapter",
                manufacturer="fake",
                operstate="up",
                link_up=True,
            ),
        )
    )
    ambiguous_driver, _, _, _ = _driver(ambiguous)
    ambiguous_result = ambiguous_driver.preflight_setup(_cell())
    ambiguous_driver.apply_setup(_cell())
    assert ambiguous_result.diagnostic.status == "error"
    assert ambiguous_driver.diagnostic().status == "error"


def test_legacy_mode_discovery_also_inventories_cell_adapters_for_migration() -> None:
    legacy_config = LegacyYAMSetupConfig(
        can_interface="can0",
        leader_port="/dev/serial/by-id/legacy-leader",
        mujoco_xml_path="/opt/legacy/yam.xml",
        leader_calibration_id="legacy-leader",
        leader_calibration_dir="/var/lib/legacy/calibration",
    )

    class LegacyDriver(MockYAMDriver):
        def setup_config(self) -> LegacyYAMSetupConfig:
            return legacy_config

        def discover_setup(self) -> YAMDiscoveryResult:
            return YAMDiscoveryResult(
                mode="hardware",
                can_interfaces=[YAMDiscoveryCandidate(id="can0", label="Legacy CAN")],
                leader_ports=[
                    YAMDiscoveryCandidate(
                        id=legacy_config.leader_port, label="Legacy leader"
                    )
                ],
                suggested_config=legacy_config,
                detail="Legacy passive discovery.",
            )

    inventory_calls = 0

    def inventory() -> SocketCANInventory:
        nonlocal inventory_calls
        inventory_calls += 1
        return FULL_INVENTORY

    driver = YAMCellDriver(LegacyDriver(), inventory_provider=inventory)
    result = driver.discover_setup()

    assert inventory_calls == 1
    assert {device.stable_identity for device in result.devices} == set(
        FULL_INVENTORY.adapters[index].stable_identity
        for index in range(len(FULL_INVENTORY.adapters))
    )
    assert [candidate.id for candidate in result.can_interfaces] == ["can0"]
    assert [candidate.id for candidate in result.leader_ports] == [
        legacy_config.leader_port
    ]
    assert result.suggested_config == legacy_config


def test_cell_auto_restore_rechecks_missing_hotplug_but_never_retries_runtime_fault() -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    inventory = [FULL_INVENTORY]

    def build_driver(factory: WorkerFactory) -> YAMCellDriver:
        return YAMCellDriver(
            MockYAMDriver(initially_connected=False),
            inventory_provider=lambda: inventory[0],
            checkout_verifier=lambda root, commit: I2RTCheckoutIdentity(
                root=root, commit=commit, read_only=True
            ),
            worker_factory=factory,
            bus_locks=FakeLocks(),
            dependency_commit_marker="a" * 40,
        )

    initial_driver = build_driver(WorkerFactory())
    initial_manager = YAMSetupManager(
        driver=initial_driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=sessions,
    )
    with sessions() as db:
        initial_manager.save(
            db,
            config=_cell(),
            auto_restore=True,
            acknowledge_automatic_motion_risk=True,
            acknowledge_gripper_calibration_motion=True,
        )

    inventory[0] = SocketCANInventory(adapters=())
    workers = WorkerFactory()
    restored_driver = build_driver(workers)
    manager = YAMSetupManager(
        driver=restored_driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=sessions,
    )

    manager._restore_boot_sync()
    assert restored_driver.diagnostic().status == "missing"
    assert workers.created == []

    inventory[0] = FULL_INVENTORY
    manager._restore_missing_once()
    assert restored_driver.diagnostic().status == "connected"
    assert len(workers.created) == 4

    workers.by_id("right-follower").faulted = True
    assert restored_driver.get_arm("right-follower").control_state == "error"
    created_after_fault = len(workers.created)
    manager._restore_missing_once()
    assert restored_driver.diagnostic().status == "error"
    assert len(workers.created) == created_after_fault
    manager._shutdown_sync()


def test_partial_connect_is_follower_first_and_rolls_back_only_new_workers() -> None:
    factory = WorkerFactory()
    driver, _, locks, _ = _driver(FULL_INVENTORY, factory)
    config = _cell()
    assert driver.apply_setup(config).status == "configured"
    driver.connect_arms(["right-follower"])
    existing = factory.by_id("right-follower")

    factory.fail_ids.add("left-leader")
    with pytest.raises(YAMDriverUnavailableError, match="worker failed"):
        driver.connect_arms(["left-leader", "left-follower"])

    assert [worker.config.logical_id for worker in factory.created] == [
        "right-follower",
        "left-follower",
        "left-leader",
    ]
    assert existing.started
    assert existing.shutdown_calls == 0
    assert factory.by_id("left-follower").shutdown_calls == 1
    assert [lease.released for lease in locks.leases] == [False, True, True]
    assert driver.get_arm("right-follower").connected
    assert not driver.get_arm("left-follower").connected


def test_apply_setup_rejects_live_cell_before_worker_teardown_or_restart() -> None:
    driver, factory, _, _ = _driver(FULL_INVENTORY)
    config = _cell()
    driver.apply_setup(config)
    driver.connect_arms(["right-follower"])
    worker = factory.by_id("right-follower")

    with pytest.raises(YAMDriverUnavailableError, match="Disconnect all current"):
        driver.apply_setup(config.model_copy(update={"name": "changed topology"}))
    with pytest.raises(YAMDriverUnavailableError, match="Disconnect all current"):
        driver.apply_setup(config)

    assert worker.shutdown_calls == 0
    assert worker.started
    assert driver.get_arm("right-follower").connected


def test_pair_mapping_precedes_limits_but_inference_does_not_remap(
    tmp_path: Path,
) -> None:
    frame_map = tmp_path / "map.json"
    frame_map.write_text(
        json.dumps(
            {"sign": [-1, 1, 1, 1, 1, 1], "offset": [0.2, 0, 0, 0, 0, 0]}
        )
    )
    limits = tmp_path / "limits.json"
    limits.write_text(
        json.dumps(
            {
                "serial": "SER-RF",
                "lower": [-0.4, None, None, None, None, None],
                "upper": [0.4, None, None, None, None, None],
            }
        )
    )
    driver, factory, _, _ = _driver(FULL_INVENTORY)
    driver.apply_setup(
        _cell(frame_map_path=str(frame_map), soft_limits_path=str(limits))
    )
    driver.connect_arms(["right-leader", "right-follower"])
    follower = factory.by_id("right-follower")

    leader_action = ArmAction(
        timestamp=NOW,
        joint_positions_radians={name: (1.0 if index == 0 else 0.0) for index, name in enumerate(JOINT_NAMES)},
        gripper_position=0.7,
    )
    prepared = driver.prepare_teleop_action(
        "right-leader", "right-follower", leader_action
    )
    assert prepared.joint_positions_radians[JOINT_NAMES[0]] == -0.4
    driver.apply_action("right-follower", prepared)
    assert follower.commands[-1][0] == -0.4
    assert follower.commands[-1][6] == 0.7

    follower_frame = ArmAction(
        timestamp=NOW,
        joint_positions_radians={name: (0.3 if index == 0 else 0.0) for index, name in enumerate(JOINT_NAMES)},
        gripper_position=0.6,
    )
    driver.apply_action("right-follower", follower_frame)
    assert follower.commands[-1][0] == 0.3


def test_one_shot_jog_is_rejected_and_action_underflow_safe_idles_before_watchdog() -> None:
    driver, factory, _, _ = _driver(FULL_INVENTORY)
    driver.apply_setup(_cell())
    driver.connect_arms(["right-follower"])
    follower = factory.by_id("right-follower")
    with pytest.raises(Exception, match="continuous fresh command stream"):
        driver.jog("right-follower", object())  # type: ignore[arg-type]
    driver.apply_action(
        "right-follower",
        ArmAction(
            timestamp=NOW,
            joint_positions_radians={name: 0.0 for name in JOINT_NAMES},
            gripper_position=0.5,
        ),
    )

    time.sleep(0.30)

    assert not follower.accepting
    assert follower.safe_idle_calls >= 1
    assert driver.get_arm("right-follower").connected
    assert driver.diagnostic().status != "error"


def test_worker_fault_latches_exact_pair_safe_idles_counterpart_and_never_respawns() -> None:
    driver, factory, _, _ = _driver(FULL_INVENTORY)
    driver.apply_setup(_cell())
    driver.connect_arms(["right-leader", "right-follower", "left-follower"])
    leader = factory.by_id("right-leader")
    right_follower = factory.by_id("right-follower")
    left_follower = factory.by_id("left-follower")
    leader.faulted = True

    snapshots = {arm.id: arm for arm in driver.list_arms()}

    assert snapshots["right-leader"].control_state == "error"
    assert "private vendor failure" not in driver.diagnostic().detail
    assert right_follower.safe_idle_calls >= 1
    assert left_follower.safe_idle_calls == 0
    with pytest.raises(YAMDriverUnavailableError, match="latched"):
        driver.apply_action(
            "right-follower",
            ArmAction(
                timestamp=NOW,
                joint_positions_radians={name: 0.0 for name in JOINT_NAMES},
                gripper_position=0.5,
            ),
        )
    before = len(factory.created)
    with pytest.raises(YAMDriverUnavailableError, match="latched"):
        driver.connect_arms(["right-leader"])
    assert len(factory.created) == before


def test_uncertain_teardown_retains_lock_and_blocks_reconnect() -> None:
    factory = WorkerFactory()
    factory.uncertain_ids.add("right-follower")
    driver, _, locks, _ = _driver(FULL_INVENTORY, factory)
    driver.apply_setup(_cell())
    driver.connect_arms(["right-follower"])

    with pytest.raises(YAMDriverUnavailableError, match="buses remain blocked"):
        driver.disconnect_arms(["right-follower"])

    assert not locks.leases[0].released
    blocked = driver.get_arm("right-follower")
    assert blocked.control_state == "error"
    assert blocked.energized and blocked.holding
    assert blocked.can is not None and blocked.can.state == "warning"
    before = len(factory.created)
    with pytest.raises(YAMDriverUnavailableError, match="latched"):
        driver.connect_arms(["right-follower"])
    assert len(factory.created) == before


def test_worker_factory_exception_releases_bus_lock() -> None:
    class RaisingFactory(WorkerFactory):
        def __call__(self, config: I2RTArmWorkerConfig) -> FakeWorker:
            raise RuntimeError("private construction failure")

    locks = FakeLocks()
    driver, _, _, _ = _driver(FULL_INVENTORY, RaisingFactory(), locks)
    driver.apply_setup(_cell())
    with pytest.raises(YAMDriverUnavailableError, match="constructed safely"):
        driver.connect_arms(["right-follower"])
    assert locks.leases[0].released


def test_external_fault_boundary_revokes_writes_and_requires_explicit_disconnect() -> None:
    driver, factory, _, _ = _driver(FULL_INVENTORY)
    driver.apply_setup(_cell())
    driver.connect_arms(["right-leader", "right-follower"])
    follower = factory.by_id("right-follower")
    follower.enable_commands()

    driver.latch_fault(["right-follower"], "/private/provider/error")

    assert not follower.accepting
    assert follower.safe_idle_calls >= 1
    assert "private" not in driver.diagnostic().detail
    with pytest.raises(YAMDriverUnavailableError, match="latched"):
        driver.connect_arms(["right-follower"])
    driver.disconnect_arms(["right-follower"])
    driver.connect_arms(["right-follower"])
    assert len(
        [worker for worker in factory.created if worker.config.logical_id == "right-follower"]
    ) == 2


def test_cross_pair_route_is_rejected_before_any_write() -> None:
    driver, factory, _, _ = _driver(FULL_INVENTORY)
    driver.apply_setup(_cell())
    driver.connect_arms(["right-follower", "left-leader"])
    action = ArmAction(
        timestamp=NOW,
        joint_positions_radians={name: 0.0 for name in JOINT_NAMES},
        gripper_position=0.5,
    )

    with pytest.raises(ValueError, match="exact declared pair"):
        driver.prepare_teleop_action("left-leader", "right-follower", action)

    assert factory.by_id("right-follower").commands == []


def test_handle_check_is_disconnected_handle_only_thresholded_and_releases_bus() -> None:
    reader = FakeHandleReader(
        HandleDiagnosticSample(True, 0.15, 0.85, (False, True), 10)
    )
    driver, workers, locks, _ = _driver(
        FULL_INVENTORY, handle_reader=reader
    )
    driver.apply_setup(_cell())

    result = driver.check_handle_range("right-leader", duration_seconds=0.1)

    assert result.healthy
    assert workers.created == []
    assert reader.configs[0].runtime_interface == "bus-blue"
    assert locks.leases[-1].released
    assert driver.get_arm("right-leader").handle.range_status == "healthy"  # type: ignore[union-attr]
    reader.result = HandleDiagnosticSample(True, 0.151, 1.0, (), 10)
    assert not driver.check_handle_range(
        "right-leader", duration_seconds=0.1
    ).healthy
    driver.connect_arms(["right-leader"])
    with pytest.raises(YAMDriverUnavailableError, match="Disconnect"):
        driver.check_handle_range("right-leader", duration_seconds=0.1)


def test_unreaped_handle_child_latches_arm_and_retains_bus_lease_until_restart() -> None:
    class UncertainReader:
        def sample(self, config: HandleDiagnosticConfig) -> HandleDiagnosticSample:
            del config
            raise YAMHandleDiagnosticTeardownUncertainError("private child state")

    driver, workers, locks, _ = _driver(
        FULL_INVENTORY, handle_reader=UncertainReader()  # type: ignore[arg-type]
    )
    driver.apply_setup(_cell())

    with pytest.raises(YAMDriverUnavailableError, match="remains blocked"):
        driver.check_handle_range("right-leader", duration_seconds=0.1)

    assert workers.created == []
    assert not locks.leases[-1].released
    assert driver.diagnostic().status == "error"
    assert driver.get_arm("right-leader").control_state == "error"
    with pytest.raises(YAMDriverUnavailableError, match="still owns"):
        driver.connect_arms(["right-leader"])
    with pytest.raises(YAMDriverUnavailableError, match="uncertain"):
        driver.disconnect_arms(["right-leader"])
    assert not locks.leases[-1].released


def test_handle_wire_protocol_only_requests_50e_reports_50f_and_closes() -> None:
    class Response:
        def __init__(self, radians: float) -> None:
            counts = round(radians * 4096.0 / (2.0 * math.pi))
            self.arbitration_id = HANDLE_REPORT_ID
            self.is_extended_id = False
            self.data = struct.pack("!BhhB", 0xFF, counts, 0, 0x02)

    class Bus:
        def __init__(self) -> None:
            self.sent: list[tuple[int, bytes]] = []
            self.responses = [Response(0.0), Response(0.7), Response(0.0)]
            self.filters: list[dict[str, int]] = []
            self.closed = False

        def set_filters(self, filters: list[dict[str, int]]) -> None:
            self.filters = filters

        def send(self, message: tuple[int, bytes]) -> None:
            self.sent.append(message)

        def recv(self, timeout: float) -> Response | None:
            del timeout
            return self.responses.pop(0) if self.responses else None

        def shutdown(self) -> None:
            self.closed = True

    bus = Bus()
    result = sample_handle_requests_only(
        HandleDiagnosticConfig(
            "right-leader", "SER-RL", "bus-blue", 0.1, sample_frequency_hz=50.0
        ),
        bus_factory=lambda interface: (
            bus,
            lambda arbitration_id, payload: (arbitration_id, payload),
        ),
    )

    assert result.observed_minimum == 0.0
    assert result.observed_maximum is not None and result.observed_maximum >= 0.99
    assert set(bus.sent) == {(HANDLE_REQUEST_ID, HANDLE_REPORT_REQUEST)}
    assert bus.filters == [{"can_id": HANDLE_REPORT_ID, "can_mask": 0x7FF}]
    assert bus.closed


def test_handle_supervisor_timeout_terminates_kills_and_reaps() -> None:
    class Process:
        def __init__(self) -> None:
            self.alive = True
            self.exitcode: int | None = None
            self.terminated = False
            self.killed = False
            self.joins = 0

        def start(self) -> None:
            return None

        def join(self, timeout: float) -> None:
            del timeout
            self.joins += 1

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.alive = False
            self.exitcode = -9

    process = Process()

    class Context:
        def Queue(self, maxsize: int) -> queue.Queue[object]:
            return queue.Queue(maxsize=maxsize)

        def Process(self, **kwargs: object) -> Process:
            del kwargs
            return process

    reader = SupervisedHandleRangeReader(
        context=Context(), teardown_grace_seconds=0.05  # type: ignore[arg-type]
    )
    with pytest.raises(YAMHandleDiagnosticError):
        reader.sample(HandleDiagnosticConfig("leader", "SER", "can0", 0.1))
    assert process.terminated and process.killed
    assert not process.alive and process.exitcode == -9 and process.joins == 3


def test_handle_supervisor_reports_unreaped_child_as_teardown_uncertainty() -> None:
    class Process:
        exitcode: int | None = None

        def start(self) -> None:
            return None

        def join(self, timeout: float) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    class Context:
        def Queue(self, maxsize: int) -> queue.Queue[object]:
            return queue.Queue(maxsize=maxsize)

        def Process(self, **kwargs: object) -> Process:
            del kwargs
            return Process()

    reader = SupervisedHandleRangeReader(
        context=Context(), teardown_grace_seconds=0.05  # type: ignore[arg-type]
    )
    with pytest.raises(YAMHandleDiagnosticTeardownUncertainError):
        reader.sample(HandleDiagnosticConfig("leader", "SER", "can0", 0.1))


def test_handle_supervisor_unexpected_post_start_error_still_kills_or_latches() -> None:
    class Process:
        exitcode: int | None = None

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def start(self) -> None:
            return None

        def join(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("private interrupted join")

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = Process()

    class Context:
        def Queue(self, maxsize: int) -> queue.Queue[object]:
            return queue.Queue(maxsize=maxsize)

        def Process(self, **kwargs: object) -> Process:
            del kwargs
            return process

    reader = SupervisedHandleRangeReader(
        context=Context(), teardown_grace_seconds=0.05  # type: ignore[arg-type]
    )
    with pytest.raises(YAMHandleDiagnosticTeardownUncertainError):
        reader.sample(HandleDiagnosticConfig("leader", "SER", "can0", 0.1))
    assert process.terminated and process.killed


def test_handle_child_parent_death_signal_closes_fork_prctl_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LibC:
        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def prctl(self, *args: int) -> int:
            self.calls.append(args)
            return 0

    libc = LibC()
    parent_ids = iter((100, 101))
    monkeypatch.setattr(handle_diagnostic.sys, "platform", "linux")
    monkeypatch.setattr(handle_diagnostic.os, "getppid", lambda: next(parent_ids))
    monkeypatch.setattr(handle_diagnostic.ctypes, "CDLL", lambda *a, **k: libc)

    with pytest.raises(RuntimeError, match="supervisor exited"):
        handle_diagnostic._set_parent_death_signal(100)

    assert libc.calls == [(1, handle_diagnostic.signal.SIGTERM, 0, 0, 0)]


def test_handle_child_rejects_an_already_orphaned_start_before_prctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LibC:
        def prctl(self, *args: int) -> int:
            raise AssertionError(f"prctl must not run for orphaned child: {args}")

    monkeypatch.setattr(handle_diagnostic.sys, "platform", "linux")
    monkeypatch.setattr(handle_diagnostic.os, "getppid", lambda: 1)
    monkeypatch.setattr(handle_diagnostic.ctypes, "CDLL", lambda *a, **k: LibC())

    with pytest.raises(RuntimeError, match="before child start"):
        handle_diagnostic._set_parent_death_signal(1234)
