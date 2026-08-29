from __future__ import annotations

from collections.abc import Generator

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ctrl_pi.api.yam_setup import get_optional_setup_db
from ctrl_pi.db import Base, get_db
from ctrl_pi.drivers.mock_yam import MOCK_CELL_CONFIG, MockYAMDriver
from ctrl_pi.drivers.yam import (
    YAMCellConfig,
    YAMDriver,
    YAMDriverDiagnostic,
    YAMPreflightResult,
)
from ctrl_pi.drivers.yam_cell import YAMCellDriver
from ctrl_pi.drivers.yam_cell_discovery import SocketCANInventory
from ctrl_pi.drivers.yam_i2rt_worker import I2RTCheckoutIdentity
from ctrl_pi.main import create_app
from ctrl_pi.models import YAMCell, YAMCellArm, YAMSetup
from ctrl_pi.rig import RigLease
from ctrl_pi.yam_setup import YAMSetupManager


CRANK_CELL_CONFIG = MOCK_CELL_CONFIG.model_copy(
    update={
        "name": "Mock crank 4310 cell",
        "arms": [
            arm.model_copy(update={"end_effector_kind": "crank_4310"})
            if arm.role == "follower"
            else arm
            for arm in MOCK_CELL_CONFIG.arms
        ],
    },
    deep=True,
)


class CrankMockYAMDriver(MockYAMDriver):
    def __init__(self) -> None:
        super().__init__(initially_connected=False)
        self._setup_config = CRANK_CELL_CONFIG
        self.connect_calls = 0

    def preflight_setup(self, config: YAMCellConfig) -> YAMPreflightResult:
        ready = config == CRANK_CELL_CONFIG
        return YAMPreflightResult(
            ready=ready,
            calibration_ready=ready,
            diagnostic=YAMDriverDiagnostic(
                status="configured" if ready else "error",
                detail=(
                    "Mock crank cell passed passive preflight."
                    if ready
                    else "Mock crank cell configuration did not match."
                ),
            ),
            i2rt_ready=True,
        )

    def apply_setup(self, config: YAMCellConfig) -> YAMDriverDiagnostic:
        result = self.preflight_setup(config)
        if result.ready:
            self._setup_config = config.model_copy(deep=True)
        return result.diagnostic

    def connect_arms(self, arm_ids=None):  # type: ignore[no-untyped-def]
        self.connect_calls += 1
        return super().connect_arms(arm_ids)


def _database():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _client(
    *, initially_connected: bool = False, driver: YAMDriver | None = None
):
    engine, factory = _database()
    driver = driver or MockYAMDriver(initially_connected=initially_connected)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)

    def db_override() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_optional_setup_db] = db_override
    return TestClient(app), engine, driver


def _save(client: TestClient):
    return client.put(
        "/api/yam/cell",
        json={"config": MOCK_CELL_CONFIG.model_dump(), "auto_restore": False},
    )


def test_cell_save_normalizes_arms_and_supersedes_legacy_setup() -> None:
    client, engine, _ = _client()
    with Session(engine) as db:
        db.add(
            YAMSetup(
                id="primary",
                mode="hardware",
                can_interface="can0",
                leader_port="/dev/serial/by-id/legacy",
                mujoco_xml_path="/opt/legacy.xml",
                leader_calibration_id="legacy",
                leader_calibration_dir="/var/lib/legacy",
                auto_restore=False,
            )
        )
        db.commit()

    response = _save(client)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["config"]["kind"] == "cell"
    assert body["configured_arm_count"] == 4
    assert body["saved"] is True
    with Session(engine) as db:
        assert db.get(YAMSetup, "primary") is None
        assert db.get(YAMCell, "primary") is not None
        arms = db.scalars(
            select(YAMCellArm).order_by(YAMCellArm.position)
        ).all()
        assert [arm.logical_id for arm in arms] == [
            "yam-leader",
            "yam-follower",
            "yam-leader-left",
            "yam-follower-left",
        ]
        assert not any(hasattr(arm, "runtime_interface") for arm in arms)


def test_cell_save_rejects_connected_topology_before_any_driver_lifecycle_or_db_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, engine, driver = _client(initially_connected=True)
    calls: list[str] = []
    for method_name in ("preflight_setup", "apply_setup", "disconnect_arms", "connect_arms"):
        original = getattr(driver, method_name)

        def tracked(*args, _name=method_name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(driver, method_name, tracked)

    response = _save(client)

    assert response.status_code == 409
    assert "Disconnect all current" in response.text
    assert calls == []
    assert all(arm.connected for arm in driver.list_arms())
    with Session(engine) as db:
        assert db.get(YAMCell, "primary") is None


def test_connected_cell_can_revoke_auto_restore_without_preflight_or_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, engine, driver = _client(initially_connected=False)
    enabled = client.put(
        "/api/yam/cell",
        json={
            "config": MOCK_CELL_CONFIG.model_dump(),
            "auto_restore": True,
            "acknowledge_automatic_motion_risk": True,
            "acknowledge_gripper_calibration_motion": True,
        },
    )
    assert enabled.status_code == 200, enabled.text
    connected = client.post(
        "/api/yam/cell/connect",
        json={
            "acknowledge_hardware_motion_risk": True,
            "acknowledge_gripper_calibration_motion": True,
        },
    )
    assert connected.status_code == 200, connected.text

    calls: list[str] = []
    for method_name in ("preflight_setup", "apply_setup", "disconnect_arms", "connect_arms"):
        original = getattr(driver, method_name)

        def tracked(*args, _name=method_name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(driver, method_name, tracked)

    revoked = _save(client)

    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["auto_restore"] is False
    assert revoked.json()["all_connected"] is True
    assert calls == []
    with Session(engine) as db:
        row = db.get(YAMCell, "primary")
        assert row is not None and row.auto_restore is False


def test_empty_cell_draft_saves_with_verified_checkout_but_cannot_connect_or_restore() -> None:
    empty = MOCK_CELL_CONFIG.model_copy(update={"arms": [], "pair_ports": {}})
    driver = YAMCellDriver(
        MockYAMDriver(initially_connected=False),
        inventory_provider=lambda: SocketCANInventory(adapters=()),
        checkout_verifier=lambda root, commit: I2RTCheckoutIdentity(
            root=root, commit=commit, read_only=True
        ),
        dependency_commit_marker=empty.i2rt_commit,
    )
    client, engine, _ = _client(driver=driver)

    preflight = client.post(
        "/api/yam/cell/preflight", json={"config": empty.model_dump()}
    )
    saved = client.put(
        "/api/yam/cell", json={"config": empty.model_dump(), "auto_restore": False}
    )
    auto_restore = client.put(
        "/api/yam/cell", json={"config": empty.model_dump(), "auto_restore": True}
    )
    connect = client.post(
        "/api/yam/cell/connect",
        json={"acknowledge_hardware_motion_risk": True},
    )

    assert preflight.status_code == 200
    assert preflight.json()["ready"] is False
    assert preflight.json()["calibration_ready"] is False
    assert preflight.json()["i2rt_ready"] is True
    assert saved.status_code == 200, saved.text
    assert saved.json()["state"] == "needs_setup"
    assert saved.json()["saved"] is True
    assert saved.json()["configured_arm_count"] == 0
    assert saved.json()["calibration_ready"] is False
    assert auto_restore.status_code == 409
    assert connect.status_code == 409
    with Session(engine) as db:
        row = db.get(YAMCell, "primary")
        assert row is not None and row.arms == [] and row.auto_restore is False


def test_selected_connect_requires_linear_ack_and_disconnect_is_explicit() -> None:
    client, _, driver = _client()
    assert _save(client).status_code == 200

    leader = client.post(
        "/api/yam/cell/connect",
        json={
            "arm_ids": ["yam-leader"],
            "acknowledge_hardware_motion_risk": True,
        },
    )
    assert leader.status_code == 200, leader.text
    assert leader.json()["connected_arm_count"] == 1
    assert leader.json()["all_connected"] is False

    rejected = client.post(
        "/api/yam/cell/connect",
        json={
            "arm_ids": ["yam-follower"],
            "acknowledge_hardware_motion_risk": True,
        },
    )
    assert rejected.status_code == 409
    assert "linear_4310" in rejected.text
    assert driver.get_arm("yam-follower").connected is False

    follower = client.post(
        "/api/yam/cell/connect",
        json={
            "arm_ids": ["yam-follower"],
            "acknowledge_hardware_motion_risk": True,
            "acknowledge_gripper_calibration_motion": True,
        },
    )
    assert follower.status_code == 200, follower.text
    follower_state = next(
        arm for arm in follower.json()["arms"] if arm["arm_id"] == "yam-follower"
    )
    assert follower_state["energized"] is True
    assert follower_state["holding"] is True

    disconnected = client.post(
        "/api/yam/cell/disconnect", json={"arm_ids": ["yam-follower"]}
    )
    assert disconnected.status_code == 200, disconnected.text
    follower_state = next(
        arm
        for arm in disconnected.json()["arms"]
        if arm["arm_id"] == "yam-follower"
    )
    assert follower_state["connected"] is False
    assert follower_state["energized"] is False
    assert follower_state["control_state"] == "disconnected"


def test_crank_4310_requires_same_calibration_consent_for_save_connect_and_restore() -> None:
    driver = CrankMockYAMDriver()
    client, engine, _ = _client(driver=driver)
    base_payload = {
        "config": CRANK_CELL_CONFIG.model_dump(),
        "auto_restore": True,
        "acknowledge_automatic_motion_risk": True,
    }

    rejected_save = client.put("/api/yam/cell", json=base_payload)
    accepted_save = client.put(
        "/api/yam/cell",
        json={
            **base_payload,
            "acknowledge_gripper_calibration_motion": True,
        },
    )

    assert rejected_save.status_code == 409
    assert "crank_4310" in rejected_save.text
    assert accepted_save.status_code == 200, accepted_save.text
    assert accepted_save.json()["auto_restore"] is True

    rejected_connect = client.post(
        "/api/yam/cell/connect",
        json={
            "arm_ids": ["yam-follower"],
            "acknowledge_hardware_motion_risk": True,
        },
    )
    assert rejected_connect.status_code == 409
    assert "yam-follower (crank_4310)" in rejected_connect.text
    assert driver.connect_calls == 0

    connected = client.post(
        "/api/yam/cell/connect",
        json={
            "arm_ids": ["yam-follower"],
            "acknowledge_hardware_motion_risk": True,
            "acknowledge_gripper_calibration_motion": True,
        },
    )
    assert connected.status_code == 200, connected.text
    assert driver.connect_calls == 1

    restored_driver = CrankMockYAMDriver()
    restore_manager = YAMSetupManager(
        driver=restored_driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
    )
    restore_manager._restore_boot_sync()

    assert restored_driver.connect_calls == 1
    assert all(arm.connected for arm in restored_driver.list_arms())


def test_reset_commit_failure_restores_topology_but_never_reconnects_without_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResetTrackingDriver(MockYAMDriver):
        def __init__(self) -> None:
            super().__init__(initially_connected=False)
            self.events: list[str] = []

        def apply_setup(self, config):  # type: ignore[no-untyped-def]
            self.events.append("apply")
            return super().apply_setup(config)

        def connect_arms(self, arm_ids=None):  # type: ignore[no-untyped-def]
            self.events.append("connect")
            return super().connect_arms(arm_ids)

        def disconnect_arms(self, arm_ids=None):  # type: ignore[no-untyped-def]
            self.events.append("disconnect")
            return super().disconnect_arms(arm_ids)

        def reset_setup(self):  # type: ignore[no-untyped-def]
            self.events.append("reset")
            super().disconnect_arms()
            return super().reset_setup()

    engine, factory = _database()
    driver = ResetTrackingDriver()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    with factory() as db:
        manager.save(db, config=MOCK_CELL_CONFIG, auto_restore=False)
        manager.connect(
            db,
            acknowledge_hardware_motion_risk=True,
            acknowledge_gripper_calibration_motion=True,
        )
    assert all(arm.connected for arm in driver.list_arms())
    driver.events.clear()

    with factory() as db:
        monkeypatch.setattr(
            db,
            "commit",
            lambda: (_ for _ in ()).throw(SQLAlchemyError("private reset commit")),
        )
        with pytest.raises(SQLAlchemyError, match="reset commit"):
            manager.reset(db)

    assert driver.setup_config() == MOCK_CELL_CONFIG
    assert not any(arm.connected for arm in driver.list_arms())
    assert "reset" in driver.events
    assert "apply" in driver.events
    assert "connect" not in driver.events
    with Session(engine) as db:
        assert db.get(YAMCell, "primary") is not None


def test_handle_range_check_is_separate_from_passive_preflight() -> None:
    client, _, _ = _client()
    assert _save(client).status_code == 200
    preflight = client.post(
        "/api/yam/cell/preflight", json={"config": MOCK_CELL_CONFIG.model_dump()}
    )
    assert preflight.status_code == 200
    leader = next(
        arm for arm in preflight.json()["arms"] if arm["arm_id"] == "yam-leader"
    )
    assert leader["handle_status"] == "not_checked"

    checked = client.post(
        "/api/yam/cell/handle-check",
        json={
            "arm_id": "yam-leader",
            "duration_seconds": 1,
            "acknowledge_active_can_diagnostic": True,
        },
    )
    assert checked.status_code == 200, checked.text
    assert checked.json()["healthy"] is True
    assert checked.json()["observed_minimum"] <= 0.15
    assert checked.json()["observed_maximum"] >= 0.85


def test_mock_mode_cell_alias_never_mutates_physical_rows() -> None:
    engine, factory = _database()
    with factory() as db:
        db.add(
            YAMSetup(
                id="primary",
                mode="hardware",
                can_interface="can7",
                leader_port="/dev/serial/by-id/physical",
                mujoco_xml_path="/opt/physical.xml",
                leader_calibration_id="physical",
                leader_calibration_dir="/var/lib/physical",
                auto_restore=True,
            )
        )
        db.commit()
    driver = MockYAMDriver()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=True,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)

    def db_override() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_optional_setup_db] = db_override
    with TestClient(app) as client:
        assert _save(client).status_code == 200
        assert client.delete("/api/yam/cell").status_code == 200

    with factory() as db:
        row = db.get(YAMSetup, "primary")
        assert row is not None
        assert row.can_interface == "can7"
        assert db.get(YAMCell, "primary") is None
