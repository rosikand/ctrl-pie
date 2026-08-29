from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import UTC, datetime
import threading
import time
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ctrl_pi.api.yam_setup import get_optional_setup_db
from ctrl_pi.db import Base, get_db
from ctrl_pi.drivers.mock_yam import MOCK_SETUP_CONFIG, MockYAMDriver
from ctrl_pi.drivers.yam import (
    YAMDiscoveryCandidate,
    YAMDiscoveryResult,
    YAMDriverDiagnostic,
    YAMPreflightResult,
    YAMSetupConfig,
)
from ctrl_pi.main import create_app
from ctrl_pi.models import YAMSetup
from ctrl_pi.rig import RigLease
from ctrl_pi.yam_setup import YAMSetupConnectError, YAMSetupManager


HARDWARE_CONFIG = YAMSetupConfig(
    can_interface="can0",
    leader_port="/dev/serial/by-id/ctrl-pi-test-leader",
    mujoco_xml_path="/opt/ctrl-pi/yam.xml",
    leader_calibration_id="yam-leader",
    leader_calibration_dir="/var/lib/ctrl-pi/calibration",
)
REPLACEMENT_CONFIG = HARDWARE_CONFIG.model_copy(
    update={"mujoco_xml_path": "/opt/ctrl-pi/yam-v2.xml"}
)


class SetupDriver(MockYAMDriver):
    """Hardware-mode lifecycle fake; it keeps the real public driver contract."""

    def __init__(
        self,
        *,
        visible: bool,
        fail_connect: bool = False,
        failure_status: Literal["missing", "error"] = "error",
        raise_connect: bool = False,
    ) -> None:
        super().__init__()
        self.visible = visible
        self.fail_connect = fail_connect
        self.failure_status = failure_status
        self.raise_connect = raise_connect
        self.current = HARDWARE_CONFIG
        self.events: list[str] = []
        self.start_calls = 0
        self._diagnostic = YAMDriverDiagnostic(
            status="configured" if visible else "missing",
            detail="Test hardware is ready." if visible else "Test hardware is absent.",
        )

    def setup_config(self) -> YAMSetupConfig:
        return self.current

    def discover_setup(self) -> YAMDiscoveryResult:
        return YAMDiscoveryResult(
            mode="hardware",
            can_interfaces=(
                [YAMDiscoveryCandidate(id="can0", label="SocketCAN interface can0")]
                if self.visible
                else []
            ),
            leader_ports=(
                [
                    YAMDiscoveryCandidate(
                        id=HARDWARE_CONFIG.leader_port,
                        label="Stable serial device 1",
                    )
                ]
                if self.visible
                else []
            ),
            suggested_config=HARDWARE_CONFIG,
            detail="Passive fake discovery.",
        )

    def preflight_setup(self, config: YAMSetupConfig) -> YAMPreflightResult:
        assert config.can_interface == HARDWARE_CONFIG.can_interface
        assert config.leader_port == HARDWARE_CONFIG.leader_port
        diagnostic = YAMDriverDiagnostic(
            status="configured" if self.visible else "missing",
            detail="Test hardware is ready." if self.visible else "Test hardware is absent.",
        )
        return YAMPreflightResult(
            ready=self.visible,
            calibration_ready=True,
            diagnostic=diagnostic,
        )

    def apply_setup(self, config: YAMSetupConfig) -> YAMDriverDiagnostic:
        self.events.append("apply")
        self.current = config
        self._diagnostic = self.preflight_setup(config).diagnostic
        return self._diagnostic

    def reset_setup(self) -> YAMDriverDiagnostic:
        self.events.append("reset")
        self._diagnostic = YAMDriverDiagnostic(
            status="missing", detail="No physical setup is selected."
        )
        return self._diagnostic

    def startup(self) -> None:
        self.events.append("startup")
        self.start_calls += 1
        if self.raise_connect:
            self._diagnostic = YAMDriverDiagnostic(
                status="missing", detail="Test startup raised after opening resources."
            )
            raise RuntimeError("private vendor startup payload")
        if self.fail_connect:
            self._diagnostic = YAMDriverDiagnostic(
                status=self.failure_status,
                detail="Test hardware connection failed safely.",
            )
        elif self.visible:
            self._diagnostic = YAMDriverDiagnostic(
                status="connected", detail="Test hardware is connected."
            )

    def shutdown(self) -> None:
        self.events.append("shutdown")
        if self._diagnostic.status == "connected":
            self._diagnostic = YAMDriverDiagnostic(
                status="configured", detail="Test hardware is stopped."
            )

    def diagnostic(self) -> YAMDriverDiagnostic:
        return self._diagnostic

    def list_arms(self):
        connected = self._diagnostic.status == "connected"
        return [
            arm.model_copy(
                update={"connected": connected, "timestamp": datetime.now(UTC)},
                deep=True,
            )
            for arm in super().list_arms()
        ]


@pytest.fixture
def setup_database():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _persist_hardware(factory, *, auto_restore: bool) -> None:
    with factory() as db:
        db.add(
            YAMSetup(
                id="primary",
                mode="hardware",
                **HARDWARE_CONFIG.model_dump(),
                auto_restore=auto_restore,
            )
        )
        db.commit()


def _database_overrides(app, factory) -> None:
    def database() -> Generator[Session, None, None]:
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    app.dependency_overrides[get_optional_setup_db] = database


def test_mock_setup_endpoints_preserve_physical_row_and_arm_behavior(
    setup_database,
) -> None:
    engine, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = MockYAMDriver()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=True,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)
    _database_overrides(app, factory)

    with TestClient(app) as client:
        initial = client.get("/api/yam/setup")
        discovery = client.post("/api/yam/setup/discover")
        saved = client.put(
            "/api/yam/setup",
            json={"config": MOCK_SETUP_CONFIG.model_dump(), "auto_restore": True},
        )
        reset = client.delete("/api/yam/setup")
        before = client.get("/api/arms/yam-follower").json()
        jogged = client.post(
            "/api/arms/yam-follower/jog",
            json={"kind": "joint", "axis": "shoulder_yaw", "delta": 0.1},
        )

    assert initial.status_code == discovery.status_code == 200
    assert saved.status_code == reset.status_code == jogged.status_code == 200
    assert initial.json()["state"] == "ready"
    assert initial.json()["saved"] is False
    assert initial.json()["requires_physical_validation"] is False
    assert discovery.json()["mode"] == "mock"
    assert jogged.json()["joints"][0]["position_radians"] == pytest.approx(
        before["joints"][0]["position_radians"] + 0.1
    )
    with Session(engine) as db:
        physical = db.get(YAMSetup, "primary")
        assert physical is not None
        assert physical.mode == "hardware"
        assert physical.can_interface == "can0"
        assert physical.auto_restore is True


def test_setup_status_remains_renderable_before_postgres_is_configured() -> None:
    driver = MockYAMDriver()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=True,
        session_factory=None,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)
    app.dependency_overrides[get_optional_setup_db] = lambda: None

    with TestClient(app) as client:
        response = client.get("/api/yam/setup")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["state"] == "ready"
    assert response.json()["saved"] is False


@pytest.mark.parametrize("with_database", [False, True])
def test_hardware_boot_never_opens_env_setup_without_saved_consent(
    setup_database,
    with_database: bool,
) -> None:
    _, factory = setup_database
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory if with_database else None,
        restore_interval_seconds=0.05,
    )

    async def scenario() -> None:
        await manager.startup()
        await asyncio.sleep(0.08)
        assert driver.start_calls == 0
        assert driver.diagnostic().status == "configured"
        await manager.shutdown()

    asyncio.run(scenario())


def test_hardware_api_requires_motion_consent_and_respects_active_rig(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=False)
    driver = SetupDriver(visible=True)
    lease = RigLease()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=lease,
        mock_mode=False,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)
    _database_overrides(app, factory)

    with TestClient(app) as client:
        no_auto_consent = client.put(
            "/api/yam/setup",
            json={"config": HARDWARE_CONFIG.model_dump(), "auto_restore": True},
        )
        no_connect_consent = client.post("/api/yam/setup/connect", json={})
        token = lease.acquire("inference", "deployment-test")
        try:
            busy = client.post(
                "/api/yam/setup/connect",
                json={"acknowledge_hardware_motion_risk": True},
            )
        finally:
            lease.release(token)
        connected = client.post(
            "/api/yam/setup/connect",
            json={"acknowledge_hardware_motion_risk": True},
        )

    assert no_auto_consent.status_code == no_connect_consent.status_code == 409
    assert "risk" in no_auto_consent.json()["detail"]
    assert "risk" in no_connect_consent.json()["detail"]
    assert busy.status_code == 409
    assert "inference" in busy.json()["detail"]
    assert connected.status_code == 200
    assert connected.json()["state"] == "ready"
    assert connected.json()["connected"] is True


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        (
            "PUT",
            "/api/yam/setup",
            {
                "config": HARDWARE_CONFIG.model_dump(),
                "auto_restore": 1,
                "acknowledge_automatic_motion_risk": True,
            },
        ),
        (
            "PUT",
            "/api/yam/setup",
            {
                "config": HARDWARE_CONFIG.model_dump(),
                "auto_restore": "yes",
                "acknowledge_automatic_motion_risk": True,
            },
        ),
        (
            "PUT",
            "/api/yam/setup",
            {
                "config": HARDWARE_CONFIG.model_dump(),
                "auto_restore": True,
                "acknowledge_automatic_motion_risk": 1,
            },
        ),
        (
            "PUT",
            "/api/yam/setup",
            {
                "config": HARDWARE_CONFIG.model_dump(),
                "auto_restore": True,
                "acknowledge_automatic_motion_risk": "yes",
            },
        ),
        (
            "POST",
            "/api/yam/setup/connect",
            {"acknowledge_hardware_motion_risk": 1},
        ),
        (
            "POST",
            "/api/yam/setup/connect",
            {"acknowledge_hardware_motion_risk": "yes"},
        ),
    ],
)
def test_hardware_api_rejects_coerced_safety_booleans_before_driver_access(
    setup_database,
    method: str,
    path: str,
    payload: dict[str, object],
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=False)
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)
    _database_overrides(app, factory)

    def unexpected_manager_call(*_args, **_kwargs):
        raise AssertionError("invalid safety booleans reached the setup manager")

    manager.save = unexpected_manager_call  # type: ignore[method-assign]
    manager.connect = unexpected_manager_call  # type: ignore[method-assign]

    with TestClient(app) as client:
        driver.events.clear()
        driver.start_calls = 0
        response = client.request(method, path, json=payload)
        assert response.status_code == 422
        assert driver.events == []
        assert driver.start_calls == 0


def test_save_commit_failure_restores_previous_connected_setup(
    setup_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory = setup_database
    _persist_hardware(factory, auto_restore=False)
    driver = SetupDriver(visible=True)
    driver.startup()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    sentinel = "postgresql://operator:secret@db/commit-failed"

    with factory() as db:
        monkeypatch.setattr(
            db,
            "commit",
            lambda: (_ for _ in ()).throw(SQLAlchemyError(sentinel)),
        )
        with pytest.raises(SQLAlchemyError, match="commit-failed"):
            manager.save(
                db,
                config=REPLACEMENT_CONFIG,
                auto_restore=False,
            )

    assert driver.setup_config() == HARDWARE_CONFIG
    assert driver.diagnostic().status == "connected"
    with Session(engine) as db:
        row = db.get(YAMSetup, "primary")
        assert row is not None
        assert row.mujoco_xml_path == HARDWARE_CONFIG.mujoco_xml_path
        assert row.auto_restore is False


def test_connect_commit_failure_guarantees_driver_shutdown(
    setup_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=False)
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )

    with factory() as db:
        monkeypatch.setattr(
            db,
            "commit",
            lambda: (_ for _ in ()).throw(SQLAlchemyError("private commit failure")),
        )
        with pytest.raises(SQLAlchemyError):
            manager.connect(db, acknowledge_hardware_motion_risk=True)

    assert driver.diagnostic().status != "connected"
    assert driver.events[-1] == "shutdown"


def test_manual_connect_clears_queued_storage_restore_without_reopening(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    manager._restore_storage_pending = True

    with factory() as db:
        result = manager.connect(db, acknowledge_hardware_motion_risk=True)

    assert result.connected is True
    assert manager._restore_storage_pending is False
    lifecycle_after_connect = list(driver.events)
    manager._restore_storage_once()
    assert driver.events == lifecycle_after_connect


def test_repeated_manual_connect_does_not_cycle_connected_hardware(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=False)
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )

    with factory() as db:
        first = manager.connect(db, acknowledge_hardware_motion_risk=True)
    lifecycle_after_first = list(driver.events)
    with factory() as db:
        second = manager.connect(db, acknowledge_hardware_motion_risk=True)

    assert first.connected is second.connected is True
    assert driver.events == lifecycle_after_first
    assert driver.events.count("startup") == 1


def test_preference_commit_failure_never_restarts_unchanged_connected_driver(
    setup_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=True)
    driver.startup()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    lifecycle_before = list(driver.events)

    with factory() as db:
        monkeypatch.setattr(
            db,
            "commit",
            lambda: (_ for _ in ()).throw(SQLAlchemyError("preference commit failed")),
        )
        with pytest.raises(SQLAlchemyError):
            manager.save(db, config=HARDWARE_CONFIG, auto_restore=False)

    assert driver.events == lifecycle_before
    assert driver.diagnostic().status == "connected"
    with factory() as db:
        row = db.get(YAMSetup, "primary")
        assert row is not None
        assert row.auto_restore is True


def test_setup_api_sanitizes_storage_and_driver_failures(setup_database) -> None:
    _, factory = setup_database
    driver = MockYAMDriver()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=True,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)
    _database_overrides(app, factory)
    sentinel = "postgresql://operator:secret@private-device/RAW_PAYLOAD"

    def broken_discovery():
        raise RuntimeError(sentinel)

    manager.discover = broken_discovery  # type: ignore[method-assign]
    with TestClient(app) as client:
        discovery = client.post("/api/yam/setup/discover")

        def broken_save(*_args, **_kwargs):
            raise RuntimeError(sentinel)

        manager.save = broken_save  # type: ignore[method-assign]
        save = client.put(
            "/api/yam/setup",
            json={"config": MOCK_SETUP_CONFIG.model_dump()},
        )

    assert discovery.status_code == save.status_code == 503
    assert sentinel not in discovery.text
    assert sentinel not in save.text


def test_hardware_status_db_failure_returns_sanitized_503(
    setup_database,
) -> None:
    _, factory = setup_database
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    app = create_app(yam_driver=driver, yam_setup_manager=manager)
    _database_overrides(app, factory)
    sentinel = "postgresql://operator:secret@private-db/status-failed"

    with TestClient(app) as client:
        manager.status = (  # type: ignore[method-assign]
            lambda _db: (_ for _ in ()).throw(SQLAlchemyError(sentinel))
        )
        response = client.get("/api/yam/setup")

    assert response.status_code == 503
    assert response.json()["detail"] == "YAM setup status is unavailable."
    assert sentinel not in response.text


def test_saved_unplugged_setup_connects_once_after_passive_visibility_transition(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=False)
    lease = RigLease()
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=lease,
        mock_mode=False,
        session_factory=factory,
        restore_interval_seconds=0.05,
    )

    async def scenario() -> None:
        await manager.startup()
        assert driver.events == ["apply"]
        token = lease.acquire("teleop", "recording-test")
        driver.visible = True
        await asyncio.sleep(0.13)
        assert driver.start_calls == 0
        lease.release(token)
        await asyncio.sleep(0.13)
        assert driver.start_calls == 1
        await asyncio.sleep(0.13)
        assert driver.start_calls == 1
        await manager.shutdown()

    asyncio.run(scenario())
    with factory() as db:
        row = db.get(YAMSetup, "primary")
        assert row is not None
        assert row.last_attempt_at is not None
        assert row.last_connected_at is not None


def test_unplugged_saved_setup_can_revoke_auto_restore_without_preflight(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=False)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    manager._restore_boot_sync()

    with factory() as db:
        status = manager.save(
            db,
            config=HARDWARE_CONFIG,
            auto_restore=False,
        )

    assert status.saved is True
    assert status.auto_restore is False
    assert status.connected is False
    assert driver.start_calls == 0
    with factory() as db:
        row = db.get(YAMSetup, "primary")
        assert row is not None
        assert row.auto_restore is False


def test_boot_restore_honors_disabled_auto_restore(setup_database) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=False)
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
        restore_interval_seconds=0.05,
    )

    async def scenario() -> None:
        await manager.startup()
        await asyncio.sleep(0.13)
        assert driver.events == ["apply"]
        with factory() as db:
            status = manager.status(db)
            assert status.restored_on_boot is True
            assert status.state == "ready_to_connect"
            assert status.auto_restore is False
        await manager.shutdown()

    asyncio.run(scenario())


def test_ready_saved_setup_restores_before_other_consumers_start(setup_database) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )

    async def scenario() -> None:
        await manager.startup()
        assert driver.events[:2] == ["apply", "startup"]
        with factory() as db:
            status = manager.status(db)
            assert status.restored_on_boot is True
            assert status.state == "ready"
        await manager.shutdown()

    asyncio.run(scenario())


def test_transient_boot_database_failure_recovers_saved_setup_without_restart(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=True)
    calls = 0

    def flaky_factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SQLAlchemyError("temporary private database failure")
        return factory()

    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=flaky_factory,
        restore_interval_seconds=0.05,
    )

    async def scenario() -> None:
        await manager.startup()
        assert driver.start_calls == 0
        assert manager._restore_storage_pending is True
        await asyncio.sleep(0.14)
        assert driver.events[:3] == ["shutdown", "apply", "startup"]
        assert driver.start_calls == 1
        assert manager._restore_storage_pending is False
        with factory() as db:
            status = manager.status(db)
            assert status.state == "ready"
            assert status.restored_on_boot is True
        await manager.shutdown()

    asyncio.run(scenario())


def test_failed_automatic_connect_latches_error_without_retry(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=False, fail_connect=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
        restore_interval_seconds=0.05,
    )

    async def scenario() -> None:
        await manager.startup()
        driver.visible = True
        await asyncio.sleep(0.13)
        assert driver.start_calls == 1
        assert driver.diagnostic().status == "error"
        await asyncio.sleep(0.15)
        assert driver.start_calls == 1
        await manager.shutdown()

    asyncio.run(scenario())


def test_missing_diagnostic_after_active_attempt_is_latched_without_retry(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(
        visible=True,
        fail_connect=True,
        failure_status="missing",
    )
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
        restore_interval_seconds=0.05,
    )

    async def scenario() -> None:
        await manager.startup()
        assert driver.start_calls == 1
        await asyncio.sleep(0.18)
        assert driver.start_calls == 1
        with factory() as db:
            status = manager.status(db)
            assert status.state == "error"
            assert "connect manually" in status.diagnostic.detail
        await manager.shutdown()

    asyncio.run(scenario())


def test_failed_manual_connect_cannot_trigger_automatic_retry(setup_database) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(
        visible=True,
        fail_connect=True,
        failure_status="missing",
    )
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )

    with factory() as db:
        with pytest.raises(YAMSetupConnectError, match="connection failed"):
            manager.connect(db, acknowledge_hardware_motion_risk=True)
    manager._restore_missing_once()

    assert driver.start_calls == 1
    with factory() as db:
        assert manager.status(db).state == "error"


def test_queued_monitor_rechecks_active_failure_latch_inside_operation(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=True)
    driver._diagnostic = YAMDriverDiagnostic(
        status="missing", detail="Test hardware is awaiting restoration."
    )
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )

    manager._operation_lock.acquire()
    worker = threading.Thread(target=manager._restore_missing_once)
    worker.start()
    time.sleep(0.03)
    manager._active_attempt_failed = True
    manager._operation_lock.release()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert driver.start_calls == 0


def test_hotplug_startup_exception_shuts_down_and_latches_without_retry(
    setup_database,
) -> None:
    _, factory = setup_database
    _persist_hardware(factory, auto_restore=True)
    driver = SetupDriver(visible=False, raise_connect=True)
    manager = YAMSetupManager(
        driver=driver,
        rig_lease=RigLease(),
        mock_mode=False,
        session_factory=factory,
    )
    manager._restore_boot_sync()
    driver.visible = True

    manager._restore_missing_once()
    manager._restore_missing_once()

    assert driver.start_calls == 1
    assert driver.events[-2:] == ["startup", "shutdown"]
    with factory() as db:
        status = manager.status(db)
        assert status.state == "error"
        assert "connect manually" in status.diagnostic.detail


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("can_interface", "can-interface-name-too-long"),
        ("leader_port", "/dev/../etc/passwd"),
        ("leader_port", "/tmp/not-a-device"),
        ("mujoco_xml_path", "relative/model.xml"),
        ("leader_calibration_id", "bad/id"),
    ],
)
def test_setup_config_rejects_hostile_or_unbounded_values(
    field: str, value: str
) -> None:
    payload = HARDWARE_CONFIG.model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        YAMSetupConfig.model_validate(payload)

    with pytest.raises(ValidationError):
        YAMSetupConfig.model_validate({**HARDWARE_CONFIG.model_dump(), "secret": "x"})
