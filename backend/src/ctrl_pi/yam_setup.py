from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
import logging
import threading
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ctrl_pi.drivers.yam import (
    YAMDiscoveryResult,
    YAMDriver,
    YAMDriverDiagnostic,
    YAMPreflightResult,
    YAMSetupConfig,
)
from ctrl_pi.models import YAMSetup
from ctrl_pi.rig import RigLease

logger = logging.getLogger(__name__)

PRIMARY_SETUP_ID = "primary"
RESTORE_INTERVAL_SECONDS = 2.0
YAMSetupState = Literal[
    "needs_setup", "awaiting_hardware", "ready_to_connect", "ready", "error"
]
SessionFactory = Callable[[], Session]


class YAMSetupStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "hardware"]
    state: YAMSetupState
    configured: bool
    saved: bool
    connected: bool
    calibration_ready: bool
    auto_restore: bool
    restored_on_boot: bool
    config: YAMSetupConfig | None
    diagnostic: YAMDriverDiagnostic
    last_attempt_at: datetime | None
    last_connected_at: datetime | None
    requires_physical_validation: bool


class YAMSetupManager:
    """Serializes setup persistence and lifecycle around one stable driver object."""

    def __init__(
        self,
        *,
        driver: YAMDriver,
        rig_lease: RigLease,
        mock_mode: bool,
        session_factory: SessionFactory | None,
        restore_interval_seconds: float = RESTORE_INTERVAL_SECONDS,
    ) -> None:
        self.driver = driver
        self.rig_lease = rig_lease
        self.mock_mode = mock_mode
        self.session_factory = session_factory
        self.restore_interval_seconds = max(0.05, restore_interval_seconds)
        self._operation_lock = threading.RLock()
        self._watch_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._restored_on_boot = False
        self._setup_error: str | None = None
        self._active_attempt_failed = False
        self._restore_storage_pending = False

    @property
    def mode(self) -> Literal["mock", "hardware"]:
        return "mock" if self.mock_mode else "hardware"

    async def startup(self) -> None:
        """Restore before other arm consumers start, then watch missing hardware."""

        self._stop_event = asyncio.Event()
        await asyncio.to_thread(self._restore_boot_sync)
        if not self.mock_mode and self.session_factory is not None:
            self._watch_task = asyncio.create_task(
                self._restore_monitor(), name="ctrl-pi-yam-restore"
            )

    async def shutdown(self) -> None:
        stop = self._stop_event
        if stop is not None:
            stop.set()
        task = self._watch_task
        if task is not None:
            await task
        self._watch_task = None
        await asyncio.to_thread(self._shutdown_sync)

    def discover(self) -> YAMDiscoveryResult:
        # This delegates only to the driver's passive discovery boundary.
        return self.driver.discover_setup()

    def preflight(self, config: YAMSetupConfig) -> YAMPreflightResult:
        if self.mock_mode:
            return self.driver.preflight_setup(config)
        candidates = self.driver.discover_setup()
        visible_can = {item.id for item in candidates.can_interfaces}
        visible_serial = {item.id for item in candidates.leader_ports}
        if config.can_interface not in visible_can or config.leader_port not in visible_serial:
            return YAMPreflightResult(
                ready=False,
                calibration_ready=False,
                diagnostic=YAMDriverDiagnostic(
                    status="missing",
                    detail=(
                        "Select a currently discovered SocketCAN interface and stable serial "
                        "device before preflight."
                    ),
                ),
            )
        return self.driver.preflight_setup(config)

    def status(self, db: Session | None) -> YAMSetupStatus:
        with self._operation_lock:
            row = (
                self._hardware_row(db)
                if not self.mock_mode and db is not None
                else None
            )
            row_config = self._config_from_row(row) if row is not None else None
            config = row_config or self.driver.setup_config()
            saved = row is not None and row_config is not None
            diagnostic = self.driver.diagnostic()
            if self._setup_error is not None:
                diagnostic = YAMDriverDiagnostic(
                    status="error",
                    detail=self._setup_error,
                )
            try:
                arms = self.driver.list_arms()
                connected = (
                    diagnostic.status == "connected"
                    and bool(arms)
                    and all(arm.connected for arm in arms)
                )
            except Exception:
                connected = False
                diagnostic = YAMDriverDiagnostic(
                    status="error", detail="YAM driver status is unavailable."
                )
            calibration_ready = False
            if config is not None:
                try:
                    calibration_ready = self.driver.preflight_setup(
                        config
                    ).calibration_ready
                except Exception:
                    calibration_ready = False
            if connected:
                state: YAMSetupState = "ready"
            elif diagnostic.status == "error":
                state = "error"
            elif config is None:
                state = "needs_setup"
            elif diagnostic.status == "configured":
                state = "ready_to_connect"
            else:
                state = "awaiting_hardware"
            return YAMSetupStatus(
                mode=self.mode,
                state=state,
                configured=config is not None,
                saved=saved,
                connected=connected,
                calibration_ready=calibration_ready,
                auto_restore=bool(row.auto_restore) if row is not None else False,
                restored_on_boot=self._restored_on_boot,
                config=config,
                diagnostic=diagnostic,
                last_attempt_at=row.last_attempt_at if row is not None else None,
                last_connected_at=(
                    row.last_connected_at if row is not None else None
                ),
                requires_physical_validation=not self.mock_mode,
            )

    def save(
        self,
        db: Session,
        *,
        config: YAMSetupConfig,
        auto_restore: bool,
        acknowledge_automatic_motion_risk: bool = False,
    ) -> YAMSetupStatus:
        if not self.mock_mode and auto_restore and not acknowledge_automatic_motion_risk:
            raise YAMSetupRejectedError(
                "Acknowledge the automatic hardware motion risk before enabling auto-restore."
            )
        if self.mock_mode:
            preflight = self.preflight(config)
            if not preflight.ready:
                raise YAMSetupRejectedError(preflight.diagnostic.detail)
            # Mock setup is built in. Never overwrite a saved physical rig.
            with self._operation("save"):
                self.driver.apply_setup(config)
            return self.status(db)
        with self._operation("save"):
            old_config = self.driver.setup_config()
            old_connected = self._is_connected()
            driver_changed = False
            try:
                row = self._hardware_row(db)
                persisted_config = self._config_from_row(row)
                if (
                    row is not None
                    and persisted_config == config
                    and not auto_restore
                ):
                    # Revoking unattended connection must remain possible while
                    # the rig is unplugged or another prerequisite is missing.
                    row.auto_restore = False
                    db.commit()
                    self._setup_error = None
                    self._restore_storage_pending = False
                    return self.status(db)
                preflight = self.preflight(config)
                if not preflight.ready:
                    raise YAMSetupRejectedError(preflight.diagnostic.detail)
                if row is None:
                    row = YAMSetup(id=PRIMARY_SETUP_ID, mode="hardware")
                    db.add(row)
                if not (old_connected and old_config == config):
                    driver_changed = True
                    diagnostic = self.driver.apply_setup(config)
                    if diagnostic.status != "configured":
                        raise YAMSetupRejectedError(diagnostic.detail)
                self._write_config(row, config)
                row.auto_restore = auto_restore
                db.commit()
                self._setup_error = None
                self._active_attempt_failed = False
                self._restore_storage_pending = False
            except Exception:
                db.rollback()
                if driver_changed:
                    self._rollback_driver(old_config, old_connected)
                raise
        return self.status(db)

    def connect(
        self,
        db: Session,
        *,
        acknowledge_hardware_motion_risk: bool = False,
    ) -> YAMSetupStatus:
        if not self.mock_mode and not acknowledge_hardware_motion_risk:
            raise YAMSetupRejectedError(
                "Acknowledge the hardware motion risk before connecting the YAM rig."
            )
        if self.mock_mode:
            with self._operation("connect"):
                self.driver.startup()
            return self.status(db)
        with self._operation("connect"):
            row = self._hardware_row(db)
            config = self._config_from_row(row) if row is not None else None
            if row is None or config is None:
                raise YAMSetupRejectedError("Save a valid YAM setup before connecting.")
            if self.driver.setup_config() == config and self._is_connected():
                # Connect is idempotent. Cycling an already-connected rig can
                # re-engage its controller without adding any useful state.
                self._setup_error = None
                self._active_attempt_failed = False
                self._restore_storage_pending = False
                return self.status(db)
            diagnostic = self.driver.apply_setup(config)
            if diagnostic.status != "configured":
                raise YAMSetupRejectedError(diagnostic.detail)
            attempted_at = datetime.now(UTC)
            row.last_attempt_at = attempted_at
            self._active_attempt_failed = True
            self._setup_error = None
            failure_detail = "YAM hardware connection failed safely."
            try:
                self.driver.startup()
                connected = self._is_connected()
                if connected:
                    row.last_connected_at = datetime.now(UTC)
                    self._active_attempt_failed = False
                else:
                    failure_detail = self.driver.diagnostic().detail
                    self.driver.shutdown()
                    self._setup_error = (
                        "YAM hardware connection failed safely; review the setup before "
                        "retrying manually."
                    )
                db.commit()
                # A successful explicit transaction proves storage is back;
                # do not let a queued startup-recovery pass re-apply and
                # reconnect the same hardware after this request returns.
                self._restore_storage_pending = False
            except Exception:
                db.rollback()
                self.driver.shutdown()
                self._active_attempt_failed = True
                self._setup_error = (
                    "YAM connection state could not be persisted safely; review the setup "
                    "before retrying manually."
                )
                raise
            if not connected:
                raise YAMSetupConnectError(failure_detail)
        return self.status(db)

    def reset(self, db: Session) -> YAMSetupStatus:
        if self.mock_mode:
            with self._operation("reset"):
                self.driver.reset_setup()
            return self.status(db)
        with self._operation("reset"):
            row = self._hardware_row(db)
            old_config = self._config_from_row(row)
            old_connected = self._is_connected()
            if row is not None:
                db.delete(row)
            try:
                self.driver.reset_setup()
                db.commit()
                self._setup_error = None
                self._active_attempt_failed = False
                self._restore_storage_pending = False
                self._restored_on_boot = False
            except Exception:
                db.rollback()
                self._rollback_driver(old_config, old_connected)
                raise
        return self.status(db)

    @contextmanager
    def _operation(self, name: str) -> Iterator[None]:
        with self._operation_lock:
            with self.rig_lease.hold("setup", f"yam-setup:{name}"):
                yield

    def _restore_boot_sync(self) -> None:
        with self._operation("boot-restore"):
            if self.mock_mode:
                self.driver.startup()
                return
            if self.session_factory is None:
                # Environment values bootstrap a selected configuration only.
                # Opening hardware always requires explicit persisted consent.
                return
            try:
                with self.session_factory() as db:
                    row = self._hardware_row(db)
                    if row is None:
                        return
                    config = self._config_from_row(row)
                    if config is None:
                        self.driver.shutdown()
                        self._setup_error = (
                            "The saved YAM setup is invalid; reset it and configure the rig again."
                        )
                        return
                    self.driver.apply_setup(config)
                    self._restored_on_boot = True
                    self._restore_storage_pending = False
                    self._setup_error = None
                    if not row.auto_restore:
                        return
                    preflight = self.driver.preflight_setup(config)
                    if not preflight.ready:
                        return
                    row.last_attempt_at = datetime.now(UTC)
                    self._active_attempt_failed = True
                    self.driver.startup()
                    connected = self._is_connected()
                    if connected:
                        row.last_connected_at = datetime.now(UTC)
                    else:
                        self.driver.shutdown()
                        self._setup_error = (
                            "Automatic YAM connection failed safely; review the setup and "
                            "connect manually."
                        )
                    db.commit()
                    if connected:
                        self._active_attempt_failed = False
            except SQLAlchemyError:
                logger.error("Saved YAM setup could not be restored from PostgreSQL")
                self.driver.shutdown()
                self._restore_storage_pending = True
                self._setup_error = "The saved YAM setup could not be restored safely."
            except Exception:
                logger.error("Saved YAM setup restoration failed safely")
                self.driver.shutdown()
                self._restore_storage_pending = False
                self._setup_error = "The saved YAM setup could not be applied safely."

    async def _restore_monitor(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.restore_interval_seconds
                )
            except TimeoutError:
                if self._restore_storage_pending:
                    await asyncio.to_thread(self._restore_storage_once)
                else:
                    await asyncio.to_thread(self._restore_missing_once)

    def _restore_storage_once(self) -> None:
        """Retry only the saved-row load after a transient startup DB failure."""

        if (
            not self._restore_storage_pending
            or self._active_attempt_failed
            or self.session_factory is None
        ):
            return
        try:
            with self._operation("storage-restore"):
                if (
                    not self._restore_storage_pending
                    or self._active_attempt_failed
                ):
                    return
                with self.session_factory() as db:
                    row = self._hardware_row(db)
                    if row is None:
                        self._restore_storage_pending = False
                        self._setup_error = None
                        return
                    config = self._config_from_row(row)
                    if config is None:
                        self._restore_storage_pending = False
                        self._setup_error = (
                            "The saved YAM setup is invalid; reset it and configure the rig again."
                        )
                        return
                    self.driver.apply_setup(config)
                    self._restored_on_boot = True
                    self._restore_storage_pending = False
                    self._setup_error = None
                    if not row.auto_restore:
                        return
                    preflight = self.driver.preflight_setup(config)
                    if not preflight.ready:
                        return
                    row.last_attempt_at = datetime.now(UTC)
                    self._active_attempt_failed = True
                    self.driver.startup()
                    connected = self._is_connected()
                    if connected:
                        row.last_connected_at = datetime.now(UTC)
                    else:
                        self.driver.shutdown()
                        self._setup_error = (
                            "Automatic YAM connection failed safely; review the setup and "
                            "connect manually."
                        )
                    db.commit()
                    if connected:
                        self._active_attempt_failed = False
        except Exception as error:
            from ctrl_pi.rig import RigLeaseConflictError

            if isinstance(error, RigLeaseConflictError):
                return
            if isinstance(error, SQLAlchemyError):
                self.driver.shutdown()
                self._restore_storage_pending = True
                self._setup_error = "The saved YAM setup could not be restored safely."
                return
            self.driver.shutdown()
            self._restore_storage_pending = False
            self._setup_error = "The saved YAM setup could not be applied safely."

    def _restore_missing_once(self) -> None:
        # Runtime/error faults are latched for manual review. Only an unplugged
        # saved rig participates in passive re-detection.
        if (
            self._active_attempt_failed
            or self.driver.diagnostic().status != "missing"
            or self.session_factory is None
        ):
            return
        try:
            with self._operation("auto-restore"):
                if (
                    self._active_attempt_failed
                    or self.driver.diagnostic().status != "missing"
                ):
                    return
                with self.session_factory() as db:
                    row = self._hardware_row(db)
                    if row is None or not row.auto_restore:
                        return
                    config = self._config_from_row(row)
                    if config is None:
                        return
                    preflight = self.driver.preflight_setup(config)
                    if not preflight.ready:
                        return
                    row.last_attempt_at = datetime.now(UTC)
                    self._active_attempt_failed = True
                    self.driver.startup()
                    connected = self._is_connected()
                    if connected:
                        row.last_connected_at = datetime.now(UTC)
                    else:
                        self.driver.shutdown()
                        self._setup_error = (
                            "Automatic YAM connection failed safely; review the setup and "
                            "connect manually."
                        )
                    db.commit()
                    if connected:
                        self._active_attempt_failed = False
        except Exception as error:
            # A busy rig is expected; a provider/open failure becomes the
            # driver's sanitized error and is intentionally not retried.
            from ctrl_pi.rig import RigLeaseConflictError

            if isinstance(error, SQLAlchemyError):
                self.driver.shutdown()
                self._restore_storage_pending = True
                self._setup_error = "The saved YAM setup could not be restored safely."
            elif not isinstance(error, RigLeaseConflictError):
                logger.error("YAM automatic restoration failed safely")
                self.driver.shutdown()
                self._active_attempt_failed = True
                self._setup_error = (
                    "Automatic YAM connection failed safely; review the setup and connect "
                    "manually."
                )

    def _shutdown_sync(self) -> None:
        with self._operation_lock:
            self.driver.shutdown()

    def _rollback_driver(
        self, old_config: YAMSetupConfig | None, old_connected: bool
    ) -> None:
        try:
            if old_config is None:
                self.driver.reset_setup()
            else:
                self.driver.apply_setup(old_config)
            if old_connected:
                self.driver.startup()
        except Exception:
            self.driver.shutdown()
            self._setup_error = (
                "YAM setup rollback failed; the rig remains disconnected for safety."
            )

    def _is_connected(self) -> bool:
        try:
            arms = self.driver.list_arms()
            return (
                self.driver.diagnostic().status == "connected"
                and bool(arms)
                and all(arm.connected for arm in arms)
            )
        except Exception:
            return False

    @staticmethod
    def _hardware_row(db: Session) -> YAMSetup | None:
        row = db.get(YAMSetup, PRIMARY_SETUP_ID)
        return row if row is not None and row.mode == "hardware" else None

    @staticmethod
    def _config_from_row(row: YAMSetup | None) -> YAMSetupConfig | None:
        if row is None:
            return None
        try:
            return YAMSetupConfig(
                can_interface=row.can_interface,
                leader_port=row.leader_port,
                mujoco_xml_path=row.mujoco_xml_path,
                leader_calibration_id=row.leader_calibration_id,
                leader_calibration_dir=row.leader_calibration_dir,
            )
        except ValueError:
            return None

    @staticmethod
    def _write_config(row: YAMSetup, config: YAMSetupConfig) -> None:
        row.mode = "hardware"
        row.can_interface = config.can_interface
        row.leader_port = config.leader_port
        row.mujoco_xml_path = config.mujoco_xml_path
        row.leader_calibration_id = config.leader_calibration_id
        row.leader_calibration_dir = config.leader_calibration_dir


class YAMSetupRejectedError(RuntimeError):
    """A sanitized setup request could not pass the non-opening safety gate."""


class YAMSetupConnectError(RuntimeError):
    """An acknowledged active connection attempt failed closed."""
