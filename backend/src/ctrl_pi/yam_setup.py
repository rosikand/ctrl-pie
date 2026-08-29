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
    ArmTelemetry,
    LegacyYAMSetupConfig,
    YAMCellArmConfig,
    YAMCellConfig,
    YAMConfiguration,
    YAMDiscoveryResult,
    YAMDriver,
    YAMDriverDiagnostic,
    YAMHandleRangeResult,
    YAMPreflightResult,
    YAMSetupConfig,
)
from ctrl_pi.models import YAMCell, YAMCellArm, YAMSetup
from ctrl_pi.rig import RigLease

logger = logging.getLogger(__name__)

PRIMARY_SETUP_ID = "primary"
RESTORE_INTERVAL_SECONDS = 2.0
YAMSetupState = Literal[
    "needs_setup",
    "awaiting_hardware",
    "ready_to_connect",
    "partially_connected",
    "ready",
    "error",
]
SessionFactory = Callable[[], Session]
CALIBRATED_4310_END_EFFECTORS = frozenset({"linear_4310", "crank_4310"})


class YAMSetupArmStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_id: str
    role: Literal["leader", "follower"]
    pair_id: str | None
    group_id: str | None
    side: str | None
    connected: bool
    control_state: str
    energized: bool
    holding: bool
    runtime_interface: str | None
    error: str | None = None


class YAMSetupStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "hardware"]
    state: YAMSetupState
    configured: bool
    saved: bool
    connected: bool
    any_connected: bool
    all_connected: bool
    configured_arm_count: int
    connected_arm_count: int
    arms: list[YAMSetupArmStatus]
    calibration_ready: bool
    auto_restore: bool
    restored_on_boot: bool
    config: YAMConfiguration | None
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

    def preflight(self, config: YAMConfiguration) -> YAMPreflightResult:
        if self.mock_mode:
            return self.driver.preflight_setup(config)
        if isinstance(config, YAMCellConfig):
            # Cell drivers perform bounded stable-identity resolution and all
            # file/dependency checks without opening hardware.
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

    def status(
        self, db: Session | None, *, refresh_preflight: bool = True
    ) -> YAMSetupStatus:
        with self._operation_lock:
            row, row_config = (
                self._saved_configuration(db)
                if not self.mock_mode and db is not None
                else (None, None)
            )
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
                connected_count = sum(arm.connected for arm in arms)
                any_connected = connected_count > 0
                all_connected = bool(arms) and connected_count == len(arms)
                arm_statuses = [self._arm_status(arm) for arm in arms]
            except Exception:
                arms = []
                connected_count = 0
                any_connected = False
                all_connected = False
                arm_statuses = []
                diagnostic = YAMDriverDiagnostic(
                    status="error", detail="YAM driver status is unavailable."
                )
            calibration_ready = all_connected
            if config is not None and refresh_preflight:
                try:
                    calibration_ready = self.driver.preflight_setup(
                        config
                    ).calibration_ready
                except Exception:
                    calibration_ready = False
            if all_connected:
                state: YAMSetupState = "ready"
            elif any_connected:
                state = "partially_connected"
            elif diagnostic.status == "error":
                state = "error"
            elif config is None:
                state = "needs_setup"
            elif isinstance(config, YAMCellConfig) and not config.arms:
                # An empty cell is a valid durable topology draft, but there
                # is intentionally nothing that can be connected yet.
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
                connected=all_connected,
                any_connected=any_connected,
                all_connected=all_connected,
                configured_arm_count=len(arms),
                connected_arm_count=connected_count,
                arms=arm_statuses,
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
        config: YAMConfiguration,
        auto_restore: bool,
        acknowledge_automatic_motion_risk: bool = False,
        acknowledge_gripper_calibration_motion: bool = False,
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
        if isinstance(config, YAMCellConfig):
            return self._save_cell(
                db,
                config=config,
                auto_restore=auto_restore,
                acknowledge_gripper_calibration_motion=(
                    acknowledge_gripper_calibration_motion
                ),
            )
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

    def _save_cell(
        self,
        db: Session,
        *,
        config: YAMCellConfig,
        auto_restore: bool,
        acknowledge_gripper_calibration_motion: bool,
    ) -> YAMSetupStatus:
        if auto_restore and not config.arms:
            raise YAMSetupRejectedError(
                "Automatic connection cannot be enabled for an empty YAM cell."
            )
        if (
            auto_restore
            and self._contains_calibrated_4310_follower(config)
            and not acknowledge_gripper_calibration_motion
        ):
            raise YAMSetupRejectedError(
                "Acknowledge that unattended auto-restore may calibrate and move "
                "each calibrated 4310 follower jaw (linear_4310 or crank_4310)."
            )
        with self._operation("save-cell"):
            persisted_row, persisted_config = self._saved_configuration(db)
            if (
                isinstance(persisted_row, YAMCell)
                and persisted_config == config
                and not auto_restore
            ):
                # Revoking unattended motion consent is a DB-only safety
                # operation. It never needs preflight or a hardware lifecycle
                # transition, even when explicitly connected arms are live.
                persisted_row.auto_restore = False
                try:
                    db.commit()
                except Exception:
                    db.rollback()
                    raise
                self._setup_error = None
                self._restore_storage_pending = False
                return self.status(db, refresh_preflight=False)
            try:
                old_connected_ids = {
                    arm.id for arm in self.driver.list_arms() if arm.connected
                }
            except Exception as error:
                raise YAMSetupRejectedError(
                    "Current YAM cell connection state is unavailable; no setup "
                    "change was attempted."
                ) from error
            if old_connected_ids:
                raise YAMSetupRejectedError(
                    "Disconnect all current YAM cell arms before saving topology, "
                    "source, or safety configuration changes."
                )
            old_config = self.driver.setup_config()
            driver_changed = False
            try:
                preflight = self.preflight(config)
                empty_draft_ready_to_save = bool(
                    not config.arms
                    and preflight.i2rt_ready is True
                    and preflight.diagnostic.status == "configured"
                )
                if not preflight.ready and not empty_draft_ready_to_save:
                    raise YAMSetupRejectedError(preflight.diagnostic.detail)
                if not (old_connected_ids and old_config == config):
                    driver_changed = True
                    diagnostic = self.driver.apply_setup(config)
                    if diagnostic.status not in {"configured", "connected"}:
                        raise YAMSetupRejectedError(diagnostic.detail)
                row = self._cell_row(db)
                if row is None:
                    row = YAMCell(
                        id=PRIMARY_SETUP_ID,
                        name=config.name,
                        i2rt_root=config.i2rt_root,
                        i2rt_commit=config.i2rt_commit,
                        pair_ports={},
                    )
                    db.add(row)
                self._write_cell(db, row, config)
                row.auto_restore = auto_restore
                legacy = self._hardware_row(db)
                if legacy is not None:
                    db.delete(legacy)
                db.commit()
                self._setup_error = None
                self._active_attempt_failed = False
                self._restore_storage_pending = False
            except Exception:
                db.rollback()
                if driver_changed:
                    self._rollback_driver_resources(old_config, old_connected_ids)
                raise
        return self.status(db)

    def connect(
        self,
        db: Session,
        *,
        arm_ids: list[str] | None = None,
        acknowledge_hardware_motion_risk: bool = False,
        acknowledge_gripper_calibration_motion: bool = False,
    ) -> YAMSetupStatus:
        if not self.mock_mode and not acknowledge_hardware_motion_risk:
            raise YAMSetupRejectedError(
                "Acknowledge the hardware motion risk before connecting the YAM rig."
            )
        if self.mock_mode:
            selected = self._normalize_arm_ids(arm_ids, self.driver.list_arms())
            with self._operation("connect", resources=selected):
                self.driver.connect_arms(sorted(selected))
            return self.status(db)
        persisted_row, persisted_config = self._saved_configuration(db)
        if isinstance(persisted_row, YAMCell) and isinstance(
            persisted_config, YAMCellConfig
        ):
            return self._connect_cell(
                db,
                config=persisted_config,
                arm_ids=arm_ids,
                acknowledge_gripper_calibration_motion=(
                    acknowledge_gripper_calibration_motion
                ),
            )
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

    def _connect_cell(
        self,
        db: Session,
        *,
        config: YAMCellConfig,
        arm_ids: list[str] | None,
        acknowledge_gripper_calibration_motion: bool,
    ) -> YAMSetupStatus:
        selected = self._normalize_config_arm_ids(arm_ids, config)
        selected_arms = [arm for arm in config.arms if arm.logical_id in selected]
        calibrated_followers = [
            arm
            for arm in selected_arms
            if self._is_calibrated_4310_follower(arm)
        ]
        if calibrated_followers and not acknowledge_gripper_calibration_motion:
            names = ", ".join(
                f"{arm.logical_id} ({arm.end_effector_kind})"
                for arm in calibrated_followers
            )
            raise YAMSetupRejectedError(
                "Acknowledge 4310 jaw calibration motion before connecting: "
                f"{names}."
            )
        with self._operation("connect-cell", resources=selected):
            row, current_config = self._saved_configuration(db)
            if not isinstance(row, YAMCell) or current_config != config:
                raise YAMSetupRejectedError(
                    "The saved YAM cell changed; refresh status before connecting."
                )
            old_connected_ids = self._connected_ids()
            if selected <= old_connected_ids:
                self._setup_error = None
                return self.status(db)
            if self.driver.setup_config() != config:
                diagnostic = self.driver.apply_setup(config)
                if diagnostic.status not in {"configured", "connected"}:
                    raise YAMSetupRejectedError(diagnostic.detail)
            preflight = self.driver.preflight_setup(config)
            if preflight.arms:
                preflight_by_id = {arm.arm_id: arm for arm in preflight.arms}
                blocked = [
                    arm_id
                    for arm_id in selected
                    if arm_id not in preflight_by_id
                    or not preflight_by_id[arm_id].ready
                ]
                if blocked:
                    raise YAMSetupRejectedError(
                        "Selected arms failed passive preflight: "
                        + ", ".join(sorted(blocked))
                    )
            elif not preflight.ready:
                raise YAMSetupRejectedError(preflight.diagnostic.detail)
            row.last_attempt_at = datetime.now(UTC)
            self._active_attempt_failed = True
            self._setup_error = None
            newly_selected = selected - old_connected_ids
            try:
                self.driver.connect_arms(sorted(newly_selected))
                connected_ids = self._connected_ids()
                if not selected <= connected_ids:
                    self.driver.disconnect_arms(sorted(newly_selected & connected_ids))
                    self._setup_error = (
                        "Selected YAM arms did not reach a connected state; review "
                        "per-arm diagnostics before retrying."
                    )
                    db.commit()
                    raise YAMSetupConnectError(self._setup_error)
                row.last_connected_at = datetime.now(UTC)
                db.commit()
                self._active_attempt_failed = False
                self._restore_storage_pending = False
            except YAMSetupConnectError:
                raise
            except Exception:
                db.rollback()
                now_connected = self._connected_ids()
                rollback_ids = newly_selected & now_connected
                if rollback_ids:
                    self.driver.disconnect_arms(sorted(rollback_ids))
                self._active_attempt_failed = True
                self._setup_error = (
                    "YAM cell connection failed safely; unrelated arms were left unchanged."
                )
                raise
        return self.status(db)

    def disconnect(
        self, db: Session, *, arm_ids: list[str] | None = None
    ) -> YAMSetupStatus:
        """Stop selected arm workers and expose the resulting limp/live state."""

        config = self.driver.setup_config()
        if isinstance(config, YAMCellConfig):
            selected = self._normalize_config_arm_ids(arm_ids, config)
        else:
            selected = self._normalize_arm_ids(arm_ids, self.driver.list_arms())
        with self._operation("disconnect", resources=selected):
            self.driver.disconnect_arms(sorted(selected))
            still_connected = self._connected_ids() & selected
            if still_connected:
                raise YAMSetupConnectError(
                    "Disconnect is incomplete for: "
                    + ", ".join(sorted(still_connected))
                    + ". Treat bus ownership and torque state as uncertain."
                )
            self._setup_error = None
            self._active_attempt_failed = False
        return self.status(db)

    def check_handle(
        self,
        *,
        arm_id: str,
        duration_seconds: float = 10.0,
        acknowledge_active_can_diagnostic: bool = False,
    ) -> YAMHandleRangeResult:
        if not self.mock_mode and not acknowledge_active_can_diagnostic:
            raise YAMSetupRejectedError(
                "Acknowledge that this separate handle diagnostic actively reads CAN "
                "input; it is not passive discovery."
            )
        with self._operation("handle-check", resources={arm_id}):
            return self.driver.check_handle_range(
                arm_id, duration_seconds=duration_seconds
            )

    def reset(self, db: Session) -> YAMSetupStatus:
        if self.mock_mode:
            with self._operation("reset"):
                self.driver.reset_setup()
            return self.status(db)
        with self._operation("reset"):
            persisted_row, old_config = self._saved_configuration(db)
            old_connected_ids = self._connected_ids()
            legacy = self._hardware_row(db)
            cell = self._cell_row(db)
            if legacy is not None:
                db.delete(legacy)
            if cell is not None:
                db.delete(cell)
            try:
                self.driver.reset_setup()
                db.commit()
                self._setup_error = None
                self._active_attempt_failed = False
                self._restore_storage_pending = False
                self._restored_on_boot = False
            except Exception:
                db.rollback()
                self._rollback_driver_resources(old_config, old_connected_ids)
                raise
        return self.status(db)

    @contextmanager
    def _operation(
        self, name: str, *, resources: set[str] | None = None
    ) -> Iterator[None]:
        with self._operation_lock:
            with self.rig_lease.hold(
                "setup", f"yam-setup:{name}", resources=resources
            ):
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
                    row, config = self._saved_configuration(db)
                    if row is None:
                        return
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
                    row, config = self._saved_configuration(db)
                    if row is None:
                        self._restore_storage_pending = False
                        self._setup_error = None
                        return
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
                    row, config = self._saved_configuration(db)
                    if row is None or not row.auto_restore:
                        return
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
        self, old_config: YAMConfiguration | None, old_connected: bool
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

    def _connected_ids(self) -> set[str]:
        try:
            return {arm.id for arm in self.driver.list_arms() if arm.connected}
        except Exception:
            return set()

    @staticmethod
    def _normalize_arm_ids(
        arm_ids: list[str] | None, arms: list[ArmTelemetry]
    ) -> set[str]:
        configured = {arm.id for arm in arms}
        selected = configured if arm_ids is None else set(arm_ids)
        if not selected:
            raise YAMSetupRejectedError("Select at least one configured YAM arm.")
        unknown = selected - configured
        if unknown:
            raise YAMSetupRejectedError(
                "Unknown YAM arm IDs: " + ", ".join(sorted(unknown))
            )
        if arm_ids is not None and len(arm_ids) != len(selected):
            raise YAMSetupRejectedError("YAM arm IDs must not contain duplicates.")
        return selected

    @staticmethod
    def _normalize_config_arm_ids(
        arm_ids: list[str] | None, config: YAMCellConfig
    ) -> set[str]:
        configured = {arm.logical_id for arm in config.arms}
        selected = configured if arm_ids is None else set(arm_ids)
        if not selected:
            raise YAMSetupRejectedError("Select at least one configured YAM arm.")
        unknown = selected - configured
        if unknown:
            raise YAMSetupRejectedError(
                "Unknown YAM arm IDs: " + ", ".join(sorted(unknown))
            )
        if arm_ids is not None and len(arm_ids) != len(selected):
            raise YAMSetupRejectedError("YAM arm IDs must not contain duplicates.")
        return selected

    @staticmethod
    def _is_calibrated_4310_follower(arm: YAMCellArmConfig) -> bool:
        return bool(
            arm.role == "follower"
            and arm.end_effector_kind in CALIBRATED_4310_END_EFFECTORS
        )

    @classmethod
    def _contains_calibrated_4310_follower(cls, config: YAMCellConfig) -> bool:
        return any(cls._is_calibrated_4310_follower(arm) for arm in config.arms)

    def _rollback_driver_resources(
        self, old_config: YAMConfiguration | None, old_connected_ids: set[str]
    ) -> None:
        del old_connected_ids
        try:
            current = {arm.id for arm in self.driver.list_arms() if arm.connected}
            if current:
                self.driver.disconnect_arms(sorted(current))
            if any(arm.connected for arm in self.driver.list_arms()):
                raise RuntimeError("YAM resources did not confirm disconnection")
            if old_config is None:
                self.driver.reset_setup()
            else:
                self.driver.apply_setup(old_config)
        except Exception:
            self.driver.shutdown()
            self._setup_error = (
                "YAM setup rollback could not restore a truthful disconnected "
                "configuration; the rig remains stopped for safety."
            )

    @staticmethod
    def _arm_status(arm: ArmTelemetry) -> YAMSetupArmStatus:
        return YAMSetupArmStatus(
            arm_id=arm.id,
            role=arm.role,
            pair_id=arm.pair_id,
            group_id=arm.group_id,
            side=arm.side,
            connected=arm.connected,
            control_state=arm.control_state,
            energized=arm.energized,
            holding=arm.holding,
            runtime_interface=arm.can.interface if arm.can is not None else None,
            error=(
                arm.warnings[0]
                if arm.control_state == "error" and arm.warnings
                else None
            ),
        )

    @staticmethod
    def _hardware_row(db: Session) -> YAMSetup | None:
        row = db.get(YAMSetup, PRIMARY_SETUP_ID)
        return row if row is not None and row.mode == "hardware" else None

    @staticmethod
    def _cell_row(db: Session) -> YAMCell | None:
        return db.get(YAMCell, PRIMARY_SETUP_ID)

    @classmethod
    def _saved_configuration(
        cls, db: Session
    ) -> tuple[YAMCell | YAMSetup | None, YAMConfiguration | None]:
        cell = cls._cell_row(db)
        if cell is not None:
            # A present cell always wins, even if corrupt. Never reveal an old
            # legacy row behind a failed V1.2 migration or partial operator edit.
            return cell, cls._cell_config_from_row(cell)
        legacy = cls._hardware_row(db)
        return legacy, cls._config_from_row(legacy)

    @staticmethod
    def _cell_config_from_row(row: YAMCell | None) -> YAMCellConfig | None:
        if row is None:
            return None
        try:
            return YAMCellConfig(
                name=row.name,
                i2rt_root=row.i2rt_root,
                i2rt_commit=row.i2rt_commit,
                pair_ports=dict(row.pair_ports),
                arms=[
                    YAMCellArmConfig(
                        logical_id=arm.logical_id,
                        name=arm.name,
                        role=arm.role,
                        pair_id=arm.pair_id,
                        group_id=arm.group_id,
                        side=arm.side,
                        transport_kind=arm.transport_kind,
                        stable_identity=arm.stable_identity,
                        end_effector_kind=arm.end_effector_kind,
                        frame_map_path=arm.frame_map_path,
                        soft_limits_path=arm.soft_limits_path,
                        mujoco_xml_path=arm.mujoco_xml_path,
                        calibration_id=arm.calibration_id,
                        calibration_dir=arm.calibration_dir,
                    )
                    for arm in row.arms
                ],
            )
        except (TypeError, ValueError):
            return None

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

    @staticmethod
    def _write_cell(db: Session, row: YAMCell, config: YAMCellConfig) -> None:
        row.name = config.name
        row.i2rt_root = config.i2rt_root
        row.i2rt_commit = config.i2rt_commit
        row.pair_ports = dict(config.pair_ports)
        if row.arms:
            row.arms.clear()
            db.flush()
        row.arms.extend(
            YAMCellArm(
                cell_id=row.id,
                position=position,
                logical_id=arm.logical_id,
                name=arm.name,
                role=arm.role,
                pair_id=arm.pair_id,
                group_id=arm.group_id,
                side=arm.side,
                transport_kind=arm.transport_kind,
                stable_identity=arm.stable_identity,
                end_effector_kind=arm.end_effector_kind,
                frame_map_path=arm.frame_map_path,
                soft_limits_path=arm.soft_limits_path,
                mujoco_xml_path=arm.mujoco_xml_path,
                calibration_id=arm.calibration_id,
                calibration_dir=arm.calibration_dir,
                config={},
            )
            for position, arm in enumerate(config.arms)
        )


class YAMSetupRejectedError(RuntimeError):
    """A sanitized setup request could not pass the non-opening safety gate."""


class YAMSetupConnectError(RuntimeError):
    """An acknowledged active connection attempt failed closed."""
