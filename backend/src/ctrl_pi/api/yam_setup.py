from __future__ import annotations

from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ctrl_pi.db import configured_engine, get_db
from ctrl_pi.drivers.yam import (
    YAMConfiguration,
    YAMDiscoveryResult,
    YAMHandleRangeResult,
    YAMPreflightResult,
)
from ctrl_pi.rig import RigLeaseConflictError
from ctrl_pi.yam_setup import (
    YAMSetupConnectError,
    YAMSetupManager,
    YAMSetupRejectedError,
    YAMSetupStatus,
)

router = APIRouter(prefix="/api/yam/setup", tags=["yam-setup"])
cell_router = APIRouter(prefix="/api/yam/cell", tags=["yam-cell"])


class YAMPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: YAMConfiguration


class YAMSetupWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: YAMConfiguration
    auto_restore: bool = Field(default=False, strict=True)
    acknowledge_automatic_motion_risk: bool = Field(default=False, strict=True)
    acknowledge_gripper_calibration_motion: bool = Field(default=False, strict=True)


class YAMConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_ids: list[str] | None = Field(default=None, min_length=1, max_length=16)
    acknowledge_hardware_motion_risk: bool = Field(default=False, strict=True)
    acknowledge_gripper_calibration_motion: bool = Field(default=False, strict=True)


class YAMDisconnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_ids: list[str] | None = Field(default=None, min_length=1, max_length=16)


class YAMHandleCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_id: str = Field(min_length=1, max_length=120)
    duration_seconds: float = Field(default=10.0, ge=1.0, le=15.0)
    acknowledge_active_can_diagnostic: bool = Field(default=False, strict=True)


def get_yam_setup_manager(request: Request) -> YAMSetupManager:
    return request.app.state.yam_setup_manager


def get_optional_setup_db() -> Generator[Session | None, None, None]:
    engine = configured_engine()
    if engine is None:
        yield None
        return
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store"


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


@router.get("", response_model=YAMSetupStatus)
@cell_router.get("", response_model=YAMSetupStatus)
def get_yam_setup(
    response: Response,
    db: Session | None = Depends(get_optional_setup_db),
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMSetupStatus:
    _no_store(response)
    try:
        return manager.status(db)
    except SQLAlchemyError:
        if not manager.mock_mode:
            raise _unavailable("YAM setup status is unavailable.") from None
        try:
            return manager.status(None)
        except Exception:
            raise _unavailable("YAM setup status is unavailable.") from None
    except Exception:
        raise _unavailable("YAM setup status is unavailable.") from None


@router.post("/discover", response_model=YAMDiscoveryResult)
@cell_router.post("/discover", response_model=YAMDiscoveryResult)
def discover_yam_setup(
    response: Response,
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMDiscoveryResult:
    _no_store(response)
    try:
        return manager.discover()
    except Exception:
        raise _unavailable("YAM setup discovery is unavailable.") from None


@router.post("/preflight", response_model=YAMPreflightResult)
@cell_router.post("/preflight", response_model=YAMPreflightResult)
def preflight_yam_setup(
    payload: YAMPreflightRequest,
    response: Response,
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMPreflightResult:
    _no_store(response)
    try:
        return manager.preflight(payload.config)
    except Exception:
        raise _unavailable("YAM setup preflight is unavailable.") from None


@router.put("", response_model=YAMSetupStatus)
@cell_router.put("", response_model=YAMSetupStatus)
def save_yam_setup(
    payload: YAMSetupWrite,
    response: Response,
    db: Session = Depends(get_db),
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMSetupStatus:
    _no_store(response)
    try:
        return manager.save(
            db,
            config=payload.config,
            auto_restore=payload.auto_restore,
            acknowledge_automatic_motion_risk=(
                payload.acknowledge_automatic_motion_risk
            ),
            acknowledge_gripper_calibration_motion=(
                payload.acknowledge_gripper_calibration_motion
            ),
        )
    except (RigLeaseConflictError, YAMSetupRejectedError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM setup could not be saved safely.") from None


@router.post("/connect", response_model=YAMSetupStatus)
@cell_router.post("/connect", response_model=YAMSetupStatus)
def connect_yam_setup(
    response: Response,
    payload: YAMConnectRequest | None = None,
    db: Session = Depends(get_db),
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMSetupStatus:
    _no_store(response)
    try:
        return manager.connect(
            db,
            arm_ids=None if payload is None else payload.arm_ids,
            acknowledge_hardware_motion_risk=(
                False
                if payload is None
                else payload.acknowledge_hardware_motion_risk
            ),
            acknowledge_gripper_calibration_motion=(
                False
                if payload is None
                else payload.acknowledge_gripper_calibration_motion
            ),
        )
    except (RigLeaseConflictError, YAMSetupRejectedError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except YAMSetupConnectError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM hardware connection failed safely.") from None


@router.delete("", response_model=YAMSetupStatus)
@cell_router.delete("", response_model=YAMSetupStatus)
def reset_yam_setup(
    response: Response,
    db: Session = Depends(get_db),
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMSetupStatus:
    _no_store(response)
    try:
        return manager.reset(db)
    except RigLeaseConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM setup could not be reset safely.") from None


@router.post("/disconnect", response_model=YAMSetupStatus)
@cell_router.post("/disconnect", response_model=YAMSetupStatus)
def disconnect_yam_setup(
    response: Response,
    payload: YAMDisconnectRequest | None = None,
    db: Session = Depends(get_db),
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMSetupStatus:
    _no_store(response)
    try:
        return manager.disconnect(
            db, arm_ids=None if payload is None else payload.arm_ids
        )
    except (RigLeaseConflictError, YAMSetupRejectedError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except YAMSetupConnectError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM hardware disconnect failed safely.") from None


@router.post("/handle-check", response_model=YAMHandleRangeResult)
@cell_router.post("/handle-check", response_model=YAMHandleRangeResult)
def check_yam_handle(
    payload: YAMHandleCheckRequest,
    response: Response,
    manager: YAMSetupManager = Depends(get_yam_setup_manager),
) -> YAMHandleRangeResult:
    _no_store(response)
    try:
        return manager.check_handle(
            arm_id=payload.arm_id,
            duration_seconds=payload.duration_seconds,
            acknowledge_active_can_diagnostic=(
                payload.acknowledge_active_can_diagnostic
            ),
        )
    except (RigLeaseConflictError, YAMSetupRejectedError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM handle range check failed safely.") from None
