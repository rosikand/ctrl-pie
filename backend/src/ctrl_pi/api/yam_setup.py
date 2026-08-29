from __future__ import annotations

from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ctrl_pi.db import configured_engine, get_db
from ctrl_pi.drivers.yam import YAMDiscoveryResult, YAMPreflightResult, YAMSetupConfig
from ctrl_pi.rig import RigLeaseConflictError
from ctrl_pi.yam_setup import (
    YAMSetupConnectError,
    YAMSetupManager,
    YAMSetupRejectedError,
    YAMSetupStatus,
)

router = APIRouter(prefix="/api/yam/setup", tags=["yam-setup"])


class YAMPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: YAMSetupConfig


class YAMSetupWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: YAMSetupConfig
    auto_restore: bool = Field(default=False, strict=True)
    acknowledge_automatic_motion_risk: bool = Field(default=False, strict=True)


class YAMConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acknowledge_hardware_motion_risk: bool = Field(default=False, strict=True)


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
        )
    except (RigLeaseConflictError, YAMSetupRejectedError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM setup could not be saved safely.") from None


@router.post("/connect", response_model=YAMSetupStatus)
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
            acknowledge_hardware_motion_risk=(
                False
                if payload is None
                else payload.acknowledge_hardware_motion_risk
            ),
        )
    except (RigLeaseConflictError, YAMSetupRejectedError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except YAMSetupConnectError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Exception:
        raise _unavailable("YAM hardware connection failed safely.") from None


@router.delete("", response_model=YAMSetupStatus)
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
