from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

from ctrl_pi.drivers.yam import (
    ArmNotFoundError,
    ArmsResponse,
    ArmTelemetry,
    JogCommand,
    JogLimitError,
    TelemetryFrame,
    YAMDriver,
)
from ctrl_pi.rig import RigLease, RigLeaseConflictError

router = APIRouter(tags=["arms"])
TELEMETRY_INTERVAL_SECONDS = 0.05


def get_yam_driver(request: Request) -> YAMDriver:
    return request.app.state.yam_driver


def get_rig_lease(request: Request) -> RigLease:
    return request.app.state.rig_lease


@router.get("/api/arms", response_model=ArmsResponse)
def list_arms(driver: YAMDriver = Depends(get_yam_driver)) -> ArmsResponse:
    return ArmsResponse(arms=driver.list_arms())


@router.get("/api/arms/{arm_id}", response_model=ArmTelemetry)
def get_arm(arm_id: str, driver: YAMDriver = Depends(get_yam_driver)) -> ArmTelemetry:
    try:
        return driver.get_arm(arm_id)
    except ArmNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Arm '{arm_id}' was not found.") from error


@router.post("/api/arms/{arm_id}/jog", response_model=ArmTelemetry)
def jog_arm(
    arm_id: str,
    command: JogCommand,
    driver: YAMDriver = Depends(get_yam_driver),
    rig_lease: RigLease = Depends(get_rig_lease),
) -> ArmTelemetry:
    try:
        with rig_lease.hold("manual", f"jog:{arm_id}"):
            return driver.jog(arm_id, command)
    except RigLeaseConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except ArmNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Arm '{arm_id}' was not found.") from error
    except JogLimitError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.websocket("/ws/arms")
async def stream_arms(websocket: WebSocket) -> None:
    driver: YAMDriver = websocket.app.state.yam_driver
    await websocket.accept()
    try:
        while True:
            frame = TelemetryFrame(timestamp=datetime.now(UTC), arms=driver.list_arms())
            await websocket.send_json(frame.model_dump(mode="json"))
            await asyncio.sleep(TELEMETRY_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return
