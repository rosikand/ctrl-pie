from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ctrl_pi.camera import MockCamera

router = APIRouter(prefix="/api/camera", tags=["camera"])


async def mjpeg_chunks(camera: MockCamera, fps: int = 15) -> AsyncIterator[bytes]:
    interval = 1.0 / fps
    while True:
        started = asyncio.get_running_loop().time()
        frame = await asyncio.to_thread(camera.capture)
        jpeg = await asyncio.to_thread(camera.jpeg, frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
            + jpeg
            + b"\r\n"
        )
        elapsed = asyncio.get_running_loop().time() - started
        await asyncio.sleep(max(0.0, interval - elapsed))


@router.get("/stream")
def camera_stream(request: Request) -> StreamingResponse:
    camera: MockCamera = request.app.state.mock_camera
    return StreamingResponse(
        mjpeg_chunks(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
