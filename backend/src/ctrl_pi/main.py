from contextlib import asynccontextmanager

from fastapi import FastAPI

from ctrl_pi.api.arms import router as arms_router
from ctrl_pi.api.camera import router as camera_router
from ctrl_pi.api.recordings import router as recordings_router
from ctrl_pi.api.settings import router as settings_router
from ctrl_pi.camera import MockCamera
from ctrl_pi.config import get_config
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import YAMDriver
from ctrl_pi.recording import RecordingManager


def create_app(
    yam_driver: YAMDriver | None = None,
    mock_camera: MockCamera | None = None,
    recording_manager: RecordingManager | None = None,
) -> FastAPI:
    driver = yam_driver or MockYAMDriver()
    camera = mock_camera or MockCamera()
    manager = recording_manager or RecordingManager(
        driver=driver,
        camera=camera,
        staging_dir=get_config().recording_staging_dir,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await application.state.recording_manager.startup()
        yield
        await application.state.recording_manager.shutdown()

    application = FastAPI(
        title="ctrl-π API",
        description="Local control plane for the ctrl-π robot-learning console.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.yam_driver = driver
    application.state.mock_camera = manager.camera
    application.state.recording_manager = manager
    application.include_router(settings_router)
    application.include_router(arms_router)
    application.include_router(recordings_router)
    application.include_router(camera_router)

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "mock"}

    return application


app = create_app()
