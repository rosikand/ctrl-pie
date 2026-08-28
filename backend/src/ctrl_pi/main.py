from contextlib import asynccontextmanager

from fastapi import FastAPI

from ctrl_pi.api.arms import router as arms_router
from ctrl_pi.api.camera import router as camera_router
from ctrl_pi.api.datasets import router as datasets_router
from ctrl_pi.api.recordings import router as recordings_router
from ctrl_pi.api.settings import router as settings_router
from ctrl_pi.camera import MockCamera
from ctrl_pi.config import get_config
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import YAMDriver
from ctrl_pi.hf import HFDatasetUploader
from ctrl_pi.hf_datasets import HFDatasetBrowser
from ctrl_pi.recording import RecordingManager


def create_app(
    yam_driver: YAMDriver | None = None,
    mock_camera: MockCamera | None = None,
    recording_manager: RecordingManager | None = None,
    hf_uploader: HFDatasetUploader | None = None,
    hf_dataset_browser: HFDatasetBrowser | None = None,
) -> FastAPI:
    config = get_config()
    driver = yam_driver or MockYAMDriver()
    camera = mock_camera or MockCamera()
    manager = recording_manager or RecordingManager(
        driver=driver,
        camera=camera,
        staging_dir=config.recording_staging_dir,
    )
    uploader = hf_uploader or HFDatasetUploader(config.recording_staging_dir)
    dataset_browser = hf_dataset_browser or HFDatasetBrowser()

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
    application.state.hf_uploader = uploader
    application.state.hf_dataset_browser = dataset_browser
    application.include_router(settings_router)
    application.include_router(arms_router)
    application.include_router(recordings_router)
    application.include_router(camera_router)
    application.include_router(datasets_router)

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "mock"}

    return application


app = create_app()
