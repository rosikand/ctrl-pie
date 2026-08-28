from fastapi import FastAPI

from ctrl_pi.api.arms import router as arms_router
from ctrl_pi.api.settings import router as settings_router
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import YAMDriver


def create_app(yam_driver: YAMDriver | None = None) -> FastAPI:
    application = FastAPI(
        title="ctrl-π API",
        description="Local control plane for the ctrl-π robot-learning console.",
        version="0.1.0",
    )
    application.state.yam_driver = yam_driver or MockYAMDriver()
    application.include_router(settings_router)
    application.include_router(arms_router)

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "mock"}

    return application


app = create_app()
