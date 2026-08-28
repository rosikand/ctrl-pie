from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

from ctrl_pi.api.arms import router as arms_router
from ctrl_pi.api.camera import router as camera_router
from ctrl_pi.api.datasets import router as datasets_router
from ctrl_pi.api.inference import router as inference_router
from ctrl_pi.api.recordings import router as recordings_router
from ctrl_pi.api.settings import router as settings_router
from ctrl_pi.api.trainer import router as trainer_router
from ctrl_pi.camera import MockCamera
from ctrl_pi.compute import ComputeTarget
from ctrl_pi.compute_stub import StubComputeTarget
from ctrl_pi.config import get_config
from ctrl_pi.db import configured_engine
from ctrl_pi.deployments import (
    DeploymentService,
    HFModelRevisionResolver,
    ModelRevisionResolver,
)
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import YAMDriver
from ctrl_pi.hf import HFDatasetUploader
from ctrl_pi.hf_datasets import HFDatasetBrowser
from ctrl_pi.hf_episodes import HFEpisodeBrowser
from ctrl_pi.hf_models import HFModelBrowser
from ctrl_pi.inference_runtime import RuntimeLoadSpec, StubInferenceRuntime
from ctrl_pi.inference_sessions import InferenceSessionManager, TransportFactory
from ctrl_pi.inference_transport import (
    InProcessInferenceTransport,
    ModalInferenceTransport,
)
from ctrl_pi.recording import RecordingManager
from ctrl_pi.rig import RigLease


def create_app(
    yam_driver: YAMDriver | None = None,
    mock_camera: MockCamera | None = None,
    recording_manager: RecordingManager | None = None,
    hf_uploader: HFDatasetUploader | None = None,
    hf_dataset_browser: HFDatasetBrowser | None = None,
    hf_episode_browser: HFEpisodeBrowser | None = None,
    hf_model_browser: HFModelBrowser | None = None,
    compute_target: ComputeTarget | None = None,
    model_revision_resolver: ModelRevisionResolver | None = None,
    deployment_service: DeploymentService | None = None,
    inference_session_manager: InferenceSessionManager | None = None,
    inference_transport_factory: TransportFactory | None = None,
) -> FastAPI:
    config = get_config()
    driver = yam_driver or MockYAMDriver()
    camera = mock_camera or MockCamera()
    if recording_manager is None:
        rig_lease = RigLease()
        manager = RecordingManager(
            driver=driver,
            camera=camera,
            staging_dir=config.recording_staging_dir,
            rig_lease=rig_lease,
        )
    else:
        manager = recording_manager
        rig_lease = manager.rig_lease
    uploader = hf_uploader or HFDatasetUploader(config.recording_staging_dir)
    dataset_browser = hf_dataset_browser or HFDatasetBrowser()
    episode_browser = hf_episode_browser or HFEpisodeBrowser()
    model_browser = hf_model_browser or HFModelBrowser()
    engine = configured_engine()
    application_session_factory = (
        None if engine is None else sessionmaker(bind=engine, expire_on_commit=False)
    )
    if deployment_service is None:
        if compute_target is not None:
            target = compute_target
        elif config.mock_mode:
            target = StubComputeTarget()
        else:
            from ctrl_pi.compute_modal import ModalComputeTarget

            target = ModalComputeTarget.from_config(config)
        deployment_service = DeploymentService(
            target,
            model_revision_resolver=(
                model_revision_resolver
                if model_revision_resolver is not None
                else HFModelRevisionResolver(
                    config.hf_token.get_secret_value()
                    if config.hf_token is not None
                    else None,
                    config.hf_namespace,
                )
            ),
            session_factory=application_session_factory,
        )
    orchestration_session_factory = deployment_service.session_factory

    if inference_transport_factory is None:

        def inference_transport_factory(record):
            revision = record.checkpoint_revision
            if revision is None:
                raise ValueError("deployment revision is unavailable")
            if record.target_kind == "stub":
                runtime = StubInferenceRuntime(runtime=record.runtime)
                runtime.load(
                    RuntimeLoadSpec(
                        model_repo=record.model_repo,
                        revision=revision,
                        local_model_path=None,
                        device="cpu",
                        actions_per_chunk=20,
                    )
                )
                return InProcessInferenceTransport(runtime)
            if record.endpoint_url is None:
                raise ValueError("deployment endpoint URL is unavailable")
            return ModalInferenceTransport(
                record.endpoint_url,
                proxy_token_id=config.modal_proxy_token_id,
                proxy_token_secret=config.modal_proxy_token_secret,
            )

    session_manager = inference_session_manager or InferenceSessionManager(
        deployment_service=deployment_service,
        driver=driver,
        camera=camera,
        rig_lease=rig_lease,
        recording_manager=manager,
        transport_factory=inference_transport_factory,
        session_factory=orchestration_session_factory,
        recording_fps=config.recording_fps,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await application.state.recording_manager.startup()
        try:
            await application.state.deployment_service.reconcile_startup()
            await application.state.inference_session_manager.startup()
            yield
        finally:
            try:
                await application.state.inference_session_manager.shutdown()
            finally:
                await application.state.recording_manager.shutdown()

    application = FastAPI(
        title="ctrl-π API",
        description="Local control plane for the ctrl-π robot-learning console.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def sanitized_validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                key: item[key]
                for key in ("loc", "msg", "type")
                if key in item
            }
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    application.state.yam_driver = driver
    application.state.rig_lease = rig_lease
    application.state.mock_camera = manager.camera
    application.state.recording_manager = manager
    application.state.hf_uploader = uploader
    application.state.hf_dataset_browser = dataset_browser
    application.state.hf_episode_browser = episode_browser
    application.state.hf_model_browser = model_browser
    application.state.deployment_service = deployment_service
    application.state.inference_session_manager = session_manager
    application.include_router(settings_router)
    application.include_router(arms_router)
    application.include_router(recordings_router)
    application.include_router(camera_router)
    application.include_router(datasets_router)
    application.include_router(trainer_router)
    application.include_router(inference_router)

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "mock"}

    return application


app = create_app()
