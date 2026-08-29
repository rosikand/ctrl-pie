import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

from ctrl_pi.api.arms import router as arms_router
from ctrl_pi.api.camera import router as camera_router
from ctrl_pi.api.datasets import router as datasets_router
from ctrl_pi.api.inference import router as inference_router
from ctrl_pi.api.managed_training import router as managed_training_router
from ctrl_pi.api.models import legacy_router as legacy_models_router
from ctrl_pi.api.models import router as models_router
from ctrl_pi.api.recordings import (
    reconcile_recordings_startup,
    router as recordings_router,
)
from ctrl_pi.api.settings import router as settings_router
from ctrl_pi.api.trainer import router as trainer_router
from ctrl_pi.api.yam_setup import router as yam_setup_router
from ctrl_pi.camera import MockCamera
from ctrl_pi.compute import ComputeTarget, TargetKind
from ctrl_pi.compute_stub import StubComputeTarget
from ctrl_pi.config import get_config
from ctrl_pi.db import configured_engine
from ctrl_pi.deployments import (
    DeploymentService,
    HFModelRevisionResolver,
    ModelRevisionResolver,
)
from ctrl_pi.drivers.real_yam import create_yam_driver
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
from ctrl_pi.managed_training import ManagedTrainingManager
from ctrl_pi.managed_training_artifacts import (
    HFManagedTrainingArtifactService,
    ManagedTrainingArtifactService,
    StubManagedTrainingArtifactService,
)
from ctrl_pi.recording import RecordingManager
from ctrl_pi.rig import RigLease
from ctrl_pi.spa import install_spa
from ctrl_pi.training_compute import ManagedTrainingTarget, TrainingTargetKind
from ctrl_pi.training_compute_stub import StubManagedTrainingTarget
from ctrl_pi.yam_setup import YAMSetupManager


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
    frontend_dist_dir: Path | None = None,
    yam_setup_manager: YAMSetupManager | None = None,
    managed_training_target: ManagedTrainingTarget | None = None,
    managed_training_artifact_service: ManagedTrainingArtifactService | None = None,
    managed_training_manager: ManagedTrainingManager | None = None,
) -> FastAPI:
    config = get_config()
    driver = yam_driver if yam_driver is not None else create_yam_driver(config)
    camera = mock_camera or MockCamera()
    if recording_manager is None:
        rig_lease = (
            yam_setup_manager.rig_lease
            if yam_setup_manager is not None
            else RigLease()
        )
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
    setup_manager = yam_setup_manager or YAMSetupManager(
        driver=driver,
        rig_lease=rig_lease,
        mock_mode=config.mock_mode,
        session_factory=application_session_factory,
    )
    if setup_manager.driver is not driver:
        raise ValueError("YAM setup manager and application must share one driver instance")
    if setup_manager.rig_lease is not rig_lease:
        raise ValueError("YAM setup manager and application must share one rig lease")
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

    if managed_training_manager is None and orchestration_session_factory is not None:
        if managed_training_target is not None:
            training_target = managed_training_target
        elif config.mock_mode:
            training_target = StubManagedTrainingTarget()
        else:
            from ctrl_pi.training_compute_modal import ModalTrainingTarget

            training_target = ModalTrainingTarget.from_config(config)
        if managed_training_artifact_service is not None:
            training_artifacts = managed_training_artifact_service
        elif training_target.kind == "stub":
            training_artifacts = StubManagedTrainingArtifactService()
        else:
            training_artifacts = HFManagedTrainingArtifactService(
                config.hf_token.get_secret_value()
                if config.hf_token is not None
                else None,
                config.hf_namespace,
            )

        def training_target_factory(
            kind: TrainingTargetKind,
        ) -> ManagedTrainingTarget | None:
            if kind == training_target.kind:
                return training_target
            if kind == "stub":
                return StubManagedTrainingTarget()
            if kind == "modal":
                from ctrl_pi.training_compute_modal import ModalTrainingTarget

                return ModalTrainingTarget.from_config(config)
            return None

        def training_artifact_factory(
            kind: TrainingTargetKind,
        ) -> ManagedTrainingArtifactService | None:
            if kind == training_target.kind:
                return training_artifacts
            if kind == "stub":
                return StubManagedTrainingArtifactService()
            if kind == "modal":
                return HFManagedTrainingArtifactService(
                    config.hf_token.get_secret_value()
                    if config.hf_token is not None
                    else None,
                    config.hf_namespace,
                )
            return None

        managed_training_manager = ManagedTrainingManager(
            training_target,
            training_artifacts,
            session_factory=orchestration_session_factory,
            target_factory=training_target_factory,
            artifact_service_factory=training_artifact_factory,
        )

    def cleanup_service_factory(
        target_kind: TargetKind,
    ) -> DeploymentService | None:
        """Lazily bind persisted rows to their original provider adapter."""

        if target_kind == deployment_service.target.kind:
            return deployment_service
        if target_kind == "stub":
            cleanup_target: ComputeTarget = StubComputeTarget()
        elif target_kind == "modal":
            from ctrl_pi.compute_modal import ModalComputeTarget

            cleanup_target = ModalComputeTarget.from_config(config)
        else:
            return None
        return DeploymentService(
            cleanup_target,
            session_factory=orchestration_session_factory,
        )

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
        cleanup_service_factory=cleanup_service_factory,
        recording_fps=config.recording_fps,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        setup_started = False
        recording_started = False
        managed_training_started = False
        try:
            setup_started = True
            await application.state.yam_setup_manager.startup()
            await application.state.recording_manager.startup()
            recording_started = True
            if callable(
                getattr(
                    application.state.recording_manager,
                    "reconcile_recording_artifacts",
                    None,
                )
            ):
                await reconcile_recordings_startup(
                    application.state.recording_session_factory,
                    application.state.recording_manager,
                )
            await application.state.inference_session_manager.startup()
            if application.state.managed_training_manager is not None:
                managed_training_started = True
                await application.state.managed_training_manager.startup()
            yield
        finally:
            try:
                if managed_training_started:
                    await application.state.managed_training_manager.shutdown()
            finally:
                try:
                    await application.state.inference_session_manager.shutdown()
                finally:
                    try:
                        if recording_started:
                            await application.state.recording_manager.shutdown()
                    finally:
                        if setup_started:
                            await application.state.yam_setup_manager.shutdown()

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
    application.state.yam_setup_manager = setup_manager
    application.state.rig_lease = rig_lease
    application.state.mock_camera = manager.camera
    application.state.recording_manager = manager
    application.state.recording_session_factory = (
        application_session_factory or orchestration_session_factory
    )
    application.state.hf_uploader = uploader
    application.state.hf_dataset_browser = dataset_browser
    application.state.hf_episode_browser = episode_browser
    application.state.hf_model_browser = model_browser
    application.state.deployment_service = deployment_service
    application.state.inference_session_manager = session_manager
    application.state.managed_training_manager = managed_training_manager
    application.include_router(settings_router)
    application.include_router(yam_setup_router)
    application.include_router(arms_router)
    application.include_router(recordings_router)
    application.include_router(camera_router)
    application.include_router(datasets_router)
    application.include_router(models_router)
    application.include_router(legacy_models_router)
    application.include_router(trainer_router)
    application.include_router(managed_training_router)
    application.include_router(inference_router)

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "mock" if config.mock_mode else "hardware"}

    install_spa(
        application,
        frontend_dist_dir
        if frontend_dist_dir is not None
        else config.frontend_dist_dir,
    )

    return application


app = create_app()
