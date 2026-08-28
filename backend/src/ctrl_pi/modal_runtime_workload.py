from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from ctrl_pi.compute import ComputeConfigurationError, ResourcePolicy
from ctrl_pi.inference_runtime import (
    MAX_IMAGE_BASE64_BYTES,
    MAX_IMAGE_FEATURES,
    ActionChunk,
    DescribeRequest,
    HealthRequest,
    InferenceRuntime,
    InferenceRuntimeError,
    ObservationRequest,
    RuntimeHealth,
    RuntimeConfigurationError,
    RuntimeLoadSpec,
    RuntimeProtocolError,
    RuntimeWireRequest,
)
from ctrl_pi.modal_workload import (
    MODAL_OWNERSHIP_TAG_KEY,
    ModalWorkload,
)

MODAL_RUNTIME_CLASS = "RuntimeServer"
MODAL_MODEL_PATH = "/opt/ctrl-pi/model"
MODAL_MODEL_MANIFEST = ".ctrl-pi-runtime.json"
MODAL_RUNTIME_PACKAGES = (
    "fastapi==0.141.1",
    "huggingface-hub==0.35.3",
    "lerobot[smolvla]==0.4.4",
)
MODAL_RUNTIME_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
MODAL_GPU_MAP = {
    "Modal: A10G": "A10G",
    "Modal: A100": "A100",
    "Modal: H100": "H100",
}

# Modal WebSockets are capped at 2 MiB. The POST route can carry the full
# bounded observation envelope (up to eight 4 MiB base64 image fields).
MAX_WEBSOCKET_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_POST_BODY_BYTES = MAX_IMAGE_FEATURES * MAX_IMAGE_BASE64_BYTES + 1024 * 1024

_REVISION = re.compile(r"[0-9a-f]{40}")
_WIRE_ADAPTER = TypeAdapter(RuntimeWireRequest)


class _Image(Protocol):
    def uv_pip_install(self, *packages: str) -> _Image: ...

    def run_function(self, function: Any, **kwargs: Any) -> _Image: ...

    def env(self, variables: dict[str, str]) -> _Image: ...

    def add_local_python_source(self, *modules: str, **kwargs: Any) -> _Image: ...


class _ModalModule(Protocol):
    App: Any
    Image: Any
    Secret: Any

    def enter(self) -> Any: ...

    def exit(self) -> Any: ...

    def asgi_app(self, *, requires_proxy_auth: bool) -> Any: ...


class _RequestTooLarge(ValueError):
    pass


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSONValue(ValueError):
    pass


def _download_model_snapshot(
    model_repo: str,
    revision: str,
    destination: str = MODAL_MODEL_PATH,
) -> None:
    """Bake one immutable Hub snapshot into an Image build layer."""

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("The Hugging Face build secret is unavailable.")
    model_path = Path(destination)
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_repo,
        revision=revision,
        token=token,
        local_dir=str(model_path),
    )
    manifest = {
        "schema": 1,
        "model_repo": model_repo,
        "revision": revision,
    }
    (model_path / MODAL_MODEL_MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_lerobot_runtime(
    *,
    model_repo: str,
    revision: str,
    actions_per_chunk: int,
    model_path: Path = Path(MODAL_MODEL_PATH),
) -> InferenceRuntime:
    """Load only the snapshot identity baked into this container."""

    try:
        manifest = json.loads(
            (model_path / MODAL_MODEL_MANIFEST).read_text(encoding="utf-8")
        )
    except Exception:
        raise RuntimeError("The baked model manifest is unavailable.") from None
    if manifest != {
        "schema": 1,
        "model_repo": model_repo,
        "revision": revision,
    }:
        raise RuntimeError("The baked model identity is invalid.")

    from ctrl_pi.runtime_lerobot import LeRobotRuntime

    runtime = LeRobotRuntime()
    runtime.load(
        RuntimeLoadSpec(
            model_repo=model_repo,
            revision=revision,
            local_model_path=model_path,
            device="cuda",
            actions_per_chunk=actions_per_chunk,
        )
    )
    return runtime


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    raise _NonFiniteJSONValue


def _decode_json(value: bytes | str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        _DuplicateJSONKey,
        _NonFiniteJSONValue,
        TypeError,
    ):
        raise ValueError("The runtime request is invalid.") from None


def _validate_wire_request(payload: object) -> HealthRequest | DescribeRequest | ObservationRequest:
    if isinstance(payload, dict) and "type" not in payload and set(payload) == {"nonce"}:
        payload = {"type": "health", **payload}
    try:
        return _WIRE_ADAPTER.validate_python(payload)
    except ValidationError:
        raise ValueError("The runtime request is invalid.") from None


class RuntimeRequestHandler:
    """Framework-independent, fixed-error runtime request dispatcher."""

    def __init__(self, runtime: InferenceRuntime) -> None:
        self._runtime = runtime

    def dispatch(self, payload: object) -> dict[str, object]:
        request = _validate_wire_request(payload)
        if isinstance(request, HealthRequest):
            descriptor = self._runtime.describe()
            return RuntimeHealth(
                healthy=True,
                echo=request.nonce,
                runtime=descriptor.runtime,
                model_repo=descriptor.model_repo,
                revision=descriptor.revision,
            ).model_dump(mode="json")
        if isinstance(request, DescribeRequest):
            self._runtime.reset_session()
            return self._runtime.describe().model_dump(mode="json")
        chunk = self._runtime.predict(request)
        return chunk.model_dump(mode="json")


async def _bounded_request_json(request: Any) -> object:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise ValueError("The runtime request is invalid.")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_content_length = int(content_length)
        except ValueError:
            raise ValueError("The runtime request is invalid.") from None
        if parsed_content_length > MAX_POST_BODY_BYTES:
            raise _RequestTooLarge
    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > MAX_POST_BODY_BYTES - len(body):
            raise _RequestTooLarge
        body.extend(chunk)
    return _decode_json(bytes(body))


def create_runtime_asgi_app(runtime: InferenceRuntime) -> Any:
    """Create the strict POST/WebSocket surface hosted by one Modal Function."""

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    handler = RuntimeRequestHandler(runtime)

    @app.post("/")
    async def request_runtime(request: Request) -> JSONResponse:
        try:
            payload = await _bounded_request_json(request)
            response = handler.dispatch(payload)
            return JSONResponse(response)
        except _RequestTooLarge:
            return JSONResponse(
                {"detail": "The runtime request is too large."}, status_code=413
            )
        except (ValueError, RuntimeProtocolError):
            return JSONResponse(
                {"detail": "The runtime request is invalid."}, status_code=422
            )
        except InferenceRuntimeError:
            return JSONResponse(
                {"detail": "The inference runtime is unavailable."}, status_code=503
            )
        except Exception:
            return JSONResponse(
                {"detail": "The inference runtime failed safely."}, status_code=500
            )

    @app.websocket("/ws")
    async def runtime_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            try:
                runtime.reset_session()
            except InferenceRuntimeError:
                await websocket.close(code=1011, reason="Runtime is unavailable.")
                return
            except Exception:
                await websocket.close(code=1011, reason="Runtime failed safely.")
                return
            while True:
                event = await websocket.receive()
                if event.get("type") == "websocket.disconnect":
                    return
                message = event.get("text")
                if not isinstance(message, str):
                    await websocket.close(
                        code=1008,
                        reason="Runtime request is invalid.",
                    )
                    return
                if len(message.encode("utf-8")) > MAX_WEBSOCKET_MESSAGE_BYTES:
                    await websocket.close(code=1009, reason="Runtime message is too large.")
                    return
                try:
                    request = _validate_wire_request(_decode_json(message))
                    if not isinstance(request, ObservationRequest):
                        raise ValueError("The runtime request is invalid.")
                    response = handler.dispatch(request.model_dump(mode="json"))
                except (ValueError, RuntimeProtocolError, ValidationError):
                    await websocket.close(code=1008, reason="Runtime request is invalid.")
                    return
                except InferenceRuntimeError:
                    await websocket.close(code=1011, reason="Runtime is unavailable.")
                    return
                except Exception:
                    await websocket.close(code=1011, reason="Runtime failed safely.")
                    return
                await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    return app


def _validate_runtime_resources(resources: ResourcePolicy) -> str:
    try:
        gpu = MODAL_GPU_MAP[resources.compute_size]
    except KeyError:
        raise ComputeConfigurationError(
            "The Modal runtime compute size is not supported."
        ) from None
    if (
        resources.min_containers != 0
        or resources.buffer_containers != 0
        or resources.max_containers != 1
        or not 2 <= resources.scaledown_window_seconds <= 60
        or not 1 <= resources.timeout_seconds <= 1800
    ):
        raise ComputeConfigurationError(
            "The Modal resource policy violates the V1 cost guardrails."
        )
    return gpu


def build_modal_runtime_workload(
    *,
    app_name: str,
    deployment_id: uuid.UUID,
    resources: ResourcePolicy,
    runtime: Literal["lerobot", "openpi"],
    model_repo: str,
    checkpoint_revision: str,
    hf_token: str,
    actions_per_chunk: int = 20,
    modal_module: _ModalModule | None = None,
) -> ModalWorkload:
    """Build, but never deploy, one revision-pinned GPU runtime App."""

    if runtime == "openpi":
        raise ComputeConfigurationError(
            "The real OpenPI Modal runtime is not available in V1."
        )
    if runtime != "lerobot":
        raise ComputeConfigurationError("The Modal inference runtime is invalid.")
    if not _REVISION.fullmatch(checkpoint_revision):
        raise ComputeConfigurationError(
            "The model revision must be an immutable Hub commit SHA."
        )
    if not hf_token.strip():
        raise ComputeConfigurationError(
            "Hugging Face credentials are required for Modal model builds."
        )
    if not 1 <= actions_per_chunk <= 100:
        raise ComputeConfigurationError(
            "actions_per_chunk must be between 1 and 100."
        )
    # Validate repo ID and the full runtime identity before creating provider objects.
    try:
        RuntimeLoadSpec(
            model_repo=model_repo,
            revision=checkpoint_revision,
            local_model_path=Path(MODAL_MODEL_PATH),
            device="cuda",
            actions_per_chunk=actions_per_chunk,
        )
    except RuntimeConfigurationError as error:
        raise ComputeConfigurationError(str(error)) from None
    gpu = _validate_runtime_resources(resources)

    if modal_module is None:
        import modal

        modal_module = modal

    build_secret = modal_module.Secret.from_dict({"HF_TOKEN": hf_token})
    image: _Image = modal_module.Image.debian_slim(
        python_version="3.11"
    ).uv_pip_install(*MODAL_RUNTIME_PACKAGES)
    image = image.run_function(
        _download_model_snapshot,
        args=(model_repo, checkpoint_revision),
        secrets=[build_secret],
    )
    # Apply offline mode only after the explicit-token build download. The
    # serving container then fails closed instead of resolving nested mutable
    # tokenizer/processor model IDs from the Hub at load or request time.
    image = image.env(MODAL_RUNTIME_OFFLINE_ENV).add_local_python_source(
        "ctrl_pi", copy=True
    )
    app = modal_module.App(
        app_name,
        image=image,
        tags={MODAL_OWNERSHIP_TAG_KEY: str(deployment_id)},
    )

    class RuntimeServer:
        @modal_module.enter()
        def load(self) -> None:
            self.runtime = _load_lerobot_runtime(
                model_repo=model_repo,
                revision=checkpoint_revision,
                actions_per_chunk=actions_per_chunk,
            )

        @modal_module.exit()
        def close(self) -> None:
            runtime_instance = getattr(self, "runtime", None)
            if runtime_instance is not None:
                runtime_instance.close()

        @modal_module.asgi_app(requires_proxy_auth=True)
        def web(self) -> Any:
            return create_runtime_asgi_app(self.runtime)

    runtime_server = app.cls(
        image=image,
        gpu=gpu,
        serialized=True,
        min_containers=resources.min_containers,
        buffer_containers=resources.buffer_containers,
        max_containers=resources.max_containers,
        scaledown_window=resources.scaledown_window_seconds,
        timeout=resources.timeout_seconds,
    )(RuntimeServer)
    return ModalWorkload(
        app=app,
        health_function=runtime_server().web,
    )
