from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import modal
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ctrl_pi.compute import ComputeConfigurationError, ResourcePolicy
from ctrl_pi.inference_runtime import (
    ObservationRequest,
    InferenceRuntimeError,
    RuntimeLoadSpec,
    RuntimeProtocolError,
    StubInferenceRuntime,
)
from ctrl_pi.modal_runtime_workload import (
    MAX_POST_BODY_BYTES,
    MODAL_GPU_MAP,
    MODAL_MODEL_MANIFEST,
    MODAL_RUNTIME_PACKAGES,
    MODAL_RUNTIME_CLASS,
    MODAL_RUNTIME_OFFLINE_ENV,
    RuntimeRequestHandler,
    _decode_json,
    _download_model_snapshot,
    build_modal_runtime_workload,
    create_runtime_asgi_app,
)
from ctrl_pi.modal_workload import MODAL_OWNERSHIP_TAG_KEY

DEPLOYMENT_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
REVISION = "a" * 40
MODEL_REPO = "ctrl-pi/tiny-act"
HF_SECRET = "hf_runtime_build_secret"


def _resources(compute_size: str = "Modal: A10G") -> ResourcePolicy:
    return ResourcePolicy(
        compute_size=compute_size,
        timeout_seconds=1800,
        scaledown_window_seconds=60,
    )


def _runtime() -> StubInferenceRuntime:
    runtime = StubInferenceRuntime(runtime="lerobot")
    runtime.load(
        RuntimeLoadSpec(
            model_repo=MODEL_REPO,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
            actions_per_chunk=3,
        )
    )
    return runtime


def _observation(sequence: int = 4) -> dict[str, object]:
    return ObservationRequest(
        request_id=uuid.UUID("44444444-4444-4444-8444-444444444444"),
        sequence=sequence,
        captured_at=datetime.now(UTC),
        vectors={"observation.state": [0.0] * 6 + [0.5]},
    ).model_dump(mode="json")


class FakeImage:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def uv_pip_install(self, *packages: str) -> FakeImage:
        self.owner.packages = packages
        return self

    def run_function(self, function: Any, **kwargs: Any) -> FakeImage:
        self.owner.build_function = function
        self.owner.build_options = kwargs
        return self

    def env(self, variables: dict[str, str]) -> FakeImage:
        self.owner.image_environment = variables
        return self

    def add_local_python_source(self, *modules: str, **kwargs: Any) -> FakeImage:
        self.owner.source_modules = modules
        self.owner.source_options = kwargs
        return self


class FakeImageAPI:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def debian_slim(self, *, python_version: str) -> FakeImage:
        self.owner.python_version = python_version
        return FakeImage(self.owner)


class FakeSecretAPI:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def from_dict(self, values: dict[str, str]) -> object:
        self.owner.build_secret_values = values
        return self.owner.build_secret


class FakeApp:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def cls(self, **kwargs: Any):
        self.owner.cls_options = kwargs
        return lambda runtime_class: runtime_class


class FakeModal:
    def __init__(self) -> None:
        self.Image = FakeImageAPI(self)
        self.Secret = FakeSecretAPI(self)
        self.build_secret = object()
        self.app_name = ""
        self.tags: dict[str, str] = {}
        self.packages: tuple[str, ...] = ()
        self.python_version = ""
        self.build_function: Any = None
        self.build_options: dict[str, Any] = {}
        self.image_environment: dict[str, str] = {}
        self.source_modules: tuple[str, ...] = ()
        self.source_options: dict[str, Any] = {}
        self.build_secret_values: dict[str, str] = {}
        self.cls_options: dict[str, Any] = {}
        self.requires_proxy_auth: list[bool] = []
        self.apps_created = 0

    def App(self, app_name: str, *, image: FakeImage, tags: dict[str, str]) -> FakeApp:
        assert isinstance(image, FakeImage)
        self.apps_created += 1
        self.app_name = app_name
        self.tags = tags
        return FakeApp(self)

    @staticmethod
    def enter():
        return lambda function: function

    @staticmethod
    def exit():
        return lambda function: function

    def asgi_app(self, *, requires_proxy_auth: bool):
        self.requires_proxy_auth.append(requires_proxy_auth)
        return lambda function: function


def test_runtime_workload_bakes_one_revision_and_applies_gpu_guardrails() -> None:
    fake_modal = FakeModal()

    workload = build_modal_runtime_workload(
        app_name=f"ctrl-pi-{DEPLOYMENT_ID}",
        deployment_id=DEPLOYMENT_ID,
        resources=_resources(),
        runtime="lerobot",
        model_repo=MODEL_REPO,
        checkpoint_revision=REVISION,
        hf_token=HF_SECRET,
        actions_per_chunk=20,
        modal_module=fake_modal,  # type: ignore[arg-type]
    )

    assert workload.app.owner is fake_modal
    assert callable(workload.health_function)
    assert fake_modal.app_name == f"ctrl-pi-{DEPLOYMENT_ID}"
    assert fake_modal.tags == {MODAL_OWNERSHIP_TAG_KEY: str(DEPLOYMENT_ID)}
    assert fake_modal.python_version == "3.11"
    assert fake_modal.packages == MODAL_RUNTIME_PACKAGES
    assert fake_modal.build_secret_values == {"HF_TOKEN": HF_SECRET}
    assert fake_modal.build_function is _download_model_snapshot
    assert fake_modal.build_options == {
        "args": (MODEL_REPO, REVISION),
        "secrets": [fake_modal.build_secret],
    }
    assert fake_modal.image_environment == MODAL_RUNTIME_OFFLINE_ENV
    assert HF_SECRET not in repr(fake_modal.build_options)
    assert fake_modal.source_modules == ("ctrl_pi",)
    assert fake_modal.source_options == {"copy": True}
    assert fake_modal.requires_proxy_auth == [True]
    assert "secrets" not in fake_modal.cls_options
    assert "env" not in fake_modal.cls_options
    assert fake_modal.cls_options == {
        "image": ANY,
        "gpu": MODAL_GPU_MAP["Modal: A10G"],
        "serialized": True,
        "min_containers": 0,
        "buffer_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "timeout": 1800,
    }
    runtime_class = workload.health_function.__self__.__class__
    closure_values = [
        cell.cell_contents
        for function in (runtime_class.load, runtime_class.close, runtime_class.web)
        for cell in (function.__closure__ or ())
    ]
    assert HF_SECRET not in repr(closure_values)


def test_runtime_workload_constructs_with_pinned_official_modal_sdk_without_network() -> None:
    workload = build_modal_runtime_workload(
        app_name=f"ctrl-pi-{DEPLOYMENT_ID}",
        deployment_id=DEPLOYMENT_ID,
        resources=_resources(),
        runtime="lerobot",
        model_repo=MODEL_REPO,
        checkpoint_revision=REVISION,
        hf_token=HF_SECRET,
        modal_module=modal,
    )

    assert isinstance(workload.app, modal.App)
    assert isinstance(workload.health_function, modal.Function)
    assert sorted(workload.app.registered_functions) == [f"{MODAL_RUNTIME_CLASS}.*"]


@pytest.mark.parametrize(
    ("runtime", "revision", "token", "compute_size"),
    [
        ("openpi", REVISION, HF_SECRET, "Modal: A10G"),
        ("lerobot", "main", HF_SECRET, "Modal: A10G"),
        ("lerobot", REVISION, "", "Modal: A10G"),
        ("lerobot", REVISION, "   ", "Modal: A10G"),
        ("lerobot", REVISION, HF_SECRET, "CPU"),
    ],
)
def test_runtime_workload_rejects_invalid_configuration_before_modal_objects(
    runtime: str,
    revision: str,
    token: str,
    compute_size: str,
) -> None:
    fake_modal = FakeModal()

    with pytest.raises(ComputeConfigurationError):
        build_modal_runtime_workload(
            app_name=f"ctrl-pi-{DEPLOYMENT_ID}",
            deployment_id=DEPLOYMENT_ID,
            resources=_resources(compute_size),
            runtime=runtime,  # type: ignore[arg-type]
            model_repo=MODEL_REPO,
            checkpoint_revision=revision,
            hf_token=token,
            modal_module=fake_modal,  # type: ignore[arg-type]
        )

    assert fake_modal.apps_created == 0
    assert fake_modal.build_secret_values == {}


def test_snapshot_download_is_explicitly_tokened_and_writes_identity_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setenv("HF_TOKEN", HF_SECRET)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    _download_model_snapshot(MODEL_REPO, REVISION, str(tmp_path))

    assert calls == [
        {
            "repo_id": MODEL_REPO,
            "revision": REVISION,
            "token": HF_SECRET,
            "local_dir": str(tmp_path),
        }
    ]
    assert json.loads((tmp_path / MODAL_MODEL_MANIFEST).read_text()) == {
        "schema": 1,
        "model_repo": MODEL_REPO,
        "revision": REVISION,
    }


def test_runtime_asgi_post_and_websocket_use_the_same_strict_envelopes() -> None:
    with TestClient(create_runtime_asgi_app(_runtime())) as client:
        health = client.post("/", json={"nonce": "proof"})
        assert health.status_code == 200
        assert health.json() == {
            "type": "health",
            "healthy": True,
            "echo": "proof",
            "runtime": "lerobot",
            "model_repo": MODEL_REPO,
            "revision": REVISION,
        }

        describe = client.post("/", json={"type": "describe"})
        assert describe.status_code == 200
        assert describe.json()["type"] == "runtime"
        assert describe.json()["revision"] == REVISION

        expected = client.post("/", json=_observation()).json()
        assert expected["type"] == "action_chunk"
        assert len(expected["actions"]) == 3
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(_observation())
            streamed = websocket.receive_json()
            assert streamed["type"] == expected["type"]
            assert streamed["request_id"] == expected["request_id"]
            assert streamed["observation_sequence"] == expected["observation_sequence"]
            assert streamed["revision"] == expected["revision"]
            assert streamed["actions"] == expected["actions"]


def test_runtime_websocket_resets_policy_once_per_connection() -> None:
    class CountingRuntime(StubInferenceRuntime):
        resets = 0

        def reset_session(self) -> None:
            super().reset_session()
            self.resets += 1

    runtime = CountingRuntime(runtime="lerobot")
    runtime.load(
        RuntimeLoadSpec(
            model_repo=MODEL_REPO,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
            actions_per_chunk=3,
        )
    )
    with TestClient(create_runtime_asgi_app(runtime)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(_observation())
            assert websocket.receive_json()["type"] == "action_chunk"
        assert runtime.resets == 1


def test_runtime_websocket_reset_failure_closes_safely_before_receive() -> None:
    secret = "reset-provider-secret"

    class FailingResetRuntime(StubInferenceRuntime):
        def reset_session(self) -> None:
            raise RuntimeProtocolError(f"reset exposed {secret}")

    runtime = FailingResetRuntime(runtime="lerobot")
    runtime.load(
        RuntimeLoadSpec(
            model_repo=MODEL_REPO,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
        )
    )
    with TestClient(create_runtime_asgi_app(runtime)) as client:
        with client.websocket_connect("/ws") as websocket:
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
    assert closed.value.code == 1011
    assert secret not in closed.value.reason


def test_runtime_post_describe_resets_session_and_sanitizes_reset_failure() -> None:
    class CountingRuntime(StubInferenceRuntime):
        resets = 0

        def reset_session(self) -> None:
            super().reset_session()
            self.resets += 1

    runtime = CountingRuntime(runtime="lerobot")
    runtime.load(
        RuntimeLoadSpec(
            model_repo=MODEL_REPO,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
        )
    )
    with TestClient(create_runtime_asgi_app(runtime)) as client:
        assert client.post("/", json={"type": "describe"}).status_code == 200
        assert runtime.resets == 1

    secret = "post-reset-provider-secret"

    class FailingRuntime(CountingRuntime):
        def reset_session(self) -> None:
            raise InferenceRuntimeError(f"reset exposed {secret}")

    failed_runtime = FailingRuntime(runtime="lerobot")
    failed_runtime.load(
        RuntimeLoadSpec(
            model_repo=MODEL_REPO,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
        )
    )
    with TestClient(create_runtime_asgi_app(failed_runtime)) as client:
        failed = client.post("/", json={"type": "describe"})
    assert failed.status_code == 503
    assert secret not in failed.text


def test_runtime_asgi_never_reflects_invalid_input_or_raw_runtime_errors() -> None:
    secret = "runtime-provider-secret"

    class FailingRuntime(StubInferenceRuntime):
        def predict(self, request: ObservationRequest):
            del request
            raise RuntimeProtocolError(f"bad payload contained {secret}")

    runtime = FailingRuntime(runtime="lerobot")
    runtime.load(
        RuntimeLoadSpec(
            model_repo=MODEL_REPO,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
        )
    )
    with TestClient(create_runtime_asgi_app(runtime)) as client:
        invalid = client.post(
            "/",
            content=json.dumps({"type": "describe", "leak": secret}),
            headers={"content-type": "application/json"},
        )
        assert invalid.status_code == 422
        assert secret not in invalid.text

        failed = client.post("/", json=_observation())
        assert failed.status_code == 422
        assert secret not in failed.text

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "describe", "leak": secret})
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
        assert closed.value.code == 1008


def test_runtime_asgi_enforces_json_body_and_websocket_frame_boundaries() -> None:
    with TestClient(create_runtime_asgi_app(_runtime())) as client:
        oversized = client.post(
            "/",
            content=b"{}",
            headers={
                "content-type": "application/json",
                "content-length": str(MAX_POST_BODY_BYTES + 1),
            },
        )
        assert oversized.status_code == 413

        duplicate = client.post(
            "/",
            content=b'{"type":"describe","type":"health"}',
            headers={"content-type": "application/json"},
        )
        assert duplicate.status_code == 422

        wrong_media = client.post(
            "/",
            content=b'{"type":"describe"}',
            headers={"content-type": "text/plain"},
        )
        assert wrong_media.status_code == 422

        with client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(b'{"type":"observation"}')
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
        assert closed.value.code == 1008


def test_framework_independent_handler_rejects_duplicate_json_keys() -> None:
    handler = RuntimeRequestHandler(_runtime())

    with pytest.raises(ValueError, match="invalid"):
        _decode_json('{"type":"describe","type":"health"}')

    for nonfinite in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="invalid"):
            _decode_json(f'{{"type":"observation","sequence":{nonfinite}}}')

    with pytest.raises(ValueError, match="invalid"):
        handler.dispatch({"type": "describe", "unexpected": True})


def test_runtime_workload_maps_invalid_repo_to_compute_configuration_error() -> None:
    fake_modal = FakeModal()

    with pytest.raises(ComputeConfigurationError, match="repo"):
        build_modal_runtime_workload(
            app_name=f"ctrl-pi-{DEPLOYMENT_ID}",
            deployment_id=DEPLOYMENT_ID,
            resources=_resources(),
            runtime="lerobot",
            model_repo="not-a-namespaced-repo",
            checkpoint_revision=REVISION,
            hf_token=HF_SECRET,
            modal_module=fake_modal,  # type: ignore[arg-type]
        )

    assert fake_modal.apps_created == 0
