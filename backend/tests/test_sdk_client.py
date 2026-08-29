from __future__ import annotations

from collections import Counter
from importlib.resources import files
import inspect
import json
from typing import get_type_hints
from uuid import UUID

import httpx
import pytest

from ctrl_pi import CtrlPiClient, CtrlPiError, YAMSetupConfig
from ctrl_pi._http import MAX_JSON_RESPONSE_BYTES
from ctrl_pi.sdk_models import (
    ArmTelemetry,
    DatasetEpisode,
    Deployment,
    HealthStatus,
    InferenceState,
    ModelPage,
    Recording,
    TrainingRun,
    YAMSetupStatus,
)


RUN_ID = "11111111-1111-4111-8111-111111111111"
RECORDING_ID = "22222222-2222-4222-8222-222222222222"
DEPLOYMENT_ID = "33333333-3333-4333-8333-333333333333"
ENDPOINT_ID = "44444444-4444-4444-8444-444444444444"
REVISION = "a" * 40
NOW = "2026-08-29T00:00:00Z"


def _yam_config() -> dict[str, object]:
    return {
        "can_interface": "can0",
        "leader_port": "/dev/serial/by-id/mock-yam",
        "mujoco_xml_path": "/opt/yam/model.xml",
        "leader_calibration_id": "leader",
        "leader_calibration_dir": "/opt/yam/calibration",
    }


def _yam_status() -> dict[str, object]:
    return {
        "mode": "mock",
        "state": "ready",
        "configured": True,
        "saved": False,
        "connected": True,
        "calibration_ready": True,
        "auto_restore": False,
        "restored_on_boot": False,
        "config": _yam_config(),
        "diagnostic": {"status": "connected", "detail": "Mock rig is ready."},
        "last_attempt_at": None,
        "last_connected_at": NOW,
        "requires_physical_validation": False,
    }


def _arm() -> dict[str, object]:
    return {
        "id": "yam-follower",
        "name": "YAM follower",
        "role": "follower",
        "driver": "mock",
        "connected": True,
        "timestamp": NOW,
        "joints": [
            {
                "name": "shoulder_yaw",
                "position_radians": 0.1,
                "velocity_radians_per_second": 0.0,
                "effort_newton_meters": None,
                "temperature_celsius": None,
            }
        ],
        "pose": {
            "x_m": 0.1,
            "y_m": 0.2,
            "z_m": 0.3,
            "roll_radians": 0.0,
            "pitch_radians": 0.0,
            "yaw_radians": 0.0,
        },
        "gripper": {
            "position": 0.5,
            "velocity": 0.0,
            "force_newtons": None,
            "is_closed": False,
        },
        "can": {
            "interface": "can0",
            "state": "active",
            "bitrate": 1_000_000,
            "tx_error_count": 0,
            "rx_error_count": 0,
        },
        "control_loop": {
            "target_frequency_hz": 100.0,
            "frequency_hz": 99.5,
            "cycle_time_ms": 10.0,
            "jitter_ms": 0.1,
            "dropped_cycles": 0,
        },
    }


def _recording() -> dict[str, object]:
    return {
        "id": RECORDING_ID,
        "name": "pick cube",
        "task": "pick the cube",
        "status": "ready",
        "leader_robot_id": "yam-leader",
        "follower_robot_id": "yam-follower",
        "episode_count": 1,
        "duration_seconds": 5.0,
        "hf_repo_id": None,
        "metadata": {},
        "created_at": NOW,
        "updated_at": NOW,
    }


def _recording_state() -> dict[str, object]:
    return {
        "recording_id": RECORDING_ID,
        "teleop_active": False,
        "episode_active": False,
        "current_episode_index": None,
        "episode_duration_seconds": 0.0,
        "episode_count": 1,
        "status": "ready",
    }


def _run() -> dict[str, object]:
    return {
        "id": RUN_ID,
        "name": "ACT demo",
        "status": "running",
        "current_step": 10,
        "dataset_repo": "acme/data",
        "base_model": "lerobot/act-base",
        "runtime": "lerobot",
        "framework": "pytorch",
        "output_model_repo": None,
        "checkpoint_revision": None,
        "config": {},
        "metrics": {"loss": [{"step": 10, "value": 0.5}]},
        "checkpoints": [],
        "created_at": NOW,
        "updated_at": NOW,
    }


def _deployment() -> dict[str, object]:
    return {
        "id": DEPLOYMENT_ID,
        "endpoint_id": ENDPOINT_ID,
        "name": "mock policy",
        "target_kind": "stub",
        "status": "running",
        "model_repo": "ctrl-pi/mock-policy",
        "checkpoint_revision": "0" * 40,
        "runtime": "stub",
        "compute_size": "CPU",
        "timeout_seconds": 1800,
        "endpoint_url": None,
        "provider_app_id": "stub-app",
        "arm_id": None,
        "record_session": False,
        "recording_id": None,
        "started_at": NOW,
        "stopped_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _inference_state() -> dict[str, object]:
    return {
        **_deployment(),
        "session_status": "running",
        "endpoint_healthy": True,
        "teardown_verified": False,
        "steps_executed": 100,
        "requests_completed": 5,
        "dropped_chunks": 0,
        "queue_depth": 0,
        "last_latency_ms": 8.5,
        "average_latency_ms": 9.0,
        "frequency_hz": 20.0,
        "last_error": None,
        "session_started_at": NOW,
        "session_stopped_at": None,
        "recording": {
            "enabled": False,
            "status": "disabled",
            "recording_id": None,
            "episode_count": 0,
            "duration_seconds": 0.0,
            "hf_repo_id": None,
        },
    }


def _response_for(request: httpx.Request) -> dict[str, object]:
    path = request.url.path
    method = request.method
    if path == "/api/health":
        return {"status": "ok", "mode": "mock"}
    if path == "/api/settings/status":
        return {
            "mode": "mock",
            "setup_complete": True,
            "services": [],
            "inference": {
                "mock_mode": True,
                "hf_configured": False,
                "modal_configured": False,
                "modal_proxy_configured": False,
            },
        }
    if path == "/api/settings":
        return {
            "hf_namespace": "acme",
            "recording_fps": 20,
            "default_runtime": "lerobot",
            "default_compute": "Modal: A10G",
            "modal_timeout_minutes": 30,
        }
    if path == "/api/yam/setup/discover":
        return {
            "mode": "mock",
            "can_interfaces": [{"id": "can0", "label": "Mock CAN"}],
            "leader_ports": [
                {"id": "/dev/serial/by-id/mock-yam", "label": "Mock leader"}
            ],
            "suggested_config": _yam_config(),
            "detail": "Mock discovery result.",
        }
    if path == "/api/yam/setup/preflight":
        return {
            "ready": True,
            "calibration_ready": True,
            "diagnostic": {"status": "configured", "detail": "Ready to connect."},
        }
    if path in {"/api/yam/setup", "/api/yam/setup/connect"}:
        return _yam_status()
    if path == "/api/arms":
        return {"arms": [_arm()]}
    if path.startswith("/api/arms/"):
        return _arm()
    if path == "/api/recordings":
        return {"recordings": [_recording()]} if method == "GET" else _recording()
    if path.endswith("/upload"):
        return {
            "recording": _recording(),
            "repo_id": "acme/pick-cube",
            "repo_url": "https://huggingface.co/datasets/acme/pick-cube",
            "revision": REVISION,
        }
    if path.endswith("/state") and path.startswith("/api/recordings/"):
        return _recording_state()
    if path.startswith("/api/recordings/") and (
        "/teleop/" in path or "/episodes/" in path
    ):
        return _recording_state()
    if path == f"/api/recordings/{RECORDING_ID}":
        return _recording()
    if path == "/api/datasets":
        return {
            "namespace": "acme",
            "datasets": [],
            "total": 0,
            "next_cursor": None,
            "fetched_at": NOW,
        }
    if path == "/api/datasets/pick-cube/episodes":
        return {
            "repo_id": "acme/pick-cube",
            "revision": REVISION,
            "fps": 20.0,
            "state_names": ["joint"],
            "action_names": ["joint"],
            "video_key": "observation.images.front",
            "total_episodes": 1,
            "episodes": [],
        }
    if path == "/api/datasets/pick-cube/episodes/0":
        summary = {
            "episode_index": 0,
            "tasks": ["pick"],
            "frame_count": 1,
            "duration_seconds": 0.05,
            "dataset_from_index": 0,
            "dataset_to_index": 1,
            "video_from_timestamp": 0.0,
            "video_to_timestamp": 0.05,
        }
        return {
            "repo_id": "acme/pick-cube",
            "revision": REVISION,
            "fps": 20.0,
            "state_names": ["joint"],
            "action_names": ["joint"],
            "video_key": "observation.images.front",
            "episode": summary,
            "frames": [
                {"timestamp": 0.0, "frame_index": 0, "state": [0.1], "action": [0.2]}
            ],
            "sampled_frame_count": 1,
            "frames_truncated": False,
            "video_url": f"/api/datasets/pick-cube/episodes/0/video?revision={REVISION}",
        }
    if path == "/api/models":
        return {
            "namespace": "acme",
            "models": [],
            "total": 0,
            "fetched_at": NOW,
        }
    if path == "/api/trainer/runs":
        return {"runs": [_run()]} if method == "GET" else _run()
    if path.endswith("/logs"):
        if method == "GET":
            return {
                "logs": [],
                "oldest_sequence": None,
                "latest_sequence": None,
                "next_sequence": 0,
                "truncated": False,
                "has_more": False,
            }
        return {
            "sequence": 1,
            "source": "stdout",
            "line": "step",
            "step": 10,
            "timestamp": NOW,
        }
    if path.startswith(f"/api/trainer/runs/{RUN_ID}"):
        return _run()
    if path == "/api/inference/deployments":
        return {"deployments": [_deployment()]} if method == "GET" else _deployment()
    if path.endswith("/state") or path.endswith("/start") or path.endswith("/stop"):
        return _inference_state()
    if path == f"/api/inference/deployments/{DEPLOYMENT_ID}":
        return _deployment()
    raise AssertionError(f"Unhandled SDK request: {method} {path}")


def test_sdk_covers_the_current_public_rest_workflows_with_typed_results() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response_for(request))

    config = YAMSetupConfig.model_validate(_yam_config())
    with CtrlPiClient(
        "http://ctrl-pi.local/", _transport=httpx.MockTransport(handler)
    ) as client:
        assert isinstance(client.health(), HealthStatus)
        client.get_system_status()
        client.get_settings()
        client.update_settings(recording_fps=30)
        assert isinstance(client.get_yam_setup(), YAMSetupStatus)
        client.discover_yam()
        client.preflight_yam(config)
        client.save_yam_setup(
            config,
            auto_restore=True,
            acknowledge_automatic_motion_risk=True,
        )
        client.connect_yam(acknowledge_hardware_motion_risk=True)
        client.reset_yam_setup()
        assert isinstance(client.list_arms()[0], ArmTelemetry)
        client.get_arm("yam-follower")
        client.jog_arm("yam-follower", kind="joint", axis="shoulder_yaw", delta=0.1)
        assert isinstance(client.list_recordings()[0], Recording)
        client.create_recording(
            "pick cube",
            task="pick",
            leader_robot_id="yam-leader",
            follower_robot_id="yam-follower",
        )
        client.get_recording(RECORDING_ID)
        client.get_recording_state(RECORDING_ID)
        client.start_teleop(RECORDING_ID)
        client.stop_teleop(RECORDING_ID)
        client.start_episode(RECORDING_ID, operator="operator")
        client.stop_episode(RECORDING_ID, success=True)
        client.upload_recording(RECORDING_ID, repo_name="pick-cube", private=True)
        client.list_datasets(limit=10, cursor="cursor", refresh=True)
        client.list_dataset_episodes("pick-cube")
        assert isinstance(
            client.get_dataset_episode("pick-cube", 0, revision=REVISION),
            DatasetEpisode,
        )
        assert isinstance(client.list_models(refresh=True), ModelPage)
        assert isinstance(client.create_run("ACT demo"), TrainingRun)
        client.list_runs(status="running")
        client.get_run(RUN_ID)
        client.update_run(RUN_ID, status="running", output_model_repo=None)
        client.log_metrics(RUN_ID, step=10, metrics={"loss": 0.5})
        client.list_console_logs(RUN_ID, after_sequence=0, limit=10)
        client.log_console(RUN_ID, line="step", step=10)
        client.register_checkpoint(
            RUN_ID, repo_id="acme/policy", revision=REVISION, step=10
        )
        assert isinstance(
            client.deploy("mock policy", model_repo="ctrl-pi/mock-policy"), Deployment
        )
        client.list_deployments()
        client.get_deployment(DEPLOYMENT_ID)
        client.get_inference_state(DEPLOYMENT_ID)
        assert isinstance(
            client.start_inference(
                DEPLOYMENT_ID,
                arm_id="yam-follower",
                task="pick",
                record_session=True,
                recording_name="inference pick",
                recording_operator="operator",
            ),
            InferenceState,
        )
        client.stop_inference(DEPLOYMENT_ID, recording_success=True)
        assert not client.is_closed
    assert client.is_closed

    calls = Counter((request.method, request.url.path) for request in requests)
    assert calls[("GET", "/api/models")] == 1
    assert calls[("GET", "/api/trainer/models")] == 0
    assert calls[("GET", f"/api/recordings/{RECORDING_ID}")] == 1
    assert calls[("GET", f"/api/trainer/runs/{RUN_ID}/logs")] == 1
    assert calls[("POST", f"/api/inference/deployments/{DEPLOYMENT_ID}/stop")] == 1

    yam_save = next(
        request
        for request in requests
        if request.method == "PUT" and request.url.path == "/api/yam/setup"
    )
    assert json.loads(yam_save.content)["acknowledge_automatic_motion_risk"] is True
    yam_connect = next(
        request
        for request in requests
        if request.url.path == "/api/yam/setup/connect"
    )
    assert json.loads(yam_connect.content) == {
        "acknowledge_hardware_motion_risk": True
    }
    upload = next(request for request in requests if request.url.path.endswith("/upload"))
    assert json.loads(upload.content)["private"] is True
    dataset_detail = next(
        request
        for request in requests
        if request.url.path == "/api/datasets/pick-cube/episodes/0"
    )
    assert dataset_detail.url.params["revision"] == REVISION


@pytest.mark.parametrize(
    "base_url",
    [
        "ctrl-pi.local:8000",
        "http://[::1",
        "http://user:secret@ctrl-pi.local",
        "http://ctrl-pi.local?token=secret",
        "http://ctrl-pi.local/#secret",
        "http://ctrl-pi.local/prefix",
    ],
)
def test_sdk_rejects_ambiguous_or_credential_bearing_base_urls(base_url: str) -> None:
    with pytest.raises(CtrlPiError, match="base URL is invalid") as caught:
        CtrlPiClient(base_url)
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_sdk_rejects_invalid_timeouts(timeout: float) -> None:
    with pytest.raises(CtrlPiError, match="timeout is invalid"):
        CtrlPiClient("http://ctrl-pi.local", timeout=timeout)


def test_sdk_does_not_follow_redirects_or_echo_error_bodies() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            302,
            headers={"location": "http://user:redirect-secret@attacker.invalid"},
            json={"detail": "body-secret"},
        )

    client = CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    )
    with pytest.raises(CtrlPiError) as caught:
        client.health()
    assert caught.value.status_code == 302
    assert "secret" not in str(caught.value)
    assert calls == 1
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert client._http._client.follow_redirects is False
    assert client._http._client.trust_env is False
    client.close()


def test_sdk_sanitizes_network_closed_and_invalid_response_errors() -> None:
    def leak(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("transport-secret", request=request)

    client = CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(leak)
    )
    with pytest.raises(CtrlPiError, match="Could not reach") as network:
        client.health()
    assert "secret" not in str(network.value)
    assert network.value.__cause__ is None
    assert network.value.__context__ is None
    client.close()

    with pytest.raises(CtrlPiError, match="Could not reach") as closed:
        client.health()
    assert closed.value.__cause__ is None
    assert closed.value.__context__ is None

    malformed = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"not json")
        ),
    )
    with pytest.raises(CtrlPiError, match="invalid response") as invalid:
        malformed.health()
    assert invalid.value.__cause__ is None
    assert invalid.value.__context__ is None
    malformed.close()

    wrong_shape = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"status": "ok", "mode": "mock", "unexpected": True}
            )
        ),
    )
    with pytest.raises(CtrlPiError, match="invalid response"):
        wrong_shape.health()
    wrong_shape.close()

    serialization = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: pytest.fail("invalid JSON must not reach the transport")
        ),
    )
    with pytest.raises(CtrlPiError, match="SDK request is invalid") as request_error:
        serialization.create_recording(
            "demo",
            task="demo",
            leader_robot_id="yam-leader",
            follower_robot_id="yam-follower",
            metadata={"private_note": object()},
        )
    assert request_error.value.__cause__ is None
    assert request_error.value.__context__ is None
    serialization.close()


def test_sdk_rejects_declared_and_streamed_oversized_json(monkeypatch: pytest.MonkeyPatch) -> None:
    declared = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-length": str(MAX_JSON_RESPONSE_BYTES + 1)},
                content=b"{}",
            )
        ),
    )
    with pytest.raises(CtrlPiError, match="too large") as caught:
        declared.health()
    assert caught.value.__context__ is None
    declared.close()

    import ctrl_pi._http as safe_http

    monkeypatch.setattr(safe_http, "MAX_JSON_RESPONSE_BYTES", 5)

    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"st'
            yield b'atus":"ok"}'

    streamed = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Chunks())
        ),
    )
    with pytest.raises(CtrlPiError, match="too large"):
        streamed.health()
    streamed.close()


def test_sdk_rejects_unsafe_identifiers_before_any_request() -> None:
    calls: list[httpx.Request] = []
    client = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: (calls.append(request), httpx.Response(200, json={}))[1]
        ),
    )

    invalid_calls = [
        lambda: client.get_run("../models"),
        lambda: client.get_recording("not-a-uuid"),
        lambda: client.get_arm("../yam-follower"),
        lambda: client.list_dataset_episodes("../private"),
        lambda: client.get_dataset_episode("safe", 0, revision="main"),
        lambda: client.list_datasets(cursor="x" * 1_025),
    ]
    for call in invalid_calls:
        with pytest.raises(CtrlPiError):
            call()
    assert calls == []
    client.close()


def test_sdk_requires_explicit_safety_and_visibility_flags() -> None:
    client = CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(lambda request: None)
    )
    with pytest.raises(TypeError):
        client.connect_yam()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        client.upload_recording(RECORDING_ID, repo_name="dataset")  # type: ignore[call-arg]
    with pytest.raises(CtrlPiError, match="visibility flag"):
        client.upload_recording(
            RECORDING_ID, repo_name="dataset", private=1  # type: ignore[arg-type]
        )
    client.close()


def test_sdk_never_retries_a_failed_motion_or_lifecycle_mutation() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"detail": "provider-secret"})

    client = CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    )
    with pytest.raises(CtrlPiError) as caught:
        client.stop_inference(DEPLOYMENT_ID)
    assert caught.value.status_code == 503
    assert len(calls) == 1
    assert "secret" not in str(caught.value)
    client.close()


def test_sdk_response_models_are_strict_and_reject_non_finite_numbers() -> None:
    coerced = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'{"status":"ok","mode":1}')
        ),
    )
    with pytest.raises(CtrlPiError, match="invalid response"):
        coerced.health()
    coerced.close()

    arm = _arm()
    arm["pose"] = {**arm["pose"], "x_m": float("nan")}  # type: ignore[dict-item]
    non_finite = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(arm).encode())
        ),
    )
    with pytest.raises(CtrlPiError, match="invalid response"):
        non_finite.get_arm("yam-follower")
    non_finite.close()


def test_public_ids_are_real_uuid_values() -> None:
    client = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_recording())
        ),
    )
    recording = client.get_recording(RECORDING_ID)
    assert recording.id == UUID(RECORDING_ID)
    client.close()


def test_sdk_is_pep_561_typed_and_update_annotations_do_not_collapse_to_object() -> None:
    assert files("ctrl_pi").joinpath("py.typed").is_file()
    update_hints = get_type_hints(CtrlPiClient.update_run)
    settings_hints = get_type_hints(CtrlPiClient.update_settings)
    assert update_hints["status"] is not object
    assert settings_hints["recording_fps"] is not object
    assert "object" not in str(inspect.signature(CtrlPiClient.update_run))
