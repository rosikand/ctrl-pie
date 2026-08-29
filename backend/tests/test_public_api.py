from __future__ import annotations

from fastapi.testclient import TestClient

from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.main import create_app


def test_openapi_covers_every_current_ctrl_pi_product_workflow() -> None:
    app = create_app(yam_driver=MockYAMDriver())
    schema = app.openapi()
    paths = schema["paths"]
    expected_methods = {
        "/api/health": {"get"},
        "/api/settings/status": {"get"},
        "/api/settings": {"get", "patch"},
        "/api/yam/setup": {"get", "put", "delete"},
        "/api/yam/setup/discover": {"post"},
        "/api/yam/setup/preflight": {"post"},
        "/api/yam/setup/connect": {"post"},
        "/api/arms": {"get"},
        "/api/arms/{arm_id}": {"get"},
        "/api/arms/{arm_id}/jog": {"post"},
        "/api/camera/stream": {"get"},
        "/api/recordings": {"get", "post"},
        "/api/recordings/{recording_id}": {"get"},
        "/api/recordings/{recording_id}/state": {"get"},
        "/api/recordings/{recording_id}/teleop/start": {"post"},
        "/api/recordings/{recording_id}/teleop/stop": {"post"},
        "/api/recordings/{recording_id}/episodes/start": {"post"},
        "/api/recordings/{recording_id}/episodes/stop": {"post"},
        "/api/recordings/{recording_id}/upload": {"post"},
        "/api/datasets": {"get"},
        "/api/datasets/{repo_name}/episodes": {"get"},
        "/api/datasets/{repo_name}/episodes/{episode_index}": {"get"},
        "/api/datasets/{repo_name}/episodes/{episode_index}/video": {"get"},
        "/api/models": {"get"},
        "/api/trainer/models": {"get"},
        "/api/trainer/runs": {"get", "post"},
        "/api/trainer/runs/{run_id}": {"get", "patch"},
        "/api/trainer/runs/{run_id}/metrics": {"post"},
        "/api/trainer/runs/{run_id}/logs": {"get", "post"},
        "/api/trainer/runs/{run_id}/checkpoints": {"post"},
        "/api/inference/deployments": {"get", "post"},
        "/api/inference/deployments/{deployment_id}": {"get"},
        "/api/inference/deployments/{deployment_id}/start": {"post"},
        "/api/inference/deployments/{deployment_id}/state": {"get"},
        "/api/inference/deployments/{deployment_id}/stop": {"post"},
    }

    for path, methods in expected_methods.items():
        assert path in paths
        assert methods.issubset(paths[path])

    operation_ids = [
        operation["operationId"]
        for path in paths.values()
        for method, operation in path.items()
        if method in {"get", "post", "put", "patch", "delete"}
    ]
    assert len(operation_ids) == len(set(operation_ids))

    registered_routes = [
        child
        for route in app.routes
        for child in (
            getattr(getattr(route, "original_router", None), "routes", None)
            or [route]
        )
    ]
    route_paths = {
        path
        for route in registered_routes
        if isinstance(path := getattr(route, "path", None), str)
    }
    assert "/ws/arms" in route_paths
    assert "/api/inference/deployments/{deployment_id}/stream" in route_paths


def test_models_has_a_first_class_contract_and_deprecated_compatibility_alias() -> None:
    schema = create_app(yam_driver=MockYAMDriver()).openapi()
    canonical = schema["paths"]["/api/models"]["get"]
    legacy = schema["paths"]["/api/trainer/models"]["get"]

    assert canonical.get("deprecated") is not True
    assert canonical["tags"] == ["models"]
    assert canonical["operationId"] == "list_models"
    assert legacy["deprecated"] is True
    assert legacy["tags"] == ["trainer"]
    assert legacy["operationId"] == "list_trainer_models_legacy"
    assert canonical["responses"]["200"] == legacy["responses"]["200"]


def test_arm_identifiers_are_bounded_before_driver_access() -> None:
    client = TestClient(create_app(yam_driver=MockYAMDriver()))
    oversized = "a" * 121

    assert client.get(f"/api/arms/{oversized}").status_code == 422
    assert client.post(
        f"/api/arms/{oversized}/jog",
        json={"kind": "joint", "axis": "shoulder_yaw", "delta": 0.1},
    ).status_code == 422

    arm_parameter = client.app.openapi()["paths"]["/api/arms/{arm_id}"]["get"][
        "parameters"
    ][0]
    assert arm_parameter["schema"]["minLength"] == 1
    assert arm_parameter["schema"]["maxLength"] == 120
