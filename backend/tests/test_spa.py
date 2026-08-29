from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ctrl_pi.config import AppConfig
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.main import create_app


REPOSITORY_ROOT = Path(__file__).parents[2]


def test_frontend_has_six_primary_routes_and_separate_training_models_owners() -> None:
    app_source = (REPOSITORY_ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    training_source = (REPOSITORY_ROOT / "frontend/src/pages/TrainingPage.tsx").read_text(encoding="utf-8")
    models_source = (REPOSITORY_ROOT / "frontend/src/pages/ModelsPage.tsx").read_text(encoding="utf-8")
    navigation_source = app_source.split(
        "const primaryNavigation: NavigationItem[] = [", 1
    )[1].split("]", 1)[0]

    expected_navigation = (
        '{ label: "Arms", path: "/arms"',
        '{ label: "Record / Teleop", mobileLabel: "Record", path: "/record"',
        '{ label: "Datasets", path: "/datasets"',
        '{ label: "Training", path: "/training"',
        '{ label: "Models", path: "/models"',
        '{ label: "Inference", path: "/inference"',
    )
    assert navigation_source.count("{ label:") == 6
    assert all(item in navigation_source for item in expected_navigation)
    assert all(
        f'<Route path="{path}"' in app_source
        for path in ("/arms", "/record", "/datasets", "/training", "/models", "/inference")
    )
    assert "grid-cols-6" in app_source

    assert "useTrainerModels" not in training_source
    assert "TrainerModelSummary" not in training_source
    assert "ModelCard" not in training_source
    assert 'from "./TrainingPage"' not in models_source
    assert "useTrainerModels" in models_source
    assert "TrainerModelSummary" in models_source
    assert "function ModelCard" in models_source
    assert "const visibleLogs = logs.slice().reverse();" in training_source
    assert "logs.slice(-500)" not in training_source
    assert "ManagedJobCard" in training_source
    assert 'job.target_kind === "modal"' in training_source
    assert "Stub · no GPU" in training_source
    assert "job.output_revision ?? job.output_marker_revision" in training_source
    assert "run.managed_job?.output_revision ?? run.managed_job?.output_marker_revision" in training_source
    assert "managedArtifactUrl ?" in training_source
    assert ") : <ArtifactLink repoId={run.output_model_repo} />" in training_source
    assert "Requested repo · existence not yet verified" in training_source
    assert "Simulation teardown complete · no provider tasks" in training_source
    assert "job.teardown_verified" in training_source
    assert "job.event_gap" in training_source
    assert "Launch a managed job from the Python SDK" in training_source
    assert "launch_managed_training" not in training_source
    assert "cancel_managed_training" not in training_source


def test_settings_exposes_service_backed_yam_onboarding_without_browser_device_logic() -> None:
    app_source = (REPOSITORY_ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    settings_source = (REPOSITORY_ROOT / "frontend/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    panel_source = (REPOSITORY_ROOT / "frontend/src/components/YamSetupPanel.tsx").read_text(encoding="utf-8")
    hook_source = (REPOSITORY_ROOT / "frontend/src/hooks/useYamSetup.ts").read_text(encoding="utf-8")
    api_source = (REPOSITORY_ROOT / "frontend/src/lib/api.ts").read_text(encoding="utf-8")
    type_source = (REPOSITORY_ROOT / "frontend/src/types/yamSetup.ts").read_text(encoding="utf-8")

    assert "<YamSetupPanel onSettingsRefresh={refresh} />" in settings_source
    assert 'id="yam-setup"' in panel_source
    assert "Discovery only lists network interfaces and stable serial candidates" in panel_source
    assert "It does not open a bus, start a controller, or move an arm." in panel_source
    assert "calibration file is readiness evidence only" in panel_source
    assert "I secured the workspace" in panel_source
    assert "emergency stop" in panel_source
    assert "requires_physical_validation" in panel_source or "Physical directions, offsets" in panel_source
    assert "Multiple candidates are intentionally not auto-selected" in panel_source
    assert "Mock mode provides a deterministic leader" in panel_source
    assert "const [autoRestore, setAutoRestore] = useState(false);" in panel_source
    assert "yam.setup.auto_restore || !yam.setup.saved" not in panel_source
    assert "canDisableAutomaticConnection" in panel_source
    assert "It does not disconnect hardware that is already connected." in panel_source
    assert 'setup.saved ? "Waiting for saved devices" : "Configured hardware missing"' in panel_source
    assert "Another robot operation is active" not in panel_source
    assert "{yam.error}" in panel_source
    assert "dirty || autoRestoreDirty || !yam.setup.calibration_ready" in panel_source
    assert "acknowledge_automatic_motion_risk" in type_source
    assert "acknowledge_hardware_motion_risk" in type_source
    assert "acknowledge_automatic_motion_risk" in hook_source
    assert "acknowledge_hardware_motion_risk" in hook_source
    assert 'maxLength={15}' in panel_source

    assert '"/api/yam/setup"' in api_source
    assert '"/api/yam/setup/discover"' in api_source
    assert '"/api/yam/setup/preflight"' in api_source
    assert '"/api/yam/setup/connect"' in api_source
    assert '`/api/models${suffix}`' in api_source
    assert '`/api/trainer/models${suffix}`' not in api_source
    assert "navigator.serial" not in panel_source
    assert "navigator.usb" not in panel_source
    assert "new WebSocket" not in panel_source

    # Refresh remains stable after state updates and every request is abortable.
    assert "const hasSetup = useRef(false);" in hook_source
    assert "currentController" in hook_source
    assert "operationRef" in hook_source
    assert "controller.signal" in hook_source
    assert 'document.visibilityState !== "visible"' in hook_source
    assert 'document.addEventListener("visibilitychange"' in hook_source
    assert 'setup.state === "error"' in hook_source
    assert "}, [setup]);" not in hook_source
    assert "lastGlobalStatus" in panel_source
    assert "onSettingsRefresh();" in panel_source
    assert "fetchYamSetup" in app_source
    assert 'location.pathname === "/settings"' in app_source
    assert 'yam.saved && yam.auto_restore' in app_source


def _frontend_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>ctrl-pi-spa</div></body></html>",
        encoding="utf-8",
    )
    (assets / "index-DEADBEEF.js").write_text(
        "globalThis.ctrlPiLoaded = true;",
        encoding="utf-8",
    )
    return dist


def test_configured_distribution_serves_root_assets_and_spa_deep_links(
    tmp_path: Path,
) -> None:
    dist = _frontend_dist(tmp_path)
    app = create_app(frontend_dist_dir=dist)

    with TestClient(app) as client:
        root = client.get("/")
        head = client.head("/inference")
        asset = client.get("/assets/index-DEADBEEF.js")
        deep_link = client.get("/datasets/example.dataset")

    assert app.state.frontend_serving_enabled is True
    assert app.state.frontend_dist_dir == dist
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert root.headers["cache-control"] == "no-cache"
    assert "ctrl-pi-spa" in root.text
    assert head.status_code == 200
    assert head.content == b""
    assert asset.status_code == 200
    assert asset.text == "globalThis.ctrlPiLoaded = true;"
    assert "ctrl-pi-spa" in deep_link.text


def test_spa_fallback_never_converts_api_socket_asset_or_media_404s_to_html(
    tmp_path: Path,
) -> None:
    app = create_app(frontend_dist_dir=_frontend_dist(tmp_path))

    with TestClient(app) as client:
        responses = [
            client.get("/api/not-a-route", headers={"accept": "text/html"}),
            client.get("/ws/not-a-socket", headers={"accept": "text/html"}),
            client.get("/assets/missing.js", headers={"accept": "text/html"}),
            client.get("/missing-recording.mp4", headers={"accept": "text/html"}),
            client.get("/favicon.svg", headers={"accept": "text/html"}),
        ]

    assert all(response.status_code == 404 for response in responses)
    assert all("ctrl-pi-spa" not in response.text for response in responses)


def test_api_and_websocket_routes_take_precedence_over_spa_fallback(
    tmp_path: Path,
) -> None:
    app = create_app(
        yam_driver=MockYAMDriver(),
        frontend_dist_dir=_frontend_dist(tmp_path),
    )

    with TestClient(app) as client:
        health = client.get("/api/health", headers={"accept": "text/html"})
        with client.websocket_connect("/ws/arms") as websocket:
            telemetry = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/not-a-socket"):
                pass

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "mode": "mock"}
    assert telemetry["type"] == "telemetry"
    assert [arm["id"] for arm in telemetry["arms"]] == [
        "yam-leader",
        "yam-follower",
    ]


@pytest.mark.parametrize("configured_path", ["missing", "incomplete"])
def test_missing_or_incomplete_distribution_keeps_api_only_startup(
    tmp_path: Path,
    configured_path: str,
) -> None:
    dist = tmp_path / configured_path
    if configured_path == "incomplete":
        (dist / "assets").mkdir(parents=True)

    app = create_app(frontend_dist_dir=dist)
    with TestClient(app) as client:
        health = client.get("/api/health")
        root = client.get("/")

    assert app.state.frontend_serving_enabled is False
    assert app.state.frontend_dist_dir is None
    assert health.status_code == 200
    assert root.status_code == 404


def test_frontend_distribution_is_opt_in_for_vite_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRONTEND_DIST_DIR", raising=False)

    config = AppConfig(_env_file=None)

    assert config.frontend_dist_dir is None
