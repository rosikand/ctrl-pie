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
