from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from ctrl_pi.camera import MockCamera
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.main import create_app
from ctrl_pi.rig import RigLease


class LifecycleDriver(MockYAMDriver):
    def __init__(self, events: list[str], *, fail_startup: bool = False) -> None:
        super().__init__()
        self.events = events
        self.fail_startup = fail_startup
        self.startup_thread: str | None = None
        self.shutdown_thread: str | None = None

    def startup(self) -> None:
        self.startup_thread = threading.current_thread().name
        self.events.append("driver.startup")
        if self.fail_startup:
            raise RuntimeError("driver startup failed")

    def shutdown(self) -> None:
        self.shutdown_thread = threading.current_thread().name
        self.events.append("driver.shutdown")


class LifecycleRecordingManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.rig_lease = RigLease()
        self.camera = MockCamera()

    async def startup(self) -> None:
        self.events.append("recording.startup")

    async def shutdown(self) -> None:
        self.events.append("recording.shutdown")


class LifecycleDeploymentService:
    session_factory = None

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def reconcile_startup(self) -> None:
        self.events.append("deployment.reconcile")


class LifecycleInferenceManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def startup(self) -> None:
        self.events.append("inference.startup")

    async def shutdown(self) -> None:
        self.events.append("inference.shutdown")


def _app(events: list[str], driver: LifecycleDriver):
    return create_app(
        yam_driver=driver,
        recording_manager=LifecycleRecordingManager(events),  # type: ignore[arg-type]
        deployment_service=LifecycleDeploymentService(events),  # type: ignore[arg-type]
        inference_session_manager=LifecycleInferenceManager(events),  # type: ignore[arg-type]
    )


def test_app_lifespan_starts_driver_first_and_stops_it_after_all_loops() -> None:
    events: list[str] = []
    driver = LifecycleDriver(events)

    with TestClient(_app(events, driver)) as client:
        assert client.get("/api/health").status_code == 200
        assert events == [
            "driver.startup",
            "recording.startup",
            "deployment.reconcile",
            "inference.startup",
        ]

    assert events == [
        "driver.startup",
        "recording.startup",
        "deployment.reconcile",
        "inference.startup",
        "inference.shutdown",
        "recording.shutdown",
        "driver.shutdown",
    ]
    assert driver.startup_thread is not None
    assert driver.shutdown_thread is not None
    assert driver.startup_thread != threading.current_thread().name
    assert driver.shutdown_thread != threading.current_thread().name


def test_app_lifespan_shuts_driver_down_when_driver_startup_raises() -> None:
    events: list[str] = []
    driver = LifecycleDriver(events, fail_startup=True)

    with pytest.raises(RuntimeError, match="driver startup failed"):
        with TestClient(_app(events, driver)):
            pass

    assert events == [
        "driver.startup",
        "inference.shutdown",
        "driver.shutdown",
    ]
