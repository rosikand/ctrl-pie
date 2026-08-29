from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.drivers.yam import (
    JOINT_NAMES,
    ActionLimitError,
    ArmAction,
    JogCommand,
    JogLimitError,
)
from ctrl_pi.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(MockYAMDriver()))


def test_list_and_detail_expose_four_metadata_rich_mock_arms(client: TestClient) -> None:
    response = client.get("/api/arms")

    assert response.status_code == 200
    arms = response.json()["arms"]
    assert [(arm["id"], arm["role"]) for arm in arms] == [
        ("yam-leader", "leader"),
        ("yam-follower", "follower"),
        ("yam-leader-left", "leader"),
        ("yam-follower-left", "follower"),
    ]
    assert [joint["name"] for joint in arms[0]["joints"]] == list(JOINT_NAMES)
    assert arms[0]["can"]["state"] == "active"
    assert arms[0]["connected"] is True
    assert arms[0]["pair_id"] == "right"
    assert arms[0]["group_id"] == "bimanual"
    assert arms[0]["end_effector_kind"] == "yam_teaching_handle"
    assert arms[0]["handle"]["reachable"] is True
    assert arms[1]["end_effector_kind"] == "linear_4310"
    assert arms[1]["control_state"] == "gravity_comp"
    assert arms[1]["holding"] is True
    assert arms[1]["warnings"] == ["NO SASH GUARD"]

    detail = client.get("/api/arms/yam-follower")
    assert detail.status_code == 200
    assert detail.json()["id"] == "yam-follower"
    assert client.get("/api/arms/not-an-arm").status_code == 404


def test_mock_safe_idle_keeps_connected_follower_visibly_holding() -> None:
    driver = MockYAMDriver()

    follower = driver.safe_idle("yam-follower")

    assert follower.connected is True
    assert follower.energized is True
    assert follower.control_state == "gravity_comp"
    assert follower.holding is True

    disconnected = driver.disconnect_arms(["yam-follower"])[0]
    assert disconnected.connected is False
    assert disconnected.energized is False
    assert disconnected.holding is False


def test_mock_fault_remains_conservatively_live_until_explicit_disconnect() -> None:
    driver = MockYAMDriver()

    driver.latch_fault(["yam-follower"], "sanitized test fault")
    faulted = driver.get_arm("yam-follower")

    assert faulted.connected is False
    assert faulted.control_state == "error"
    assert faulted.energized is True
    assert faulted.holding is True

    disconnected = driver.disconnect_arms(["yam-follower"])[0]
    assert disconnected.energized is False
    assert disconnected.holding is False


def test_joint_jog_changes_only_process_local_live_state(client: TestClient) -> None:
    before = client.get("/api/arms/yam-follower").json()
    shoulder_before = before["joints"][0]["position_radians"]
    pose_before = before["pose"]

    response = client.post(
        "/api/arms/yam-follower/jog",
        json={"kind": "joint", "axis": "shoulder_yaw", "delta": 0.1},
    )

    assert response.status_code == 200
    shoulder_after = response.json()["joints"][0]["position_radians"]
    assert shoulder_after == pytest.approx(shoulder_before + 0.1)
    assert response.json()["joints"][0]["velocity_radians_per_second"] > 0
    assert response.json()["pose"] != pose_before
    assert (
        client.get("/api/arms/yam-follower").json()["joints"][0]["position_radians"]
        == pytest.approx(shoulder_after)
    )


def test_mock_leaders_are_observation_only_for_jog_and_actions() -> None:
    driver = MockYAMDriver()
    leader = driver.get_arm("yam-leader")

    with pytest.raises(JogLimitError, match="observation-only"):
        driver.jog(
            leader.id,
            JogCommand(kind="joint", axis="shoulder_yaw", delta=0.1),
        )
    with pytest.raises(ActionLimitError, match="observation-only"):
        driver.apply_action(leader.id, ArmAction.from_telemetry(leader))

    after = driver.get_arm(leader.id)
    assert [joint.position_radians for joint in after.joints] == [
        joint.position_radians for joint in leader.joints
    ]
    assert after.gripper.position == leader.gripper.position
    assert driver.action_counts[leader.id] == 0


def test_jog_rejects_a_busy_rig_before_mutating_the_arm(client: TestClient) -> None:
    before = client.get("/api/arms/yam-follower").json()
    token = client.app.state.rig_lease.acquire("inference", "deployment-1")
    try:
        response = client.post(
            "/api/arms/yam-follower/jog",
            json={"kind": "joint", "axis": "shoulder_yaw", "delta": 0.1},
        )
    finally:
        client.app.state.rig_lease.release(token)

    after = client.get("/api/arms/yam-follower").json()
    assert response.status_code == 409
    assert "controlled by inference" in response.json()["detail"]
    assert after["joints"][0]["position_radians"] == pytest.approx(
        before["joints"][0]["position_radians"]
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"kind": "joint", "axis": "x", "delta": 0.1}, "axis must be one of"),
        ({"kind": "joint", "axis": "shoulder_yaw", "delta": 0.26}, "no greater than 0.25"),
        ({"kind": "cartesian", "axis": "x", "delta": 0.051}, "no greater than 0.05"),
        ({"kind": "gripper", "axis": "position", "delta": 0}, "non-zero"),
    ],
)
def test_jog_rejects_unallowlisted_or_unbounded_commands(
    client: TestClient, payload: dict[str, object], message: str
) -> None:
    response = client.post("/api/arms/yam-follower/jog", json=payload)

    assert response.status_code == 422
    assert message in response.text


def test_jog_rejects_motion_beyond_arm_safety_limits(client: TestClient) -> None:
    for _ in range(12):
        response = client.post(
            "/api/arms/yam-follower/jog",
            json={"kind": "gripper", "axis": "position", "delta": 0.1},
        )
        if response.status_code == 409:
            break

    assert response.status_code == 409
    assert "outside its safe range" in response.json()["detail"]


def test_websocket_streams_low_latency_all_arm_snapshots(client: TestClient) -> None:
    with client.websocket_connect("/ws/arms") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["type"] == "telemetry"
    assert [arm["id"] for arm in first["arms"]] == [
        "yam-leader",
        "yam-follower",
        "yam-leader-left",
        "yam-follower-left",
    ]
    first_timestamp = datetime.fromisoformat(first["timestamp"])
    second_timestamp = datetime.fromisoformat(second["timestamp"])
    assert 0 < (second_timestamp - first_timestamp).total_seconds() < 0.2
