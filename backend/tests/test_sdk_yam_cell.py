from __future__ import annotations

import json

import httpx
from pydantic import ValidationError
import pytest

from ctrl_pi import (
    ArmTelemetry,
    CtrlPiClient,
    CtrlPiError,
    YAMCellArmConfig,
    YAMCellConfig,
    YAMCellDiscovery,
    YAMCellPreflight,
    YAMCellStatus,
    YAMHandleRangeResult,
    YAMSetupConfig,
)


NOW = "2026-08-29T12:00:00Z"


def _cell() -> YAMCellConfig:
    return YAMCellConfig(
        name="Test cell",
        i2rt_root="/opt/operator/i2rt",
        i2rt_commit="a" * 40,
        arms=[
            YAMCellArmConfig(
                logical_id="right-leader",
                name="Right leader",
                role="leader",
                pair_id="right-pair",
                group_id="bimanual",
                side="right",
                transport_kind="socketcan",
                stable_identity="adapter-right-leader",
                end_effector_kind="yam_teaching_handle",
            ),
            YAMCellArmConfig(
                logical_id="right-follower",
                name="Right follower",
                role="follower",
                pair_id="right-pair",
                group_id="bimanual",
                side="right",
                transport_kind="socketcan",
                stable_identity="adapter-right-follower",
                end_effector_kind="linear_4310",
                frame_map_path="/opt/operator/maps/right.json",
                soft_limits_path="/opt/operator/limits/right.json",
            ),
        ],
        pair_ports={"right-pair": 5001},
    )


def _status(cell: YAMCellConfig) -> dict[str, object]:
    return {
        "mode": "hardware",
        "state": "ready",
        "configured": True,
        "saved": True,
        "connected": True,
        "any_connected": True,
        "all_connected": True,
        "configured_arm_count": 2,
        "connected_arm_count": 2,
        "arms": [
            {
                "arm_id": "right-leader",
                "role": "leader",
                "pair_id": "right-pair",
                "group_id": "bimanual",
                "side": "right",
                "connected": True,
                "control_state": "gravity_comp",
                "energized": True,
                "holding": False,
                "runtime_interface": "can7",
                "error": None,
            },
            {
                "arm_id": "right-follower",
                "role": "follower",
                "pair_id": "right-pair",
                "group_id": "bimanual",
                "side": "right",
                "connected": True,
                "control_state": "position_control",
                "energized": True,
                "holding": True,
                "runtime_interface": "can2",
                "error": None,
            },
        ],
        "calibration_ready": True,
        "auto_restore": False,
        "restored_on_boot": False,
        "config": cell.model_dump(mode="json"),
        "diagnostic": {"status": "connected", "detail": "Two arms connected."},
        "last_attempt_at": NOW,
        "last_connected_at": NOW,
        "requires_physical_validation": True,
    }


def _discovery(cell: YAMCellConfig) -> dict[str, object]:
    return {
        "mode": "hardware",
        "devices": [
            {
                "transport_kind": "socketcan",
                "stable_identity": "adapter-right-leader",
                "product": "USB-CAN adapter",
                "runtime_interface": "can7",
                "link_state": "up",
                "duplicate_identity": False,
            }
        ],
        "resolutions": [
            {
                "arm_id": "right-leader",
                "transport_kind": "socketcan",
                "stable_identity": "adapter-right-leader",
                "runtime_interface": "can7",
                "resolved": True,
                "conflict": False,
                "detail": "Resolved without opening the device.",
            }
        ],
        "can_interfaces": [],
        "leader_ports": [],
        "suggested_config": cell.model_dump(mode="json"),
        "detail": "Passive discovery completed.",
    }


def _preflight() -> dict[str, object]:
    return {
        "ready": True,
        "calibration_ready": True,
        "diagnostic": {"status": "configured", "detail": "Cell is ready."},
        "i2rt_ready": True,
        "arms": [
            {
                "arm_id": "right-follower",
                "ready": True,
                "runtime_interface": "can2",
                "link_state": "up",
                "frame_map_status": "active",
                "soft_limits_status": "missing",
                "handle_status": "not_applicable",
                "warnings": ["NO SASH GUARD"],
                "diagnostic": {
                    "status": "configured",
                    "detail": "Follower passively resolved.",
                },
            }
        ],
        "warnings": ["NO SASH GUARD"],
    }


def test_cell_sdk_uses_canonical_routes_and_explicit_motion_boundaries() -> None:
    cell = _cell()
    status = _status(cell)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        route = (request.method, request.url.path)
        if route == ("POST", "/api/yam/cell/discover"):
            return httpx.Response(200, json=_discovery(cell))
        if route == ("POST", "/api/yam/cell/preflight"):
            return httpx.Response(200, json=_preflight())
        if route == ("POST", "/api/yam/cell/handle-check"):
            return httpx.Response(
                200,
                json={
                    "arm_id": "right-leader",
                    "reachable": True,
                    "observed_minimum": 0.02,
                    "observed_maximum": 0.98,
                    "healthy": True,
                    "detail": "Teaching handle range is healthy.",
                },
            )
        return httpx.Response(200, json=status)

    with CtrlPiClient(
        "http://ctrl-pi.local", _transport=httpx.MockTransport(handler)
    ) as client:
        assert isinstance(client.get_yam_cell(), YAMCellStatus)
        assert isinstance(client.discover_yam_cell(), YAMCellDiscovery)
        assert isinstance(client.preflight_yam_cell(cell), YAMCellPreflight)
        client.save_yam_cell(
            cell,
            auto_restore=False,
            acknowledge_automatic_motion_risk=False,
            acknowledge_gripper_calibration_motion=True,
        )
        client.connect_yam_arms(
            arm_ids=["right-follower"],
            acknowledge_hardware_motion_risk=True,
            acknowledge_gripper_calibration_motion=True,
        )
        client.disconnect_yam_arms()
        result = client.check_yam_handle(
            "right-leader",
            duration_seconds=3,
            acknowledge_active_can_diagnostic=True,
        )
        assert isinstance(result, YAMHandleRangeResult)
        assert result.healthy is True

    calls = {(request.method, request.url.path) for request in requests}
    assert calls == {
        ("GET", "/api/yam/cell"),
        ("POST", "/api/yam/cell/discover"),
        ("POST", "/api/yam/cell/preflight"),
        ("PUT", "/api/yam/cell"),
        ("POST", "/api/yam/cell/connect"),
        ("POST", "/api/yam/cell/disconnect"),
        ("POST", "/api/yam/cell/handle-check"),
    }
    connect = next(
        request for request in requests if request.url.path == "/api/yam/cell/connect"
    )
    assert json.loads(connect.content) == {
        "arm_ids": ["right-follower"],
        "acknowledge_hardware_motion_risk": True,
        "acknowledge_gripper_calibration_motion": True,
    }
    disconnect = next(
        request
        for request in requests
        if request.url.path == "/api/yam/cell/disconnect"
    )
    assert json.loads(disconnect.content) == {"arm_ids": None}
    handle = next(
        request
        for request in requests
        if request.url.path == "/api/yam/cell/handle-check"
    )
    assert json.loads(handle.content) == {
        "arm_id": "right-leader",
        "duration_seconds": 3.0,
        "acknowledge_active_can_diagnostic": True,
    }


def test_cell_sdk_accepts_additive_arm_telemetry() -> None:
    telemetry = ArmTelemetry.model_validate_json(
        json.dumps(
            {
            "id": "right-leader",
            "name": "Right leader",
            "role": "leader",
            "pair_id": "right-pair",
            "group_id": "bimanual",
            "side": "right",
            "transport_kind": "socketcan",
            "stable_identity": "adapter-right-leader",
            "end_effector_kind": "yam_teaching_handle",
            "driver": "i2rt-worker",
            "connected": True,
            "control_state": "gravity_comp",
            "energized": True,
            "holding": False,
            "timestamp": NOW,
            "joints": [],
            "pose": {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
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
                "interface": "can7",
                "state": "active",
                "bitrate": None,
                "tx_error_count": None,
                "rx_error_count": None,
            },
            "control_loop": {
                "target_frequency_hz": 425.0,
                "frequency_hz": 421.0,
                "cycle_time_ms": 2.37,
                "jitter_ms": 0.12,
                "dropped_cycles": 0,
                "source": "i2rt-motor-chain",
            },
            "handle": {
                "reachable": True,
                "trigger_position": 0.25,
                "buttons": [False, True],
                "range_status": "healthy",
                "observed_minimum": 0.01,
                "observed_maximum": 0.99,
                "calibration_warning": None,
            },
            "frame_map_active": False,
            "soft_limits_active": False,
            "warnings": [],
            }
        )
    )
    assert telemetry.handle is not None
    assert telemetry.handle.range_status == "healthy"
    assert telemetry.control_loop.source == "i2rt-motor-chain"


def test_cell_sdk_fails_closed_before_transport_on_invalid_selection() -> None:
    calls: list[httpx.Request] = []
    client = CtrlPiClient(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: (calls.append(request), httpx.Response(200, json={}))[1]
        ),
    )

    with pytest.raises(CtrlPiError, match="selection"):
        client.connect_yam_arms(
            arm_ids=[],
            acknowledge_hardware_motion_risk=True,
            acknowledge_gripper_calibration_motion=True,
        )
    with pytest.raises(CtrlPiError, match="duplicates"):
        client.disconnect_yam_arms(arm_ids=["right-leader", "right-leader"])
    with pytest.raises(CtrlPiError, match="acknowledgement"):
        client.connect_yam_arms(
            acknowledge_hardware_motion_risk=1,  # type: ignore[arg-type]
            acknowledge_gripper_calibration_motion=True,
        )
    with pytest.raises(CtrlPiError, match="duration"):
        client.check_yam_handle(
            "right-leader",
            duration_seconds=0.5,
            acknowledge_active_can_diagnostic=True,
        )
    with pytest.raises(CtrlPiError, match="cell config"):
        client.preflight_yam_cell(  # type: ignore[arg-type]
            YAMSetupConfig(
                can_interface="can0",
                leader_port="/dev/serial/by-id/legacy",
                mujoco_xml_path="/opt/yam/model.xml",
                leader_calibration_id="legacy",
                leader_calibration_dir="/opt/yam/calibration",
            )
        )
    assert calls == []
    client.close()


def test_cell_config_rejects_ephemeral_can_identity_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="ephemeral canN"):
        YAMCellArmConfig(
            logical_id="leader",
            name="Leader",
            role="leader",
            transport_kind="socketcan",
            stable_identity="can0",
            end_effector_kind="yam_teaching_handle",
        )

    payload = _cell().model_dump(mode="json")
    payload["operator_secret"] = "must-not-be-accepted"
    with pytest.raises(ValidationError, match="Extra inputs"):
        YAMCellConfig.model_validate(payload)


def test_cell_domain_accepts_an_empty_topology_draft() -> None:
    cell = YAMCellConfig(
        name="Unassigned cell",
        i2rt_root="/opt/operator/i2rt",
        i2rt_commit="a" * 40,
        arms=[],
    )

    assert cell.arms == []
    assert cell.pair_ports == {}
