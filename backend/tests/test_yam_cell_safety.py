from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ctrl_pi.drivers.yam_cell_safety import (
    MAX_SAFETY_CONFIG_BYTES,
    NO_SASH_GUARD,
    YAMSafetyConfigError,
    load_yam_follower_safety,
    load_yam_frame_map,
    load_yam_soft_limits,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_absent_frame_map_is_valid_intentional_identity(tmp_path: Path) -> None:
    frame_map = load_yam_frame_map(None)

    assert not frame_map.present
    assert not frame_map.active
    assert frame_map.apply([1, 2, 3, 4, 5, 6, 0.25]) == [
        1,
        2,
        3,
        4,
        5,
        6,
        0.25,
    ]


def test_explicit_identity_map_is_present_but_not_active(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "identity.json",
        {"sign": [1, 1, 1, 1, 1, 1], "offset": [0, 0, 0, 0, 0, 0]},
    )

    frame_map = load_yam_frame_map(path)

    assert frame_map.present
    assert not frame_map.active


def test_reference_frame_map_metadata_is_accepted_without_affecting_mapping(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "reference-map.json",
        {
            "sign": [-1, 1, 1, 1, 1, 1],
            "offset": [1.5, 0, 0, 0, 0, 0],
            "leader": "operator-label-only",
            "follower": "operator-label-only",
            "notes": "field artifact metadata",
        },
    )

    frame_map = load_yam_frame_map(path)

    assert frame_map.apply([0.5, 0, 0, 0, 0, 0, 0.25]) == [
        1.0,
        0,
        0,
        0,
        0,
        0,
        0.25,
    ]


def test_frame_map_is_an_involution_and_never_maps_gripper(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "map.json",
        {"sign": [-1, 1, -1, 1, 1, 1], "offset": [math.pi, 0, 1, 0, 0, 0]},
    )
    frame_map = load_yam_frame_map(path)
    command = [0.2, 0.3, -0.4, 0.5, 0.6, 0.7, 0.8]

    mapped = frame_map.apply(command)
    round_trip = frame_map.apply(mapped)

    assert frame_map.active
    assert mapped[:6] == pytest.approx(
        [math.pi - 0.2, 0.3, 1.4, 0.5, 0.6, 0.7]
    )
    assert mapped[6] == 0.8
    assert round_trip == pytest.approx(command)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"sign": [1] * 6},
        {"sign": [1] * 5, "offset": [0] * 6},
        {"sign": [1, 1, 1, 1, 1, 2], "offset": [0] * 6},
        {"sign": [1] * 6, "offset": [0, 0, 0, 0, 0, True]},
        {"sign": [1] * 6, "offset": [0, 0, 0, 0, 0, float("nan")]},
        {"sign": [1] * 6, "offset": [0, 0, 0, 0, 0, 0.1]},
    ],
)
def test_malformed_or_non_involutive_frame_maps_are_errors(
    tmp_path: Path, value: object
) -> None:
    path = _write_json(tmp_path / "bad-map.json", value)

    with pytest.raises(YAMSafetyConfigError):
        load_yam_frame_map(path)


def test_invalid_json_frame_map_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bad-json.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(YAMSafetyConfigError, match="not valid JSON"):
        load_yam_frame_map(path)


def test_configured_missing_safety_artifacts_are_errors(tmp_path: Path) -> None:
    with pytest.raises(YAMSafetyConfigError, match="frame map file does not exist"):
        load_yam_frame_map(tmp_path / "missing-map.json")
    with pytest.raises(YAMSafetyConfigError, match="soft limits file does not exist"):
        load_yam_soft_limits(tmp_path / "missing-limits.json")


def test_absent_soft_limits_emit_exact_no_sash_guard_warning(tmp_path: Path) -> None:
    limits = load_yam_soft_limits(None)

    assert not limits.present
    assert not limits.active
    assert limits.warning == NO_SASH_GUARD == "NO SASH GUARD"
    assert limits.clamp([1, 2, 3, 4, 5, 6, 0.5]) == [1, 2, 3, 4, 5, 6, 0.5]


def test_all_null_soft_limits_do_not_claim_an_active_guard(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "unbounded.json",
        {"lower": [None] * 6, "upper": [None] * 6},
    )

    limits = load_yam_soft_limits(path)

    assert limits.present
    assert not limits.active
    assert limits.warning == "NO SASH GUARD"


def test_soft_limits_clamp_only_first_six_joints(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "limits.json",
        {
            "lower": [0, None, -1, None, None, None],
            "upper": [1, None, 1, None, None, None],
        },
    )
    limits = load_yam_soft_limits(path)

    clamped = limits.clamp([-2, 2, 3, 4, 5, 6, 0.75])

    assert limits.present and limits.active
    assert limits.warning is None
    assert clamped == [0, 2, 1, 4, 5, 6, 0.75]


def test_reference_soft_limit_serial_is_verified_and_channel_is_not_identity(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "stamped-limits.json",
        {
            "lower": [-1] * 6,
            "upper": [1] * 6,
            "serial": "USB-CAN-DURABLE-ID",
            "channel": "can17",
        },
    )

    limits = load_yam_soft_limits(
        path, expected_stable_identity="USB-CAN-DURABLE-ID"
    )

    assert limits.active
    with pytest.raises(YAMSafetyConfigError, match="adapter serial does not match"):
        load_yam_soft_limits(path, expected_stable_identity="DIFFERENT-ADAPTER")


def test_combined_guard_maps_before_clamping_and_preserves_gripper(
    tmp_path: Path,
) -> None:
    frame_map = _write_json(
        tmp_path / "map.json",
        {"sign": [-1, 1, 1, 1, 1, 1], "offset": [2, 0, 0, 0, 0, 0]},
    )
    limits = _write_json(
        tmp_path / "limits.json",
        {"lower": [0, None, None, None, None, None], "upper": [1] * 6},
    )
    safety = load_yam_follower_safety(
        frame_map_path=frame_map, soft_limits_path=limits
    )

    guarded = safety.apply([3, 0, 0, 0, 0, 0, 0.75])

    assert guarded == [0, 0, 0, 0, 0, 0, 0.75]
    assert safety.warnings == ()


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"lower": [None] * 6},
        {"lower": [None] * 5, "upper": [None] * 6},
        {"lower": [None] * 6, "upper": [None, None, None, None, None, True]},
        {"lower": [None] * 6, "upper": [None, None, None, None, None, math.inf]},
        {"lower": [1, None, None, None, None, None], "upper": [0] * 6},
    ],
)
def test_malformed_or_mismatched_soft_limits_are_errors(
    tmp_path: Path, value: object
) -> None:
    path = _write_json(tmp_path / "bad-limits.json", value)

    with pytest.raises(YAMSafetyConfigError):
        load_yam_soft_limits(path)


def test_safety_artifacts_and_commands_are_bounded(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_SAFETY_CONFIG_BYTES + 1))

    with pytest.raises(YAMSafetyConfigError, match="bounded size"):
        load_yam_soft_limits(oversized)

    identity = load_yam_frame_map(None)
    with pytest.raises(YAMSafetyConfigError, match="at least six"):
        identity.apply([0] * 5)
    with pytest.raises(YAMSafetyConfigError, match="finite number"):
        identity.apply([0, 0, 0, 0, 0, math.nan])

    assert identity.apply(np.zeros(6, dtype=np.float32)) == [0.0] * 6


def test_present_but_unreadable_artifact_is_not_treated_as_absent(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "not-a-file"
    directory.mkdir()

    with pytest.raises(YAMSafetyConfigError, match="not readable"):
        load_yam_frame_map(directory)
