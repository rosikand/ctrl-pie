from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


YAM_ARM_JOINT_COUNT = 6
MAX_SAFETY_CONFIG_BYTES = 64 * 1024
NO_SASH_GUARD = "NO SASH GUARD"


class YAMSafetyConfigError(ValueError):
    """A frame-map or soft-limit artifact is present but unsafe to use."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_SAFETY_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise YAMSafetyConfigError(f"{label} file is not readable") from exc
    if len(payload) > MAX_SAFETY_CONFIG_BYTES:
        raise YAMSafetyConfigError(f"{label} file exceeds the bounded size limit")
    try:
        parsed = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise YAMSafetyConfigError(f"{label} file is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise YAMSafetyConfigError(f"{label} must be a JSON object")
    return parsed


def _require_keys(
    raw: dict[str, Any], *, required: frozenset[str], label: str
) -> None:
    missing = required - set(raw)
    if missing:
        fields = ", ".join(sorted(missing))
        raise YAMSafetyConfigError(f"{label} is missing required fields: {fields}")


def _require_six_entries(value: Any, *, field: str, label: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != YAM_ARM_JOINT_COUNT:
        raise YAMSafetyConfigError(
            f"{label} '{field}' must contain exactly six entries"
        )
    return value


def _finite_number(value: Any, *, field: str, index: int, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise YAMSafetyConfigError(
            f"{label} '{field}' entry {index + 1} must be a finite number"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise YAMSafetyConfigError(
            f"{label} '{field}' entry {index + 1} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise YAMSafetyConfigError(
            f"{label} '{field}' entry {index + 1} must be a finite number"
        )
    return number


def _finite_command(command: Sequence[float]) -> list[float]:
    if len(command) < YAM_ARM_JOINT_COUNT:
        raise YAMSafetyConfigError("follower command must contain at least six joints")
    values: list[float] = []
    for index, value in enumerate(command):
        number = _finite_number(
            value, field="command", index=index, label="follower command"
        )
        values.append(number)
    return values


@dataclass(frozen=True, slots=True)
class YAMFrameMap:
    sign: tuple[float, float, float, float, float, float]
    offset: tuple[float, float, float, float, float, float]
    present: bool
    active: bool
    source_path: str | None = None

    @classmethod
    def identity(
        cls, *, source_path: str | None = None, present: bool = False
    ) -> YAMFrameMap:
        return cls(
            sign=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            present=present,
            active=False,
            source_path=source_path,
        )

    def apply(self, command: Sequence[float]) -> list[float]:
        values = _finite_command(command)
        for index in range(YAM_ARM_JOINT_COUNT):
            values[index] = values[index] * self.sign[index] + self.offset[index]
        return values


@dataclass(frozen=True, slots=True)
class YAMSoftJointLimits:
    lower: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]
    upper: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]
    present: bool
    active: bool
    source_path: str | None = None

    @classmethod
    def absent(cls, *, source_path: str | None = None) -> YAMSoftJointLimits:
        empty = (None, None, None, None, None, None)
        return cls(
            lower=empty,
            upper=empty,
            present=False,
            active=False,
            source_path=source_path,
        )

    @property
    def warning(self) -> str | None:
        return None if self.active else NO_SASH_GUARD

    def clamp(self, command: Sequence[float]) -> list[float]:
        values = _finite_command(command)
        for index in range(YAM_ARM_JOINT_COUNT):
            lower = self.lower[index]
            upper = self.upper[index]
            if lower is not None:
                values[index] = max(values[index], lower)
            if upper is not None:
                values[index] = min(values[index], upper)
        return values


@dataclass(frozen=True, slots=True)
class YAMFollowerSafety:
    frame_map: YAMFrameMap
    soft_limits: YAMSoftJointLimits

    @property
    def warnings(self) -> tuple[str, ...]:
        warning = self.soft_limits.warning
        return () if warning is None else (warning,)

    def apply(self, command: Sequence[float]) -> list[float]:
        """Map leader frame first, then clamp follower joints; preserve gripper."""

        return self.soft_limits.clamp(self.frame_map.apply(command))


def load_yam_frame_map(path: str | Path | None) -> YAMFrameMap:
    """Load a strict six-joint involution; absence intentionally means identity."""

    if path is None:
        return YAMFrameMap.identity()
    source = str(path)
    raw = _load_json_object(Path(path), label="frame map")
    if raw is None:
        raise YAMSafetyConfigError("configured frame map file does not exist")
    _require_keys(
        raw, required=frozenset({"sign", "offset"}), label="frame map"
    )
    raw_sign = _require_six_entries(raw["sign"], field="sign", label="frame map")
    raw_offset = _require_six_entries(
        raw["offset"], field="offset", label="frame map"
    )

    sign_values: list[float] = []
    offset_values: list[float] = []
    for index, value in enumerate(raw_sign):
        sign = _finite_number(value, field="sign", index=index, label="frame map")
        if sign not in {-1.0, 1.0}:
            raise YAMSafetyConfigError(
                f"frame map 'sign' entry {index + 1} must be +1 or -1"
            )
        sign_values.append(sign)
    for index, value in enumerate(raw_offset):
        offset_values.append(
            _finite_number(value, field="offset", index=index, label="frame map")
        )

    for index, (sign, offset) in enumerate(zip(sign_values, offset_values, strict=True)):
        if sign == 1.0 and offset != 0.0:
            raise YAMSafetyConfigError(
                "frame map must be an involution; "
                f"joint {index + 1} has sign=+1 with non-zero offset"
            )

    sign_tuple = tuple(sign_values)
    offset_tuple = tuple(offset_values)
    identity = all(value == 1.0 for value in sign_tuple) and all(
        value == 0.0 for value in offset_tuple
    )
    return YAMFrameMap(
        sign=sign_tuple,  # type: ignore[arg-type]
        offset=offset_tuple,  # type: ignore[arg-type]
        present=True,
        active=not identity,
        source_path=source,
    )


def load_yam_soft_limits(
    path: str | Path | None,
    *,
    expected_stable_identity: str | None = None,
) -> YAMSoftJointLimits:
    """Load strict follower joint bounds; absence is unbounded and loudly warned."""

    if path is None:
        return YAMSoftJointLimits.absent()
    source = str(path)
    raw = _load_json_object(Path(path), label="soft limits")
    if raw is None:
        raise YAMSafetyConfigError("configured soft limits file does not exist")
    _require_keys(
        raw, required=frozenset({"lower", "upper"}), label="soft limits"
    )
    stamped_identity = raw.get("serial")
    if stamped_identity is not None:
        if not isinstance(stamped_identity, str) or not stamped_identity:
            raise YAMSafetyConfigError("soft limits 'serial' must be a non-empty string")
        if (
            expected_stable_identity is not None
            and stamped_identity != expected_stable_identity
        ):
            raise YAMSafetyConfigError(
                "soft limits adapter serial does not match the configured follower"
            )
    stamped_channel = raw.get("channel")
    if stamped_channel is not None and not isinstance(stamped_channel, str):
        raise YAMSafetyConfigError("soft limits 'channel' metadata must be a string")
    raw_lower = _require_six_entries(
        raw["lower"], field="lower", label="soft limits"
    )
    raw_upper = _require_six_entries(
        raw["upper"], field="upper", label="soft limits"
    )

    def parse_bound(value: Any, *, field: str, index: int) -> float | None:
        if value is None:
            return None
        return _finite_number(value, field=field, index=index, label="soft limits")

    lower_values = tuple(
        parse_bound(value, field="lower", index=index)
        for index, value in enumerate(raw_lower)
    )
    upper_values = tuple(
        parse_bound(value, field="upper", index=index)
        for index, value in enumerate(raw_upper)
    )
    for index, (lower, upper) in enumerate(
        zip(lower_values, upper_values, strict=True)
    ):
        if lower is not None and upper is not None and lower > upper:
            raise YAMSafetyConfigError(
                f"soft limits lower exceeds upper for joint {index + 1}"
            )

    active = any(value is not None for value in (*lower_values, *upper_values))
    return YAMSoftJointLimits(
        lower=lower_values,  # type: ignore[arg-type]
        upper=upper_values,  # type: ignore[arg-type]
        present=True,
        active=active,
        source_path=source,
    )


def load_yam_follower_safety(
    *,
    frame_map_path: str | Path | None,
    soft_limits_path: str | Path | None,
    expected_stable_identity: str | None = None,
) -> YAMFollowerSafety:
    return YAMFollowerSafety(
        frame_map=load_yam_frame_map(frame_map_path),
        soft_limits=load_yam_soft_limits(
            soft_limits_path,
            expected_stable_identity=expected_stable_identity,
        ),
    )
