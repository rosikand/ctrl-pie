from __future__ import annotations

import importlib
from importlib import metadata
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import socket
import stat
import threading
import time
from typing import Any, Literal, Protocol

import numpy as np

from ctrl_pi.config import AppConfig
from ctrl_pi.drivers.yam import (
    JOINT_NAMES,
    ActionLimitError,
    ArmAction,
    ArmNotFoundError,
    ArmRole,
    ArmTelemetry,
    CANTelemetry,
    ControlLoopTelemetry,
    EndEffectorPose,
    GripperTelemetry,
    JogCommand,
    JogLimitError,
    JointTelemetry,
    YAMDiscoveryCandidate,
    YAMDiscoveryResult,
    YAMDriver,
    YAMDriverDiagnostic,
    YAMDriverUnavailableError,
    YAMPreflightResult,
    YAMSetupConfig,
)

logger = logging.getLogger(__name__)

LEADER_ID = "yam-leader"
FOLLOWER_ID = "yam-follower"
SAMPLE_FREQUENCY_HZ = 50.0
SAMPLE_INTERVAL_SECONDS = 1.0 / SAMPLE_FREQUENCY_HZ
MAX_ACTION_JOINT_DELTA_RADIANS = 0.35
MAX_ACTION_GRIPPER_DELTA = 0.20
MAX_SAMPLE_AGE_SECONDS = 0.25
PINNED_YAM_DISTRIBUTIONS = {
    "lerobot-robot-yam": "0.1.1",
    "lerobot-teleoperator-yam-gello": "0.1.1",
    "yam-common": "0.1.1",
}
MAX_CALIBRATION_FILE_BYTES = 64 * 1024
SYS_CLASS_NET_ROOT = Path("/sys/class/net")
STABLE_SERIAL_ROOT = Path("/dev/serial/by-id")
CONVENTIONAL_LEADER_DEVICE = Path("/dev/yam-leader")
LEADER_CALIBRATION_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
    "gripper",
)

# The pinned plugin speaks this order. ctrl-pi deliberately puts wrist roll
# before wrist pitch, so every read and write must map by name.
VENDOR_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
)
VENDOR_TO_CTRL_JOINT = {
    "shoulder_pan": "shoulder_yaw",
    "shoulder_lift": "shoulder_pitch",
    "elbow_flex": "elbow_pitch",
    "wrist_flex": "wrist_pitch",
    "wrist_roll": "wrist_roll",
    "wrist_yaw": "wrist_yaw",
}
CTRL_TO_VENDOR_JOINT = {value: key for key, value in VENDOR_TO_CTRL_JOINT.items()}

# These are the hard limits shipped by lerobot-robot-yam 0.1.1. They are
# repeated here because this driver is the final independent safety boundary.
VENDOR_JOINT_LIMITS_RADIANS = {
    "shoulder_pan": (-2.767, 3.28),
    "shoulder_lift": (-0.15, 3.8),
    "elbow_flex": (-0.15, 3.28),
    "wrist_flex": (-1.72, 1.72),
    "wrist_roll": (-1.72, 1.72),
    "wrist_yaw": (-2.24, 2.24),
}
CTRL_JOINT_LIMITS_RADIANS = {
    VENDOR_TO_CTRL_JOINT[name]: bounds
    for name, bounds in VENDOR_JOINT_LIMITS_RADIANS.items()
}


@dataclass(frozen=True)
class RealYAMConfig:
    can_interface: str | None
    leader_port: str | None
    mujoco_xml_path: str | None
    gripper_type: str
    leader_calibration_id: str | None
    leader_calibration_dir: str | None

    @classmethod
    def from_app_config(cls, config: AppConfig) -> RealYAMConfig:
        return cls(
            can_interface=config.yam_can_interface,
            leader_port=config.yam_leader_port,
            mujoco_xml_path=config.yam_mujoco_xml_path,
            gripper_type=config.yam_gripper_type,
            leader_calibration_id=config.yam_leader_calibration_id,
            leader_calibration_dir=config.yam_leader_calibration_dir,
        )

    @classmethod
    def from_setup_config(cls, config: YAMSetupConfig) -> RealYAMConfig:
        return cls(
            can_interface=config.can_interface,
            leader_port=config.leader_port,
            mujoco_xml_path=config.mujoco_xml_path,
            gripper_type="crank_4310",
            leader_calibration_id=config.leader_calibration_id,
            leader_calibration_dir=config.leader_calibration_dir,
        )

    def public_setup_config(self) -> YAMSetupConfig | None:
        if self.gripper_type != "crank_4310" or any(
            value is None
            for value in (
                self.can_interface,
                self.leader_port,
                self.mujoco_xml_path,
                self.leader_calibration_id,
                self.leader_calibration_dir,
            )
        ):
            return None
        try:
            return YAMSetupConfig(
                can_interface=self.can_interface,
                leader_port=self.leader_port,
                mujoco_xml_path=self.mujoco_xml_path,
                leader_calibration_id=self.leader_calibration_id,
                leader_calibration_dir=self.leader_calibration_dir,
            )
        except ValueError:
            return None


class LeaderDevice(Protocol):
    def connect(self, calibrate: bool = True) -> None: ...

    def get_action(self) -> Mapping[str, float]: ...

    def disconnect(self) -> None: ...


class FollowerDevice(Protocol):
    def connect(self, calibrate: bool = True) -> None: ...

    def get_observations(self) -> Mapping[str, Any]: ...

    def worker_healthy(self) -> bool: ...

    def set_position_control_mode(self) -> None: ...

    def command_joint_pos(self, joint_pos: np.ndarray) -> None: ...

    def zero_torque_mode(self) -> None: ...

    def close(self) -> None: ...


class PoseSolver(Protocol):
    def pose(self, joint_positions_radians: Sequence[float]) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class VendorDevices:
    leader: LeaderDevice
    follower: FollowerDevice
    pose_solver: PoseSolver


class VendorFactory(Protocol):
    def create(self, config: RealYAMConfig) -> VendorDevices: ...


class LeaderCalibrationUnavailableError(RuntimeError):
    pass


class DefaultVendorFactory:
    """Lazy import/construction boundary for the pinned third-party plugins."""

    def create(self, config: RealYAMConfig) -> VendorDevices:
        leader_module = importlib.import_module(
            "lerobot_teleoperator_yam_gello.yam_leader"
        )
        leader_config_module = importlib.import_module(
            "lerobot_teleoperator_yam_gello.config_yam_leader"
        )
        follower_module = importlib.import_module("lerobot_robot_yam.yam_follower")
        follower_config_module = importlib.import_module(
            "lerobot_robot_yam.config_yam_follower"
        )

        assert config.leader_port is not None
        assert config.leader_calibration_id is not None
        assert config.leader_calibration_dir is not None
        assert config.can_interface is not None
        assert config.mujoco_xml_path is not None

        calibration_dir = Path(config.leader_calibration_dir).expanduser()
        calibration_file = calibration_dir / f"{config.leader_calibration_id}.json"
        if not calibration_file.is_file():
            raise LeaderCalibrationUnavailableError

        leader_config = leader_config_module.YAMLeaderTeleopConfig(
            port=config.leader_port,
            id=config.leader_calibration_id,
            calibration_dir=calibration_dir,
        )
        leader = leader_module.YAMLeader(leader_config)
        if not getattr(leader, "calibration", None):
            raise LeaderCalibrationUnavailableError

        follower_config = follower_config_module.YAMFollowerRobotConfig(
            id=FOLLOWER_ID,
            port=config.can_interface,
            mujoco_xml_path=config.mujoco_xml_path,
            gripper_type=config.gripper_type,
            zero_gravity_mode=True,
        )
        follower = _PinnedFollowerDevice(follower_module.YAMFollower(follower_config))
        return VendorDevices(
            leader=leader,
            follower=follower,
            pose_solver=_MujocoPoseSolver(config.mujoco_xml_path),
        )


class _PinnedFollowerDevice:
    """Keep the pinned plugin's private worker check out of the driver core."""

    def __init__(self, follower: Any) -> None:
        self._follower = follower

    def connect(self, calibrate: bool = True) -> None:
        self._follower.connect(calibrate=calibrate)

    def get_observations(self) -> Mapping[str, Any]:
        return self._follower.get_observations()

    def worker_healthy(self) -> bool:
        motor_chain_robot = getattr(self._follower, "_motor_chain_robot", None)
        worker = getattr(motor_chain_robot, "_server_thread", None)
        motor_chain = getattr(motor_chain_robot, "motor_chain", None)
        return bool(
            motor_chain_robot is not None
            and worker is not None
            and worker.is_alive()
            and motor_chain is not None
            and getattr(motor_chain, "running", False)
        )

    def set_position_control_mode(self) -> None:
        self._follower.set_position_control_mode()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._follower.command_joint_pos(joint_pos)

    def zero_torque_mode(self) -> None:
        self._follower.zero_torque_mode()

    def close(self) -> None:
        self._follower.close()


class _MujocoPoseSolver:
    """Small forward-kinematics reader over the explicitly selected model."""

    def __init__(self, xml_path: str) -> None:
        mujoco = importlib.import_module("mujoco")
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser()))
        self._data = mujoco.MjData(self._model)
        try:
            self._qpos_addresses = tuple(
                int(self._model.joint(f"joint{index}").qposadr[0])
                for index in range(1, 7)
            )
            self._site_id = int(self._model.site("grasp_site").id)
        except Exception as error:
            raise RuntimeError("the YAM model lacks the expected joints or grasp site") from error

    def pose(self, joint_positions_radians: Sequence[float]) -> tuple[float, ...]:
        if len(joint_positions_radians) != 6:
            raise ValueError("expected six joint positions")
        for address, position in zip(self._qpos_addresses, joint_positions_radians):
            self._data.qpos[address] = float(position)
        self._mujoco.mj_forward(self._model, self._data)
        translation = self._data.site_xpos[self._site_id]
        rotation = self._data.site_xmat[self._site_id].reshape(3, 3)
        roll, pitch, yaw = _rotation_matrix_to_rpy(rotation)
        result = (
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            roll,
            pitch,
            yaw,
        )
        if not all(math.isfinite(value) for value in result):
            raise ValueError("forward kinematics returned non-finite values")
        return result


def _rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if horizontal > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = 0.0
    return roll, pitch, yaw


class RealYAMDriver(YAMDriver):
    """Cached, fail-closed adapter for one YAM leader/follower pair."""

    def __init__(
        self,
        config: RealYAMConfig,
        *,
        vendor_factory: VendorFactory | None = None,
    ) -> None:
        self.config = config
        self._initial_config = config
        self._vendor_factory = vendor_factory or DefaultVendorFactory()
        self._lifecycle_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._leader_io_lock = threading.RLock()
        self._follower_io_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._io_thread: threading.Thread | None = None
        self._leader: LeaderDevice | None = None
        self._follower: FollowerDevice | None = None
        self._pose_solver: PoseSolver | None = None
        self._accepting_commands = False
        self._position_control_enabled = False
        self._last_sample_monotonic: float | None = None
        self._dropped_cycles = 0
        self._snapshots = {
            LEADER_ID: self._disconnected_snapshot(LEADER_ID, "YAM Leader", "leader"),
            FOLLOWER_ID: self._disconnected_snapshot(
                FOLLOWER_ID, "YAM Follower", "follower"
            ),
        }
        self._diagnostic = self.preflight()

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        *,
        vendor_factory: VendorFactory | None = None,
    ) -> RealYAMDriver:
        return cls(
            RealYAMConfig.from_app_config(config),
            vendor_factory=vendor_factory,
        )

    def preflight(self) -> YAMDriverDiagnostic:
        """Validate installation and visibility without importing/opening hardware."""

        try:
            return self._preflight_unchecked()
        except Exception:
            return YAMDriverDiagnostic(
                status="error",
                detail="YAM hardware configuration could not be inspected safely.",
            )

    def setup_config(self) -> YAMSetupConfig | None:
        with self._state_lock:
            config = self.config
        return config.public_setup_config()

    def discover_setup(self) -> YAMDiscoveryResult:
        """List bounded OS-level candidates without importing or opening a device."""

        interfaces: list[str] = []
        try:
            for _, name in sorted(socket.if_nameindex(), key=lambda item: item[1]):
                if len(interfaces) >= 32 or len(name.encode("utf-8")) > 15:
                    continue
                type_path = SYS_CLASS_NET_ROOT / name / "type"
                try:
                    with type_path.open("rb") as handle:
                        raw_type = handle.read(16)
                    if raw_type.strip() == b"280":  # Linux ARPHRD_CAN
                        interfaces.append(name)
                except OSError:
                    continue
        except OSError:
            interfaces = []
        ports: list[str] = []
        try:
            serial_root = STABLE_SERIAL_ROOT
            if serial_root.is_dir():
                for path in sorted(serial_root.iterdir(), key=lambda item: item.name):
                    if len(ports) >= 32 or not path.is_symlink():
                        continue
                    try:
                        if stat.S_ISCHR(path.stat().st_mode):
                            ports.append(str(path))
                    except OSError:
                        continue
            conventional = CONVENTIONAL_LEADER_DEVICE
            if (
                len(ports) < 32
                and conventional not in (Path(item) for item in ports)
                and conventional.exists()
                and stat.S_ISCHR(conventional.stat().st_mode)
            ):
                ports.append(str(conventional))
        except OSError:
            ports = []

        current = self.setup_config()
        suggested = current
        if suggested is None:
            with self._state_lock:
                raw = self.config
            try:
                suggested = YAMSetupConfig(
                    can_interface=raw.can_interface or (interfaces[0] if interfaces else ""),
                    leader_port=raw.leader_port or (ports[0] if ports else ""),
                    mujoco_xml_path=raw.mujoco_xml_path or "",
                    leader_calibration_id=raw.leader_calibration_id or "yam-leader",
                    leader_calibration_dir=raw.leader_calibration_dir or "",
                )
            except ValueError:
                suggested = None
        return YAMDiscoveryResult(
            mode="hardware",
            can_interfaces=[
                YAMDiscoveryCandidate(
                    id=name,
                    label=f"SocketCAN interface {name}",
                )
                for name in interfaces
            ],
            leader_ports=[
                YAMDiscoveryCandidate(
                    id=path,
                    label=f"Stable serial device {index + 1}",
                )
                for index, path in enumerate(ports)
            ],
            suggested_config=suggested,
            detail=(
                "Discovery found OS-level YAM connection candidates without opening them."
                if interfaces or ports
                else "No OS-level YAM connection candidates are currently visible."
            ),
        )

    def preflight_setup(self, config: YAMSetupConfig) -> YAMPreflightResult:
        candidate = RealYAMDriver(
            RealYAMConfig.from_setup_config(config),
            vendor_factory=self._vendor_factory,
        )
        diagnostic = candidate.diagnostic()
        calibration_ready = candidate._calibration_file_ready()
        return YAMPreflightResult(
            ready=diagnostic.status == "configured",
            calibration_ready=calibration_ready,
            diagnostic=diagnostic,
        )

    def apply_setup(self, config: YAMSetupConfig) -> YAMDriverDiagnostic:
        self._replace_config(RealYAMConfig.from_setup_config(config))
        return self.diagnostic()

    def reset_setup(self) -> YAMDriverDiagnostic:
        self._replace_config(self._initial_config)
        return self.diagnostic()

    def _replace_config(self, config: RealYAMConfig) -> None:
        with self._lifecycle_lock:
            self.shutdown()
            with self._state_lock:
                if self._io_thread is not None and self._io_thread.is_alive():
                    raise YAMDriverUnavailableError(
                        "YAM setup could not change before the hardware loop stopped."
                    )
                self.config = config
                self._snapshots = {
                    LEADER_ID: self._disconnected_snapshot(
                        LEADER_ID, "YAM Leader", "leader"
                    ),
                    FOLLOWER_ID: self._disconnected_snapshot(
                        FOLLOWER_ID, "YAM Follower", "follower"
                    ),
                }
                self._diagnostic = self.preflight()

    def _preflight_unchecked(self) -> YAMDriverDiagnostic:
        problem = self._configuration_problem()
        if problem is not None:
            return problem
        for distribution, expected_version in PINNED_YAM_DISTRIBUTIONS.items():
            try:
                installed_version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                return YAMDriverDiagnostic(
                    status="missing",
                    detail="Pinned YAM hardware packages are not installed.",
                )
            if installed_version != expected_version:
                return YAMDriverDiagnostic(
                    status="error",
                    detail="Installed YAM hardware package versions do not match V1 pins.",
                )

        assert self.config.mujoco_xml_path is not None
        model_path = Path(self.config.mujoco_xml_path).expanduser()
        if not model_path.is_file() or not os.access(model_path, os.R_OK):
            return YAMDriverDiagnostic(
                status="missing",
                detail="The configured YAM MuJoCo model is unavailable or unreadable.",
            )

        assert self.config.leader_calibration_dir is not None
        assert self.config.leader_calibration_id is not None
        if not self._calibration_file_ready():
            return YAMDriverDiagnostic(
                status="missing",
                detail=(
                    "The configured YAM leader calibration file is unavailable, unreadable, "
                    "or invalid."
                ),
            )

        assert self.config.leader_port is not None
        leader_device = Path(self.config.leader_port).expanduser()
        if not leader_device.exists():
            return YAMDriverDiagnostic(
                status="missing",
                detail="The configured YAM leader serial device is not visible.",
            )
        try:
            serial_is_character = self._serial_device_is_character(leader_device)
        except OSError:
            return YAMDriverDiagnostic(
                status="missing",
                detail="The configured YAM leader serial device is not visible.",
            )
        if not serial_is_character:
            return YAMDriverDiagnostic(
                status="error",
                detail="The configured YAM leader path is not a serial device.",
            )
        if not os.access(leader_device, os.R_OK | os.W_OK):
            return YAMDriverDiagnostic(
                status="error",
                detail="The configured YAM leader serial device is not readable and writable.",
            )

        assert self.config.can_interface is not None
        can_kind = self._network_interface_kind(self.config.can_interface)
        if can_kind == "missing":
            return YAMDriverDiagnostic(
                status="missing",
                detail="The configured YAM SocketCAN interface is not visible.",
            )
        if can_kind == "other":
            return YAMDriverDiagnostic(
                status="error",
                detail="The configured network interface is not SocketCAN.",
            )
        return YAMDriverDiagnostic(
            status="configured",
            detail=(
                "YAM packages, configuration, model, calibration file, and devices passed "
                "non-opening preflight."
            ),
        )

    @staticmethod
    def _serial_device_is_character(path: Path) -> bool:
        return stat.S_ISCHR(path.stat().st_mode)

    @staticmethod
    def _network_interface_kind(name: str) -> Literal["can", "other", "missing"]:
        try:
            socket.if_nametoindex(name)
            type_path = SYS_CLASS_NET_ROOT / name / "type"
            with type_path.open("rb") as handle:
                raw_type = handle.read(16)
        except OSError:
            return "missing"
        return "can" if raw_type.strip() == b"280" else "other"

    def _calibration_file_ready(self) -> bool:
        """Validate the pinned leader artifact without constructing vendor objects."""

        directory = self.config.leader_calibration_dir
        calibration_id = self.config.leader_calibration_id
        if directory is None or calibration_id is None:
            return False
        try:
            path = Path(directory).expanduser() / f"{calibration_id}.json"
            with path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_size <= 0
                    or opened.st_size > MAX_CALIBRATION_FILE_BYTES
                ):
                    return False
                raw = handle.read(MAX_CALIBRATION_FILE_BYTES + 1)
            if not raw or len(raw) > MAX_CALIBRATION_FILE_BYTES:
                return False
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict) or set(payload) != set(LEADER_CALIBRATION_MOTORS):
            return False
        expected_ids = {
            name: index for index, name in enumerate(LEADER_CALIBRATION_MOTORS, start=1)
        }
        required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
        for name, calibration in payload.items():
            if not isinstance(calibration, dict) or set(calibration) != required:
                return False
            values = tuple(calibration[field] for field in required)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                return False
            if calibration["id"] != expected_ids[name]:
                return False
            if calibration["drive_mode"] != 0:
                return False
            if any(abs(calibration[field]) > 2_147_483_647 for field in required):
                return False
            if calibration["range_min"] >= calibration["range_max"]:
                return False
        return True

    def startup(self) -> None:
        with self._lifecycle_lock:
            self._startup_locked()

    def _startup_locked(self) -> None:
        with self._state_lock:
            if self._accepting_commands:
                return
            if self._io_thread is not None and self._io_thread.is_alive():
                return
            preflight = self.preflight()
            if preflight.status != "configured":
                self._diagnostic = preflight
                self._mark_disconnected_locked()
                return
            self._stop_event.clear()
            self._position_control_enabled = False
        try:
            devices = self._vendor_factory.create(self.config)
            with self._state_lock:
                self._leader = devices.leader
                self._follower = devices.follower
                self._pose_solver = devices.pose_solver
            with self._leader_io_lock:
                devices.leader.connect(calibrate=False)
            with self._follower_io_lock:
                devices.follower.connect(calibrate=False)
            self._sample_io(initial=True)
        except Exception as error:
            logger.error("YAM hardware startup failed safely")
            self._disconnect_devices()
            with self._state_lock:
                self._diagnostic = self._sanitized_startup_error(error)
                self._mark_disconnected_locked()
            return

        with self._state_lock:
            self._accepting_commands = True
            self._diagnostic = YAMDriverDiagnostic(
                status="connected",
                detail="YAM leader and follower are connected.",
            )
            thread = threading.Thread(
                target=self._sample_loop,
                name="ctrl-pi-yam-telemetry",
                daemon=True,
            )
            self._io_thread = thread
        try:
            thread.start()
        except Exception:
            logger.error("YAM telemetry thread could not start safely")
            with self._state_lock:
                self._accepting_commands = False
                self._stop_event.set()
            # A normal Thread.start failure leaves the thread unstarted. If an
            # implementation starts it before raising, let its finally block
            # own device cleanup and wait only for the existing safety bound.
            if thread.ident is not None and thread is not threading.current_thread():
                try:
                    thread.join(timeout=2.0)
                except RuntimeError:
                    pass
            if not thread.is_alive():
                self._disconnect_devices()
            with self._state_lock:
                self._io_thread = thread if thread.is_alive() else None
                self._diagnostic = YAMDriverDiagnostic(
                    status="error",
                    detail=(
                        "YAM telemetry could not start; the devices were closed safely."
                        if not thread.is_alive()
                        else (
                            "YAM telemetry could not start cleanly; device cleanup remains "
                            "with the I/O owner."
                        )
                    ),
                )
                self._mark_disconnected_locked()

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        with self._state_lock:
            self._accepting_commands = False
            self._stop_event.set()
            thread = self._io_thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.ident is not None
        ):
            try:
                thread.join(timeout=2.0)
            except RuntimeError:
                logger.error("YAM telemetry thread could not be joined safely")
            if thread.is_alive():
                logger.error("YAM telemetry thread missed its shutdown deadline")
                with self._state_lock:
                    self._diagnostic = YAMDriverDiagnostic(
                        status="error",
                        detail=(
                            "YAM telemetry did not stop before the safety deadline; "
                            "device cleanup remains with the I/O owner."
                        ),
                    )
                    self._mark_disconnected_locked()
                return
        self._disconnect_devices()
        with self._state_lock:
            self._io_thread = None
            self._mark_disconnected_locked()
            if self._diagnostic.status == "connected":
                self._diagnostic = YAMDriverDiagnostic(
                    status="configured",
                    detail="YAM hardware driver is stopped.",
                )

    def diagnostic(self) -> YAMDriverDiagnostic:
        with self._state_lock:
            return self._diagnostic.model_copy(deep=True)

    def list_arms(self) -> list[ArmTelemetry]:
        with self._state_lock:
            return [snapshot.model_copy(deep=True) for snapshot in self._snapshots.values()]

    def get_arm(self, arm_id: str) -> ArmTelemetry:
        with self._state_lock:
            return self._find_snapshot_locked(arm_id).model_copy(deep=True)

    def jog(self, arm_id: str, command: JogCommand) -> ArmTelemetry:
        with self._state_lock:
            current = self._require_commandable_locked(arm_id, action=False)
            if command.kind == "cartesian":
                raise JogLimitError(
                    "Cartesian jog is unavailable for the real YAM driver."
                )
            targets = {
                joint.name: joint.position_radians for joint in current.joints
            }
            gripper = current.gripper.position
            if command.kind == "joint":
                target = targets[command.axis] + command.delta
                lower, upper = CTRL_JOINT_LIMITS_RADIANS[command.axis]
                if not lower <= target <= upper:
                    raise JogLimitError(
                        f"jog would move {command.axis} outside its safe range "
                        f"[{lower:.2f}, {upper:.2f}]"
                    )
                targets[command.axis] = target
            else:
                gripper += command.delta
                if not 0.0 <= gripper <= 1.0:
                    raise JogLimitError(
                        "jog would move gripper position outside its safe range [0.00, 1.00]"
                    )
        self._write(targets, gripper, action=False)
        with self._state_lock:
            return self._snapshots[FOLLOWER_ID].model_copy(deep=True)

    def apply_action(self, arm_id: str, action: ArmAction) -> ArmTelemetry:
        with self._state_lock:
            current = self._require_commandable_locked(arm_id, action=True)
            current_joints = {
                joint.name: joint.position_radians for joint in current.joints
            }
            for name in JOINT_NAMES:
                target = action.joint_positions_radians[name]
                lower, upper = CTRL_JOINT_LIMITS_RADIANS[name]
                if not lower <= target <= upper:
                    raise ActionLimitError(
                        f"action target for {name} is outside its safe range"
                    )
                if abs(target - current_joints[name]) > MAX_ACTION_JOINT_DELTA_RADIANS:
                    raise ActionLimitError(
                        f"action step for {name} exceeds the hardware safety bound"
                    )
            if (
                abs(action.gripper_position - current.gripper.position)
                > MAX_ACTION_GRIPPER_DELTA
            ):
                raise ActionLimitError(
                    "gripper action step exceeds the hardware safety bound"
                )
            targets = dict(action.joint_positions_radians)
            gripper = action.gripper_position
        self._write(targets, gripper, action=True)
        with self._state_lock:
            return self._snapshots[FOLLOWER_ID].model_copy(deep=True)

    def _sample_loop(self) -> None:
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                deadline += SAMPLE_INTERVAL_SECONDS
                try:
                    with self._state_lock:
                        if not self._accepting_commands:
                            return
                    self._sample_io()
                except Exception:
                    logger.error("YAM telemetry sampling failed safely")
                    self._fault(
                        "YAM telemetry failed; the follower entered zero-torque mode."
                    )
                    return
                wait = deadline - time.monotonic()
                if wait > 0:
                    self._stop_event.wait(wait)
                else:
                    deadline = time.monotonic()
        finally:
            if self._stop_event.is_set():
                self._disconnect_devices()

    def _sample_io(self, *, initial: bool = False) -> None:
        with self._state_lock:
            leader = self._leader
            follower = self._follower
            pose_solver = self._pose_solver
        if leader is None or follower is None or pose_solver is None:
            raise RuntimeError("YAM devices are unavailable")

        started = time.monotonic()
        with self._leader_io_lock:
            leader_action = leader.get_action()
        with self._follower_io_lock:
            if not follower.worker_healthy():
                raise RuntimeError("YAM follower worker is not healthy")
            follower_observation = follower.get_observations()
            if not follower.worker_healthy():
                raise RuntimeError("YAM follower worker stopped during sampling")
        finished = time.monotonic()

        with self._state_lock:
            previous_sample = self._last_sample_monotonic
        period = (
            SAMPLE_INTERVAL_SECONDS
            if previous_sample is None
            else max(finished - previous_sample, 1e-9)
        )
        with self._state_lock:
            self._last_sample_monotonic = finished
            if period > SAMPLE_INTERVAL_SECONDS * 1.5:
                self._dropped_cycles += max(
                    1, int(period / SAMPLE_INTERVAL_SECONDS) - 1
                )
            dropped_cycles = self._dropped_cycles
        frequency = min(1.0 / period, SAMPLE_FREQUENCY_HZ)
        cycle_time_ms = max((finished - started) * 1000.0, 0.0)
        loop = ControlLoopTelemetry(
            target_frequency_hz=SAMPLE_FREQUENCY_HZ,
            frequency_hz=frequency,
            cycle_time_ms=cycle_time_ms,
            jitter_ms=abs(period - SAMPLE_INTERVAL_SECONDS) * 1000.0,
            dropped_cycles=dropped_cycles,
        )

        leader_positions, leader_gripper = self._leader_positions(leader_action)
        (
            follower_positions,
            follower_gripper,
            follower_gripper_velocity,
            velocities,
            efforts,
            temperatures,
        ) = self._follower_positions(follower_observation)
        leader_pose = self._validated_pose(
            pose_solver.pose(
                [
                    leader_positions[VENDOR_TO_CTRL_JOINT[vendor_name]]
                    for vendor_name in VENDOR_JOINT_NAMES
                ]
            )
        )
        follower_pose = self._validated_pose(
            pose_solver.pose(
                [
                    follower_positions[VENDOR_TO_CTRL_JOINT[vendor_name]]
                    for vendor_name in VENDOR_JOINT_NAMES
                ]
            )
        )
        timestamp = datetime.now(UTC)

        with self._state_lock:
            prior_leader = self._snapshots[LEADER_ID]
        prior_positions = {
            joint.name: joint.position_radians for joint in prior_leader.joints
        }
        leader_velocities = {
            name: (
                0.0
                if not prior_leader.connected
                else (leader_positions[name] - prior_positions[name]) / period
            )
            for name in JOINT_NAMES
        }
        leader_gripper_velocity = (
            0.0
            if not prior_leader.connected
            else (leader_gripper - prior_leader.gripper.position) / period
        )

        leader_snapshot = self._connected_snapshot(
            arm_id=LEADER_ID,
            name="YAM Leader",
            role="leader",
            timestamp=timestamp,
            positions=leader_positions,
            velocities=leader_velocities,
            efforts=dict.fromkeys(JOINT_NAMES, None),
            temperatures=dict.fromkeys(JOINT_NAMES, None),
            pose=leader_pose,
            gripper=leader_gripper,
            gripper_velocity=leader_gripper_velocity,
            can_interface="leader-serial",
            bitrate=57_600,
            loop=loop,
        )
        follower_snapshot = self._connected_snapshot(
            arm_id=FOLLOWER_ID,
            name="YAM Follower",
            role="follower",
            timestamp=timestamp,
            positions=follower_positions,
            velocities=velocities,
            efforts=efforts,
            temperatures=temperatures,
            pose=follower_pose,
            gripper=follower_gripper,
            gripper_velocity=follower_gripper_velocity,
            can_interface=self.config.can_interface or "unconfigured",
            bitrate=1_000_000,
            loop=loop,
        )
        with self._state_lock:
            if (
                self._leader is not leader
                or self._follower is not follower
                or (not initial and not self._accepting_commands)
            ):
                return
            self._snapshots[LEADER_ID] = leader_snapshot
            self._snapshots[FOLLOWER_ID] = follower_snapshot

    def _write(
        self,
        ctrl_targets: Mapping[str, float],
        gripper_position: float,
        *,
        action: bool,
    ) -> None:
        command = np.array(
            [
                *(float(ctrl_targets[VENDOR_TO_CTRL_JOINT[name]]) for name in VENDOR_JOINT_NAMES),
                1.0 - float(gripper_position),
            ],
            dtype=np.float64,
        )
        if command.shape != (7,) or not np.all(np.isfinite(command)):
            self._raise_command_limit(
                action, "YAM command targets must be seven finite values"
            )
        unavailable_message: str | None = None
        with self._follower_io_lock:
            with self._state_lock:
                current = self._require_commandable_locked(FOLLOWER_ID, action=action)
                sample_age = (
                    math.inf
                    if self._last_sample_monotonic is None
                    else time.monotonic() - self._last_sample_monotonic
                )
                if sample_age > MAX_SAMPLE_AGE_SECONDS:
                    unavailable_message = (
                        "YAM telemetry is stale; motion was disabled before the command."
                    )
                    self._latch_fault_state_locked(
                        "YAM telemetry became stale; the follower entered zero-torque mode."
                    )
                if unavailable_message is None:
                    current_positions = {
                        joint.name: joint.position_radians for joint in current.joints
                    }
                    if set(ctrl_targets) != set(JOINT_NAMES):
                        self._raise_command_limit(
                            action, "YAM command must contain exactly six joints"
                        )
                    for name in JOINT_NAMES:
                        target = float(ctrl_targets[name])
                        lower, upper = CTRL_JOINT_LIMITS_RADIANS[name]
                        if not math.isfinite(target) or not lower <= target <= upper:
                            self._raise_command_limit(
                                action,
                                f"command target for {name} is outside its safe range",
                            )
                        if (
                            abs(target - current_positions[name])
                            > MAX_ACTION_JOINT_DELTA_RADIANS
                        ):
                            self._raise_command_limit(
                                action,
                                f"command step for {name} exceeds the hardware safety bound",
                            )
                    if (
                        not math.isfinite(gripper_position)
                        or not 0.0 <= gripper_position <= 1.0
                        or abs(gripper_position - current.gripper.position)
                        > MAX_ACTION_GRIPPER_DELTA
                    ):
                        self._raise_command_limit(
                            action, "gripper command exceeds the hardware safety bound"
                        )
                follower = self._follower
                position_control_enabled = self._position_control_enabled
            if follower is None:
                raise YAMDriverUnavailableError("The YAM follower is disconnected.")
            if unavailable_message is None:
                try:
                    if not follower.worker_healthy():
                        raise RuntimeError("YAM follower worker is unavailable")
                    if not position_control_enabled:
                        follower.set_position_control_mode()
                        with self._state_lock:
                            self._position_control_enabled = True
                    follower.command_joint_pos(command)
                    if not follower.worker_healthy():
                        raise RuntimeError("YAM follower worker stopped during command")
                except Exception:
                    logger.error("YAM follower command failed safely")
                    unavailable_message = (
                        "The YAM command failed and motion was disabled."
                    )
                    with self._state_lock:
                        self._latch_fault_state_locked(
                            "YAM command failed; the follower entered zero-torque mode."
                        )
            if unavailable_message is not None:
                self._disconnect_follower_io()
        if unavailable_message is not None:
            raise YAMDriverUnavailableError(unavailable_message)

    @staticmethod
    def _raise_command_limit(action: bool, message: str) -> None:
        if action:
            raise ActionLimitError(message)
        raise JogLimitError(message)

    def _require_commandable_locked(
        self, arm_id: str, *, action: bool
    ) -> ArmTelemetry:
        snapshot = self._find_snapshot_locked(arm_id)
        if arm_id != FOLLOWER_ID or snapshot.role != "follower":
            message = "Only the connected YAM follower accepts commands."
            if action:
                raise ActionLimitError(message)
            raise JogLimitError(message)
        if not self._accepting_commands or not snapshot.connected:
            raise YAMDriverUnavailableError("The YAM follower is disconnected.")
        return snapshot

    def _find_snapshot_locked(self, arm_id: str) -> ArmTelemetry:
        try:
            return self._snapshots[arm_id]
        except KeyError as error:
            raise ArmNotFoundError(arm_id) from error

    def _leader_positions(
        self, action: Mapping[str, float]
    ) -> tuple[dict[str, float], float]:
        expected = {f"{name}.pos" for name in (*VENDOR_JOINT_NAMES, "gripper")}
        if not expected.issubset(action):
            raise ValueError("leader sample is missing required joints")
        normalized: dict[str, float] = {}
        for vendor_name in VENDOR_JOINT_NAMES:
            value = float(action[f"{vendor_name}.pos"])
            if not math.isfinite(value) or not -100.0 <= value <= 100.0:
                raise ValueError("leader joint sample is outside its normalized range")
            lower, upper = VENDOR_JOINT_LIMITS_RADIANS[vendor_name]
            normalized[VENDOR_TO_CTRL_JOINT[vendor_name]] = (
                lower + ((value + 100.0) / 200.0) * (upper - lower)
            )
        gripper = float(action["gripper.pos"])
        if not math.isfinite(gripper) or not 0.0 <= gripper <= 100.0:
            raise ValueError("leader gripper sample is outside its normalized range")
        # The crank plugin's command space is 0=open, 1=closed; ctrl-pi's
        # public convention is the inverse (0=closed, 1=open). The GELLO
        # plugin is designed to feed that follower command space directly.
        return normalized, 1.0 - (gripper / 100.0)

    def _follower_positions(
        self, observation: Mapping[str, Any]
    ) -> tuple[
        dict[str, float],
        float,
        float,
        dict[str, float],
        dict[str, float],
        dict[str, float | None],
    ]:
        positions_raw = self._vector(observation, "joint_pos", 6)
        velocities_raw = self._vector(observation, "joint_vel", 7, minimum_length=True)
        efforts_raw = self._vector(observation, "joint_eff", 7, minimum_length=True)
        temperatures_raw = self._optional_vector(observation, "temp_mos", 6)
        gripper_raw = self._vector(observation, "gripper_pos", 1)[0]
        if not 0.0 <= gripper_raw <= 1.0:
            raise ValueError("follower gripper sample is outside its normalized range")

        positions: dict[str, float] = {}
        velocities: dict[str, float] = {}
        efforts: dict[str, float] = {}
        temperatures: dict[str, float | None] = {}
        for index, vendor_name in enumerate(VENDOR_JOINT_NAMES):
            ctrl_name = VENDOR_TO_CTRL_JOINT[vendor_name]
            position = positions_raw[index]
            lower, upper = VENDOR_JOINT_LIMITS_RADIANS[vendor_name]
            if not lower <= position <= upper:
                raise ValueError("follower joint sample is outside its hard limit")
            positions[ctrl_name] = position
            velocities[ctrl_name] = velocities_raw[index]
            efforts[ctrl_name] = efforts_raw[index]
            temperatures[ctrl_name] = temperatures_raw[index]
        return (
            positions,
            1.0 - gripper_raw,
            -velocities_raw[6],
            velocities,
            efforts,
            temperatures,
        )

    @staticmethod
    def _vector(
        observation: Mapping[str, Any],
        key: str,
        length: int,
        *,
        minimum_length: bool = False,
    ) -> list[float]:
        if key not in observation:
            raise ValueError("follower sample is missing a required field")
        array = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        if (minimum_length and len(array) < length) or (
            not minimum_length and len(array) != length
        ):
            raise ValueError("follower sample has an invalid shape")
        values = [float(item) for item in array[:length]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("follower sample contains non-finite values")
        return values

    @classmethod
    def _optional_vector(
        cls, observation: Mapping[str, Any], key: str, length: int
    ) -> list[float | None]:
        if key not in observation:
            return [None] * length
        return cls._vector(observation, key, length, minimum_length=True)

    @staticmethod
    def _validated_pose(values: Sequence[float]) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if len(result) != 6 or not all(math.isfinite(value) for value in result):
            raise ValueError("YAM forward kinematics returned an invalid pose")
        return result

    def _configuration_problem(self) -> YAMDriverDiagnostic | None:
        missing: list[str] = []
        for name, value in (
            ("YAM_CAN_INTERFACE", self.config.can_interface),
            ("YAM_LEADER_PORT", self.config.leader_port),
            ("YAM_MUJOCO_XML_PATH", self.config.mujoco_xml_path),
            ("YAM_LEADER_CALIBRATION_ID", self.config.leader_calibration_id),
            ("YAM_LEADER_CALIBRATION_DIR", self.config.leader_calibration_dir),
        ):
            if value is None or not str(value).strip():
                missing.append(name)
        if missing:
            return YAMDriverDiagnostic(
                status="missing",
                detail=f"Set {', '.join(missing)} for YAM hardware mode.",
            )
        if self.config.gripper_type != "crank_4310":
            return YAMDriverDiagnostic(
                status="error",
                detail=(
                    "YAM_GRIPPER_TYPE must be crank_4310 with the pinned V1 driver; "
                    "linear grippers are not safely supported."
                ),
            )
        calibration_id = self.config.leader_calibration_id or ""
        if (
            len(calibration_id) > 64
            or not calibration_id[0].isalnum()
            or any(
                not (character.isalnum() or character in "._-")
                for character in calibration_id
            )
        ):
            return YAMDriverDiagnostic(
                status="error",
                detail="YAM_LEADER_CALIBRATION_ID is invalid.",
            )
        for value in (self.config.can_interface, self.config.leader_port):
            assert value is not None
            if len(value) > 200 or any(character in value for character in "\r\n\0"):
                return YAMDriverDiagnostic(
                    status="error",
                    detail="YAM hardware configuration contains an invalid device value.",
                )
        for value in (
            self.config.mujoco_xml_path,
            self.config.leader_calibration_dir,
        ):
            assert value is not None
            if len(value) > 1024 or any(character in value for character in "\r\n\0"):
                return YAMDriverDiagnostic(
                    status="error",
                    detail="YAM hardware configuration contains an invalid path value.",
                )
        return None

    @staticmethod
    def _sanitized_startup_error(error: Exception) -> YAMDriverDiagnostic:
        if isinstance(error, LeaderCalibrationUnavailableError):
            detail = (
                "The configured YAM leader calibration is unavailable or invalid."
            )
            status = "missing"
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            detail = "YAM hardware packages are unavailable; install the pinned dependencies."
            status = "missing"
        elif isinstance(error, FileNotFoundError):
            detail = "A configured YAM device, model, or calibration file is unavailable."
            status = "missing"
        elif isinstance(error, PermissionError):
            detail = "YAM device access was denied; check host and container permissions."
            status = "error"
        else:
            detail = "YAM hardware startup failed; check the devices, model, and calibration."
            status = "error"
        return YAMDriverDiagnostic(status=status, detail=detail)

    def _latch_fault_state_locked(self, detail: str) -> None:
        """Latch a public fault while the caller owns the state lock."""

        self._accepting_commands = False
        self._stop_event.set()
        self._diagnostic = YAMDriverDiagnostic(status="error", detail=detail)
        self._mark_disconnected_locked()

    def _fault(self, detail: str) -> None:
        with self._state_lock:
            self._latch_fault_state_locked(detail)
        self._disconnect_devices()

    def latch_fault(self, arm_ids: list[str], detail: str) -> None:
        del arm_ids
        self._fault(detail)
        if any(arm.connected for arm in self.list_arms()):
            raise YAMDriverUnavailableError(
                "The legacy YAM command path could not be fault-latched safely."
            )

    def _disconnect_devices(self) -> None:
        """Close both buses while acquiring device locks in one fixed order."""

        with self._leader_io_lock:
            with self._follower_io_lock:
                with self._state_lock:
                    follower = self._follower
                    leader = self._leader
                    self._follower = None
                    self._leader = None
                    self._pose_solver = None
                    self._position_control_enabled = False
                self._close_follower(follower)
                self._close_leader(leader)

    def _disconnect_follower_io(self) -> None:
        """Close only the follower while its I/O lock is already held."""

        with self._state_lock:
            follower = self._follower
            self._follower = None
            self._pose_solver = None
            self._position_control_enabled = False
        self._close_follower(follower)

    @staticmethod
    def _close_follower(follower: FollowerDevice | None) -> None:
        if follower is not None:
            try:
                follower.zero_torque_mode()
            except Exception:
                logger.error("YAM follower zero-torque cleanup failed safely")
            try:
                follower.close()
            except Exception:
                logger.error("YAM follower close failed safely")

    @staticmethod
    def _close_leader(leader: LeaderDevice | None) -> None:
        if leader is not None:
            try:
                leader.disconnect()
            except Exception:
                logger.error("YAM leader disconnect failed safely")

    def _mark_disconnected_locked(self) -> None:
        timestamp = datetime.now(UTC)
        for arm_id, snapshot in tuple(self._snapshots.items()):
            self._snapshots[arm_id] = snapshot.model_copy(
                update={
                    "connected": False,
                    "timestamp": timestamp,
                    "can": snapshot.can.model_copy(update={"state": "disconnected"}),
                    "control_loop": snapshot.control_loop.model_copy(
                        update={"frequency_hz": 0.0}
                    ),
                },
                deep=True,
            )

    def _disconnected_snapshot(
        self, arm_id: str, name: str, role: ArmRole
    ) -> ArmTelemetry:
        return ArmTelemetry(
            id=arm_id,
            name=name,
            role=role,
            driver="yam",
            connected=False,
            timestamp=datetime.now(UTC),
            joints=[
                JointTelemetry(
                    name=joint_name,
                    position_radians=0.0,
                    velocity_radians_per_second=0.0,
                    effort_newton_meters=None,
                    temperature_celsius=None,
                )
                for joint_name in JOINT_NAMES
            ],
            pose=EndEffectorPose(
                x_m=0.0,
                y_m=0.0,
                z_m=0.0,
                roll_radians=0.0,
                pitch_radians=0.0,
                yaw_radians=0.0,
            ),
            gripper=GripperTelemetry(
                position=0.0,
                velocity=0.0,
                force_newtons=None,
                is_closed=False,
            ),
            can=CANTelemetry(
                interface=(
                    "leader-serial"
                    if role == "leader"
                    else (self.config.can_interface or "unconfigured")
                ),
                state="disconnected",
                bitrate=57_600 if role == "leader" else 1_000_000,
                tx_error_count=None,
                rx_error_count=None,
            ),
            control_loop=ControlLoopTelemetry(
                target_frequency_hz=SAMPLE_FREQUENCY_HZ,
                frequency_hz=0.0,
                cycle_time_ms=0.0,
                jitter_ms=0.0,
                dropped_cycles=0,
            ),
        )

    @staticmethod
    def _connected_snapshot(
        *,
        arm_id: str,
        name: str,
        role: ArmRole,
        timestamp: datetime,
        positions: Mapping[str, float],
        velocities: Mapping[str, float],
        efforts: Mapping[str, float | None],
        temperatures: Mapping[str, float | None],
        pose: Sequence[float],
        gripper: float,
        gripper_velocity: float,
        can_interface: str,
        bitrate: int,
        loop: ControlLoopTelemetry,
    ) -> ArmTelemetry:
        if not math.isfinite(gripper_velocity):
            raise ValueError("gripper velocity must be finite")
        return ArmTelemetry(
            id=arm_id,
            name=name,
            role=role,
            driver="yam",
            connected=True,
            timestamp=timestamp,
            joints=[
                JointTelemetry(
                    name=joint_name,
                    position_radians=positions[joint_name],
                    velocity_radians_per_second=velocities[joint_name],
                    effort_newton_meters=efforts[joint_name],
                    temperature_celsius=temperatures[joint_name],
                )
                for joint_name in JOINT_NAMES
            ],
            pose=EndEffectorPose(
                x_m=pose[0],
                y_m=pose[1],
                z_m=pose[2],
                roll_radians=pose[3],
                pitch_radians=pose[4],
                yaw_radians=pose[5],
            ),
            gripper=GripperTelemetry(
                position=gripper,
                velocity=gripper_velocity,
                force_newtons=None,
                is_closed=gripper <= 0.15,
            ),
            can=CANTelemetry(
                interface=can_interface,
                state="active",
                bitrate=bitrate,
                tx_error_count=None,
                rx_error_count=None,
            ),
            control_loop=loop,
        )


def create_yam_driver(
    config: AppConfig,
    *,
    vendor_factory: VendorFactory | None = None,
) -> YAMDriver:
    if config.mock_mode:
        from ctrl_pi.drivers.mock_yam import MockYAMDriver

        return MockYAMDriver()
    # Hardware mode is always cell-capable.  The original adapter remains the
    # compatibility delegate when the V1.1 environment selects a complete
    # serial-GELLO + CAN-follower setup; newly persisted all-CAN cells are
    # applied in place by YAMSetupManager without restarting FastAPI.
    from ctrl_pi.drivers.yam_cell import YAMCellDriver

    return YAMCellDriver(
        RealYAMDriver.from_app_config(config, vendor_factory=vendor_factory)
    )
