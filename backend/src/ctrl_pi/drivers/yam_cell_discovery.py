from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Sequence


DEFAULT_SYSFS_ROOT = Path("/sys")
MAX_CAN_INTERFACES = 64
MAX_NET_INTERFACES = 4_096
MAX_USB_ANCESTORS = 8
MAX_SYSFS_VALUE_BYTES = 512

_ARPHRD_CAN = 280
_IFF_UP = 0x1

IssueSeverity = Literal["warning", "error"]


@dataclass(frozen=True, slots=True)
class SocketCANDiscoveryIssue:
    """A bounded, display-safe passive discovery finding."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    logical_id: str | None = None
    interface: str | None = None
    stable_identity: str | None = None


@dataclass(frozen=True, slots=True)
class SocketCANAdapter:
    """One kernel SocketCAN interface and its durable USB identity, if known."""

    interface: str
    stable_identity: str | None
    product: str | None
    manufacturer: str | None
    operstate: str | None
    link_up: bool | None


@dataclass(frozen=True, slots=True)
class SocketCANInventory:
    adapters: tuple[SocketCANAdapter, ...]
    issues: tuple[SocketCANDiscoveryIssue, ...] = ()
    truncated: bool = False

    @property
    def errors(self) -> tuple[SocketCANDiscoveryIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ConfiguredSocketCANArm:
    logical_id: str
    stable_identity: str


@dataclass(frozen=True, slots=True)
class ResolvedSocketCANArm:
    logical_id: str
    stable_identity: str
    runtime_interface: str
    product: str | None
    manufacturer: str | None
    operstate: str | None
    link_up: bool | None


@dataclass(frozen=True, slots=True)
class SocketCANResolution:
    arms: tuple[ResolvedSocketCANArm, ...]
    inventory: SocketCANInventory
    issues: tuple[SocketCANDiscoveryIssue, ...]

    @property
    def errors(self) -> tuple[SocketCANDiscoveryIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def ready(self) -> bool:
        return not self.errors

    @property
    def by_logical_id(self) -> dict[str, ResolvedSocketCANArm]:
        return {arm.logical_id: arm for arm in self.arms}


class _PassiveReadError(RuntimeError):
    pass


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_sysfs_value(
    path: Path,
    *,
    sysfs_root: Path,
    maximum_bytes: int = MAX_SYSFS_VALUE_BYTES,
) -> str:
    """Read one small sysfs attribute without following it outside sysfs."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    if not _is_within(resolved, sysfs_root):
        raise _PassiveReadError("sysfs attribute resolves outside the configured root")
    try:
        with resolved.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise _PassiveReadError("sysfs attribute is not readable") from exc
    if len(raw) > maximum_bytes:
        raise _PassiveReadError("sysfs attribute exceeds the bounded read limit")
    try:
        value = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise _PassiveReadError("sysfs attribute is not valid UTF-8") from exc
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _PassiveReadError("sysfs attribute contains control characters")
    return value


def _optional_sysfs_value(path: Path, *, sysfs_root: Path) -> str | None:
    try:
        value = _read_sysfs_value(path, sysfs_root=sysfs_root)
    except FileNotFoundError:
        return None
    return value or None


def _usb_attributes(
    device_link: Path,
    *,
    sysfs_root: Path,
) -> tuple[str | None, str | None, str | None]:
    """Find the bounded USB ancestor that owns the adapter serial attribute."""

    try:
        device = device_link.resolve(strict=True)
    except OSError:
        return None, None, None
    if not _is_within(device, sysfs_root):
        raise _PassiveReadError("CAN device resolves outside the configured sysfs root")

    cursor = device
    for _ in range(MAX_USB_ANCESTORS + 1):
        serial = _optional_sysfs_value(cursor / "serial", sysfs_root=sysfs_root)
        if serial is not None:
            product = _optional_sysfs_value(cursor / "product", sysfs_root=sysfs_root)
            manufacturer = _optional_sysfs_value(
                cursor / "manufacturer", sysfs_root=sysfs_root
            )
            return serial, product, manufacturer
        if cursor == sysfs_root or cursor.parent == cursor:
            break
        cursor = cursor.parent
    return None, None, None


def _network_interface_names(net_root: Path) -> tuple[list[str], bool, bool]:
    names: list[str] = []
    try:
        with os.scandir(net_root) as entries:
            for entry in entries:
                if (
                    not entry.name
                    or len(entry.name.encode("utf-8")) > 15
                    or any(
                        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                        for character in entry.name
                    )
                ):
                    continue
                names.append(entry.name)
                if len(names) > MAX_NET_INTERFACES:
                    break
    except OSError:
        return [], False, False
    def order(name: str) -> tuple[int, int | str, str]:
        suffix = name[3:] if name.startswith("can") else ""
        if suffix.isdigit():
            return (0, int(suffix), name)
        return (1, name, name)

    names.sort(key=order)
    return names[:MAX_NET_INTERFACES], len(names) > MAX_NET_INTERFACES, True


def discover_socketcan_adapters(
    *,
    sysfs_root: str | Path = DEFAULT_SYSFS_ROOT,
    maximum_interfaces: int = MAX_CAN_INTERFACES,
) -> SocketCANInventory:
    """Passively inventory SocketCAN adapters using bounded, read-only sysfs reads.

    This function never imports a CAN/vendor package, opens a CAN or serial device,
    executes a process, changes link state, pings a motor, or constructs a robot.
    ``sysfs_root`` is injectable so the complete path is testable without hardware.
    """

    if maximum_interfaces < 1 or maximum_interfaces > MAX_CAN_INTERFACES:
        raise ValueError(
            f"maximum_interfaces must be between 1 and {MAX_CAN_INTERFACES}"
        )

    root = Path(sysfs_root).resolve()
    net_root = root / "class" / "net"
    names, net_truncated, sysfs_available = _network_interface_names(net_root)
    truncated = net_truncated
    adapters: list[SocketCANAdapter] = []
    issues: list[SocketCANDiscoveryIssue] = []

    if not sysfs_available:
        issues.append(
            SocketCANDiscoveryIssue(
                code="sysfs_unavailable",
                message="SocketCAN sysfs inventory is not readable.",
            )
        )

    if truncated:
        issues.append(
            SocketCANDiscoveryIssue(
                code="inventory_limit_exceeded",
                message=(
                    "SocketCAN inventory exceeded the bounded interface limit; "
                    "discovery is incomplete."
                ),
            )
        )

    for interface in names:
        interface_root = net_root / interface
        try:
            hardware_type = _optional_sysfs_value(
                interface_root / "type", sysfs_root=root
            )
            if hardware_type is None or int(hardware_type, 10) != _ARPHRD_CAN:
                continue
        except (ValueError, _PassiveReadError):
            issues.append(
                SocketCANDiscoveryIssue(
                    code="hardware_type_unreadable",
                    message="Network interface hardware type could not be read safely.",
                    severity="warning",
                    interface=interface,
                )
            )
            continue
        if len(adapters) >= maximum_interfaces:
            truncated = True
            continue
        try:
            stable_identity, product, manufacturer = _usb_attributes(
                interface_root / "device", sysfs_root=root
            )
        except _PassiveReadError:
            stable_identity = product = manufacturer = None
            issues.append(
                SocketCANDiscoveryIssue(
                    code="identity_unreadable",
                    message="USB adapter identity could not be read safely from sysfs.",
                    severity="warning",
                    interface=interface,
                )
            )

        if stable_identity is None:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="identity_missing",
                    message="SocketCAN interface has no readable USB adapter serial.",
                    severity="warning",
                    interface=interface,
                )
            )
        elif len(stable_identity.encode("utf-8")) > 256:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="identity_invalid",
                    message="USB adapter serial exceeds the supported identity length.",
                    interface=interface,
                )
            )
            stable_identity = None

        operstate: str | None
        link_up: bool | None
        try:
            operstate = _optional_sysfs_value(
                interface_root / "operstate", sysfs_root=root
            )
        except _PassiveReadError:
            operstate = None
            issues.append(
                SocketCANDiscoveryIssue(
                    code="operstate_unreadable",
                    message="SocketCAN operstate could not be read safely from sysfs.",
                    severity="warning",
                    interface=interface,
                    stable_identity=stable_identity,
                )
            )

        try:
            flags = _optional_sysfs_value(interface_root / "flags", sysfs_root=root)
            link_up = None if flags is None else bool(int(flags, 0) & _IFF_UP)
        except (ValueError, _PassiveReadError):
            link_up = None
            issues.append(
                SocketCANDiscoveryIssue(
                    code="link_flags_unreadable",
                    message="SocketCAN link flags could not be read safely from sysfs.",
                    severity="warning",
                    interface=interface,
                    stable_identity=stable_identity,
                )
            )

        adapters.append(
            SocketCANAdapter(
                interface=interface,
                stable_identity=stable_identity,
                product=product,
                manufacturer=manufacturer,
                operstate=operstate,
                link_up=link_up,
            )
        )

    if truncated and not any(
        issue.code == "inventory_limit_exceeded" for issue in issues
    ):
        issues.append(
            SocketCANDiscoveryIssue(
                code="inventory_limit_exceeded",
                message=(
                    "SocketCAN inventory exceeded the bounded interface limit; "
                    "discovery is incomplete."
                ),
            )
        )

    identities = Counter(
        adapter.stable_identity
        for adapter in adapters
        if adapter.stable_identity is not None
    )
    for identity, count in sorted(identities.items()):
        if count > 1:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="duplicate_discovered_identity",
                    message=(
                        "One USB adapter serial resolves to multiple runtime "
                        "SocketCAN interfaces."
                    ),
                    stable_identity=identity,
                )
            )

    return SocketCANInventory(
        adapters=tuple(adapters), issues=tuple(issues), truncated=truncated
    )


def resolve_configured_socketcan_arms(
    configured_arms: Sequence[ConfiguredSocketCANArm],
    inventory: SocketCANInventory,
    *,
    require_link_up: bool = True,
) -> SocketCANResolution:
    """Resolve durable arm identities and reject every ambiguous assignment."""

    issues = list(inventory.issues)
    resolved: list[ResolvedSocketCANArm] = []

    logical_id_counts = Counter(arm.logical_id for arm in configured_arms)
    identity_counts = Counter(arm.stable_identity for arm in configured_arms)
    duplicate_logical_ids = {
        logical_id for logical_id, count in logical_id_counts.items() if count > 1
    }
    duplicate_identities = {
        identity for identity, count in identity_counts.items() if count > 1
    }

    for logical_id in sorted(duplicate_logical_ids):
        issues.append(
            SocketCANDiscoveryIssue(
                code="duplicate_logical_id",
                message="Configured SocketCAN arm logical IDs must be unique.",
                logical_id=logical_id,
            )
        )
    for identity in sorted(duplicate_identities):
        issues.append(
            SocketCANDiscoveryIssue(
                code="duplicate_configured_identity",
                message="Configured SocketCAN arms must have unique durable identities.",
                stable_identity=identity,
            )
        )

    discovered_by_identity: dict[str, list[SocketCANAdapter]] = defaultdict(list)
    for adapter in inventory.adapters:
        if adapter.stable_identity is not None:
            discovered_by_identity[adapter.stable_identity].append(adapter)

    for arm in configured_arms:
        if arm.logical_id in duplicate_logical_ids:
            continue
        if arm.stable_identity in duplicate_identities:
            continue
        if not arm.logical_id or not arm.stable_identity:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="invalid_configured_identity",
                    message="Configured SocketCAN arm identity fields may not be empty.",
                    logical_id=arm.logical_id or None,
                    stable_identity=arm.stable_identity or None,
                )
            )
            continue

        matches = discovered_by_identity.get(arm.stable_identity, [])
        if not matches:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="configured_identity_missing",
                    message="Configured USB adapter serial is not currently discovered.",
                    logical_id=arm.logical_id,
                    stable_identity=arm.stable_identity,
                )
            )
            continue
        if len(matches) != 1:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="configured_identity_ambiguous",
                    message=(
                        "Configured USB adapter serial does not resolve to exactly one "
                        "runtime interface."
                    ),
                    logical_id=arm.logical_id,
                    stable_identity=arm.stable_identity,
                )
            )
            continue

        match = matches[0]
        if require_link_up and match.link_up is not True:
            issues.append(
                SocketCANDiscoveryIssue(
                    code=(
                        "runtime_link_down"
                        if match.link_up is False
                        else "runtime_link_unknown"
                    ),
                    message=(
                        "Resolved SocketCAN interface is not administratively UP."
                        if match.link_up is False
                        else "Resolved SocketCAN link state could not be verified."
                    ),
                    logical_id=arm.logical_id,
                    interface=match.interface,
                    stable_identity=arm.stable_identity,
                )
            )
        resolved.append(
            ResolvedSocketCANArm(
                logical_id=arm.logical_id,
                stable_identity=arm.stable_identity,
                runtime_interface=match.interface,
                product=match.product,
                manufacturer=match.manufacturer,
                operstate=match.operstate,
                link_up=match.link_up,
            )
        )

    owners_by_interface: dict[str, list[ResolvedSocketCANArm]] = defaultdict(list)
    for arm in resolved:
        owners_by_interface[arm.runtime_interface].append(arm)
    for interface, owners in sorted(owners_by_interface.items()):
        if len(owners) > 1:
            issues.append(
                SocketCANDiscoveryIssue(
                    code="runtime_interface_collision",
                    message="Multiple configured arms resolve to one runtime CAN bus.",
                    interface=interface,
                )
            )

    return SocketCANResolution(
        arms=tuple(resolved), inventory=inventory, issues=tuple(issues)
    )


def discover_and_resolve_socketcan_arms(
    configured_arms: Sequence[ConfiguredSocketCANArm],
    *,
    sysfs_root: str | Path = DEFAULT_SYSFS_ROOT,
    maximum_interfaces: int = MAX_CAN_INTERFACES,
    require_link_up: bool = True,
) -> SocketCANResolution:
    inventory = discover_socketcan_adapters(
        sysfs_root=sysfs_root, maximum_interfaces=maximum_interfaces
    )
    return resolve_configured_socketcan_arms(
        configured_arms, inventory, require_link_up=require_link_up
    )
