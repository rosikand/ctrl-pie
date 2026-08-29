from __future__ import annotations

from pathlib import Path

import pytest

from ctrl_pi.drivers.yam_cell_discovery import (
    ConfiguredSocketCANArm,
    SocketCANAdapter,
    SocketCANInventory,
    discover_and_resolve_socketcan_arms,
    discover_socketcan_adapters,
    resolve_configured_socketcan_arms,
)


def _make_adapter(
    sysfs_root: Path,
    *,
    interface: str,
    serial: str | None,
    flags: str | None = "0x1",
    operstate: str | None = "unknown",
    product: str = "Test USB-CAN",
) -> None:
    net_device = sysfs_root / "class" / "net" / interface
    net_device.mkdir(parents=True)
    (net_device / "type").write_text("280", encoding="utf-8")
    if flags is not None:
        (net_device / "flags").write_text(flags, encoding="utf-8")
    if operstate is not None:
        (net_device / "operstate").write_text(operstate, encoding="utf-8")

    usb_device = sysfs_root / "devices" / f"usb-{interface}"
    usb_interface = usb_device / f"usb-{interface}:1.0"
    usb_interface.mkdir(parents=True)
    if serial is not None:
        (usb_device / "serial").write_text(serial, encoding="utf-8")
        (usb_device / "product").write_text(product, encoding="utf-8")
        (usb_device / "manufacturer").write_text("Test Maker", encoding="utf-8")
    (net_device / "device").symlink_to(usb_interface, target_is_directory=True)


def _codes(result: SocketCANInventory | object) -> set[str]:
    issues = getattr(result, "issues")
    return {issue.code for issue in issues}


def test_discovery_is_read_only_sysfs_and_uses_natural_interface_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sysfs_root = tmp_path / "sys"
    _make_adapter(sysfs_root, interface="can10", serial="SERIAL-B")
    _make_adapter(sysfs_root, interface="can2", serial="SERIAL-A")
    (sysfs_root / "class" / "net" / "eth0").mkdir()

    opened: list[tuple[Path, str]] = []
    original_open = Path.open

    def guarded_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        resolved = path.resolve(strict=True)
        resolved.relative_to(sysfs_root.resolve())
        assert mode == "rb"
        opened.append((resolved, mode))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    inventory = discover_socketcan_adapters(sysfs_root=sysfs_root)

    assert inventory.ok
    assert [adapter.interface for adapter in inventory.adapters] == ["can2", "can10"]
    assert [adapter.stable_identity for adapter in inventory.adapters] == [
        "SERIAL-A",
        "SERIAL-B",
    ]
    assert inventory.adapters[0].product == "Test USB-CAN"
    assert inventory.adapters[0].manufacturer == "Test Maker"
    assert inventory.adapters[0].link_up is True
    assert opened


def test_stable_identity_survives_shuffled_runtime_can_assignments(
    tmp_path: Path,
) -> None:
    requests = [
        ConfiguredSocketCANArm("right-leader", "LEADER-SERIAL"),
        ConfiguredSocketCANArm("right-follower", "FOLLOWER-SERIAL"),
    ]
    first_root = tmp_path / "first" / "sys"
    _make_adapter(first_root, interface="can0", serial="LEADER-SERIAL")
    _make_adapter(first_root, interface="can1", serial="FOLLOWER-SERIAL")
    second_root = tmp_path / "second" / "sys"
    _make_adapter(second_root, interface="can9", serial="LEADER-SERIAL")
    _make_adapter(second_root, interface="can3", serial="FOLLOWER-SERIAL")

    first = discover_and_resolve_socketcan_arms(requests, sysfs_root=first_root)
    second = discover_and_resolve_socketcan_arms(requests, sysfs_root=second_root)

    assert first.ready and second.ready
    assert first.by_logical_id["right-leader"].runtime_interface == "can0"
    assert first.by_logical_id["right-follower"].runtime_interface == "can1"
    assert second.by_logical_id["right-leader"].runtime_interface == "can9"
    assert second.by_logical_id["right-follower"].runtime_interface == "can3"
    assert second.by_logical_id["right-leader"].stable_identity == "LEADER-SERIAL"


def test_discovery_uses_hardware_type_instead_of_assuming_canN_names(
    tmp_path: Path,
) -> None:
    sysfs_root = tmp_path / "sys"
    _make_adapter(sysfs_root, interface="yam_bus_right", serial="RIGHT-SERIAL")

    result = discover_and_resolve_socketcan_arms(
        [ConfiguredSocketCANArm("right", "RIGHT-SERIAL")],
        sysfs_root=sysfs_root,
    )

    assert result.ready
    assert result.by_logical_id["right"].runtime_interface == "yam_bus_right"


def test_missing_configured_adapter_fails_closed(tmp_path: Path) -> None:
    sysfs_root = tmp_path / "sys"
    _make_adapter(sysfs_root, interface="can0", serial="PRESENT")

    result = discover_and_resolve_socketcan_arms(
        [
            ConfiguredSocketCANArm("present", "PRESENT"),
            ConfiguredSocketCANArm("missing", "ABSENT"),
        ],
        sysfs_root=sysfs_root,
    )

    assert not result.ready
    assert _codes(result) >= {"configured_identity_missing"}
    assert set(result.by_logical_id) == {"present"}


def test_duplicate_configured_identity_and_logical_id_fail_closed() -> None:
    inventory = SocketCANInventory(
        adapters=(
            SocketCANAdapter("can0", "SAME", None, None, "unknown", True),
        )
    )

    result = resolve_configured_socketcan_arms(
        [
            ConfiguredSocketCANArm("same-id", "SAME"),
            ConfiguredSocketCANArm("same-id", "OTHER"),
            ConfiguredSocketCANArm("third", "SAME"),
        ],
        inventory,
    )

    assert not result.ready
    assert _codes(result) >= {
        "duplicate_logical_id",
        "duplicate_configured_identity",
    }
    assert result.arms == ()


def test_duplicate_discovered_identity_is_ambiguous_and_fails_closed() -> None:
    inventory = SocketCANInventory(
        adapters=(
            SocketCANAdapter("can0", "DUPLICATE", None, None, "unknown", True),
            SocketCANAdapter("can4", "DUPLICATE", None, None, "unknown", True),
        )
    )

    result = resolve_configured_socketcan_arms(
        [ConfiguredSocketCANArm("arm", "DUPLICATE")], inventory
    )

    assert not result.ready
    assert "configured_identity_ambiguous" in _codes(result)
    assert result.arms == ()


def test_two_identities_cannot_own_one_runtime_interface() -> None:
    inventory = SocketCANInventory(
        adapters=(
            SocketCANAdapter("can0", "SERIAL-A", None, None, "unknown", True),
            SocketCANAdapter("can0", "SERIAL-B", None, None, "unknown", True),
        )
    )

    result = resolve_configured_socketcan_arms(
        [
            ConfiguredSocketCANArm("arm-a", "SERIAL-A"),
            ConfiguredSocketCANArm("arm-b", "SERIAL-B"),
        ],
        inventory,
    )

    assert not result.ready
    assert "runtime_interface_collision" in _codes(result)


@pytest.mark.parametrize(
    ("flags", "expected_code"),
    [("0x0", "runtime_link_down"), (None, "runtime_link_unknown")],
)
def test_resolved_link_must_be_verifiably_up(
    tmp_path: Path, flags: str | None, expected_code: str
) -> None:
    sysfs_root = tmp_path / "sys"
    _make_adapter(sysfs_root, interface="can0", serial="SERIAL", flags=flags)

    result = discover_and_resolve_socketcan_arms(
        [ConfiguredSocketCANArm("arm", "SERIAL")], sysfs_root=sysfs_root
    )

    assert not result.ready
    assert expected_code in _codes(result)
    assert result.by_logical_id["arm"].runtime_interface == "can0"


def test_missing_usb_serial_is_reported_without_guessing(tmp_path: Path) -> None:
    sysfs_root = tmp_path / "sys"
    _make_adapter(sysfs_root, interface="can0", serial=None)

    inventory = discover_socketcan_adapters(sysfs_root=sysfs_root)

    assert inventory.adapters[0].stable_identity is None
    assert "identity_missing" in _codes(inventory)
    assert not any(adapter.stable_identity == "can0" for adapter in inventory.adapters)


def test_discovery_fails_if_device_symlink_escapes_injected_sysfs(
    tmp_path: Path,
) -> None:
    sysfs_root = tmp_path / "sys"
    net_device = sysfs_root / "class" / "net" / "can0"
    net_device.mkdir(parents=True)
    (net_device / "type").write_text("280", encoding="utf-8")
    (net_device / "flags").write_text("0x1", encoding="utf-8")
    (net_device / "operstate").write_text("unknown", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "serial").write_text("MUST-NOT-READ", encoding="utf-8")
    (net_device / "device").symlink_to(outside, target_is_directory=True)

    inventory = discover_socketcan_adapters(sysfs_root=sysfs_root)

    assert inventory.adapters[0].stable_identity is None
    assert _codes(inventory) >= {"identity_unreadable", "identity_missing"}


def test_inventory_is_bounded_and_incomplete_inventory_is_an_error(
    tmp_path: Path,
) -> None:
    sysfs_root = tmp_path / "sys"
    for index in range(3):
        _make_adapter(sysfs_root, interface=f"can{index}", serial=f"SERIAL-{index}")

    inventory = discover_socketcan_adapters(
        sysfs_root=sysfs_root, maximum_interfaces=2
    )

    assert inventory.truncated
    assert not inventory.ok
    assert len(inventory.adapters) == 2
    assert "inventory_limit_exceeded" in _codes(inventory)


def test_unavailable_sysfs_and_invalid_bound_are_explicit(tmp_path: Path) -> None:
    unavailable = discover_socketcan_adapters(sysfs_root=tmp_path / "absent")

    assert not unavailable.ok
    assert "sysfs_unavailable" in _codes(unavailable)
    with pytest.raises(ValueError, match="between 1 and 64"):
        discover_socketcan_adapters(sysfs_root=tmp_path, maximum_interfaces=0)
