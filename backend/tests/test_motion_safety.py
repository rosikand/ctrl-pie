from __future__ import annotations

from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.motion_safety import release_motion_ownership
from ctrl_pi.rig import RigLease


class _UnsafeIdleDriver(MockYAMDriver):
    def __init__(self, *, latch_fails: bool = False) -> None:
        super().__init__()
        self.latch_fails = latch_fails

    def safe_idle(self, arm_id: str):
        del arm_id
        raise RuntimeError("injected safe-idle failure")

    def latch_fault(self, arm_ids: list[str], detail: str) -> None:
        if self.latch_fails:
            raise RuntimeError("injected latch failure")
        super().latch_fault(arm_ids, detail)


def test_release_confirms_safe_idle_before_lease_release() -> None:
    driver = MockYAMDriver()
    lease = RigLease()
    token = lease.acquire("test", "clean", resources={"yam-follower"})

    result = release_motion_ownership(
        driver=driver,
        arm_ids=["yam-follower"],
        rig_lease=lease,
        token=token,
        fault_detail="sanitized test fault",
    )

    assert result.clean
    assert lease.current("yam-follower") is None
    assert driver.get_arm("yam-follower").control_state == "gravity_comp"


def test_release_fault_latches_before_releasing_after_safe_idle_failure() -> None:
    driver = _UnsafeIdleDriver()
    lease = RigLease()
    token = lease.acquire("test", "latched", resources={"yam-follower"})

    result = release_motion_ownership(
        driver=driver,
        arm_ids=["yam-follower"],
        rig_lease=lease,
        token=token,
        fault_detail="sanitized test fault",
    )

    assert result.fault_latched and result.lease_released
    assert lease.current("yam-follower") is None
    arm = driver.get_arm("yam-follower")
    assert arm.control_state == "error" and not arm.connected


def test_release_retains_lease_when_safe_idle_and_fault_latch_are_uncertain() -> None:
    driver = _UnsafeIdleDriver(latch_fails=True)
    lease = RigLease()
    token = lease.acquire("test", "uncertain", resources={"yam-follower"})

    result = release_motion_ownership(
        driver=driver,
        arm_ids=["yam-follower"],
        rig_lease=lease,
        token=token,
        fault_detail="sanitized test fault",
    )

    assert not result.command_path_blocked
    assert not result.lease_released
    assert lease.current("yam-follower") == token
