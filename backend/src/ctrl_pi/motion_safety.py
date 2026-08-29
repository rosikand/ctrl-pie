"""Shared fail-closed boundary for releasing robot command ownership."""

from __future__ import annotations

from dataclasses import dataclass

from ctrl_pi.drivers.yam import YAMDriver
from ctrl_pi.rig import RigLease, RigLeaseOwnershipError, RigLeaseToken


@dataclass(frozen=True, slots=True)
class MotionReleaseResult:
    safe_idle_confirmed: bool
    fault_latched: bool
    lease_released: bool
    detail: str

    @property
    def command_path_blocked(self) -> bool:
        return self.safe_idle_confirmed or self.fault_latched

    @property
    def clean(self) -> bool:
        return self.safe_idle_confirmed and self.lease_released


def release_motion_ownership(
    *,
    driver: YAMDriver,
    arm_ids: list[str] | tuple[str, ...],
    rig_lease: RigLease,
    token: RigLeaseToken,
    fault_detail: str,
) -> MotionReleaseResult:
    """Revoke writes before releasing a lease, or retain the lease on uncertainty.

    A confirmed safe-idle transition permits release. If safe idle fails, the
    driver must first fault-latch the selected command paths so subsequent
    command APIs reject them. If both operations are uncertain, this function
    deliberately retains the lease, preventing another writer from acquiring
    the affected resources.
    """

    selected = list(dict.fromkeys(arm_ids))
    if not selected:
        raise ValueError("at least one motion arm is required")

    safe_idle_confirmed = False
    fault_latched = False
    try:
        for arm_id in selected:
            driver.safe_idle(arm_id)
        safe_idle_confirmed = True
    except Exception:
        try:
            driver.latch_fault(selected, fault_detail)
            fault_latched = True
        except Exception:
            return MotionReleaseResult(
                safe_idle_confirmed=False,
                fault_latched=False,
                lease_released=False,
                detail=(
                    "Safe idle and driver fault-latch are uncertain; command "
                    "ownership remains blocked."
                ),
            )

    try:
        rig_lease.release(token)
    except RigLeaseOwnershipError:
        return MotionReleaseResult(
            safe_idle_confirmed=safe_idle_confirmed,
            fault_latched=fault_latched,
            lease_released=False,
            detail=(
                "The command path is blocked, but lease release could not be confirmed."
            ),
        )

    return MotionReleaseResult(
        safe_idle_confirmed=safe_idle_confirmed,
        fault_latched=fault_latched,
        lease_released=True,
        detail=(
            "Safe idle was confirmed before command ownership was released."
            if safe_idle_confirmed
            else "Safe idle failed; the driver fault-latched the command path before release."
        ),
    )
