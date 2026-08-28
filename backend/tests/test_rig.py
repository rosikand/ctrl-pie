from __future__ import annotations

import pytest

from ctrl_pi.rig import (
    RigLease,
    RigLeaseConflictError,
    RigLeaseOwnershipError,
)


def test_rig_lease_is_nonblocking_and_owner_checked() -> None:
    lease = RigLease()
    teleop = lease.acquire("teleop", "recording-1")

    with pytest.raises(RigLeaseConflictError, match="controlled by teleop"):
        lease.acquire("inference", "deployment-1")

    foreign = RigLease().acquire("manual", "jog:yam-follower")
    with pytest.raises(RigLeaseOwnershipError, match="does not own"):
        lease.release(foreign)

    lease.release(teleop)
    assert lease.current() is None
    with pytest.raises(RigLeaseOwnershipError, match="no active lease"):
        lease.release(teleop)


def test_rig_hold_releases_after_failure() -> None:
    lease = RigLease()

    with pytest.raises(RuntimeError, match="driver failed"):
        with lease.hold("manual", "jog:yam-follower"):
            raise RuntimeError("driver failed")

    inference = lease.acquire("inference", "deployment-1")
    lease.release(inference)
