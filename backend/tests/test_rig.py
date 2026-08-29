from __future__ import annotations

import pytest

from ctrl_pi.rig import (
    RIG_RESOURCE_WILDCARD,
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


def test_disjoint_resource_sets_can_be_held_concurrently() -> None:
    lease = RigLease()
    right = lease.acquire(
        "teleop",
        "recording-right",
        {"arm:right-leader", "arm:right-follower"},
    )
    left = lease.acquire(
        "inference",
        "deployment-left",
        {"arm:left-follower"},
    )

    assert right.resources == frozenset(
        {"arm:right-leader", "arm:right-follower"}
    )
    assert lease.owns(right)
    assert lease.owns(right, {"arm:right-leader", "arm:right-follower"})
    assert not lease.owns(right, "arm:left-follower")
    assert lease.current("arm:right-leader") == right
    assert lease.current("arm:left-follower") == left
    assert lease.active() == (right, left)
    assert lease.active("arm:right-follower") == (right,)

    lease.release(right)
    assert lease.current() == left
    assert not lease.owns(right)
    lease.release(left)
    assert lease.active() == ()


def test_resource_set_acquisition_is_atomic_on_conflict() -> None:
    lease = RigLease()
    right = lease.acquire("manual", "right-jog", "arm:right-follower")

    with pytest.raises(RigLeaseConflictError, match="controlled by manual"):
        lease.acquire(
            "teleop",
            "bimanual-recording",
            {"arm:right-follower", "arm:left-follower"},
        )

    # The failed multi-resource acquire did not reserve its non-conflicting arm.
    left = lease.acquire("inference", "left-deployment", "arm:left-follower")
    assert lease.active() == (right, left)


def test_legacy_and_explicit_wildcard_leases_are_cell_exclusive() -> None:
    lease = RigLease()
    legacy = lease.acquire("setup", "saved-cell")

    assert legacy.resources == frozenset({RIG_RESOURCE_WILDCARD})
    assert lease.owns(legacy, {"arm:left", "arm:right"})
    with pytest.raises(RigLeaseConflictError, match="controlled by setup"):
        lease.acquire("manual", "left-jog", "arm:left")

    lease.release(legacy)
    arm = lease.acquire("manual", "left-jog", "arm:left")
    with pytest.raises(RigLeaseConflictError, match="controlled by manual"):
        lease.acquire("setup", "saved-cell", {RIG_RESOURCE_WILDCARD})
    assert lease.current(RIG_RESOURCE_WILDCARD) == arm


def test_resources_are_canonical_and_invalid_sets_are_rejected() -> None:
    lease = RigLease()
    token = lease.acquire(
        "teleop",
        " pair-right ",
        [" arm:right-leader ", "arm:right-follower", "arm:right-leader"],
    )

    assert token.owner_id == "pair-right"
    assert token.resources == frozenset(
        {"arm:right-leader", "arm:right-follower"}
    )
    lease.release(token)

    with pytest.raises(ValueError, match="must not be empty"):
        lease.acquire("manual", "empty", [])
    with pytest.raises(ValueError, match="cannot be combined"):
        lease.acquire("setup", "mixed", {RIG_RESOURCE_WILDCARD, "arm:left"})
    with pytest.raises(ValueError, match="1-200"):
        lease.acquire("manual", "blank", {"  "})
    with pytest.raises(TypeError, match="must be strings"):
        lease.acquire("manual", "typed", {"arm:left", 1})  # type: ignore[arg-type]
