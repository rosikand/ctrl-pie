from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

RigOwner = Literal["teleop", "inference", "manual", "setup"]
RigResource = str
RIG_RESOURCE_WILDCARD = "*"


class RigLeaseConflictError(RuntimeError):
    """The process-local arm rig is already owned by another control mode."""


class RigLeaseOwnershipError(RuntimeError):
    """A caller attempted to release a lease it does not own."""


@dataclass(frozen=True)
class RigLeaseToken:
    owner: RigOwner
    owner_id: str
    nonce: uuid.UUID
    resources: frozenset[RigResource] = field(
        default_factory=lambda: frozenset({RIG_RESOURCE_WILDCARD})
    )


def _normalize_resources(
    resources: Iterable[RigResource] | RigResource | None,
) -> frozenset[RigResource]:
    if resources is None:
        return frozenset({RIG_RESOURCE_WILDCARD})
    candidates = (resources,) if isinstance(resources, str) else resources
    normalized: set[RigResource] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise TypeError("rig lease resources must be strings")
        resource = candidate.strip()
        if not resource or len(resource) > 200:
            raise ValueError("rig lease resources must be 1-200 characters")
        normalized.add(resource)
    if not normalized:
        raise ValueError("rig lease resources must not be empty")
    if RIG_RESOURCE_WILDCARD in normalized and len(normalized) != 1:
        raise ValueError("the rig lease wildcard cannot be combined with resources")
    return frozenset(normalized)


def _resources_conflict(
    left: frozenset[RigResource], right: frozenset[RigResource]
) -> bool:
    return (
        RIG_RESOURCE_WILDCARD in left
        or RIG_RESOURCE_WILDCARD in right
        or not left.isdisjoint(right)
    )


def _resources_cover(
    held: frozenset[RigResource], requested: frozenset[RigResource]
) -> bool:
    if RIG_RESOURCE_WILDCARD in held:
        return True
    if RIG_RESOURCE_WILDCARD in requested:
        return False
    return requested.issubset(held)


class RigLease:
    """Non-blocking, process-local arbitration for commands sent to the rig.

    The lease deliberately contains no async primitives: synchronous jog handlers,
    async teleoperation, and the inference worker all share the same tiny critical
    section. Long-running owners keep only the token, never the internal lock.

    A call without ``resources`` retains the original cell-exclusive behavior by
    acquiring the wildcard resource. Explicit stable logical resource IDs allow
    disjoint arm pairs to be controlled concurrently. A multi-resource acquisition
    either reserves the complete normalized set or does not mutate lease state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[uuid.UUID, RigLeaseToken] = {}

    def acquire(
        self,
        owner: RigOwner,
        owner_id: str,
        resources: Iterable[RigResource] | RigResource | None = None,
    ) -> RigLeaseToken:
        owner_id = owner_id.strip()
        if not owner_id or len(owner_id) > 200:
            raise ValueError("rig lease owner ID must be 1-200 characters")
        normalized = _normalize_resources(resources)
        token = RigLeaseToken(
            owner=owner,
            owner_id=owner_id,
            nonce=uuid.uuid4(),
            resources=normalized,
        )
        with self._lock:
            conflict = next(
                (
                    active
                    for active in self._active.values()
                    if _resources_conflict(active.resources, normalized)
                ),
                None,
            )
            if conflict is not None:
                shared = sorted(conflict.resources & normalized)
                resource = shared[0] if shared else RIG_RESOURCE_WILDCARD
                raise RigLeaseConflictError(
                    f"the rig resource {resource!r} is already controlled by "
                    f"{conflict.owner}"
                )
            self._active[token.nonce] = token
        return token

    def release(self, token: RigLeaseToken) -> None:
        with self._lock:
            if not self._active:
                raise RigLeaseOwnershipError("the rig has no active lease")
            if self._active.get(token.nonce) != token:
                raise RigLeaseOwnershipError("the rig lease token does not own the rig")
            del self._active[token.nonce]

    def owns(
        self,
        token: RigLeaseToken,
        resources: Iterable[RigResource] | RigResource | None = None,
    ) -> bool:
        """Return whether ``token`` is active and covers the requested resources."""

        requested = None if resources is None else _normalize_resources(resources)
        with self._lock:
            active = self._active.get(token.nonce)
            return active == token and (
                requested is None or _resources_cover(token.resources, requested)
            )

    def current(self, resource: RigResource | None = None) -> RigLeaseToken | None:
        """Return the active owner for one resource.

        With no resource this preserves the original single-token query. If
        resource-scoped callers hold multiple leases, it returns the oldest active
        token; use :meth:`active` when the complete snapshot matters.
        """

        requested = None if resource is None else _normalize_resources(resource)
        with self._lock:
            for token in self._active.values():
                if requested is None or _resources_conflict(token.resources, requested):
                    return token
            return None

    def active(
        self,
        resources: Iterable[RigResource] | RigResource | None = None,
    ) -> tuple[RigLeaseToken, ...]:
        """Return an acquisition-ordered immutable snapshot of active leases."""

        requested = None if resources is None else _normalize_resources(resources)
        with self._lock:
            return tuple(
                token
                for token in self._active.values()
                if requested is None
                or _resources_conflict(token.resources, requested)
            )

    @contextmanager
    def hold(
        self,
        owner: RigOwner,
        owner_id: str,
        resources: Iterable[RigResource] | RigResource | None = None,
    ) -> Iterator[RigLeaseToken]:
        token = self.acquire(owner, owner_id, resources)
        try:
            yield token
        finally:
            self.release(token)
