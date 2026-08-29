from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

RigOwner = Literal["teleop", "inference", "manual", "setup"]


class RigLeaseConflictError(RuntimeError):
    """The process-local arm rig is already owned by another control mode."""


class RigLeaseOwnershipError(RuntimeError):
    """A caller attempted to release a lease it does not own."""


@dataclass(frozen=True)
class RigLeaseToken:
    owner: RigOwner
    owner_id: str
    nonce: uuid.UUID


class RigLease:
    """Non-blocking, process-local arbitration for commands sent to the rig.

    The lease deliberately contains no async primitives: synchronous jog handlers,
    async teleoperation, and the inference worker all share the same tiny critical
    section. Long-running owners keep only the token, never the internal lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: RigLeaseToken | None = None

    def acquire(self, owner: RigOwner, owner_id: str) -> RigLeaseToken:
        owner_id = owner_id.strip()
        if not owner_id or len(owner_id) > 200:
            raise ValueError("rig lease owner ID must be 1-200 characters")
        token = RigLeaseToken(owner=owner, owner_id=owner_id, nonce=uuid.uuid4())
        with self._lock:
            if self._active is not None:
                raise RigLeaseConflictError(
                    f"the rig is already controlled by {self._active.owner}"
                )
            self._active = token
        return token

    def release(self, token: RigLeaseToken) -> None:
        with self._lock:
            if self._active is None:
                raise RigLeaseOwnershipError("the rig has no active lease")
            if self._active != token:
                raise RigLeaseOwnershipError("the rig lease token does not own the rig")
            self._active = None

    def current(self) -> RigLeaseToken | None:
        with self._lock:
            return self._active

    @contextmanager
    def hold(self, owner: RigOwner, owner_id: str) -> Iterator[RigLeaseToken]:
        token = self.acquire(owner, owner_id)
        try:
            yield token
        finally:
            self.release(token)
