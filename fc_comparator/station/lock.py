"""Station lock: after an NG the station stays locked until a supervisor
acknowledges with their PIN or a re-inspection of the same part passes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .security import verify_secret


@dataclass
class LockState:
    locked: bool = False
    reason: str = ""
    part_number: str = ""
    inspection_id: int | None = None
    since: datetime | None = None


class StationLock:
    def __init__(self, pin_hash: str, enabled: bool = True):
        self.pin_hash = pin_hash
        self.enabled = enabled
        self.state = LockState()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[LockState], None]] = []

    @property
    def locked(self) -> bool:
        return self.state.locked

    def subscribe(self, fn: Callable[[LockState], None]) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            fn(self.state)

    def lock(self, reason: str, part_number: str, inspection_id: int | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.state = LockState(True, reason, part_number, inspection_id, datetime.now())
        self._notify()

    def release_by_pass(self, part_number: str) -> bool:
        """A passing re-inspection of the *same* part number releases the lock."""
        with self._lock:
            if not self.state.locked or part_number != self.state.part_number:
                return False
            self.state = LockState()
        self._notify()
        return True

    def acknowledge(self, pin: str) -> bool:
        """Supervisor acknowledgement. Returns False on a wrong PIN."""
        if not verify_secret(pin, self.pin_hash):
            return False
        with self._lock:
            self.state = LockState()
        self._notify()
        return True
