"""Alert interface and controller.

A backend only knows how to set three outputs: green lamp, red lamp and
buzzer. ``AlertController`` implements the station behaviour on top (OK/NG,
buzzer pulse or hold-until-ack, idle) and never lets a hardware fault crash
the inspection - faults are logged and exposed via ``last_error``.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import AlertConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Outputs:
    green: bool = False
    red: bool = False
    buzzer: bool = False


class Alert(ABC):
    """Hardware backend: drive a green/red tower light and a buzzer."""

    name = "alert"

    @abstractmethod
    def set_outputs(self, outputs: Outputs) -> None:
        """Apply the output state. May raise on hardware/communication failure."""

    def close(self) -> None:
        """Release hardware (outputs should be switched off first by the caller)."""


class AlertController:
    def __init__(self, backend: Alert, cfg: AlertConfig):
        self.backend = backend
        self.cfg = cfg
        self.state = Outputs()
        self.last_error = ""
        self.last_latency_ms = 0.0
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._generation = 0

    def _apply(self, outputs: Outputs) -> None:
        with self._lock:
            t = time.perf_counter()
            try:
                self.backend.set_outputs(outputs)
                self.last_error = ""
            except Exception as exc:
                self.last_error = f"{self.backend.name}: {exc}"
                log.error("Alert output failed: %s", self.last_error)
            self.last_latency_ms = (time.perf_counter() - t) * 1000
            self.state = outputs

    def _after(self, seconds: float, outputs: Outputs) -> None:
        """Switch to ``outputs`` after ``seconds`` unless another command came first."""
        self._cancel_timer()
        gen = self._generation

        def fire():
            with self._lock:
                if gen == self._generation:
                    self._apply(outputs)

        self._timer = threading.Timer(seconds, fire)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def ok(self) -> None:
        with self._lock:
            self._generation += 1
            self._apply(Outputs(green=True))
            if self.cfg.ok_light_seconds > 0:
                self._after(self.cfg.ok_light_seconds, Outputs())
            else:
                self._cancel_timer()

    def ng(self) -> None:
        with self._lock:
            self._generation += 1
            self._apply(Outputs(red=True, buzzer=True))
            if self.cfg.buzzer_mode == "pulse":
                self._after(self.cfg.buzzer_seconds, Outputs(red=True))
            else:
                self._cancel_timer()

    def silence(self) -> None:
        """Stop the buzzer, keep the lamps."""
        with self._lock:
            self._generation += 1
            self._cancel_timer()
            self._apply(Outputs(green=self.state.green, red=self.state.red))

    def idle(self) -> None:
        with self._lock:
            self._generation += 1
            self._cancel_timer()
            self._apply(Outputs())

    def close(self) -> None:
        self.idle()
        try:
            self.backend.close()
        except Exception as exc:
            log.warning("Closing alert backend failed: %s", exc)
