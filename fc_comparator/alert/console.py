"""Console / no-op alert backends for development and testing."""

from __future__ import annotations

import logging

from .base import Alert, Outputs

log = logging.getLogger(__name__)


class ConsoleAlert(Alert):
    name = "console"

    def __init__(self):
        self.history: list[Outputs] = []

    def set_outputs(self, outputs: Outputs) -> None:
        self.history.append(outputs)
        lamps = "GREEN" if outputs.green else "RED" if outputs.red else "off"
        log.info("[ALERT] lamp=%s buzzer=%s", lamps, "ON" if outputs.buzzer else "off")


class NullAlert(Alert):
    name = "none"

    def set_outputs(self, outputs: Outputs) -> None:
        pass
