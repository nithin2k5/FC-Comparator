"""Raspberry Pi GPIO backend (gpiozero) and GPIO foot-pedal input.

Drive the tower light and buzzer through a relay/transistor board; never
connect 24 V tower-light loads directly to the Pi's 3.3 V GPIO (see README).
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import GpioAlertConfig
from .base import Alert, Outputs


class GpioAlert(Alert):
    name = "gpio"

    def __init__(self, cfg: GpioAlertConfig, device_factory=None):
        if device_factory is None:
            from gpiozero import DigitalOutputDevice as device_factory  # noqa: N813
        mk = lambda pin: device_factory(pin, active_high=cfg.active_high, initial_value=False)  # noqa: E731
        self.green = mk(cfg.green_pin)
        self.red = mk(cfg.red_pin)
        self.buzzer = mk(cfg.buzzer_pin)

    def set_outputs(self, outputs: Outputs) -> None:
        for dev, on in ((self.green, outputs.green), (self.red, outputs.red), (self.buzzer, outputs.buzzer)):
            dev.on() if on else dev.off()

    def close(self) -> None:
        for dev in (self.green, self.red, self.buzzer):
            dev.close()


class GpioPedal:
    """Foot pedal / push button on a GPIO pin (to GND, internal pull-up) that triggers inspection."""

    def __init__(self, pin: int, callback: Callable[[], None], bounce_time: float = 0.05, button_factory=None):
        if button_factory is None:
            from gpiozero import Button as button_factory  # noqa: N813
        self.button = button_factory(pin, pull_up=True, bounce_time=bounce_time)
        self.button.when_pressed = callback

    def close(self) -> None:
        self.button.close()
