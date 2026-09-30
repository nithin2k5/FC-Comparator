"""Tower light / buzzer alerting."""

from __future__ import annotations

from ...config import AlertConfig
from .base import Alert, AlertController, Outputs
from .console import ConsoleAlert, NullAlert

__all__ = [
    "Alert",
    "AlertController",
    "ConsoleAlert",
    "NullAlert",
    "Outputs",
    "create_alert",
]


def create_alert(cfg: AlertConfig) -> Alert:
    b = cfg.backend
    if b == "console":
        return ConsoleAlert()
    if b == "none":
        return NullAlert()
    if b == "gpio":
        from .gpio import GpioAlert

        return GpioAlert(cfg.gpio)
    if b == "usb_relay":
        from .usb_relay import UsbRelayAlert

        return UsbRelayAlert(cfg.usb_relay)
    if b == "modbus":
        from .modbus import ModbusAlert

        return ModbusAlert(cfg.modbus)
    raise ValueError(f"Unknown alert.backend {b!r} (console | gpio | usb_relay | modbus | none)")
