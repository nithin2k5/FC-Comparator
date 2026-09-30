"""USB relay board backend (CH340 serial "LCUS" boards, 1-8 channels).

Protocol: 4 bytes per command ``A0 <channel> <state> <checksum>`` where
checksum = (0xA0 + channel + state) & 0xFF. These boards appear as a serial
port (COMx on Windows, /dev/ttyUSBx on Linux).
"""

from __future__ import annotations

import threading

from ...config import UsbRelayAlertConfig
from .base import Alert, Outputs


def relay_command(channel: int, on: bool) -> bytes:
    if not 1 <= channel <= 8:
        raise ValueError(f"Relay channel must be 1..8, got {channel}")
    state = 1 if on else 0
    return bytes([0xA0, channel, state, (0xA0 + channel + state) & 0xFF])


class UsbRelayAlert(Alert):
    name = "usb_relay"

    def __init__(self, cfg: UsbRelayAlertConfig, port=None):
        self.cfg = cfg
        if port is None:
            import serial

            port = serial.Serial(cfg.port, cfg.baudrate, timeout=0.2, write_timeout=0.2)
        self.port = port
        self._lock = threading.Lock()
        self._last: dict[int, bool] = {}

    def set_outputs(self, outputs: Outputs) -> None:
        wanted = {
            self.cfg.green_channel: outputs.green,
            self.cfg.red_channel: outputs.red,
            self.cfg.buzzer_channel: outputs.buzzer,
        }
        with self._lock:
            for ch, on in wanted.items():
                if self._last.get(ch) != on:
                    self.port.write(relay_command(ch, on))
                    self._last[ch] = on
            self.port.flush()

    def close(self) -> None:
        self.port.close()
