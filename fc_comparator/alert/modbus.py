"""Modbus TCP backend: write coils on a PLC which drives the tower light/buzzer.

A minimal, dependency-free Modbus TCP client (function 05 "Write Single Coil")
so the station has no version coupling to a Modbus library and works offline.
"""

from __future__ import annotations

import socket
import struct
import threading

from ..config import ModbusAlertConfig
from .base import Alert, Outputs

WRITE_SINGLE_COIL = 0x05


class ModbusError(RuntimeError):
    pass


class ModbusExceptionResponse(ModbusError):
    """The PLC answered with a Modbus exception (bad address, ...): retrying will not help."""


class ModbusTcpClient:
    def __init__(self, host: str, port: int = 502, unit_id: int = 1, timeout: float = 0.5):
        self.host, self.port, self.unit_id, self.timeout = host, port, unit_id, timeout
        self._sock: socket.socket | None = None
        self._tid = 0
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ModbusError("connection closed by PLC")
            buf += chunk
        return buf

    def write_coil(self, address: int, value: bool) -> None:
        with self._lock:
            for attempt in (1, 2):  # one transparent reconnect (PLC restarted, idle timeout, ...)
                try:
                    self._write_coil(address, value)
                    return
                except ModbusExceptionResponse:
                    raise
                except (OSError, ModbusError):
                    self.close()
                    if attempt == 2:
                        raise

    def _write_coil(self, address: int, value: bool) -> None:
        sock = self._connect()
        self._tid = (self._tid + 1) & 0xFFFF
        pdu = struct.pack(">BHH", WRITE_SINGLE_COIL, address, 0xFF00 if value else 0x0000)
        mbap = struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.unit_id)
        sock.sendall(mbap + pdu)
        tid, _pid, length, _unit = struct.unpack(">HHHB", self._recv_exact(sock, 7))
        body = self._recv_exact(sock, length - 1)
        if tid != self._tid:
            raise ModbusError(f"transaction id mismatch ({tid} != {self._tid})")
        if body[0] & 0x80:
            raise ModbusExceptionResponse(f"PLC returned exception code {body[1]} for coil {address}")
        if body[0] != WRITE_SINGLE_COIL:
            raise ModbusError(f"unexpected function code {body[0]}")


class ModbusAlert(Alert):
    name = "modbus"

    def __init__(self, cfg: ModbusAlertConfig, client: ModbusTcpClient | None = None):
        self.cfg = cfg
        self.client = client or ModbusTcpClient(cfg.host, cfg.port, cfg.unit_id, cfg.timeout)

    def set_outputs(self, outputs: Outputs) -> None:
        c = self.cfg
        # Switch lamps off before on so the tower never shows green and red together.
        writes = [(c.green_coil, outputs.green), (c.red_coil, outputs.red), (c.buzzer_coil, outputs.buzzer)]
        for addr, on in sorted(writes, key=lambda w: w[1]):
            self.client.write_coil(addr, on)

    def close(self) -> None:
        self.client.close()
