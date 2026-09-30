import socket
import struct
import threading
import time

import pytest

from fc_comparator.config import (
    AlertConfig,
    GpioAlertConfig,
    ModbusAlertConfig,
    UsbRelayAlertConfig,
)
from fc_comparator.station.alerts import (
    AlertController,
    ConsoleAlert,
    Outputs,
    create_alert,
)
from fc_comparator.station.alerts.gpio import GpioAlert, GpioPedal
from fc_comparator.station.alerts.modbus import (
    ModbusAlert,
    ModbusExceptionResponse,
    ModbusTcpClient,
)
from fc_comparator.station.alerts.usb_relay import UsbRelayAlert, relay_command
from fc_comparator.station.lock import StationLock
from fc_comparator.station.security import hash_secret


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---- controller ---------------------------------------------------------
def test_ng_pulses_buzzer_then_keeps_red():
    backend = ConsoleAlert()
    ctl = AlertController(backend, AlertConfig(buzzer_mode="pulse", buzzer_seconds=0.1))
    ctl.ng()
    assert backend.history[-1] == Outputs(red=True, buzzer=True)
    assert wait_for(lambda: backend.history[-1] == Outputs(red=True))
    ctl.ok()
    assert backend.history[-1] == Outputs(green=True)


def test_until_ack_holds_buzzer_until_silenced():
    backend = ConsoleAlert()
    ctl = AlertController(backend, AlertConfig(buzzer_mode="until_ack", buzzer_seconds=0.05))
    ctl.ng()
    time.sleep(0.15)
    assert backend.history[-1] == Outputs(red=True, buzzer=True)
    ctl.silence()
    assert backend.history[-1] == Outputs(red=True)


def test_stale_timer_does_not_override_newer_state():
    backend = ConsoleAlert()
    ctl = AlertController(backend, AlertConfig(buzzer_seconds=0.1))
    ctl.ng()
    ctl.ok()  # new board passed before the buzzer pulse ended
    time.sleep(0.2)
    assert backend.history[-1] == Outputs(green=True)


def test_hardware_fault_is_reported_not_raised():
    class Broken(ConsoleAlert):
        name = "broken"

        def set_outputs(self, outputs):
            raise OSError("relay unplugged")

    ctl = AlertController(Broken(), AlertConfig())
    ctl.ng()
    assert "relay unplugged" in ctl.last_error
    assert ctl.state == Outputs(red=True, buzzer=True)


def test_factory():
    assert create_alert(AlertConfig(backend="console")).name == "console"
    assert create_alert(AlertConfig(backend="none")).name == "none"
    with pytest.raises(ValueError):
        create_alert(AlertConfig(backend="smoke-signals"))


# ---- GPIO (fake gpiozero devices) -----------------------------------------
class FakeDevice:
    def __init__(self, pin, **kw):
        self.pin, self.kw, self.value, self.closed = pin, kw, False, False
        self.when_pressed = None

    def on(self):
        self.value = True

    def off(self):
        self.value = False

    def close(self):
        self.closed = True


def test_gpio_backend():
    g = GpioAlert(GpioAlertConfig(green_pin=5, red_pin=6, buzzer_pin=13, active_high=False), device_factory=FakeDevice)
    g.set_outputs(Outputs(red=True, buzzer=True))
    assert (g.green.value, g.red.value, g.buzzer.value) == (False, True, True)
    assert g.red.pin == 6 and g.red.kw["active_high"] is False
    g.close()
    assert g.buzzer.closed


def test_gpio_pedal():
    hits = []
    pedal = GpioPedal(26, lambda: hits.append(1), button_factory=FakeDevice)
    pedal.button.when_pressed()
    assert hits == [1] and pedal.button.kw["pull_up"] is True


# ---- USB relay (fake serial port) ---------------------------------------
class FakeSerial:
    def __init__(self):
        self.data = b""

    def write(self, b):
        self.data += b

    def flush(self):
        pass

    def close(self):
        pass


def test_usb_relay_protocol_and_dedup():
    assert relay_command(1, True) == bytes([0xA0, 0x01, 0x01, 0xA2])
    assert relay_command(2, False) == bytes([0xA0, 0x02, 0x00, 0xA2])
    port = FakeSerial()
    relay = UsbRelayAlert(UsbRelayAlertConfig(green_channel=1, red_channel=2, buzzer_channel=3), port=port)
    relay.set_outputs(Outputs(red=True, buzzer=True))
    assert port.data == relay_command(1, False) + relay_command(2, True) + relay_command(3, True)
    port.data = b""
    relay.set_outputs(Outputs(red=True))  # only the buzzer changes
    assert port.data == relay_command(3, False)
    with pytest.raises(ValueError):
        relay_command(9, True)


# ---- Modbus TCP (fake PLC) ----------------------------------------------
class FakePlc(threading.Thread):
    def __init__(self, bad_coil=None):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen()
        self.port = self.srv.getsockname()[1]
        self.coils: dict[int, bool] = {}
        self.bad_coil = bad_coil
        self.connections = 0

    def run(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            self.connections += 1
            with conn:
                while True:
                    head = conn.recv(7)
                    if len(head) < 7:
                        break
                    tid, pid, length, unit = struct.unpack(">HHHB", head)
                    pdu = conn.recv(length - 1)
                    fc, addr, val = struct.unpack(">BHH", pdu)
                    if addr == self.bad_coil:
                        resp = struct.pack(">BB", fc | 0x80, 2)
                    else:
                        self.coils[addr] = val == 0xFF00
                        resp = pdu
                    conn.sendall(struct.pack(">HHHB", tid, pid, len(resp) + 1, unit) + resp)

    def stop(self):
        self.srv.close()


def test_modbus_alert_writes_coils():
    plc = FakePlc()
    plc.start()
    try:
        cfg = ModbusAlertConfig(host="127.0.0.1", port=plc.port, green_coil=10, red_coil=11, buzzer_coil=12)
        alert = ModbusAlert(cfg)
        ctl = AlertController(alert, AlertConfig(buzzer_seconds=5))
        t = time.perf_counter()
        ctl.ng()
        assert time.perf_counter() - t < 1.0
        assert ctl.last_error == ""
        assert plc.coils == {10: False, 11: True, 12: True}
        ctl.ok()
        assert plc.coils == {10: True, 11: False, 12: False}
        assert plc.connections == 1  # connection is reused
        ctl.close()
    finally:
        plc.stop()


def test_modbus_exception_response():
    plc = FakePlc(bad_coil=11)
    plc.start()
    try:
        client = ModbusTcpClient("127.0.0.1", plc.port)
        client.write_coil(10, True)
        with pytest.raises(ModbusExceptionResponse):
            client.write_coil(11, True)
        client.close()
    finally:
        plc.stop()


def test_modbus_unreachable_plc_reports_error_quickly():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here
    ctl = AlertController(ModbusAlert(ModbusAlertConfig(host="127.0.0.1", port=port, timeout=0.3)), AlertConfig())
    t = time.perf_counter()
    ctl.ng()
    assert ctl.last_error and time.perf_counter() - t < 2.0


# ---- station lock ---------------------------------------------------------
def test_station_lock_flow():
    lock = StationLock(hash_secret("4321", iterations=1000))
    events = []
    lock.subscribe(lambda s: events.append(s.locked))
    lock.lock("3 NG positions", "P001", 7)
    assert lock.locked and lock.state.inspection_id == 7
    assert not lock.release_by_pass("P002")  # a different part cannot clear it
    assert not lock.acknowledge("0000")
    assert lock.locked
    assert lock.release_by_pass("P001")
    assert not lock.locked
    lock.lock("NG", "P001")
    assert lock.acknowledge("4321") and not lock.locked
    assert events == [True, False, True, False]


def test_disabled_lock_never_locks():
    lock = StationLock("", enabled=False)
    lock.lock("NG", "P001")
    assert not lock.locked
