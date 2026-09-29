"""Settings screen: camera, thresholds, comparison mode, alert hardware, storage, security."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import save_config
from ..pipeline import Station
from ..security import hash_secret
from .widgets import PinDialog, big_button, error_box, info_box


def _spin(lo: int, hi: int, value: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(int(value))
    return s


def _dspin(lo: float, hi: float, value: float, step: float = 0.05, decimals: int = 2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(float(value))
    return s


def _combo(items: list[str], value: str) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    c.setCurrentText(value)
    return c


class SettingsScreen(QWidget):
    config_changed = Signal()

    def __init__(self, station: Station, parent=None):
        super().__init__(parent)
        self.station = station
        self.cfg = station.cfg
        c = self.cfg

        g_cam = QGroupBox("Camera")
        f = QFormLayout(g_cam)
        self.cam_source = _combo(["camera", "file"], c.camera.source)
        self.cam_index = _spin(0, 16, c.camera.index)
        self.cam_backend = _combo(["auto", "dshow", "msmf", "v4l2", "gstreamer"], c.camera.backend)
        self.cam_w = _spin(160, 8192, c.camera.width)
        self.cam_h = _spin(120, 8192, c.camera.height)
        self.cam_autofocus = QCheckBox()
        self.cam_autofocus.setChecked(c.camera.autofocus)
        self.cam_file = QLineEdit(c.camera.file_path)
        for lbl, w in (("Source", self.cam_source), ("Camera index", self.cam_index), ("Backend", self.cam_backend),
                       ("Width", self.cam_w), ("Height", self.cam_h), ("Autofocus", self.cam_autofocus),
                       ("Image file (file source)", self.cam_file)):
            f.addRow(lbl, w)

        g_cls = QGroupBox("Classification & comparison")
        f = QFormLayout(g_cls)
        self.backend = _combo(["auto", "yolo", "template"], c.classifier.backend)
        self.model_path = QLineEdit(c.classifier.model_path)
        self.threshold = _dspin(0.0, 1.0, c.classifier.confidence_threshold)
        self.mode = _combo(["master", "cross", "both"], c.compare.mode)
        self.align = QCheckBox()
        self.align.setChecked(c.alignment.enabled)
        self.align_ng = QCheckBox()
        self.align_ng.setChecked(c.alignment.fail_as_ng)
        self.max_shift = _dspin(1, 500, c.alignment.max_shift_px, 5, 0)
        for lbl, w in (("Classifier", self.backend), ("YOLO model path", self.model_path),
                       ("Confidence threshold (below = uncertain = NG)", self.threshold),
                       ("Compare mode (master = A, cross = B)", self.mode), ("Align to reference", self.align),
                       ("Alignment failure = NG", self.align_ng), ("Max board shift (px)", self.max_shift)):
            f.addRow(lbl, w)

        g_alert = QGroupBox("Alert hardware (tower light + buzzer)")
        f = QFormLayout(g_alert)
        a = c.alert
        self.alert_backend = _combo(["console", "gpio", "usb_relay", "modbus", "none"], a.backend)
        self.buzzer_mode = _combo(["pulse", "until_ack"], a.buzzer_mode)
        self.buzzer_s = _dspin(0.1, 60, a.buzzer_seconds, 0.5, 1)
        self.gpio_g, self.gpio_r, self.gpio_b = (_spin(0, 40, p) for p in (a.gpio.green_pin, a.gpio.red_pin, a.gpio.buzzer_pin))
        self.gpio_active_high = QCheckBox()
        self.gpio_active_high.setChecked(a.gpio.active_high)
        self.relay_port = QLineEdit(a.usb_relay.port)
        self.relay_g, self.relay_r, self.relay_b = (
            _spin(1, 8, ch) for ch in (a.usb_relay.green_channel, a.usb_relay.red_channel, a.usb_relay.buzzer_channel)
        )
        self.mb_host = QLineEdit(a.modbus.host)
        self.mb_port = _spin(1, 65535, a.modbus.port)
        self.mb_unit = _spin(0, 255, a.modbus.unit_id)
        self.mb_g, self.mb_r, self.mb_b = (_spin(0, 65535, x) for x in (a.modbus.green_coil, a.modbus.red_coil, a.modbus.buzzer_coil))
        rows = [
            ("Backend", self.alert_backend), ("Buzzer mode", self.buzzer_mode), ("Buzzer pulse (s)", self.buzzer_s),
            ("GPIO green / red / buzzer pin (BCM)", self._row(self.gpio_g, self.gpio_r, self.gpio_b)),
            ("GPIO active high", self.gpio_active_high),
            ("USB relay port", self.relay_port),
            ("USB relay green / red / buzzer channel", self._row(self.relay_g, self.relay_r, self.relay_b)),
            ("Modbus PLC host", self.mb_host), ("Modbus port / unit id", self._row(self.mb_port, self.mb_unit)),
            ("Modbus green / red / buzzer coil", self._row(self.mb_g, self.mb_r, self.mb_b)),
        ]
        for lbl, w in rows:
            f.addRow(lbl, w)
        b_test = big_button("Test tower light (green → red+buzzer → off)")
        b_test.clicked.connect(self.test_alert)
        f.addRow(b_test)

        g_misc = QGroupBox("Station, storage & security")
        f = QFormLayout(g_misc)
        self.station_name = QLineEdit(c.station.name)
        self.ok_every = _spin(0, 10000, c.storage.save_ok_every_n)
        self.trigger_key = QLineEdit(c.trigger.key)
        self.pedal_pin = _spin(-1, 40, c.trigger.gpio_pin if c.trigger.gpio_pin is not None else -1)
        self.lock_on_ng = QCheckBox()
        self.lock_on_ng.setChecked(c.security.lock_on_ng)
        self.fullscreen = QCheckBox()
        self.fullscreen.setChecked(c.ui.fullscreen)
        b_pin = big_button("Change supervisor PIN")
        b_pin.clicked.connect(lambda: self.change_secret("pin"))
        b_pw = big_button("Change setup password")
        b_pw.clicked.connect(lambda: self.change_secret("setup"))
        for lbl, w in (("Station name", self.station_name), ("Save every Nth OK image (0 = none)", self.ok_every),
                       ("Inspect key / foot pedal key", self.trigger_key), ("Foot pedal GPIO pin (-1 = none)", self.pedal_pin),
                       ("Lock station on NG", self.lock_on_ng), ("Fullscreen (restart)", self.fullscreen)):
            f.addRow(lbl, w)
        f.addRow(self._row(b_pin, b_pw))

        b_save = big_button("Save & apply", "primary")
        b_save.clicked.connect(self.save)

        body = QWidget()
        grid = QGridLayout(body)
        grid.addWidget(g_cam, 0, 0)
        grid.addWidget(g_cls, 1, 0)
        grid.addWidget(g_misc, 2, 0)
        grid.addWidget(g_alert, 0, 1, 3, 1)
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll, 1)
        lay.addWidget(b_save)

    @staticmethod
    def _row(*widgets) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w

    def save(self) -> None:
        c = self.cfg
        c.camera.source = self.cam_source.currentText()
        c.camera.index = self.cam_index.value()
        c.camera.backend = self.cam_backend.currentText()
        c.camera.width, c.camera.height = self.cam_w.value(), self.cam_h.value()
        c.camera.autofocus = self.cam_autofocus.isChecked()
        c.camera.file_path = self.cam_file.text().strip()
        c.classifier.backend = self.backend.currentText()
        c.classifier.model_path = self.model_path.text().strip()
        c.classifier.confidence_threshold = self.threshold.value()
        c.compare.mode = self.mode.currentText()
        c.alignment.enabled = self.align.isChecked()
        c.alignment.fail_as_ng = self.align_ng.isChecked()
        c.alignment.max_shift_px = self.max_shift.value()
        a = c.alert
        a.backend = self.alert_backend.currentText()
        a.buzzer_mode = self.buzzer_mode.currentText()
        a.buzzer_seconds = self.buzzer_s.value()
        a.gpio.green_pin, a.gpio.red_pin, a.gpio.buzzer_pin = self.gpio_g.value(), self.gpio_r.value(), self.gpio_b.value()
        a.gpio.active_high = self.gpio_active_high.isChecked()
        a.usb_relay.port = self.relay_port.text().strip()
        a.usb_relay.green_channel, a.usb_relay.red_channel, a.usb_relay.buzzer_channel = (
            self.relay_g.value(), self.relay_r.value(), self.relay_b.value())
        a.modbus.host = self.mb_host.text().strip()
        a.modbus.port, a.modbus.unit_id = self.mb_port.value(), self.mb_unit.value()
        a.modbus.green_coil, a.modbus.red_coil, a.modbus.buzzer_coil = self.mb_g.value(), self.mb_r.value(), self.mb_b.value()
        c.station.name = self.station_name.text().strip() or "Station-1"
        c.storage.save_ok_every_n = self.ok_every.value()
        c.trigger.key = self.trigger_key.text().strip() or "F9"
        c.trigger.gpio_pin = None if self.pedal_pin.value() < 0 else self.pedal_pin.value()
        c.security.lock_on_ng = self.lock_on_ng.isChecked()
        c.ui.fullscreen = self.fullscreen.isChecked()

        problems = [p for p in c.validate() if "ROI" not in p]
        if problems:
            error_box(self, "\n".join(problems))
            return
        save_config(c)
        issues = self.station.rebuild_io()
        try:
            self.station.reload()
        except Exception as exc:
            issues.append(f"classifier: {exc}")
        self.station.store.log_event("settings", "", "settings saved")
        self.config_changed.emit()
        if issues:
            error_box(self, "Saved, but:\n" + "\n".join(issues))
        else:
            info_box(self, "Settings saved and applied.")

    def test_alert(self) -> None:
        from PySide6.QtCore import QTimer

        ctl = self.station.alert
        ctl.ok()
        QTimer.singleShot(1500, ctl.ng)
        QTimer.singleShot(3500, ctl.idle)
        QTimer.singleShot(3600, lambda: ctl.last_error and error_box(self, f"Alert fault: {ctl.last_error}"))

    def change_secret(self, which: str) -> None:
        label = "supervisor PIN" if which == "pin" else "setup password"
        first = PinDialog.ask(self, f"New {label}", numeric_only=which == "pin")
        if first is None:
            return
        if len(first) < 4:
            error_box(self, "Use at least 4 characters.")
            return
        if PinDialog.ask(self, f"Repeat new {label}", numeric_only=which == "pin") != first:
            error_box(self, "Entries do not match; unchanged.")
            return
        if which == "pin":
            self.cfg.security.supervisor_pin = hash_secret(first)
            self.station.lock.pin_hash = self.cfg.security.supervisor_pin
        else:
            self.cfg.security.setup_password = hash_secret(first)
        save_config(self.cfg)
        self.station.store.log_event("security", "", f"{label} changed")
        info_box(self, f"{label.capitalize()} changed.")
