"""Settings screen: camera, thresholds, comparison mode, alert hardware, storage, security."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ..config import save_config
from ..pipeline import Station
from ..security import hash_secret
from .widgets import PinDialog, ScrollFrame, error_box, info_box


class SettingsScreen(ttk.Frame):
    def __init__(self, master, station: Station, on_config_changed: Callable[[], None]):
        super().__init__(master)
        self.station = station
        self.cfg = station.cfg
        self.on_config_changed = on_config_changed
        self.v: dict[str, tk.Variable] = {}  # dotted config path -> variable

        scroll = ScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        c = self.cfg
        cam = self._group(body, "Camera", 0, 0)
        self._combo(cam, "camera.source", "Source", ["camera", "file"])
        self._spin(cam, "camera.index", "Camera index", 0, 16)
        self._combo(cam, "camera.backend", "Backend", ["auto", "dshow", "msmf", "v4l2", "gstreamer"])
        self._spin(cam, "camera.width", "Width", 160, 8192)
        self._spin(cam, "camera.height", "Height", 120, 8192)
        self._check(cam, "camera.autofocus", "Autofocus")
        self._entry(cam, "camera.file_path", "Image file (file source)")

        cls = self._group(body, "Classification & comparison", 1, 0)
        self._combo(cls, "classifier.backend", "Classifier", ["auto", "yolo", "template"])
        self._entry(cls, "classifier.model_path", "YOLO model path")
        self._spin(cls, "classifier.confidence_threshold", "Confidence threshold\n(below = uncertain = NG)", 0.0, 1.0, 0.05, float)
        self._combo(cls, "compare.mode", "Compare mode\n(master = A, cross = B)", ["master", "cross", "both"])
        self._check(cls, "alignment.enabled", "Align to reference")
        self._check(cls, "alignment.fail_as_ng", "Alignment failure = NG")
        self._spin(cls, "alignment.max_shift_px", "Max board shift (px)", 1, 500, 5, float)

        misc = self._group(body, "Station, storage & security", 2, 0)
        self._entry(misc, "station.name", "Station name")
        self._spin(misc, "storage.save_ok_every_n", "Save every Nth OK image\n(0 = none)", 0, 10000)
        self._entry(misc, "trigger.key", "Inspect / foot pedal key")
        self.v["trigger.gpio_pin"] = tk.IntVar(value=c.trigger.gpio_pin if c.trigger.gpio_pin is not None else -1)
        self._row(misc, "Foot pedal GPIO pin\n(-1 = none)", ttk.Spinbox(misc, from_=-1, to=40, textvariable=self.v["trigger.gpio_pin"]))
        self._check(misc, "security.lock_on_ng", "Lock station on NG")
        self._check(misc, "ui.fullscreen", "Fullscreen (restart)")
        btns = ttk.Frame(misc)
        btns.grid(row=misc.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Change supervisor PIN", command=lambda: self.change_secret("pin")).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btns, text="Change setup password", command=lambda: self.change_secret("setup")).pack(
            side="left", fill="x", expand=True)

        al = self._group(body, "Alert hardware (tower light + buzzer)", 0, 1, rowspan=3)
        self._combo(al, "alert.backend", "Backend", ["console", "gpio", "usb_relay", "modbus", "none"])
        self._combo(al, "alert.buzzer_mode", "Buzzer mode", ["pulse", "until_ack"])
        self._spin(al, "alert.buzzer_seconds", "Buzzer pulse (s)", 0.1, 60, 0.5, float)
        ttk.Separator(al).grid(row=al.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=6)
        self._spin(al, "alert.gpio.green_pin", "GPIO green pin (BCM)", 0, 40)
        self._spin(al, "alert.gpio.red_pin", "GPIO red pin", 0, 40)
        self._spin(al, "alert.gpio.buzzer_pin", "GPIO buzzer pin", 0, 40)
        self._check(al, "alert.gpio.active_high", "GPIO active high")
        ttk.Separator(al).grid(row=al.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=6)
        self._entry(al, "alert.usb_relay.port", "USB relay port")
        self._spin(al, "alert.usb_relay.green_channel", "Relay green channel", 1, 8)
        self._spin(al, "alert.usb_relay.red_channel", "Relay red channel", 1, 8)
        self._spin(al, "alert.usb_relay.buzzer_channel", "Relay buzzer channel", 1, 8)
        ttk.Separator(al).grid(row=al.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=6)
        self._entry(al, "alert.modbus.host", "Modbus PLC host")
        self._spin(al, "alert.modbus.port", "Modbus port", 1, 65535)
        self._spin(al, "alert.modbus.unit_id", "Modbus unit id", 0, 255)
        self._spin(al, "alert.modbus.green_coil", "Green coil", 0, 65535)
        self._spin(al, "alert.modbus.red_coil", "Red coil", 0, 65535)
        self._spin(al, "alert.modbus.buzzer_coil", "Buzzer coil", 0, 65535)
        ttk.Button(al, text="Test tower light (green → red+buzzer → off)", command=self.test_alert).grid(
            row=al.grid_size()[1], column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Button(self, text="Save & apply", style="Primary.TButton", command=self.save).pack(fill="x", pady=(8, 0))

    # -- form builders (bound to dotted config paths) -----------------------
    def _get(self, path: str):
        obj = self.cfg
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def _set(self, path: str, value) -> None:
        *head, last = path.split(".")
        obj = self.cfg
        for part in head:
            obj = getattr(obj, part)
        setattr(obj, last, value)

    @staticmethod
    def _group(parent, title: str, row: int, col: int, rowspan: int = 1) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text=title, padding=8)
        g.grid(row=row, column=col, rowspan=rowspan, sticky="nsew", padx=4, pady=4)
        g.columnconfigure(1, weight=1)
        return g

    @staticmethod
    def _row(g, label: str, widget) -> None:
        r = g.grid_size()[1]
        ttk.Label(g, text=label).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
        widget.grid(row=r, column=1, sticky="ew", pady=3)

    def _entry(self, g, path: str, label: str) -> None:
        self.v[path] = tk.StringVar(value=str(self._get(path) or ""))
        self._row(g, label, ttk.Entry(g, textvariable=self.v[path]))

    def _combo(self, g, path: str, label: str, values: list[str]) -> None:
        self.v[path] = tk.StringVar(value=str(self._get(path)))
        self._row(g, label, ttk.Combobox(g, textvariable=self.v[path], values=values, state="readonly"))

    def _spin(self, g, path: str, label: str, lo, hi, step=1, kind=int) -> None:
        var = tk.DoubleVar(value=float(self._get(path))) if kind is float else tk.IntVar(value=int(self._get(path)))
        self.v[path] = var
        self._row(g, label, ttk.Spinbox(g, from_=lo, to=hi, increment=step, textvariable=var))

    def _check(self, g, path: str, label: str) -> None:
        self.v[path] = tk.BooleanVar(value=bool(self._get(path)))
        self._row(g, label, ttk.Checkbutton(g, variable=self.v[path]))

    # -- actions -----------------------------------------------------------
    def save(self) -> None:
        values = {}
        for path, var in self.v.items():
            try:
                values[path] = var.get()
            except (tk.TclError, ValueError):
                error_box(self, f"Invalid value for {path}")
                return
        pin = values.pop("trigger.gpio_pin")
        for path, value in values.items():
            if isinstance(value, str):
                value = value.strip()
            self._set(path, value)
        self.cfg.trigger.gpio_pin = None if pin < 0 else int(pin)
        self.cfg.trigger.key = self.cfg.trigger.key or "F9"
        self.cfg.station.name = self.cfg.station.name or "Station-1"

        problems = [p for p in self.cfg.validate() if "ROI" not in p]
        if problems:
            error_box(self, "\n".join(problems))
            return
        save_config(self.cfg)
        issues = self.station.rebuild_io()
        try:
            self.station.reload()
        except Exception as exc:
            issues.append(f"classifier: {exc}")
        self.station.store.log_event("settings", "", "settings saved")
        self.on_config_changed()
        if issues:
            error_box(self, "Saved, but:\n" + "\n".join(issues))
        else:
            info_box(self, "Settings saved and applied.")

    def test_alert(self) -> None:
        ctl = self.station.alert
        ctl.ok()
        self.after(1500, ctl.ng)
        self.after(3500, ctl.idle)
        self.after(3600, lambda: ctl.last_error and error_box(self, f"Alert fault: {ctl.last_error}"))

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
