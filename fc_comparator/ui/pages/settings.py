"""Model Setup - station settings: camera, detection, tower light, model login and supervisor PIN.

Everything is written back to config.yaml. Rarely changed hardware details (GPIO pins,
relay channels, Modbus coils ...) are edited in config/config.yaml directly.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ...config import save_config
from ...station.security import hash_secret
from ..widgets import ask_string, show_error, show_info

# (section, config path, label, type, choices)
FIELDS = [
    ("Station & camera", "station.name", "Station name", str, None),
    ("Station & camera", "trigger.key", "Inspect key (foot pedal)", str, None),
    ("Station & camera", "camera.source", "Image source", str, ["camera", "file"]),
    ("Station & camera", "camera.index", "Camera number", int, None),
    ("Station & camera", "camera.width", "Camera width", int, None),
    ("Station & camera", "camera.height", "Camera height", int, None),
    ("Station & camera", "camera.file_path", "Image file (file source)", str, None),
    ("Station & camera", "storage.save_ok_every_n", "Save every Nth OK image", int, None),
    ("Station & camera", "ui.fullscreen", "Fullscreen (after restart)", bool, None),
    ("Detection", "detector.backend", "Detection", str, ["yolo", "template"]),
    ("Detection", "detector.confidence_threshold", "Confidence threshold", float, None),
    ("Detection", "detector.min_model_accuracy", "Accuracy a model needs", float, None),
    ("Detection", "detector.device", "Device (cpu / 0 = GPU)", str, None),
    ("Detection", "training.epochs", "Training epochs", int, None),
    ("Detection", "training.imgsz", "Training image size", int, None),
    ("Alerts", "alert.backend", "Tower light", str, ["console", "gpio", "usb_relay", "modbus", "none"]),
    ("Alerts", "alert.buzzer_mode", "Buzzer", str, ["pulse", "until_ack"]),
    ("Alerts", "alert.buzzer_seconds", "Buzzer pulse (s)", float, None),
    ("Alerts", "alert.usb_relay.port", "USB relay port", str, None),
    ("Alerts", "alert.modbus.host", "PLC address (Modbus)", str, None),
    ("Security", "security.lock_on_ng", "Lock the station after an NG", bool, None),
    ("Security", "auth.setup_timeout_min", "Model Setup closes after (min)", float, None),
]


class SettingsPanel(ttk.Frame):
    def __init__(self, master, page):
        super().__init__(master, padding=12)
        self.page = page
        self.app = page.app
        self.station = page.station
        self.cfg = page.cfg
        self.vars: dict[str, tuple[tk.Variable, type]] = {}

        sections: dict[str, ttk.LabelFrame] = {}
        for i, name in enumerate(["Station & camera", "Detection", "Alerts", "Security"]):
            f = ttk.LabelFrame(self, text=name, padding=10)
            f.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            f.grid_columnconfigure(1, weight=1)
            sections[name] = f
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        for section, path, label, kind, choices in FIELDS:
            f = sections[section]
            r = f.grid_size()[1]
            if kind is bool:
                var = tk.BooleanVar()
                ttk.Checkbutton(f, text=label, variable=var).grid(row=r, column=0, columnspan=2, sticky="w", pady=3)
            else:
                var = tk.StringVar()
                ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
                w = (ttk.Combobox(f, textvariable=var, values=choices, state="readonly") if choices
                     else ttk.Entry(f, textvariable=var))
                w.grid(row=r, column=1, sticky="ew", pady=3)
            self.vars[path] = (var, kind)

        sec = sections["Security"]
        r = sec.grid_size()[1]
        ttk.Button(sec, text="Change model login...", command=self.change_login).grid(row=r, column=0, sticky="w",
                                                                                      pady=(8, 2))
        ttk.Button(sec, text="Change supervisor PIN...", command=self.change_pin).grid(row=r + 1, column=0, sticky="w")
        alerts = sections["Alerts"]
        ttk.Button(alerts, text="Test tower light", command=self.test_alert).grid(
            row=alerts.grid_size()[1], column=0, sticky="w", pady=(8, 0))
        det = sections["Detection"]
        ttk.Label(det, text="yolo = each part's trained model;  template = template matching on the part's labelled "
                            "images (trials only)", style="Muted.TLabel", wraplength=420, justify="left").grid(
            row=det.grid_size()[1], column=0, columnspan=2, sticky="w", pady=(8, 0))

        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(bar, text="Save", style="Accent.TButton", command=self.save).pack(side="left", padx=6)
        ttk.Button(bar, text="Undo changes", command=self.load).pack(side="left")
        ttk.Label(bar, text=f"Config file: {self.cfg.path}", style="Muted.TLabel").pack(side="right")
        self.load()

    # -- values -------------------------------------------------------------------
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

    def load(self) -> None:
        for path, (var, kind) in self.vars.items():
            v = self._get(path)
            var.set(bool(v) if kind is bool else ("" if v is None else str(v)))

    def on_show(self) -> None:
        self.load()

    def save(self) -> bool:
        values = {}
        for path, (var, kind) in self.vars.items():
            raw = var.get()
            try:
                values[path] = bool(raw) if kind is bool else kind(str(raw).strip())
            except ValueError:
                show_error(self, f"'{raw}' is not a valid value for {path}.")
                return False
        old = {p: self._get(p) for p in values}
        for path, value in values.items():
            self._set(path, value)
        problems = self.cfg.validate()
        if problems:
            for path, value in old.items():
                self._set(path, value)
            show_error(self, "\n".join(problems[:8]), "Please fix")
            return False
        save_config(self.cfg)
        issues = self.station.rebuild_io()
        self.station.reload()
        self.station.store.log_event("settings", self.page.user, "settings saved")
        self.app.on_config_changed()
        if issues:
            show_error(self, "Saved, but:\n" + "\n".join(issues))
        else:
            show_info(self, "Settings saved and applied.", "Saved")
        return True

    # -- security ----------------------------------------------------------------------
    def change_login(self) -> None:
        user = ask_string(self, "Change model login", "User name:", self.cfg.auth.user)
        if user is None:
            return
        first = ask_string(self, "Change model login", "New password:", secret=True)
        if first is None:
            return
        if not user.strip() or len(first) < 4:
            show_error(self, "Enter a user name and a password of at least 4 characters.")
            return
        if ask_string(self, "Change model login", "Repeat the new password:", secret=True) != first:
            show_error(self, "The passwords did not match; nothing was changed.")
            return
        self.cfg.auth.user = user.strip()
        self.cfg.auth.password = hash_secret(first)
        save_config(self.cfg)
        self.station.store.log_event("security", self.page.user, "model login changed")
        show_info(self, "The model login was changed.", "Saved")

    def change_pin(self) -> None:
        first = ask_string(self, "Supervisor PIN", "New supervisor PIN:", secret=True)
        if first is None:
            return
        if len(first) < 4:
            show_error(self, "Use at least 4 digits.")
            return
        if ask_string(self, "Supervisor PIN", "Repeat the new PIN:", secret=True) != first:
            show_error(self, "The entries did not match; nothing was changed.")
            return
        self.cfg.security.supervisor_pin = hash_secret(first)
        self.station.lock.pin_hash = self.cfg.security.supervisor_pin
        save_config(self.cfg)
        self.station.store.log_event("security", self.page.user, "supervisor PIN changed")
        show_info(self, "The supervisor PIN was changed.", "Saved")

    def test_alert(self) -> None:
        ctl = self.station.alert
        ctl.ok()
        self.after(1500, ctl.ng)
        self.after(3500, ctl.idle)
        self.after(3600, lambda: ctl.last_error and show_error(self, f"Alert fault: {ctl.last_error}"))
