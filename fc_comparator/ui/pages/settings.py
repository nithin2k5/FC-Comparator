"""Settings page: station & camera, detection, clip types, alerts, security."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from ...config import save_config
from ...models import Taxonomy
from ...pipeline import Station
from ...security import hash_secret
from .. import theme
from ..widgets import Dispatcher, PinDialog, button, muted, show_error, show_info

TABS = ("Station & camera", "Detection", "Clip types", "Alerts", "Security")


class SettingsPage(ctk.CTkFrame):
    title = "Settings"

    def __init__(self, master, app, station: Station, dispatcher: Dispatcher):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.cfg = station.cfg
        self.dispatcher = dispatcher
        self.v: dict[str, tuple[tk.Variable, type]] = {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tabs = ctk.CTkTabview(self, fg_color=theme.SURFACE, segmented_button_selected_color=theme.ACCENT,
                                   corner_radius=14, border_width=1, border_color=theme.BORDER)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.body: dict[str, ctk.CTkScrollableFrame] = {}
        for t in TABS:
            scroll = ctk.CTkScrollableFrame(self.tabs.add(t), fg_color="transparent")
            scroll.pack(fill="both", expand=True)
            self.body[t] = scroll
        self._build()
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        button(bar, "Save & apply", self.save, "primary", width=200, height=46).pack(side="right")
        self.hint = muted(bar, "", 13)
        self.hint.pack(side="left")

    # -- form helpers -------------------------------------------------------
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

    def _tab(self, name: str) -> ctk.CTkScrollableFrame:
        return self.body[name]

    def _section(self, tab: str, title: str) -> ctk.CTkFrame:
        holder = self._tab(tab)
        f = ctk.CTkFrame(holder, fg_color=theme.SURFACE_2, corner_radius=12)
        f.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(f, text=title, font=theme.font(14, "bold"), text_color=theme.TEXT).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 4))
        f.grid_columnconfigure(1, weight=1)
        return f

    def _row(self, f, label: str, widget, help_text: str = "") -> None:
        r = f.grid_size()[1]
        ctk.CTkLabel(f, text=label, font=theme.font(13), text_color=theme.TEXT, anchor="w", width=250).grid(
            row=r, column=0, sticky="w", padx=14, pady=4)
        widget.grid(row=r, column=1, sticky="w", padx=(0, 14), pady=4)
        if help_text:
            muted(f, help_text, 11).grid(row=r + 1, column=1, sticky="w", padx=(0, 14))

    def _entry(self, f, path, label, width=260, kind=str, help_text=""):
        value = self._get(path)
        var = tk.StringVar(value="" if value is None else str(value))
        self.v[path] = (var, kind)
        self._row(f, label, ctk.CTkEntry(f, textvariable=var, width=width, height=34), help_text)

    def _option(self, f, path, label, values, help_text=""):
        var = tk.StringVar(value=str(self._get(path)))
        self.v[path] = (var, str)
        self._row(f, label, ctk.CTkOptionMenu(f, variable=var, values=values, width=220, height=34), help_text)

    def _switch(self, f, path, label, help_text=""):
        var = tk.BooleanVar(value=bool(self._get(path)))
        self.v[path] = (var, bool)
        self._row(f, label, ctk.CTkSwitch(f, text="", variable=var), help_text)

    def _build(self) -> None:
        s = self._section("Station & camera", "Station")
        self._entry(s, "station.name", "Station name")
        self._entry(s, "trigger.key", "Inspect key / foot pedal key", 120, help_text="Tk key name, e.g. F9 or space")
        self._entry(s, "trigger.gpio_pin", "Foot pedal GPIO pin (BCM)", 120, kind=int, help_text="empty = none (Raspberry Pi only)")
        self._switch(s, "ui.fullscreen", "Fullscreen (after restart)")
        s = self._section("Station & camera", "Camera")
        self._option(s, "camera.source", "Source", ["camera", "file"])
        self._entry(s, "camera.index", "Camera index", 120, int)
        self._option(s, "camera.backend", "Backend", ["auto", "dshow", "msmf", "v4l2", "gstreamer"])
        self._entry(s, "camera.width", "Width", 120, int)
        self._entry(s, "camera.height", "Height", 120, int)
        self._switch(s, "camera.autofocus", "Autofocus", "off is recommended: fixed focus gives repeatable images")
        self._entry(s, "camera.file_path", "Image file (file source)", 420)
        s = self._section("Station & camera", "Storage")
        self._entry(s, "storage.save_ok_every_n", "Save every Nth OK image", 120, int, "NG images are always saved; 0 = no OK images")

        s = self._section("Detection", "Detector")
        self._option(s, "detector.backend", "Detector", ["auto", "yolo", "template"],
                     "auto = trained model when good enough, else template matching")
        self._entry(s, "detector.model_path", "Model file", 420)
        self._entry(s, "detector.confidence_threshold", "Confidence threshold", 120, float,
                    "below this a clip is 'uncertain' = NG")
        self._entry(s, "detector.min_model_accuracy", "Minimum model accuracy for auto", 120, float)
        self._entry(s, "detector.imgsz", "Model image size", 120, int)
        self._entry(s, "detector.device", "Device", 120, help_text="cpu, or 0 for the first GPU")
        s = self._section("Detection", "Comparison & layout")
        self._option(s, "compare.mode", "Compare mode", ["master", "cross", "both"],
                     "master = against the part's pattern; cross = cables must match each other")
        self._entry(s, "layout.max_shift_px", "Max board shift (px)", 120, float)
        self._entry(s, "layout.max_rotation_deg", "Max board rotation (deg)", 120, float)

        self._build_taxonomy()

        s = self._section("Alerts", "Tower light & buzzer")
        self._option(s, "alert.backend", "Backend", ["console", "gpio", "usb_relay", "modbus", "none"])
        self._option(s, "alert.buzzer_mode", "Buzzer", ["pulse", "until_ack"])
        self._entry(s, "alert.buzzer_seconds", "Buzzer pulse (s)", 120, float)
        s = self._section("Alerts", "Raspberry Pi GPIO (BCM pins)")
        self._entry(s, "alert.gpio.green_pin", "Green", 120, int)
        self._entry(s, "alert.gpio.red_pin", "Red", 120, int)
        self._entry(s, "alert.gpio.buzzer_pin", "Buzzer", 120, int)
        self._switch(s, "alert.gpio.active_high", "Active high", "most relay modules are active low: switch off")
        s = self._section("Alerts", "USB relay board")
        self._entry(s, "alert.usb_relay.port", "Serial port", 160, help_text="COM3, /dev/ttyUSB0 ...")
        self._entry(s, "alert.usb_relay.green_channel", "Green channel", 120, int)
        self._entry(s, "alert.usb_relay.red_channel", "Red channel", 120, int)
        self._entry(s, "alert.usb_relay.buzzer_channel", "Buzzer channel", 120, int)
        s = self._section("Alerts", "Modbus TCP (PLC)")
        self._entry(s, "alert.modbus.host", "PLC address", 200)
        self._entry(s, "alert.modbus.port", "Port", 120, int)
        self._entry(s, "alert.modbus.unit_id", "Unit id", 120, int)
        self._entry(s, "alert.modbus.green_coil", "Green coil", 120, int)
        self._entry(s, "alert.modbus.red_coil", "Red coil", 120, int)
        self._entry(s, "alert.modbus.buzzer_coil", "Buzzer coil", 120, int)
        button(self._tab("Alerts"), "Test tower light  (green → red + buzzer → off)", self.test_alert,
               "secondary").pack(anchor="w", padx=12, pady=8)

        s = self._section("Security", "Access")
        self._switch(s, "security.lock_on_ng", "Lock the station after an NG")
        row = ctk.CTkFrame(self._tab("Security"), fg_color="transparent")
        row.pack(anchor="w", padx=12, pady=8)
        button(row, "Change supervisor PIN", lambda: self.change_secret("pin"), "secondary").pack(side="left")
        button(row, "Change setup password", lambda: self.change_secret("setup"), "secondary").pack(side="left", padx=8)

    # -- clip types ----------------------------------------------------------
    def _build_taxonomy(self) -> None:
        tab = self._tab("Clip types")
        muted(tab, "Clip types the detector learns. Adding a type: mark examples, then train. Renaming updates all "
                   "markings and part patterns.", 12, wraplength=900).pack(anchor="w", padx=14, pady=(8, 4))
        self.class_box = ctk.CTkFrame(tab, fg_color=theme.SURFACE_2, corner_radius=12)
        self.class_box.pack(fill="x", padx=12, pady=6)
        self.class_rows: list[tuple[str, ctk.CTkEntry, ctk.CTkFrame]] = []
        for name in self.cfg.taxonomy.classes:
            self._add_class_row(name)
        button(tab, "+ Add clip type", lambda: self._add_class_row(""), "secondary", width=160).pack(anchor="w", padx=12)
        s = self._section("Clip types", "Groups and mirror pairs")
        self.groups_var = tk.StringVar(value="; ".join(f"{g} = {', '.join(m)}" for g, m in self.cfg.taxonomy.groups.items()))
        seen, pairs = set(), []
        for a, b in self.cfg.taxonomy.mirror.items():
            if (b, a) not in seen:
                pairs.append(f"{a} <-> {b}")
                seen.add((a, b))
        self.mirror_var = tk.StringVar(value="; ".join(pairs))
        self._row(s, "Groups", ctk.CTkEntry(s, textvariable=self.groups_var, width=460, height=34),
                  "name = type, type; ...   e.g. fork = fork_left, fork_right (accepts either)")
        self._row(s, "Mirror pairs", ctk.CTkEntry(s, textvariable=self.mirror_var, width=460, height=34),
                  "a <-> b; ...   types that are left/right mirror images (used for training)")

    def _add_class_row(self, name: str) -> None:
        r = ctk.CTkFrame(self.class_box, fg_color="transparent")
        r.pack(fill="x", padx=12, pady=4)
        color = theme.class_color(self.cfg.taxonomy.classes, name) if name else "#64748b"
        ctk.CTkFrame(r, width=14, height=14, corner_radius=7, fg_color=color).pack(side="left", padx=(0, 10))
        e = ctk.CTkEntry(r, width=220, height=34)
        e.insert(0, name)
        e.pack(side="left")
        entry = (name, e, r)
        button(r, "Remove", lambda: self._remove_class_row(entry), "ghost", width=90).pack(side="left", padx=8)
        self.class_rows.append(entry)

    def _remove_class_row(self, entry) -> None:
        name = entry[0]
        store = self.station.annotations
        used_marks = store.stats()["per_class"].get(name, 0) if name else 0
        used_parts = [c for c, p in self.cfg.parts.items() if name and name in p.pattern]
        if used_marks or used_parts:
            show_error(self, f"'{name}' is still used by {used_marks} marked clip(s)"
                             + (f" and parts {', '.join(used_parts[:5])}" if used_parts else "")
                             + ". Re-mark or change those first.", "Clip type in use")
            return
        entry[2].destroy()
        self.class_rows.remove(entry)

    def _read_taxonomy(self) -> tuple[Taxonomy, dict[str, str]]:
        classes, renames = [], {}
        for old, e, _r in self.class_rows:
            new = e.get().strip()
            if not new:
                continue
            classes.append(new)
            if old and old != new:
                renames[old] = new
        groups = {}
        for part in filter(None, (p.strip() for p in self.groups_var.get().split(";"))):
            if "=" not in part:
                raise ValueError(f"Group '{part}' needs the form name = type, type")
            g, members = part.split("=", 1)
            groups[g.strip()] = [renames.get(m.strip(), m.strip()) for m in members.split(",") if m.strip()]
        mirror = {}
        for part in filter(None, (p.strip() for p in self.mirror_var.get().split(";"))):
            if "<->" not in part:
                raise ValueError(f"Mirror pair '{part}' needs the form a <-> b")
            a, b = (renames.get(x.strip(), x.strip()) for x in part.split("<->", 1))
            mirror[a], mirror[b] = b, a
        return Taxonomy(classes, groups, mirror), renames

    # -- actions -------------------------------------------------------------
    def save(self) -> None:
        values = {}
        for path, (var, kind) in self.v.items():
            raw = var.get()
            try:
                if kind is bool:
                    values[path] = bool(raw)
                elif isinstance(raw, str) and raw.strip() == "" and path in ("trigger.gpio_pin",):
                    values[path] = None
                else:
                    values[path] = kind(str(raw).strip())
            except ValueError:
                show_error(self, f"'{raw}' is not a valid value for {path}.")
                return
        try:
            taxonomy, renames = self._read_taxonomy()
        except ValueError as exc:
            show_error(self, str(exc))
            return
        problems = taxonomy.validate()
        if problems:
            show_error(self, "\n".join(problems), "Clip types")
            return
        old_classes = list(self.cfg.taxonomy.classes)
        for path, value in values.items():
            self._set(path, value)
        self.cfg.taxonomy = taxonomy
        for old, new in renames.items():
            self.station.annotations.rename_label(old, new)
            for p in self.cfg.parts.values():
                p.pattern = [new if x == old else x for x in p.pattern]
        problems = [p for p in self.cfg.validate()]
        if problems:
            show_error(self, "\n".join(problems[:8]), "Please fix")
            return
        save_config(self.cfg)
        issues = self.station.rebuild_io()
        self.station.store.log_event("settings", "", "settings saved")
        self.app.pages["data"].refresh()
        self.app.rebuild_detector()
        self.app.on_config_changed()
        msg = "Settings saved and applied."
        if taxonomy.classes != old_classes:
            msg += "\n\nClip types changed: mark examples of new types and train the model again."
        if issues:
            show_error(self, "Saved, but:\n" + "\n".join(issues))
        else:
            show_info(self, msg, "Saved")

    def test_alert(self) -> None:
        ctl = self.station.alert
        ctl.ok()
        self.after(1500, ctl.ng)
        self.after(3500, ctl.idle)
        self.after(3600, lambda: ctl.last_error and show_error(self, f"Alert fault: {ctl.last_error}"))

    def change_secret(self, which: str) -> None:
        label = "supervisor PIN" if which == "pin" else "setup password"
        first = PinDialog.ask(self, f"New {label}")
        if first is None:
            return
        if len(first) < 4:
            show_error(self, "Use at least 4 digits.")
            return
        if PinDialog.ask(self, f"Repeat the new {label}") != first:
            show_error(self, "The entries did not match; nothing was changed.")
            return
        if which == "pin":
            self.cfg.security.supervisor_pin = hash_secret(first)
            self.station.lock.pin_hash = self.cfg.security.supervisor_pin
        else:
            self.cfg.security.setup_password = hash_secret(first)
        save_config(self.cfg)
        self.station.store.log_event("security", "", f"{label} changed")
        show_info(self, f"The {label} was changed.", "Saved")
