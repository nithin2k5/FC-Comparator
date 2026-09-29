"""Main window: navigation rail + screens. Setup, Part numbers and Settings need the setup password."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from ..pipeline import Station
from ..security import verify_secret
from .history_screen import HistoryScreen
from .inspect_screen import InspectScreen
from .parts_screen import PartsScreen
from .settings_screen import SettingsScreen
from .setup_screen import SetupScreen
from .widgets import Dispatcher, PinDialog, error_box

log = logging.getLogger(__name__)

PROTECTED = {"setup", "parts", "settings"}
NAV = (("inspect", "Inspect"), ("setup", "Setup / Teach"), ("parts", "Part numbers"),
       ("history", "History / Reports"), ("settings", "Settings"))


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Tk, station: Station):
        super().__init__(root, padding=8)
        self.root = root
        self.station = station
        self.cfg = station.cfg
        self.admin = False
        self.current = "inspect"
        self._pedal = None
        self.dispatcher = Dispatcher(root)
        root.title(f"FC-Comparator - {self.cfg.station.name}")

        nav = ttk.Frame(self, width=230)
        nav.pack(side="left", fill="y", padx=(0, 8))
        nav.pack_propagate(False)
        self.nav_buttons: dict[str, ttk.Button] = {}
        for key, text in NAV:
            b = ttk.Button(nav, text=text, style="Nav.TButton", command=lambda k=key: self.go(k))
            b.pack(fill="x", pady=3)
            self.nav_buttons[key] = b
        self.leave_btn = ttk.Button(nav, text="Leave setup mode", command=self.leave_admin)
        self.status = ttk.Label(nav, style="Muted.TLabel", wraplength=210, justify="left")
        self.status.pack(side="bottom", fill="x", pady=4)

        content = ttk.Frame(self)
        content.pack(side="left", fill="both", expand=True)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        self.inspect = InspectScreen(content, station, self.dispatcher, app=self)
        self.setup = SetupScreen(content, station, self.on_config_changed)
        self.parts = PartsScreen(content, station, self.on_config_changed)
        self.history = HistoryScreen(content, station)
        self.settings = SettingsScreen(content, station, self.on_config_changed)
        self.pages = {"inspect": self.inspect, "setup": self.setup, "parts": self.parts,
                      "history": self.history, "settings": self.settings}
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self._install_triggers()
        self.go("inspect")
        self.set_navigation_locked(station.lock.locked)
        self.update_status()

    # -- triggers (foot pedal) -------------------------------------------------
    def _install_triggers(self) -> None:
        key = self.cfg.trigger.key.strip().strip("<>")
        try:
            self.root.bind_all(f"<{key}>", lambda _e: self.pedal_pressed())
        except tk.TclError:
            log.error("Invalid trigger.key %r (use a Tk key name such as F9, space, Return)", key)
        if self.cfg.trigger.gpio_pin is not None:
            try:
                from ..alert.gpio import GpioPedal

                # gpiozero calls back on its own thread: hand over to the Tk main loop.
                self._pedal = GpioPedal(self.cfg.trigger.gpio_pin, lambda: self.dispatcher.call(self.pedal_pressed))
                log.info("Foot pedal on GPIO %s", self.cfg.trigger.gpio_pin)
            except Exception as exc:
                log.error("GPIO foot pedal unavailable: %s", exc)

    def pedal_pressed(self) -> None:
        if self.current == "inspect":
            self.inspect.inspect()

    # -- navigation --------------------------------------------------------
    def go(self, key: str) -> bool:
        if key in PROTECTED and not self.admin:
            if self.station.lock.locked:
                error_box(self, "Station is locked after an NG. Resolve it first.")
                return False
            pw = PinDialog.ask(self, "Setup password")
            if pw is None:
                return False
            if not verify_secret(pw, self.cfg.security.setup_password):
                self.station.store.log_event("setup_denied", "", "wrong setup password")
                error_box(self, "Wrong password.")
                return False
            self.admin = True
            self.leave_btn.pack(fill="x", pady=(20, 3))
            self.station.store.log_event("setup_login", "", "")
        page = self.pages[key]
        if hasattr(page, "refresh"):
            page.refresh()
        page.tkraise()
        self.current = key
        for k, b in self.nav_buttons.items():
            b.configure(style="NavActive.TButton" if k == key else "Nav.TButton")
        if key == "inspect":
            self.inspect.focus_scan()
        self.update_status()
        return True

    def leave_admin(self) -> None:
        self.admin = False
        self.leave_btn.pack_forget()
        self.go("inspect")

    def set_navigation_locked(self, locked: bool) -> None:
        for key, b in self.nav_buttons.items():
            b.state(["disabled"] if locked and key not in ("inspect", "history") else ["!disabled"])

    def on_config_changed(self) -> None:
        self.inspect.refresh_parts()
        self.update_status()

    def update_status(self) -> None:
        clf = self.station.inspector.classifier
        align = "on" if self.station.inspector.aligner is not None else "off"
        self.status.configure(
            text=f"Classifier: {clf.name}\nAlignment: {align}\nAlert: {self.cfg.alert.backend}\n"
            f"Mode: {self.cfg.compare.mode}\nSource: {self.cfg.camera.source}"
        )

    def shutdown(self) -> None:
        self.inspect.stop()
        self.dispatcher.stop()
        if self._pedal is not None:
            self._pedal.close()
