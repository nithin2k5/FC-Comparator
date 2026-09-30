"""Main window: header bar + one tab per job (Inspect, Parts, Training, History, Settings)."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from ..station import Station
from .widgets import Dispatcher, show_error

log = logging.getLogger(__name__)

# Tabs that stay usable while the station is locked after an NG.
OPEN_WHEN_LOCKED = ("inspect", "history")


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Misc, station: Station, user: str = "", on_logout=None):
        super().__init__(root)
        self.root = root
        self.station = station
        self.cfg = station.cfg
        self.user = user
        self.on_logout = on_logout
        self._pedal = None
        self.dispatcher = Dispatcher(root)

        self._build_header()
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        from .pages.history import HistoryPage
        from .pages.inspect import InspectPage
        from .pages.parts import PartsPage
        from .pages.settings import SettingsPage
        from .pages.training import TrainingPage

        self.pages = {
            "inspect": InspectPage(self.tabs, self),
            "parts": PartsPage(self.tabs, self),
            "training": TrainingPage(self.tabs, self),
            "history": HistoryPage(self.tabs, self),
            "settings": SettingsPage(self.tabs, self),
        }
        titles = {"inspect": "  Inspect  ", "parts": "  Parts  ", "training": "  Training  ",
                  "history": "  History  ", "settings": "  Settings  "}
        for key, page in self.pages.items():
            self.tabs.add(page, text=titles[key])
        self.inspect = self.pages["inspect"]
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        self._current = "inspect"

        self._install_triggers()
        self.set_locked(station.lock.locked)
        self.update_status()
        self.inspect.focus_scan()

    # -- header ----------------------------------------------------------------
    def _build_header(self) -> None:
        bar = ttk.Frame(self, style="Header.TFrame", padding=(14, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="FC-Comparator", style="Header.TLabel").pack(side="left")
        ttk.Label(bar, text=f"   {self.cfg.station.name}", style="HeaderMuted.TLabel").pack(side="left")
        self.logout_btn = ttk.Button(bar, text="Log out", style="Header.TButton", command=self.logout)
        self.logout_btn.pack(side="right")
        ttk.Label(bar, text=f"User: {self.user}   " if self.user else "", style="HeaderMuted.TLabel").pack(side="right")
        self.status = ttk.Label(bar, text="", style="HeaderMuted.TLabel")
        self.status.pack(side="right", padx=20)

    def update_status(self) -> None:
        det = self.station.inspector.detector
        name = {"yolo": "trained model", "template": "template matching"}.get(det.name, det.name)
        state = "ready" if det.ready else "NOT READY"
        self.status.configure(text=f"Detector: {name} ({state})  ·  {len(self.cfg.parts)} parts  ·  "
                                   f"alert: {self.cfg.alert.backend}  ·  camera: {self.cfg.camera.source}")

    # -- tabs --------------------------------------------------------------------
    @property
    def current(self) -> str:
        return self._current

    def go(self, key: str) -> bool:
        if self.station.lock.locked and key not in OPEN_WHEN_LOCKED:
            show_error(self, "The station is locked after an NG. Re-inspect the part or get the supervisor PIN.",
                       "Station locked")
            return False
        self.tabs.select(self.pages[key])
        self._tab_changed()
        return True

    def _tab_changed(self, _e=None) -> None:
        widget = self.root.nametowidget(self.tabs.select())
        key = next(k for k, p in self.pages.items() if p is widget)
        if key == self._current:
            return
        old = self.pages[self._current]
        if hasattr(old, "on_hide"):
            old.on_hide()
        self._current = key
        if hasattr(widget, "on_show"):
            widget.on_show()

    def set_locked(self, locked: bool) -> None:
        for key, page in getattr(self, "pages", {}).items():  # pages call this while being built
            self.tabs.tab(page, state="disabled" if locked and key not in OPEN_WHEN_LOCKED else "normal")

    def on_config_changed(self) -> None:
        for page in self.pages.values():
            if hasattr(page, "on_config_changed"):
                page.on_config_changed()
        self.update_status()

    def rebuild_detector(self, then=None) -> None:
        """Reload the detector in the background (loading a model takes seconds)."""
        self.status.configure(text="Detector: loading ...")

        def done(_r):
            self.update_status()
            if then:
                then()

        def failed(exc):
            self.update_status()
            show_error(self, f"Could not load the detector: {exc}")

        self.dispatcher.run_task(self.station.reload, done, failed)

    # -- foot pedal / trigger key ------------------------------------------------
    def _install_triggers(self) -> None:
        key = self.cfg.trigger.key.strip().strip("<>")
        try:
            self.root.bind_all(f"<{key}>", lambda _e: self.pedal_pressed())
        except tk.TclError:
            log.error("Invalid trigger.key %r (use a Tk key name such as F9, space, Return)", key)
        if self.cfg.trigger.gpio_pin is not None:
            try:
                from ..station.alerts.gpio import GpioPedal

                # gpiozero calls back on its own thread: hand over to the Tk main loop.
                self._pedal = GpioPedal(self.cfg.trigger.gpio_pin, lambda: self.dispatcher.call(self.pedal_pressed))
            except Exception as exc:
                log.error("GPIO foot pedal unavailable: %s", exc)

    def pedal_pressed(self) -> None:
        if self._current == "inspect":
            self.inspect.inspect()

    # -- session -------------------------------------------------------------------
    def logout(self) -> None:
        self.station.store.log_event("logout", self.user, "")
        if self.on_logout:
            self.on_logout()

    def shutdown(self) -> None:
        for page in self.pages.values():
            if hasattr(page, "stop"):
                page.stop()
        self.dispatcher.stop()
        try:
            self.root.unbind_all(f"<{self.cfg.trigger.key.strip().strip('<>')}>")
        except tk.TclError:
            pass
        if self._pedal is not None:
            self._pedal.close()
