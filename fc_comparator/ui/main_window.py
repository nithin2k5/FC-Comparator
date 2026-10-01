"""Main window: header bar + three tabs - Inspect, History and Model Setup.

Inspect and History are open to everyone. Model Setup asks for the model login
when it is opened; the session ends when another tab is chosen or after
``auth.setup_timeout_min`` minutes without input. While the station is locked
after an NG, Model Setup cannot be opened.
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import ttk

from ..station import Station
from ..station.security import verify_secret
from .widgets import Dispatcher, ask_login, show_error

log = logging.getLogger(__name__)

SETUP = "setup"
CHECK_MS = 5000  # how often the setup session timeout is checked


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Misc, station: Station):
        super().__init__(root)
        self.root = root
        self.station = station
        self.cfg = station.cfg
        self.session_user: str | None = None  # logged in to Model Setup
        self._last_input = time.monotonic()
        self._pedal = None
        self.dispatcher = Dispatcher(root)

        self._build_header()
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        from .pages.history import HistoryPage
        from .pages.inspect import InspectPage
        from .pages.setup import ModelSetupPage

        self.pages = {
            "inspect": InspectPage(self.tabs, self),
            "history": HistoryPage(self.tabs, self),
            SETUP: ModelSetupPage(self.tabs, self),
        }
        titles = {"inspect": "  Inspect  ", "history": "  History  ", SETUP: "  Model Setup  "}
        for key, page in self.pages.items():
            self.tabs.add(page, text=titles[key])
        self.inspect = self.pages["inspect"]
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        self._current = "inspect"

        for seq in ("<KeyPress>", "<ButtonPress>", "<MouseWheel>"):
            self.root.bind_all(seq, self._touch, add="+")
        self._timer = self.after(CHECK_MS, self._check_timeout)
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
        self.logout_btn = ttk.Button(bar, text="Leave Model Setup", style="Header.TButton", command=self.end_session)
        self.session_label = ttk.Label(bar, text="", style="HeaderMuted.TLabel")
        self.session_label.pack(side="right")
        self.status = ttk.Label(bar, text="", style="HeaderMuted.TLabel")
        self.status.pack(side="right", padx=20)

    def update_status(self) -> None:
        n = len(self.station.parts.codes())
        self.status.configure(text=f"{n} part number{'s' if n != 1 else ''}  ·  detection: {self.cfg.detector.backend}  ·  "
                                   f"alert: {self.cfg.alert.backend}  ·  camera: {self.cfg.camera.source}")
        if self.session_user:
            self.session_label.configure(text=f"Model Setup: {self.session_user}   ")
            self.logout_btn.pack(side="right")
        else:
            self.session_label.configure(text="")
            self.logout_btn.pack_forget()

    # -- tabs --------------------------------------------------------------------
    @property
    def current(self) -> str:
        return self._current

    def go(self, key: str) -> bool:
        if key == SETUP and self.station.lock.locked:
            show_error(self, "The station is locked after an NG. Re-inspect the part or get the supervisor PIN.",
                       "Station locked")
            return False
        self.tabs.select(self.pages[key])
        self._tab_changed()
        return self._current == key

    def _tab_changed(self, _e=None) -> None:
        widget = self.root.nametowidget(self.tabs.select())
        key = next(k for k, p in self.pages.items() if p is widget)
        if key == self._current:
            return
        if key == SETUP and not self.session_user and not self.login():
            self.tabs.select(self.pages[self._current])  # stay where we were
            return
        old, self._current = self._current, key
        if hasattr(self.pages[old], "on_hide"):
            self.pages[old].on_hide()
        if old == SETUP:
            self.end_session("left Model Setup")
        if hasattr(widget, "on_show"):
            widget.on_show()

    def set_locked(self, locked: bool) -> None:
        if hasattr(self, "pages"):  # pages call this while being built
            self.tabs.tab(self.pages[SETUP], state="disabled" if locked else "normal")

    # -- Model Setup session ---------------------------------------------------------
    def check_login(self, user: str, password: str) -> bool:
        return user == self.cfg.auth.user and verify_secret(password, self.cfg.auth.password)

    def login(self) -> bool:
        """Ask for the model login. Returns True when it was right."""
        creds = ask_login(self, "Model login", "Model Setup changes what the station inspects against.")
        if creds is None:
            return False
        user, password = creds
        if not self.check_login(user, password):
            self.station.store.log_event("setup_login_denied", user, "wrong user name or password")
            show_error(self, "Wrong user name or password.", "Model login")
            return False
        self.session_user = user
        self._last_input = time.monotonic()
        self.station.store.log_event("setup_login", user, "")
        self.update_status()
        return True

    def end_session(self, reason: str = "logged out") -> None:
        if not self.session_user:
            return
        user, self.session_user = self.session_user, None
        self.station.store.log_event("setup_logout", user, reason)
        self.pages[SETUP].on_logout()
        self.update_status()
        if self._current == SETUP:
            self.tabs.select(self.pages["inspect"])
            self._tab_changed()

    def _touch(self, _e=None) -> None:
        self._last_input = time.monotonic()

    def _check_timeout(self) -> None:
        if self.session_user and time.monotonic() - self._last_input > self.cfg.auth.setup_timeout_min * 60:
            self.end_session(f"no input for {self.cfg.auth.setup_timeout_min:g} min")
        try:
            self._timer = self.after(CHECK_MS, self._check_timeout)
        except tk.TclError:
            pass

    # -- changes made in Model Setup ---------------------------------------------------
    def parts_changed(self, code: str | None = None) -> None:
        """A part was created, deleted, or got a new model / master."""
        self.inspect.on_parts_changed(code)
        self.update_status()

    def on_config_changed(self) -> None:
        for page in self.pages.values():
            if hasattr(page, "on_config_changed"):
                page.on_config_changed()
        self.update_status()

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

    def shutdown(self) -> None:
        self.end_session("closed")
        for page in self.pages.values():
            if hasattr(page, "stop"):
                page.stop()
        self.dispatcher.stop()
        try:
            self.after_cancel(self._timer)
        except tk.TclError:
            pass
        for seq in ("<KeyPress>", "<ButtonPress>", "<MouseWheel>",
                    f"<{self.cfg.trigger.key.strip().strip('<>')}>"):
            try:
                self.root.unbind_all(seq)
            except tk.TclError:
                pass
        if self._pedal is not None:
            self._pedal.close()
