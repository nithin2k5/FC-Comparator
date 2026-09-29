"""Main window: sidebar navigation + pages. Setup pages need the setup password."""

from __future__ import annotations

import logging
import tkinter as tk

import customtkinter as ctk

from ..pipeline import Station
from ..security import verify_secret
from . import theme
from .widgets import Dispatcher, PinDialog, show_error

log = logging.getLogger(__name__)

# key, label, icon, protected
NAV = [
    ("inspect", "Inspect", "◉", False),
    ("data", "Training data", "✎", True),
    ("train", "Train model", "⚙", True),
    ("parts", "Part numbers", "☰", True),
    ("history", "History & reports", "⏱", False),
    ("settings", "Settings", "⚒", True),
]


class MainWindow(ctk.CTkFrame):
    def __init__(self, root: tk.Misc, station: Station):
        super().__init__(root, fg_color=theme.BG, corner_radius=0)
        self.root = root
        self.station = station
        self.cfg = station.cfg
        self.admin = False
        self.current = ""
        self._pedal = None
        self.dispatcher = Dispatcher(root)
        root.title(f"FC-Comparator - {self.cfg.station.name}")
        theme.style_ttk(root)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=24, pady=20)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)
        self.header = ctk.CTkLabel(self.content, text="", font=theme.font(24, "bold"), text_color=theme.TEXT, anchor="w")
        self.header.grid(row=0, column=0, sticky="w", pady=(0, 14))

        from .pages.data import TrainingDataPage
        from .pages.history import HistoryPage
        from .pages.inspect import InspectPage
        from .pages.parts import PartsPage
        from .pages.settings import SettingsPage
        from .pages.train import TrainPage

        self.pages = {
            "inspect": InspectPage(self.content, self, station, self.dispatcher),
            "data": TrainingDataPage(self.content, self, station, self.dispatcher),
            "train": TrainPage(self.content, self, station, self.dispatcher),
            "parts": PartsPage(self.content, self, station, self.dispatcher),
            "history": HistoryPage(self.content, self, station),
            "settings": SettingsPage(self.content, self, station, self.dispatcher),
        }
        for page in self.pages.values():
            page.grid(row=1, column=0, sticky="nsew")
        self.inspect = self.pages["inspect"]

        self._install_triggers()
        self.go("inspect")
        self.set_navigation_locked(station.lock.locked)
        self.update_status()

    # -- sidebar -------------------------------------------------------------
    def _build_sidebar(self) -> None:
        side = ctk.CTkFrame(self, fg_color=theme.SIDEBAR, corner_radius=0, width=250)
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(len(NAV) + 2, weight=1)
        ctk.CTkLabel(side, text="FC-Comparator", font=theme.font(21, "bold"), text_color="#ffffff", anchor="w").grid(
            row=0, column=0, sticky="ew", padx=22, pady=(24, 0))
        ctk.CTkLabel(side, text=self.cfg.station.name, font=theme.font(13), text_color="#7c8aa5", anchor="w").grid(
            row=1, column=0, sticky="ew", padx=22, pady=(0, 20))
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for i, (key, label, icon, protected) in enumerate(NAV):
            text = f"  {icon}   {label}" + ("   \U0001f512" if protected else "")
            b = ctk.CTkButton(side, text=text, anchor="w", height=48, corner_radius=10, fg_color="transparent",
                              hover_color="#1e293b", text_color=theme.SIDEBAR_TEXT, font=theme.font(15),
                              command=lambda k=key: self.go(k))
            b.grid(row=i + 2, column=0, sticky="ew", padx=12, pady=2)
            self.nav_buttons[key] = b

        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.grid(row=len(NAV) + 3, column=0, sticky="ew", padx=12, pady=16)
        self.leave_btn = ctk.CTkButton(bottom, text="Leave setup mode", height=38, corner_radius=10, fg_color="#1e293b",
                                       hover_color="#334155", text_color="#ffffff", command=self.leave_admin)
        status = ctk.CTkFrame(bottom, fg_color="#172033", corner_radius=12)
        status.pack(fill="x", pady=(10, 10))
        self.status_title = ctk.CTkLabel(status, text="", font=theme.font(13, "bold"), text_color="#ffffff", anchor="w")
        self.status_title.pack(anchor="w", padx=12, pady=(10, 0))
        self.status_text = ctk.CTkLabel(status, text="", font=theme.font(12), text_color="#94a3b8", anchor="w",
                                        justify="left", wraplength=200)
        self.status_text.pack(anchor="w", padx=12, pady=(2, 10))
        self.mode_switch = ctk.CTkSwitch(bottom, text="Dark mode", text_color="#cbd5e1", command=self._toggle_mode)
        if ctk.get_appearance_mode() == "Dark":
            self.mode_switch.select()
        self.mode_switch.pack(anchor="w", padx=6)
        self._bottom = bottom

    def _toggle_mode(self) -> None:
        mode = "dark" if self.mode_switch.get() else "light"
        ctk.set_appearance_mode(mode)
        self.cfg.ui.appearance = mode
        theme.style_ttk(self.root)
        try:
            from ..config import save_config

            save_config(self.cfg)
        except Exception as exc:
            log.warning("Could not save appearance: %s", exc)

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
            except Exception as exc:
                log.error("GPIO foot pedal unavailable: %s", exc)

    def pedal_pressed(self) -> None:
        if self.current == "inspect":
            self.inspect.inspect()

    # -- navigation --------------------------------------------------------
    def go(self, key: str) -> bool:
        protected = next(p for k, _l, _i, p in NAV if k == key)
        if protected and not self.admin:
            if self.station.lock.locked:
                show_error(self, "The station is locked after an NG. Re-inspect or acknowledge it first.", "Station locked")
                return False
            pw = PinDialog.ask(self, "Setup password")
            if pw is None:
                return False
            if not verify_secret(pw, self.cfg.security.setup_password):
                self.station.store.log_event("setup_denied", "", "wrong setup password")
                show_error(self, "That password is not correct.", "Wrong password")
                return False
            self.admin = True
            self.leave_btn.pack(fill="x", before=self._bottom.winfo_children()[1])
            self.station.store.log_event("setup_login", "", "")
        if self.current and hasattr(self.pages[self.current], "on_hide"):
            self.pages[self.current].on_hide()
        page = self.pages[key]
        if hasattr(page, "refresh"):
            page.refresh()
        page.tkraise()
        self.current = key
        self.header.configure(text=page.title)
        for k, b in self.nav_buttons.items():
            active = k == key
            b.configure(fg_color=theme.SIDEBAR_ACTIVE if active else "transparent",
                        text_color="#ffffff" if active else theme.SIDEBAR_TEXT,
                        hover_color="#1d4ed8" if active else "#1e293b")
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
            b.configure(state="disabled" if locked and key not in ("inspect", "history") else "normal")

    def on_config_changed(self) -> None:
        self.inspect.refresh()
        self.update_status()

    def update_status(self) -> None:
        det = self.station.inspector.detector
        name = {"yolo": "Trained model", "template": "Template matching"}.get(det.name, det.name)
        self.status_title.configure(text=f"●  {name}" if det.ready else "●  Not ready")
        self.status_title.configure(text_color="#4ade80" if det.ready else "#f87171")
        self.status_text.configure(
            text=f"{det.note}\nParts: {len(self.cfg.parts)}  ·  alert: {self.cfg.alert.backend}\n"
                 f"Source: {self.cfg.camera.source}")

    def rebuild_detector(self, then=None) -> None:
        """Reload the detector in the background (template mining / model loading take seconds)."""
        self.status_title.configure(text="●  Updating detector ...", text_color="#facc15")

        def done(_r):
            self.update_status()
            if then:
                then()

        def failed(exc):
            self.update_status()
            show_error(self, f"Could not load the detector: {exc}")

        self.dispatcher.run_task(self.station.reload, done, failed)

    def shutdown(self) -> None:
        for page in self.pages.values():
            if hasattr(page, "stop"):
                page.stop()
        self.dispatcher.stop()
        if self._pedal is not None:
            self._pedal.close()
