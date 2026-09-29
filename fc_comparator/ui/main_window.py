"""Main window: navigation rail + screens. Setup, Part numbers and Settings need the setup password."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from ..pipeline import Station
from ..security import verify_secret
from .history_screen import HistoryScreen
from .inspect_screen import InspectScreen
from .parts_screen import PartsScreen
from .settings_screen import SettingsScreen
from .setup_screen import SetupScreen
from .widgets import STYLE, PinDialog, big_button, error_box

log = logging.getLogger(__name__)

PROTECTED = {"setup", "parts", "settings"}


class MainWindow(QMainWindow):
    def __init__(self, station: Station):
        super().__init__()
        self.station = station
        self.cfg = station.cfg
        self.setWindowTitle(f"FC-Comparator - {self.cfg.station.name}")
        self.setStyleSheet(STYLE)
        self._admin = False
        self._pedal = None

        self.inspect = InspectScreen(station)
        self.setup = SetupScreen(station)
        self.parts = PartsScreen(station)
        self.history = HistoryScreen(station)
        self.settings = SettingsScreen(station)
        self.pages = {
            "inspect": self.inspect, "setup": self.setup, "parts": self.parts,
            "history": self.history, "settings": self.settings,
        }
        self.stack = QStackedWidget()
        for w in self.pages.values():
            self.stack.addWidget(w)
        for w in (self.setup, self.parts, self.settings):
            w.config_changed.connect(self._on_config_changed)

        nav = QVBoxLayout()
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = {}
        for key, text in (("inspect", "Inspect"), ("setup", "Setup / Teach"), ("parts", "Part numbers"),
                          ("history", "History / Reports"), ("settings", "Settings")):
            b = big_button(text, checkable=True)
            b.setMinimumHeight(80)
            b.clicked.connect(lambda _=False, k=key: self.go(k))
            self.nav_group.addButton(b)
            self.nav_buttons[key] = b
            nav.addWidget(b)
        nav.addStretch(1)
        self.lock_admin_btn = big_button("Leave setup mode")
        self.lock_admin_btn.clicked.connect(self._leave_admin)
        self.lock_admin_btn.hide()
        nav.addWidget(self.lock_admin_btn)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #4a5568; font-size: 13px;")
        nav.addWidget(self.status)
        nav_w = QWidget()
        nav_w.setLayout(nav)
        nav_w.setFixedWidth(230)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(nav_w)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self._install_triggers()
        self.go("inspect")
        self.set_navigation_locked(self.station.lock.locked)
        self._update_status()

    # -- triggers (foot pedal) -------------------------------------------------
    def _install_triggers(self) -> None:
        sc = QShortcut(QKeySequence(self.cfg.trigger.key), self)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(self._pedal_pressed)
        self._shortcut = sc
        if self.cfg.trigger.gpio_pin is not None:
            try:
                from ..alert.gpio import GpioPedal

                # The gpiozero callback runs on its own thread: go through a Qt signal.
                self._pedal = GpioPedal(self.cfg.trigger.gpio_pin, self.inspect.trigger_requested.emit)
                log.info("Foot pedal on GPIO %s", self.cfg.trigger.gpio_pin)
            except Exception as exc:
                log.error("GPIO foot pedal unavailable: %s", exc)

    def _pedal_pressed(self) -> None:
        if self.stack.currentWidget() is self.inspect:
            self.inspect.inspect()

    # -- navigation --------------------------------------------------------
    def go(self, key: str) -> None:
        if key in PROTECTED and not self._admin:
            if self.station.lock.locked:
                error_box(self, "Station is locked after an NG. Resolve it first.")
                self.nav_buttons[self._current_key()].setChecked(True)
                return
            pw = PinDialog.ask(self, "Setup password")
            if pw is None or not verify_secret(pw, self.cfg.security.setup_password):
                if pw is not None:
                    self.station.store.log_event("setup_denied", "", "wrong setup password")
                    error_box(self, "Wrong password.")
                self.nav_buttons[self._current_key()].setChecked(True)
                return
            self._admin = True
            self.lock_admin_btn.show()
            self.station.store.log_event("setup_login", "", "")
        page = self.pages[key]
        if hasattr(page, "refresh"):
            page.refresh()
        self.stack.setCurrentWidget(page)
        self.nav_buttons[key].setChecked(True)
        if key == "inspect":
            self.inspect.focus_scan()
        self._update_status()

    def _current_key(self) -> str:
        cur = self.stack.currentWidget()
        return next((k for k, w in self.pages.items() if w is cur), "inspect")

    def _leave_admin(self) -> None:
        self._admin = False
        self.lock_admin_btn.hide()
        self.go("inspect")

    def set_navigation_locked(self, locked: bool) -> None:
        for key, b in self.nav_buttons.items():
            b.setEnabled(not locked or key in ("inspect", "history"))

    def _on_config_changed(self) -> None:
        self.inspect.refresh_parts()
        self._update_status()

    def _update_status(self) -> None:
        clf = self.station.inspector.classifier
        align = "on" if self.station.inspector.aligner is not None else "off"
        self.status.setText(
            f"Classifier: {clf.name}\nAlignment: {align}\nAlert: {self.cfg.alert.backend}\n"
            f"Mode: {self.cfg.compare.mode}\nSource: {self.cfg.camera.source}"
        )

    def closeEvent(self, event):  # noqa: N802
        self.inspect.stop()
        if self._pedal is not None:
            self._pedal.close()
        super().closeEvent(event)
