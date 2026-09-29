"""Start the station UI."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def run(config_path: str) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The UI needs PySide6:  pip install PySide6", file=sys.stderr)
        return 2

    from ..config import load_config
    from ..pipeline import Station
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("FC-Comparator")
    cfg = load_config(config_path)
    station = Station(cfg)
    station.open()
    win = MainWindow(station)
    if cfg.ui.fullscreen:
        win.showFullScreen()
    else:
        win.showMaximized()
    try:
        return app.exec()
    finally:
        station.close()
