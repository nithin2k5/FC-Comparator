"""Start the station UI: login screen, then the main window."""

from __future__ import annotations

import logging
import sys
import tkinter as tk

log = logging.getLogger(__name__)


def _dpi_aware() -> None:
    """Sharp text on scaled Windows displays (otherwise Windows stretches the whole window)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # older Windows
        pass


def run(config_path: str) -> int:
    from ..config import load_config
    from ..station import Station
    from . import style
    from .login import LoginPage
    from .main_window import MainWindow
    from .widgets import center_window

    cfg = load_config(config_path)
    _dpi_aware()
    root = tk.Tk()
    root.title(f"FC-Comparator - {cfg.station.name}")
    root.minsize(1200, 760)
    style.apply(root)
    station = Station(cfg)
    station.open()
    win: MainWindow | None = None

    def login(user: str) -> None:
        nonlocal win
        login_page.pack_forget()
        win = MainWindow(root, station, user=user, on_logout=logout)
        win.pack(fill="both", expand=True)

    def logout() -> None:
        nonlocal win
        if win is not None:
            win.shutdown()
            win.destroy()
            win = None
        login_page.reset()
        login_page.pack(fill="both", expand=True)

    login_page = LoginPage(root, cfg, login, station.store.log_event)
    login_page.pack(fill="both", expand=True)
    login_page.reset()

    if cfg.ui.fullscreen:
        root.attributes("-fullscreen", True)
    else:
        try:
            root.state("zoomed")  # Windows
        except tk.TclError:
            center_window(root, None, 1400, 900)

    def on_close() -> None:
        if win is not None:
            win.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
    try:
        root.mainloop()
    finally:
        station.close()
    return 0
