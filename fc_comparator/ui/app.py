"""Start the station UI (tkinter, part of the Python standard library)."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def run(config_path: str) -> int:
    try:
        import tkinter as tk
    except ImportError:
        print("The UI needs tkinter (Debian/Raspberry Pi OS: sudo apt install python3-tk)", file=sys.stderr)
        return 2

    from ..config import load_config
    from ..pipeline import Station
    from .main_window import MainWindow
    from .widgets import setup_style

    cfg = load_config(config_path)
    if sys.platform == "win32":
        try:  # crisp rendering on scaled displays instead of bitmap-stretched
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.minsize(1024, 700)
    setup_style(root, cfg.ui.font_scale)
    station = Station(cfg)
    station.open()
    win = MainWindow(root, station)
    win.pack(fill="both", expand=True)

    if cfg.ui.fullscreen:
        root.attributes("-fullscreen", True)
    else:
        try:
            root.state("zoomed")  # Windows / macOS
        except tk.TclError:
            try:
                root.attributes("-zoomed", True)  # X11
            except tk.TclError:
                pass

    def on_close() -> None:
        win.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
    try:
        root.mainloop()
    finally:
        station.close()
    return 0
