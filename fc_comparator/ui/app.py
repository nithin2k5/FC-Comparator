"""Start the station UI."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def run(config_path: str) -> int:
    try:
        import customtkinter as ctk
    except ImportError:
        print("The UI needs CustomTkinter:  pip install customtkinter", file=sys.stderr)
        return 2

    from ..config import load_config
    from ..pipeline import Station
    from .main_window import MainWindow

    cfg = load_config(config_path)
    ctk.set_appearance_mode(cfg.ui.appearance if cfg.ui.appearance in ("light", "dark", "system") else "light")
    ctk.set_default_color_theme("blue")
    if cfg.ui.scale and cfg.ui.scale != 1.0:
        ctk.set_widget_scaling(cfg.ui.scale)

    root = ctk.CTk()
    root.withdraw()  # show the window only once the station UI is built
    root.minsize(1200, 760)
    station = Station(cfg)
    station.open()
    win = MainWindow(root, station)
    win.pack(fill="both", expand=True)
    root.deiconify()

    if cfg.ui.fullscreen:
        root.attributes("-fullscreen", True)
    else:
        try:
            root.after(0, lambda: root.state("zoomed"))  # Windows / macOS
        except Exception:
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
