"""Touch-friendly tkinter building blocks, styling and thread marshalling.

tkinter is not thread-safe: worker threads (inspection, GPIO pedal, lock
listeners) must never touch widgets. They hand callables to ``Dispatcher``,
which runs them on the Tk main loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import cv2
import numpy as np

try:  # Pillow is much faster than PPM for live preview; it is optional
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = ImageTk = None

log = logging.getLogger(__name__)

OK_COLOR = "#1e9e3a"
NG_COLOR = "#d62828"
IDLE_COLOR = "#4a5568"
WARN_COLOR = "#d98e04"
DARK_BG = "#1a202c"
PRIMARY = "#2b6cb0"


def font(size: int, bold: bool = False) -> tuple:
    """Font tuple in the platform's default UI family."""
    try:
        family = tkfont.nametofont("TkDefaultFont").actual("family")
    except (tk.TclError, RuntimeError):
        family = "Helvetica"
    return (family, size, "bold") if bold else (family, size)


def setup_style(root: tk.Tk, scale: float = 1.0) -> None:
    base = max(9, int(round(14 * scale)))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
        try:
            tkfont.nametofont(name).configure(size=base)
        except tk.TclError:
            pass
    family = tkfont.nametofont("TkDefaultFont").actual("family")
    bold = (family, base, "bold")
    root.option_add("*TCombobox*Listbox.font", (family, base + 2))
    style = ttk.Style(root)
    style.theme_use("clam")  # the only built-in theme that honours button colours everywhere
    style.configure("TButton", padding=(14, 12))
    style.configure("TEntry", padding=8)
    style.configure("TCombobox", padding=6)
    style.configure("TSpinbox", padding=6)
    style.configure("TLabelframe.Label", font=bold)
    style.configure("Primary.TButton", font=(family, base + 10, "bold"), padding=(14, 24), background=PRIMARY, foreground="white")
    style.map("Primary.TButton", background=[("disabled", "#a0aec0"), ("active", "#2c5282")])
    style.configure("Danger.TButton", font=bold, padding=(14, 16), background="#c53030", foreground="white")
    style.map("Danger.TButton", background=[("active", "#9b2c2c")])
    style.configure("Nav.TButton", font=bold, padding=(12, 24), anchor="w")
    style.configure("NavActive.TButton", font=bold, padding=(12, 24), anchor="w", background=PRIMARY, foreground="white")
    style.map("NavActive.TButton", background=[("active", "#2c5282")])
    style.configure("Treeview", rowheight=int(32 * scale))
    style.configure("Treeview.Heading", font=bold)
    style.configure("Muted.TLabel", foreground="#718096")


# -- images ------------------------------------------------------------------
def fit(img: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    """Resize to fit inside max_w x max_h keeping aspect; returns (image, scale)."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    if s <= 0:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp), s


def to_photo(img_bgr: np.ndarray) -> tk.PhotoImage:
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if ImageTk is not None:
        return ImageTk.PhotoImage(Image.fromarray(rgb))
    h, w = rgb.shape[:2]
    return tk.PhotoImage(data=b"P6 %d %d 255 " % (w, h) + rgb.tobytes(), format="ppm")


class ImageView(tk.Label):
    """Shows a BGR image scaled to the widget, keeping the aspect ratio."""

    def __init__(self, master, placeholder: str = "No image"):
        super().__init__(master, text=placeholder, bg=DARK_BG, fg="#a0aec0", font=font(18))
        self._img: np.ndarray | None = None
        self._photo = None
        self._pending = None
        self.bind("<Configure>", lambda _e: self._schedule())

    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            return
        self._img = img
        self._render()

    def clear(self, text: str = "") -> None:
        self._img = None
        self._photo = None
        self.configure(image="", text=text)

    def _schedule(self) -> None:
        if self._pending is None:
            self._pending = self.after(60, self._render)

    def _render(self) -> None:
        self._pending = None
        if self._img is None:
            return
        w, h = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        small, _ = fit(self._img, w - 4, h - 4)
        self._photo = to_photo(small)  # keep a reference or Tk frees the image
        self.configure(image=self._photo, text="")


class Banner(tk.Label):
    """Large OK / NG / status banner."""

    def __init__(self, master):
        super().__init__(master, fg="white", font=font(36, True), height=2, wraplength=540, justify="center")
        self.show_state("READY", IDLE_COLOR)

    def show_state(self, text: str, color: str, detail: str = "") -> None:
        if detail:
            self.configure(text=f"{text}\n{detail}", bg=color, font=font(26, True))
        else:
            self.configure(text=text, bg=color, font=font(40, True))

    @property
    def text(self) -> str:
        return str(self.cget("text"))


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame; put children into ``.body``."""

    def __init__(self, master, width: int | None = None):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=width or 400)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)
        self.body.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        # touch/mouse wheel scrolling while the pointer is over the panel
        self.body.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.body.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e) -> None:
        self.canvas.yview_scroll(int(-e.delta / 120), "units")


class PinDialog(tk.Toplevel):
    """Modal touch keypad for PINs / passwords (a physical keyboard works too)."""

    def __init__(self, master, title: str, numeric_only: bool = False):
        super().__init__(master)
        self.title(title)
        self.transient(master.winfo_toplevel())
        self.resizable(False, False)
        self.result: str | None = None
        self.value = tk.StringVar()
        ttk.Label(self, text=title, font=font(16, True)).grid(row=0, column=0, columnspan=3, pady=8)
        vcmd = (self.register(lambda s: s.isdigit() or s == ""), "%P") if numeric_only else None
        entry = ttk.Entry(self, textvariable=self.value, show="•", font=font(22), width=12, justify="center")
        if vcmd:
            entry.configure(validate="key", validatecommand=vcmd)
        entry.grid(row=1, column=0, columnspan=3, padx=10, pady=6, sticky="ew")
        keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "⌫", "0", "OK"]
        for i, k in enumerate(keys):
            ttk.Button(self, text=k, width=5, command=lambda key=k: self._key(key), takefocus=False).grid(
                row=2 + i // 3, column=i % 3, padx=4, pady=4, ipady=10
            )
        ttk.Button(self, text="Cancel", command=self.destroy).grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=8)
        entry.bind("<Return>", lambda _e: self._key("OK"))
        self.bind("<Escape>", lambda _e: self.destroy())
        entry.focus_set()

    def _key(self, key: str) -> None:
        if key == "OK":
            self.result = self.value.get()
            self.destroy()
        elif key == "⌫":
            self.value.set(self.value.get()[:-1])
        else:
            self.value.set(self.value.get() + key)

    @staticmethod
    def ask(master, title: str, numeric_only: bool = False) -> str | None:
        d = PinDialog(master, title, numeric_only)
        d.wait_visibility()
        d.grab_set()
        master.wait_window(d)
        return d.result


def error_box(parent, text: str, title: str = "Error") -> None:
    messagebox.showerror(title, text, parent=parent)


def info_box(parent, text: str, title: str = "Information") -> None:
    messagebox.showinfo(title, text, parent=parent)


def confirm(parent, text: str, title: str = "Confirm") -> bool:
    return bool(messagebox.askyesno(title, text, parent=parent))


class Dispatcher:
    """Run callables on the Tk main loop; ``call`` is safe from any thread."""

    def __init__(self, root: tk.Misc, interval_ms: int = 25):
        self.root = root
        self.interval = interval_ms
        self._q: queue.Queue = queue.Queue()
        self._alive = True
        root.after(interval_ms, self._pump)

    def call(self, fn: Callable, *args) -> None:
        self._q.put((fn, args))

    def stop(self) -> None:
        self._alive = False

    def _pump(self) -> None:
        while True:
            try:
                fn, args = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:
                log.exception("UI callback failed")
        if self._alive:
            self.root.after(self.interval, self._pump)

    def run_task(self, fn: Callable[[], object], on_done: Callable[[object], None], on_error: Callable[[BaseException], None]):
        """Run ``fn`` on a worker thread; deliver the result/exception on the main loop."""

        def worker():
            try:
                result = fn()
            except BaseException as exc:  # delivered to the UI, never lost
                self.call(on_error, exc)
            else:
                self.call(on_done, result)

        t = threading.Thread(target=worker, name="ui-task", daemon=True)
        t.start()
        return t


def labeled(parent, row: int, label: str, widget: tk.Widget, **grid) -> tk.Widget:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(4, 10), pady=4)
    widget.grid(row=row, column=1, sticky="ew", pady=4, **grid)
    return widget
