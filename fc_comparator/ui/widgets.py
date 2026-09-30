"""Small building blocks shared by the pages (plain tkinter/ttk, no image libraries besides OpenCV)."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import messagebox, ttk

import cv2
import numpy as np

from . import style

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threads -> Tk main loop
# ---------------------------------------------------------------------------
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
            try:
                self.root.after(self.interval, self._pump)
            except tk.TclError:  # window closed
                pass

    def run_task(self, fn: Callable[[], object], on_done: Callable[[object], None],
                 on_error: Callable[[BaseException], None]) -> threading.Thread:
        """Run ``fn`` on a worker thread; deliver the result or exception on the main loop."""

        def worker():
            try:
                result = fn()
            except BaseException as exc:  # delivered to the UI, never lost
                log.exception("Background task failed")
                self.call(on_error, exc)
            else:
                self.call(on_done, result)

        t = threading.Thread(target=worker, name="ui-task", daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def to_photo(img_bgr: np.ndarray, max_w: int, max_h: int) -> tuple[tk.PhotoImage, float]:
    """Scale a BGR image to fit (max_w, max_h) and wrap it as a Tk PhotoImage (PPM, no Pillow needed)."""
    h, w = img_bgr.shape[:2]
    k = min(max_w / w, max_h / h)
    if abs(k - 1) > 1e-3:
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * k)), max(1, int(h * k))),
                             interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    header = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255 ".encode()
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM"), k


class ImageView(tk.Canvas):
    """Shows one image scaled to fit, or a placeholder text."""

    def __init__(self, master, placeholder: str = "", **kw):
        kw.setdefault("bg", style.CANVAS_BG)
        super().__init__(master, highlightthickness=0, **kw)
        self.image: np.ndarray | None = None
        self.placeholder = placeholder
        self._photo: tk.PhotoImage | None = None
        self.bind("<Configure>", lambda _e: self._draw())

    def set_image(self, image: np.ndarray | None) -> None:
        self.image = image
        self._draw()

    def clear(self, text: str | None = None) -> None:
        if text is not None:
            self.placeholder = text
        self.set_image(None)

    def _draw(self) -> None:
        self.delete("all")
        w, h = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self.image is None:
            self._photo = None
            self.create_text(w // 2, h // 2, text=self.placeholder, fill="#9ca3af", font=style.font(12),
                             justify="center", width=max(100, w - 40))
            return
        if w < 10 or h < 10:
            return
        self._photo, _k = to_photo(self.image, w, h)
        self.create_image(w // 2, h // 2, image=self._photo, anchor="center")


# ---------------------------------------------------------------------------
# Result banner, tables, forms
# ---------------------------------------------------------------------------
class Banner(tk.Frame):
    """Big coloured verdict box: OK / NG / READY ..."""

    def __init__(self, master):
        super().__init__(master, bg=style.IDLE)
        self.title = tk.Label(self, text="READY", bg=style.IDLE, fg="#ffffff", font=style.font(40, "bold"))
        self.title.pack(padx=16, pady=(14, 0))
        self.detail = tk.Label(self, text="", bg=style.IDLE, fg="#ffffff", font=style.font(11), wraplength=360,
                               justify="center")
        self.detail.pack(padx=16, pady=(0, 14))

    def show(self, title: str, color: str, detail: str = "") -> None:
        for w in (self, self.title, self.detail):
            w.configure(bg=color)
        self.title.configure(text=title)
        self.detail.configure(text=detail)

    @property
    def text(self) -> str:
        return str(self.title.cget("text"))


class Table(ttk.Frame):
    """Treeview with a vertical scrollbar. ``columns`` = [(key, heading, width), ...]."""

    def __init__(self, master, columns: Sequence[tuple[str, str, int]], height: int = 10,
                 on_select: Callable[[str | None], None] | None = None, selectmode: str = "browse"):
        super().__init__(master)
        self.tree = ttk.Treeview(self, columns=[c[0] for c in columns], show="headings", height=height,
                                 selectmode=selectmode)
        for key, heading, width in columns:
            self.tree.heading(key, text=heading, anchor="w")
            self.tree.column(key, width=width, anchor="w", stretch=True)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.tree.tag_configure("ng", foreground=style.NG)
        self.tree.tag_configure("ok", foreground=style.OK)
        self.tree.tag_configure("muted", foreground=style.MUTED)
        self._on_select = on_select
        if on_select:
            self.tree.bind("<<TreeviewSelect>>", lambda _e: on_select(self.selected()))

    def set_rows(self, rows: Sequence[tuple[str, Sequence, Sequence[str]]], keep_selection: bool = True) -> None:
        """rows = [(iid, values, tags), ...]"""
        sel = self.selected() if keep_selection else None
        self.tree.delete(*self.tree.get_children())
        for iid, values, tags in rows:
            self.tree.insert("", "end", iid=iid, values=list(values), tags=tuple(tags))
        if sel and self.tree.exists(sel):
            self.tree.selection_set(sel)
            self.tree.see(sel)

    def selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def select(self, iid: str) -> None:
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)


def card(master, title: str = "", subtitle: str = "") -> ttk.Frame:
    """A white panel with an optional heading. Returns the body frame; the panel is ``body.panel``."""
    panel = ttk.Frame(master, style="Card.TFrame", padding=12)
    if title:
        ttk.Label(panel, text=title, style="CardTitle.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(panel, text=subtitle, style="CardMuted.TLabel", wraplength=420, justify="left").pack(anchor="w")
    body = ttk.Frame(panel, style="Card.TFrame", padding=0)
    body.configure(borderwidth=0)
    body.pack(fill="both", expand=True, pady=(8 if title else 0, 0))
    body.panel = panel  # type: ignore[attr-defined]
    return body


def text_box(master, height: int = 8, **kw) -> tk.Text:
    t = tk.Text(master, height=height, font=style.MONO, wrap="word", relief="solid", borderwidth=1,
                highlightthickness=0, padx=6, pady=4, **kw)
    t.configure(state="disabled")
    return t


def set_text(box: tk.Text, text: str, append: bool = False) -> None:
    box.configure(state="normal")
    if not append:
        box.delete("1.0", "end")
    box.insert("end", text)
    box.see("end")
    box.configure(state="disabled")


# ---------------------------------------------------------------------------
# Dialogs (thin wrappers so tests can replace them)
# ---------------------------------------------------------------------------
def show_info(parent, text: str, title: str = "Done") -> None:
    messagebox.showinfo(title, text, parent=parent)


def show_error(parent, text: str, title: str = "Error") -> None:
    messagebox.showerror(title, text, parent=parent)


def ask_yes_no(parent, text: str, title: str = "Please confirm") -> bool:
    return bool(messagebox.askyesno(title, text, parent=parent))


def center_window(win: tk.Toplevel, parent: tk.Misc | None = None, width: int | None = None,
                  height: int | None = None) -> None:
    """Place ``win`` in the centre of the application window (or of the screen), kept on screen."""
    win.update_idletasks()
    w = width or win.winfo_reqwidth()
    h = height or win.winfo_reqheight()
    top = parent.winfo_toplevel() if parent is not None else None
    if top is not None and top.winfo_viewable():
        cx, cy = top.winfo_rootx() + top.winfo_width() // 2, top.winfo_rooty() + top.winfo_height() // 2
    else:
        cx, cy = win.winfo_screenwidth() // 2, win.winfo_screenheight() // 2
    x = max(0, min(cx - w // 2, win.winfo_screenwidth() - w))
    y = max(0, min(cy - h // 2, win.winfo_screenheight() - h))
    win.geometry(f"{w}x{h}+{x}+{y}")


class InputDialog(tk.Toplevel):
    """One-line input (text, password or PIN), modal and centred on the application window."""

    def __init__(self, parent, title: str, prompt: str, initial: str = "", secret: bool = False):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.resizable(False, False)
        self.configure(bg=style.BG)
        self.transient(parent.winfo_toplevel())
        self.result: str | None = None
        body = ttk.Frame(self, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=prompt).pack(anchor="w")
        self.entry = ttk.Entry(body, width=34, font=style.font(12), show="*" if secret else "")
        self.entry.insert(0, initial)
        self.entry.pack(fill="x", pady=(4, 12))
        row = ttk.Frame(body)
        row.pack(fill="x")
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(row, text="OK", style="Accent.TButton", command=self._ok).pack(side="right", padx=6)
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        center_window(self, parent)
        self.deiconify()
        self.entry.focus_set()
        self.entry.select_range(0, "end")

    def _ok(self) -> None:
        self.result = self.entry.get()
        self.destroy()

    def run(self) -> str | None:
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.master.wait_window(self)
        return self.result


def ask_string(parent, title: str, prompt: str, initial: str = "", secret: bool = False) -> str | None:
    return InputDialog(parent, title, prompt, initial, secret).run()
