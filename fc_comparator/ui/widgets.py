"""Reusable widgets: cards, image view, verdict banner, dialogs, part picker, tables.

tkinter is not thread-safe: worker threads (inspection, training, detector
rebuilds, GPIO pedal, lock listeners) never touch widgets. They hand callables
to ``Dispatcher``, which runs them on the Tk main loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import ttk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from . import theme

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threading
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
        """Run ``fn`` on a worker thread; deliver the result/exception on the main loop."""

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
# Layout helpers
# ---------------------------------------------------------------------------
class Card(ctk.CTkFrame):
    """A rounded surface with an optional title; put children into ``.body``."""

    def __init__(self, master, title: str = "", subtitle: str = "", **kw):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=14, border_width=1,
                         border_color=theme.BORDER, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        if title:
            head = ctk.CTkFrame(self, fg_color="transparent")
            head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
            ctk.CTkLabel(head, text=title, font=theme.font(15, "bold"), text_color=theme.TEXT, anchor="w").pack(anchor="w")
            if subtitle:
                ctk.CTkLabel(head, text=subtitle, font=theme.font(12), text_color=theme.MUTED, anchor="w",
                             justify="left", wraplength=520).pack(anchor="w")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(4 if title else 14, 14))


def page_header(master, title: str, subtitle: str = "") -> ctk.CTkFrame:
    f = ctk.CTkFrame(master, fg_color="transparent")
    ctk.CTkLabel(f, text=title, font=theme.font(24, "bold"), text_color=theme.TEXT, anchor="w").pack(anchor="w")
    if subtitle:
        ctk.CTkLabel(f, text=subtitle, font=theme.font(13), text_color=theme.MUTED, anchor="w").pack(anchor="w")
    return f


def button(master, text: str, command=None, kind: str = "secondary", **kw) -> ctk.CTkButton:
    """kind: primary | secondary | danger | success | ghost"""
    styles = {
        "primary": dict(fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER, text_color="#ffffff"),
        "success": dict(fg_color=theme.OK, hover_color=theme.OK_HOVER, text_color="#ffffff"),
        "danger": dict(fg_color=theme.NG, hover_color=theme.NG_HOVER, text_color="#ffffff"),
        "secondary": dict(fg_color=theme.SURFACE_2, hover_color=theme.BORDER, text_color=theme.TEXT,
                          border_width=1, border_color=theme.BORDER),
        "ghost": dict(fg_color="transparent", hover_color=theme.SURFACE_2, text_color=theme.TEXT),
    }
    opts = {"height": 40, "corner_radius": 10, "font": theme.font(14, "bold" if kind != "ghost" else "normal")}
    opts.update(styles[kind])
    opts.update(kw)
    return ctk.CTkButton(master, text=text, command=command, **opts)


def muted(master, text: str = "", size: int = 12, **kw) -> ctk.CTkLabel:
    return ctk.CTkLabel(master, text=text, font=theme.font(size), text_color=theme.MUTED, anchor="w", justify="left", **kw)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def widget_scaling(widget: tk.Misc) -> float:
    try:
        return ctk.ScalingTracker.get_widget_scaling(widget)
    except Exception:
        return 1.0


def fit(img: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    if s <= 0:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp), s


def to_pil(img_bgr: np.ndarray) -> Image.Image:
    if img_bgr.ndim == 2:
        return Image.fromarray(img_bgr)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


class ImageView(ctk.CTkLabel):
    """Shows a BGR image scaled to fit, keeping the aspect ratio (HiDPI aware)."""

    def __init__(self, master, placeholder: str = "No image", **kw):
        super().__init__(master, text=placeholder, fg_color=theme.CANVAS_BG, corner_radius=12,
                         text_color="#94a3b8", font=theme.font(16), **kw)
        self._img: np.ndarray | None = None
        self._ctk_image = None
        self._pending = None
        self.bind("<Configure>", lambda _e: self._schedule())

    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            return
        self._img = img
        self._render()

    def clear(self, text: str = "") -> None:
        self._img = None
        self._ctk_image = None
        self.configure(image=None, text=text)

    def _schedule(self) -> None:
        if self._pending is None:
            self._pending = self.after(80, self._render)

    def _render(self) -> None:
        self._pending = None
        if self._img is None:
            return
        s = widget_scaling(self)
        w, h = max(self.winfo_width(), 60), max(self.winfo_height(), 60)
        small, _ = fit(self._img, w - 8, h - 8)  # physical pixels: sharp on scaled displays
        pil = to_pil(small)
        self._ctk_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=(pil.width / s, pil.height / s))
        self.configure(image=self._ctk_image, text="")


class Banner(ctk.CTkFrame):
    """Large OK / NG / status banner."""

    def __init__(self, master, **kw):
        super().__init__(master, corner_radius=14, fg_color=theme.IDLE, **kw)
        self.title = ctk.CTkLabel(self, text="READY", font=theme.font(46, "bold"), text_color="#ffffff")
        self.title.pack(pady=(14, 0))
        self.detail = ctk.CTkLabel(self, text="", font=theme.font(15), text_color="#ffffff", wraplength=380)
        self.detail.pack(pady=(0, 14))

    def show(self, text: str, color, detail: str = "") -> None:
        self.configure(fg_color=color)
        self.title.configure(text=text)
        self.detail.configure(text=detail)

    @property
    def text(self) -> str:
        return str(self.title.cget("text"))


def chip(master, text: str, color: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(master, text=f" {text} ", fg_color=color, text_color="#ffffff", corner_radius=8,
                        font=theme.font(12, "bold"), height=24)


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------
class ModalDialog(ctk.CTkToplevel):
    def __init__(self, master, title: str):
        super().__init__(master)
        self.title(title)
        self.configure(fg_color=theme.SURFACE)
        self.resizable(False, False)
        self.transient(master.winfo_toplevel())
        self.result = None
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda _e: self.cancel())

    def cancel(self) -> None:
        self.result = None
        self.destroy()

    def run(self):
        self.update_idletasks()
        top = self.master.winfo_toplevel()
        x = top.winfo_rootx() + (top.winfo_width() - self.winfo_width()) // 2
        y = top.winfo_rooty() + (top.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.focus_force()
        self.master.wait_window(self)
        return self.result


class MessageDialog(ModalDialog):
    def __init__(self, master, title: str, text: str, kind: str = "info", buttons: Sequence[tuple[str, object, str]] = ()):
        super().__init__(master, title)
        color = {"info": theme.ACCENT, "error": theme.NG, "warning": theme.WARN, "question": theme.ACCENT}[kind]
        icon = {"info": "i", "error": "!", "warning": "!", "question": "?"}[kind]
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(padx=24, pady=(22, 12), fill="both")
        ctk.CTkLabel(body, text=icon, width=44, height=44, corner_radius=22, fg_color=color, text_color="#ffffff",
                     font=theme.font(22, "bold")).pack(side="left", anchor="n", padx=(0, 16))
        text_box = ctk.CTkFrame(body, fg_color="transparent")
        text_box.pack(side="left", fill="both")
        ctk.CTkLabel(text_box, text=title, font=theme.font(16, "bold"), text_color=theme.TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(text_box, text=text, font=theme.font(13), text_color=theme.TEXT, anchor="w", justify="left",
                     wraplength=420).pack(anchor="w", pady=(4, 0))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(4, 20))
        for label, value, kind_ in reversed(buttons or [("OK", True, "primary")]):
            button(row, label, lambda v=value: self._choose(v), kind_, width=110).pack(side="right", padx=(8, 0))
        self.bind("<Return>", lambda _e: self._choose((buttons or [("OK", True, "")])[0][1]))

    def _choose(self, value) -> None:
        self.result = value
        self.destroy()


def show_info(master, text: str, title: str = "Done") -> None:
    MessageDialog(master, title, text, "info").run()


def show_error(master, text: str, title: str = "Something went wrong") -> None:
    MessageDialog(master, title, text, "error").run()


def ask_yes_no(master, text: str, title: str = "Please confirm", yes: str = "Yes", danger: bool = False) -> bool:
    buttons = [(yes, True, "danger" if danger else "primary"), ("Cancel", False, "secondary")]
    return bool(MessageDialog(master, title, text, "question", buttons).run())


class InputDialog(ModalDialog):
    def __init__(self, master, title: str, fields: Sequence[tuple[str, str]], ok_text: str = "OK"):
        """fields: (label, initial value)"""
        super().__init__(master, title)
        ctk.CTkLabel(self, text=title, font=theme.font(16, "bold"), text_color=theme.TEXT).pack(anchor="w", padx=24, pady=(20, 8))
        self.entries = []
        for label, value in fields:
            muted(self, label).pack(anchor="w", padx=24)
            e = ctk.CTkEntry(self, width=360, height=40, font=theme.font(14))
            e.insert(0, value)
            e.pack(padx=24, pady=(2, 10))
            self.entries.append(e)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(4, 20))
        button(row, ok_text, self._ok, "primary", width=110).pack(side="right", padx=(8, 0))
        button(row, "Cancel", self.cancel, "secondary", width=110).pack(side="right")
        self.bind("<Return>", lambda _e: self._ok())
        if self.entries:
            self.after(50, self.entries[0].focus_set)

    def _ok(self) -> None:
        self.result = [e.get().strip() for e in self.entries]
        self.destroy()


def ask_fields(master, title: str, fields: Sequence[tuple[str, str]], ok_text: str = "OK") -> list[str] | None:
    return InputDialog(master, title, fields, ok_text).run()


class PinDialog(ModalDialog):
    """Touch keypad for PINs/passwords (a physical keyboard works too)."""

    def __init__(self, master, title: str, numeric_only: bool = True):
        super().__init__(master, title)
        self.numeric_only = numeric_only
        ctk.CTkLabel(self, text=title, font=theme.font(17, "bold"), text_color=theme.TEXT).pack(pady=(20, 8))
        self.var = tk.StringVar()
        self.entry = ctk.CTkEntry(self, textvariable=self.var, show="•", width=260, height=52, justify="center",
                                  font=theme.font(26))
        self.entry.pack(padx=24)
        self.entry.bind("<Return>", lambda _e: self._key("OK"))
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(padx=24, pady=14)
        for i, k in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", "⌫", "0", "OK"]):
            kind = "primary" if k == "OK" else "secondary"
            button(grid, k, lambda key=k: self._key(key), kind, width=80, height=62, font=theme.font(22, "bold")).grid(
                row=i // 3, column=i % 3, padx=5, pady=5)
        button(self, "Cancel", self.cancel, "ghost", width=260).pack(pady=(0, 18))
        self.after(50, self.entry.focus_set)

    def _key(self, key: str) -> None:
        if key == "OK":
            self.result = self.var.get()
            self.destroy()
        elif key == "⌫":
            self.var.set(self.var.get()[:-1])
        else:
            self.var.set(self.var.get() + key)

    @staticmethod
    def ask(master, title: str, numeric_only: bool = True) -> str | None:
        return PinDialog(master, title, numeric_only).run()


class PartPicker(ModalDialog):
    """Searchable part list - works with hundreds of part numbers."""

    def __init__(self, master, parts: dict, title: str = "Select part number"):
        super().__init__(master, title)
        self.parts = parts
        self.geometry("520x560")
        ctk.CTkLabel(self, text=title, font=theme.font(17, "bold"), text_color=theme.TEXT).pack(anchor="w", padx=20, pady=(18, 6))
        self.search = ctk.CTkEntry(self, placeholder_text="Search part number or description", height=42, font=theme.font(14))
        self.search.pack(fill="x", padx=20)
        self.search.bind("<KeyRelease>", lambda _e: self._fill())
        self.search.bind("<Return>", lambda _e: self._pick_first())
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=12)
        self.table = Table(frame, [("code", "Part number", 140), ("desc", "Description", 260), ("size", "Layout", 80)])
        self.table.pack(fill="both", expand=True)
        self.table.tree.bind("<Double-1>", lambda _e: self._pick_selected())
        self.table.tree.bind("<Return>", lambda _e: self._pick_selected())
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(0, 18))
        button(row, "Select", self._pick_selected, "primary", width=120).pack(side="right", padx=(8, 0))
        button(row, "Cancel", self.cancel, "secondary", width=120).pack(side="right")
        self._fill()
        self.after(50, self.search.focus_set)

    def _matches(self) -> list[str]:
        q = self.search.get().strip().lower()
        return [c for c in sorted(self.parts) if not q or q in c.lower() or q in self.parts[c].description.lower()]

    def _fill(self) -> None:
        self.table.set_rows([(c, (c, self.parts[c].description, f"{self.parts[c].cables}x{self.parts[c].rows}"), ())
                             for c in self._matches()])

    def _pick_first(self) -> None:
        m = self._matches()
        if m:
            self.result = m[0]
            self.destroy()

    def _pick_selected(self) -> None:
        sel = self.table.selected()
        if sel:
            self.result = sel
            self.destroy()


# ---------------------------------------------------------------------------
# Table (ttk.Treeview styled like the rest; fast for long lists)
# ---------------------------------------------------------------------------
class Table(ctk.CTkFrame):
    def __init__(self, master, columns: Sequence[tuple[str, str, int]], height: int = 10, **kw):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=10, border_width=1, border_color=theme.BORDER, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self, columns=[c[0] for c in columns], show="headings", height=height, selectmode="browse")
        for key, text, width in columns:
            self.tree.heading(key, text=text, anchor="w")
            self.tree.column(key, width=width, minwidth=40, anchor="w", stretch=True)
        self.tree.tag_configure("ng", foreground=theme.NG)
        self.tree.tag_configure("ok", foreground=theme.OK)
        self.tree.tag_configure("muted", foreground="#94a3b8")
        bar = ctk.CTkScrollbar(self, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        bar.grid(row=0, column=1, sticky="ns", pady=6)

    def set_rows(self, rows: Sequence[tuple[str, Sequence, Sequence[str]]], keep_selection: bool = True) -> None:
        """rows: (iid, values, tags)"""
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
            self.tree.see(iid)

    def on_select(self, fn: Callable[[str | None], None]) -> None:
        self.tree.bind("<<TreeviewSelect>>", lambda _e: fn(self.selected()))
